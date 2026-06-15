"""Verify the admin setDefaultModel action end to end (developers-group gated).

Sequence (uses the daonkenti-dev profile for admin group ops + manifest read):
  1. SRP login as the test user -> access token
  2. remove user from 'developers', call setDefault -> expect 403 (valid token,
     but not a developer — proves the server-side group check, not just UI gating)
  3. add user to 'developers', call setDefault(<target>) -> 200, manifest.default
     updated (the Lambda checks LIVE group membership, so no re-login needed)
  4. no token / bad token -> 403
  5. restore the original default

Usage: python tools/verify_set_default.py <email> <password> <target_model_id>
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.credentials import Credentials
from pycognito import Cognito

ROOT = Path(__file__).resolve().parent.parent
PROFILE = "daonkenti-dev"
GROUP = "developers"


def main() -> None:
    if len(sys.argv) < 4:
        sys.exit("usage: python tools/verify_set_default.py <email> <password> <target_model_id>")
    email, pw, target = sys.argv[1], sys.argv[2], sys.argv[3]

    out = json.loads((ROOT / "amplify_outputs.json").read_text(encoding="utf-8"))
    region = out["custom"].get("inferenceRegion") or out["storage"]["aws_region"]
    url = out["custom"]["inferenceUrl"]
    bucket = out["storage"]["bucket_name"]
    pool = out["auth"]["user_pool_id"]
    client = out["auth"]["user_pool_client_id"]
    idpool = out["auth"]["identity_pool_id"]

    u = Cognito(pool, client, username=email, user_pool_region=region)
    u.authenticate(password=pw)
    access_token = u.access_token
    print("SRP login OK")

    admin = boto3.Session(profile_name=PROFILE, region_name=region)
    s3 = admin.client("s3")
    idp = admin.client("cognito-idp")
    ci = admin.client("cognito-identity")
    prov = f"cognito-idp.{region}.amazonaws.com/{pool}"
    iid = ci.get_id(IdentityPoolId=idpool, Logins={prov: u.id_token})["IdentityId"]
    c = ci.get_credentials_for_identity(IdentityId=iid, Logins={prov: u.id_token})["Credentials"]
    creds = Credentials(c["AccessKeyId"], c["SecretKey"], c["SessionToken"])

    def manifest_default():
        obj = s3.get_object(Bucket=bucket, Key="models/manifest.json")
        return json.loads(obj["Body"].read()).get("default")

    def call(body):
        payload = json.dumps(body).encode()
        req = AWSRequest(method="POST", url=url, data=payload, headers={"content-type": "application/json"})
        SigV4Auth(creds, "lambda", region).add_auth(req)
        p = req.prepare()
        r = urllib.request.Request(p.url, data=payload, headers=dict(p.headers), method="POST")
        try:
            with urllib.request.urlopen(r, timeout=60) as resp:
                return resp.status, resp.read().decode()
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode()

    original = manifest_default()
    print(f"default before: {original}  | target: {target}")
    fails = []

    # 2. not a developer -> 403
    try:
        idp.admin_remove_user_from_group(UserPoolId=pool, Username=email, GroupName=GROUP)
    except Exception:  # noqa: BLE001
        pass
    st, body = call({"action": "setDefaultModel", "modelId": target, "accessToken": access_token})
    print(f"[non-developer]  status={st} (expect 403)  {body[:80]}")
    if st != 403:
        fails.append(f"non-developer should be 403, got {st}")

    # 3. developer -> 200 + manifest updated (live group check, same token)
    idp.admin_add_user_to_group(UserPoolId=pool, Username=email, GroupName=GROUP)
    st, body = call({"action": "setDefaultModel", "modelId": target, "accessToken": access_token})
    print(f"[developer]      status={st} (expect 200)  {body[:80]}")
    if st != 200:
        fails.append(f"developer setDefault failed: {st} {body}")
    if manifest_default() != target:
        fails.append(f"manifest.default not updated (={manifest_default()})")
    else:
        print(f"  -> manifest.default is now '{target}'")

    # 4. no token / bad token -> 403
    st, _ = call({"action": "setDefaultModel", "modelId": target})
    print(f"[no token]       status={st} (expect 403)")
    if st != 403:
        fails.append(f"no-token should be 403, got {st}")
    st, _ = call({"action": "setDefaultModel", "modelId": target, "accessToken": "not.a.token"})
    print(f"[bad token]      status={st} (expect 403)")
    if st != 403:
        fails.append(f"bad-token should be 403, got {st}")

    # 5. restore original default
    st, _ = call({"action": "setDefaultModel", "modelId": original, "accessToken": access_token})
    print(f"[restore -> {original}] status={st}")
    if manifest_default() != original:
        fails.append(f"could not restore default (={manifest_default()})")

    print()
    if fails:
        print("FAILURES:")
        for f in fails:
            print("  -", f)
        sys.exit(1)
    print("SET-DEFAULT VERIFIED")


if __name__ == "__main__":
    main()
