"""Verify the REAL user path with Cognito authenticated-role credentials
(what the browser uses), end to end and headless.

  SRP login -> identity-pool creds -> as that user:
    * read models/manifest.json                         (models/* authenticated read)
    * upload audio/{identityId}/<id>.wav                (own prefix write)
    * results/ write is DENIED                          (user is read-only there)
    * another entity's audio write is DENIED            (per-user isolation)
    * SigV4-invoke the Function URL {audioKey, modelId} (auth-role InvokeFunctionUrl)
    * read back results/{identityId}/<id>.json          (Lambda wrote it; user can read)

Usage:  python tools/verify_auth_role.py <email> <password>
Uses the daonkenti-dev profile only for cognito-identity + cleanup.
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.credentials import Credentials
from pycognito import Cognito

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from test_endpoint import make_wav  # noqa: E402

PROFILE = "daonkenti-dev"


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit("usage: python tools/verify_auth_role.py <email> <password>")
    email, pw = sys.argv[1], sys.argv[2]

    out = json.loads((ROOT / "amplify_outputs.json").read_text(encoding="utf-8"))
    region = out["custom"].get("inferenceRegion") or out["storage"]["aws_region"]
    url = out["custom"]["inferenceUrl"]
    bucket = out["storage"]["bucket_name"]
    pool = out["auth"]["user_pool_id"]
    client = out["auth"]["user_pool_client_id"]
    idpool = out["auth"]["identity_pool_id"]

    fails: list[str] = []

    # 1. SRP login
    u = Cognito(pool, client, username=email, user_pool_region=region)
    u.authenticate(password=pw)
    print("SRP login OK")

    # 2. identity-pool credentials for the authenticated role
    admin = boto3.Session(profile_name=PROFILE, region_name=region)
    ci = admin.client("cognito-identity")
    prov = f"cognito-idp.{region}.amazonaws.com/{pool}"
    identity_id = ci.get_id(IdentityPoolId=idpool, Logins={prov: u.id_token})["IdentityId"]
    c = ci.get_credentials_for_identity(IdentityId=identity_id, Logins={prov: u.id_token})["Credentials"]
    creds = Credentials(c["AccessKeyId"], c["SecretKey"], c["SessionToken"])
    print(f"identityId = {identity_id}")

    s3u = boto3.client(
        "s3", region_name=region,
        aws_access_key_id=c["AccessKeyId"],
        aws_secret_access_key=c["SecretKey"],
        aws_session_token=c["SessionToken"],
    )

    # read manifest (models/* authenticated read)
    try:
        s3u.get_object(Bucket=bucket, Key="models/manifest.json")
        print("OK  read models/manifest.json")
    except Exception as e:  # noqa: BLE001
        fails.append(f"read manifest denied: {e}")

    # upload to own audio/{identityId}/ prefix
    tid = uuid.uuid4().hex
    akey = f"audio/{identity_id}/{tid}.wav"
    try:
        s3u.put_object(Bucket=bucket, Key=akey, Body=make_wav(), ContentType="audio/wav")
        print(f"OK  upload {akey}")
    except Exception as e:  # noqa: BLE001
        fails.append(f"upload own audio denied: {e}")

    # labels/{identityId}/ write + read (再学習フォームの保存先)
    lkey = f"labels/{identity_id}/{tid}.json"
    try:
        s3u.put_object(Bucket=bucket, Key=lkey,
                       Body=b'{"actualKpa":240,"tireType":"test"}',
                       ContentType="application/json")
        s3u.get_object(Bucket=bucket, Key=lkey)
        s3u.delete_object(Bucket=bucket, Key=lkey)
        print(f"OK  write+read labels/{identity_id}/<id>.json")
    except Exception as e:  # noqa: BLE001
        fails.append(f"labels write/read denied: {e}")

    # negative: user must NOT write results/ (read-only)
    try:
        s3u.put_object(Bucket=bucket, Key=f"results/{identity_id}/{tid}.json", Body=b"{}")
        fails.append("SECURITY: user could write results/ (should be Lambda-only)")
    except Exception:  # noqa: BLE001
        print("OK  results/ write denied for user")

    # negative: user must NOT write another entity's audio
    try:
        s3u.put_object(Bucket=bucket, Key="audio/not-me/x.wav", Body=b"x")
        fails.append("SECURITY: user could write another entity's audio")
    except Exception:  # noqa: BLE001
        print("OK  cross-entity audio write denied")

    # SigV4-invoke the Function URL with the authenticated-role creds
    payload = json.dumps({"audioKey": akey, "modelId": "exp3v2-cal"}).encode()
    req = AWSRequest(method="POST", url=url, data=payload, headers={"content-type": "application/json"})
    SigV4Auth(creds, "lambda", region).add_auth(req)
    p = req.prepare()
    r = urllib.request.Request(p.url, data=payload, headers=dict(p.headers), method="POST")
    try:
        with urllib.request.urlopen(r, timeout=120) as resp:
            status, body = resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        status, body = e.code, e.read().decode()
    print(f"invoke status={status} pressureKpa="
          f"{body.get('pressureKpa') if isinstance(body, dict) else body}")
    if status != 200 or not isinstance(body, dict) or body.get("pressureKpa") is None:
        fails.append(f"auth-role invoke failed: {status} {body}")

    # user can read the result the Lambda wrote
    try:
        s3u.get_object(Bucket=bucket, Key=f"results/{identity_id}/{tid}.json")
        print("OK  read own results/ JSON")
    except Exception as e:  # noqa: BLE001
        fails.append(f"user cannot read own result: {e}")

    # cleanup (user deletes own audio; admin removes the result)
    try:
        s3u.delete_object(Bucket=bucket, Key=akey)
        admin.client("s3").delete_object(Bucket=bucket, Key=f"results/{identity_id}/{tid}.json")
    except Exception:  # noqa: BLE001
        pass

    print()
    if fails:
        print("FAILURES:")
        for f in fails:
            print("  -", f)
        sys.exit(1)
    print("AUTH-ROLE PATH VERIFIED")


if __name__ == "__main__":
    main()
