"""Verify the deployed takao16-raw model end-to-end on the (sandbox) Lambda.

  1. decode a 16-inch exp3 recording -> 48kHz 16-bit WAV -> upload to
     audio/devtest/takao_check.wav  (daonkenti-dev profile, admin S3 access)
  2. SigV4-invoke the Function URL for each modelId and print the result
     (admin profile creds; the Function URL is AWS_IAM auth)

Confirms: takao_v1 profile imports + the takao16 ONNX runs through the live Lambda,
and that the existing models still work (no regression). Calibration-free model, so
pressureKpa == pressureRawKpa.

Usage: python tools/verify_takao_deploy.py
"""
from __future__ import annotations
import io, json, sys, wave
from pathlib import Path

import boto3
import numpy as np
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
import urllib.error
import urllib.request

import av

PROFILE = "daonkenti-dev"
RESEARCH = Path(r"C:/Users/tk_satou/DaonKenti")
ROOT = Path(__file__).resolve().parent.parent
KEY = "audio/devtest/takao_check.wav"


def decode_wav_bytes():
    d = json.load(io.open(RESEARCH / "outputs" / "exp3_inventory.json", encoding="utf-8"))
    exp3 = [p for p in (RESEARCH / "Data").iterdir() if p.is_dir() and p.name.endswith("3")][0]
    r = next(x for x in d["records"] if x.get("size") == 16 and x.get("gauge") == 240)
    c = av.open(str(exp3 / r["session"] / r["file"])); st = c.streams.audio[0]
    sr = int(st.codec_context.sample_rate); parts = []
    for fr in c.decode(st):
        a = fr.to_ndarray()
        a = a.mean(axis=0) if (a.ndim == 2 and a.shape[0] > 1) else a.reshape(-1)
        parts.append(a.astype(np.float32))
    c.close()
    y = np.concatenate(parts)
    if y.size and np.max(np.abs(y)) > 1.5:
        y = y / 32768.0
    pcm = (np.clip(y, -1, 1) * 32767).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(pcm)
    return buf.getvalue(), r["gauge"], sr


def main():
    out = json.loads((ROOT / "amplify_outputs.json").read_text(encoding="utf-8"))
    region = out["custom"].get("inferenceRegion") or out["storage"]["aws_region"]
    url = out["custom"]["inferenceUrl"]
    bucket = out["storage"]["bucket_name"]

    sess = boto3.Session(profile_name=PROFILE, region_name=region)
    s3 = sess.client("s3")
    creds = sess.get_credentials().get_frozen_credentials()

    wav, true_kpa, sr = decode_wav_bytes()
    s3.put_object(Bucket=bucket, Key=KEY, Body=wav, ContentType="audio/wav")
    print(f"uploaded {KEY}  ({len(wav)} bytes, {sr}Hz, true gauge={true_kpa}kPa)\n")

    def call(model_id):
        body = json.dumps({"audioKey": KEY, "modelId": model_id}).encode()
        req = AWSRequest(method="POST", url=url, data=body, headers={"content-type": "application/json"})
        SigV4Auth(creds, "lambda", region).add_auth(req)
        p = req.prepare()
        r = urllib.request.Request(p.url, data=body, headers=dict(p.headers), method="POST")
        try:
            with urllib.request.urlopen(r, timeout=120) as resp:
                return resp.status, json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode()

    fails = []
    for mid in ("takao16-hyb", "exp3v2-cal", "revtail120-raw"):
        st, res = call(mid)
        if isinstance(res, dict):
            print(f"[{mid}] status={st} pressure={res.get('pressureKpa')}kPa "
                  f"hits={res.get('hitsUsed')} preprocess={res.get('preprocess')} "
                  f"model={res.get('modelId')} err={res.get('error')}")
            if st != 200 or not res.get("hitsUsed"):
                fails.append(f"{mid}: status={st} err={res.get('error')}")
            if mid == "takao16-hyb" and res.get("preprocess") != "takao_v1":
                fails.append(f"{mid}: preprocess={res.get('preprocess')} (expected takao_v1)")
        else:
            print(f"[{mid}] status={st} body={res[:200]}")
            fails.append(f"{mid}: status={st}")

    print()
    if fails:
        print("FAILURES:")
        for f in fails:
            print("  -", f)
        sys.exit(1)
    print("TAKAO DEPLOY VERIFIED (takao16-raw live + existing models OK)")


if __name__ == "__main__":
    main()
