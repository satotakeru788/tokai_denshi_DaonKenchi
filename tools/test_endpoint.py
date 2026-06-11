"""Server-side smoke test of the deployed inference endpoint (no browser).

Steps (uses the daonkenti-dev profile = admin creds; the real app uses the
Cognito authenticated role, but this exercises the same Function URL + handler):
  1. read amplify_outputs.json  (bucket, region, inferenceUrl)
  2. synthesize a multi-hit WAV (pure stdlib) and upload to audio/smoketest/<id>.wav
  3. SigV4-sign (service=lambda) a POST {audioKey, modelId} and call the Function URL
     - modelId = exp3v2-cal  -> expect 200 + pressureKpa + hitsUsed>=4
     - modelId = exp3v2-raw  -> expect a (different) pressure  (proves model switch)
  4. UNSIGNED POST -> expect 403  (proves AWS_IAM auth is enforced)
  5. confirm results/smoketest/<id>.json was written by the Lambda

Run AFTER `npm run sandbox:once` and `python tools/seed_models.py`.
Usage:  python tools/test_endpoint.py
"""
from __future__ import annotations

import io
import json
import math
import random
import struct
import sys
import urllib.error
import urllib.request
import uuid
import wave
from pathlib import Path

import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

PROFILE = "daonkenti-dev"
ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "amplify_outputs.json"


def make_wav(sr=22050, n_hits=6, gap_s=3.5, seed=1) -> bytes:
    """Background noise + exponentially-decaying transients (= synthetic hits).
    gap (3.5s) > the detector's 3.0s refractory so every hit is kept."""
    rnd = random.Random(seed)
    total = int((gap_s * n_hits + 2) * sr)
    x = [rnd.uniform(-1.0, 1.0) * 0.002 for _ in range(total)]
    blen = int(0.15 * sr)
    for k in range(n_hits):
        c = int((1.5 + k * gap_s) * sr)
        for t in range(blen):
            if c + t < total:
                x[c + t] += rnd.gauss(0.0, 1.0) * math.exp(-t / (0.02 * sr)) * 1.5
    peak = max(abs(v) for v in x) + 1e-9
    pcm = b"".join(struct.pack("<h", int(max(-1.0, min(1.0, v / peak)) * 32767)) for v in x)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)
    return buf.getvalue()


def load_outputs() -> dict:
    if not OUTPUTS.exists():
        sys.exit(f"{OUTPUTS} not found. Run `npm run sandbox:once` first.")
    return json.loads(OUTPUTS.read_text(encoding="utf-8"))


def signed_post(url: str, payload: bytes, creds, region: str):
    req = AWSRequest(method="POST", url=url, data=payload,
                     headers={"content-type": "application/json"})
    SigV4Auth(creds, "lambda", region).add_auth(req)
    prepared = req.prepare()
    r = urllib.request.Request(prepared.url, data=payload,
                               headers=dict(prepared.headers), method="POST")
    try:
        with urllib.request.urlopen(r, timeout=120) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def unsigned_post(url: str, payload: bytes):
    r = urllib.request.Request(url, data=payload,
                               headers={"content-type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(r, timeout=30) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def main() -> None:
    out = load_outputs()
    url = out["custom"]["inferenceUrl"]
    region = out["custom"].get("inferenceRegion") or out["storage"]["aws_region"]
    bucket = out["storage"]["bucket_name"]

    session = boto3.Session(profile_name=PROFILE, region_name=region)
    creds = session.get_credentials().get_frozen_credentials()
    s3 = session.client("s3")

    failures = []

    # 1-2. synthesize + upload
    wav = make_wav()
    test_id = uuid.uuid4().hex
    audio_key = f"audio/smoketest/{test_id}.wav"
    s3.put_object(Bucket=bucket, Key=audio_key, Body=wav, ContentType="audio/wav")
    print(f"uploaded s3://{bucket}/{audio_key}  ({len(wav)} bytes)")
    print(f"endpoint: {url}\n")

    # 3. signed calls: calibrated + raw
    def call(model_id):
        payload = json.dumps({"audioKey": audio_key, "modelId": model_id}).encode()
        status, body = signed_post(url, payload, creds, region)
        return status, json.loads(body) if body else {}

    status_cal, cal = call("exp3v2-cal")
    print(f"[cal] status={status_cal} pressureKpa={cal.get('pressureKpa')} "
          f"raw={cal.get('pressureRawKpa')} hits={cal.get('hitsUsed')} "
          f"calibrated={cal.get('calibrated')}")
    if status_cal != 200 or cal.get("pressureKpa") is None:
        failures.append(f"calibrated call failed: {status_cal} {cal}")
    if (cal.get("hitsUsed") or 0) < 4:
        failures.append(f"expected >=4 hits, got {cal.get('hitsUsed')}")

    status_raw, raw = call("exp3v2-raw")
    print(f"[raw] status={status_raw} pressureKpa={raw.get('pressureKpa')} "
          f"calibrated={raw.get('calibrated')}")
    if status_raw != 200 or raw.get("pressureKpa") is None:
        failures.append(f"raw call failed: {status_raw} {raw}")
    if raw.get("calibrated") is not False:
        failures.append("raw model should report calibrated=false")
    # raw == pressureRawKpa of the cal run (same ONNX, calibration stripped)
    if raw.get("pressureKpa") == cal.get("pressureKpa"):
        failures.append("model switch had no effect (cal == raw)")
    print(f"  -> model switch changes output: {cal.get('pressureKpa')} (cal) vs "
          f"{raw.get('pressureKpa')} (raw)\n")

    # 4. unsigned -> 403
    status_un, body_un = unsigned_post(url, json.dumps({"audioKey": audio_key}).encode())
    print(f"[unsigned] status={status_un} (expect 403)")
    if status_un != 403:
        failures.append(f"unsigned call should be 403, got {status_un}: {body_un[:200]}")

    # 5. result object written by Lambda
    result_key = f"results/smoketest/{test_id}.json"
    try:
        s3.head_object(Bucket=bucket, Key=result_key)
        print(f"[result] s3://{bucket}/{result_key} written OK\n")
    except Exception as e:  # noqa: BLE001
        failures.append(f"result object missing: {result_key} ({e})")

    if failures:
        print("FAILURES:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
