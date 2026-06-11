"""Measure the inference Lambda's cold vs warm latency (vs the 5s target).

Forces a cold start (config bump), times a cold invocation + 3 warm ones through
the IAM Function URL (SigV4), then pulls Init/Duration from CloudWatch.

Run after `aws sso login --profile daonkenti-dev`.  python tools/measure_latency.py
"""
from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path

import boto3

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from test_endpoint import make_wav, signed_post  # noqa: E402

PROFILE = "daonkenti-dev"


def main() -> None:
    out = json.loads((ROOT / "amplify_outputs.json").read_text(encoding="utf-8"))
    region = out["custom"].get("inferenceRegion") or out["storage"]["aws_region"]
    url = out["custom"]["inferenceUrl"]
    bucket = out["storage"]["bucket_name"]

    session = boto3.Session(profile_name=PROFILE, region_name=region)
    creds = session.get_credentials().get_frozen_credentials()
    s3 = session.client("s3")
    lam = session.client("lambda")
    logs = session.client("logs")

    fns = [
        f["FunctionName"]
        for f in lam.list_functions()["Functions"]
        if "daonkentimvp" in f["FunctionName"] and "InferFn" in f["FunctionName"]
    ]
    if not fns:
        sys.exit("InferFn not found")
    fn = fns[0]
    print(f"function: {fn}")

    # synthetic ~19.5s, 6-hit WAV uploaded to S3
    tid = uuid.uuid4().hex
    akey = f"audio/perf/{tid}.wav"
    s3.put_object(Bucket=bucket, Key=akey, Body=make_wav(), ContentType="audio/wav")
    payload = json.dumps({"audioKey": akey, "modelId": "exp3v2-cal"}).encode()

    def timed(label: str) -> float:
        t0 = time.time()
        status, body = signed_post(url, payload, creds, region)
        dt = time.time() - t0
        d = json.loads(body) if body else {}
        print(f"  {label:6s} {dt:5.2f}s  (status={status}, kPa={d.get('pressureKpa')}, hits={d.get('hitsUsed')})")
        return dt

    # force a cold start by bumping an env var. Skipped with "nocold" — for
    # SnapStart the URL points to the alias (a fixed published version), so a
    # $LATEST config bump won't recycle it; instead the first invoke right after
    # deploy is already a restore (cold), so just measure that.
    force = "nocold" not in sys.argv
    orig_env = None
    if force:
        cfg = lam.get_function_configuration(FunctionName=fn)
        orig_env = dict(cfg.get("Environment", {}).get("Variables", {}))
        bumped = dict(orig_env)
        bumped["COLD_BUST"] = str(int(time.time()))
        lam.update_function_configuration(FunctionName=fn, Environment={"Variables": bumped})
        lam.get_waiter("function_updated").wait(FunctionName=fn)
        time.sleep(2)
    else:
        print("(nocold: measuring the first invoke after deploy as cold/restore)")

    print("\nwall-clock (client -> Function URL -> Lambda):")
    cold = timed("COLD")
    warm = [timed(f"warm{i+1}") for i in range(3)]

    # restore original env (only if we bumped it)
    if force and orig_env is not None:
        try:
            lam.update_function_configuration(FunctionName=fn, Environment={"Variables": orig_env})
            lam.get_waiter("function_updated").wait(FunctionName=fn)
        except Exception as e:  # noqa: BLE001
            print(f"(warning: could not restore env: {e})")

    s3.delete_object(Bucket=bucket, Key=akey)

    # CloudWatch REPORT lines (Init Duration = cold init; Duration = handler)
    print("\nCloudWatch REPORT (Init Duration = コールド初期化, Duration = ハンドラ実行):")
    time.sleep(10)
    lg = f"/aws/lambda/{fn}"
    try:
        streams = logs.describe_log_streams(
            logGroupName=lg, orderBy="LastEventTime", descending=True, limit=5
        )["logStreams"]
        for st in streams:
            ev = logs.get_log_events(
                logGroupName=lg, logStreamName=st["logStreamName"], limit=30, startFromHead=False
            )["events"]
            for e in ev:
                m = e["message"].strip()
                if m.startswith("REPORT"):
                    print("  " + " ".join(m.replace("\t", " ").split()))
    except Exception as e:  # noqa: BLE001
        print(f"(could not read logs: {e})")

    print(f"\nSUMMARY:  cold={cold:.2f}s   warm avg={sum(warm)/len(warm):.2f}s   (target <= 5s)")


if __name__ == "__main__":
    main()
