"""Upload model files + manifest to the storage bucket under models/.

Reads the bucket name from amplify_outputs.json (written by `ampx sandbox`),
then syncs seed/models/ -> s3://<bucket>/models/ with the daonkenti-dev profile.

Layout uploaded:
  models/manifest.json          ドロップダウン用レジストリ（id/name/desc/path/calibrated）
  models/exp3v2/model.onnx
  models/exp3v2/model.onnx.data
  models/exp3v2/meta.json

Add a model later = drop files under seed/models/<id>/ and add a manifest entry,
then re-run this script (no redeploy needed).

Usage:  python tools/seed_models.py   (or: npm run seed:models)
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

PROFILE = "daonkenti-dev"
REGION = "ap-northeast-1"
ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "amplify_outputs.json"
SRC = ROOT / "seed" / "models"


def bucket_name() -> str:
    data = json.loads(OUTPUTS.read_text(encoding="utf-8"))
    st = data.get("storage", {})
    name = st.get("bucket_name")
    if not name:
        for b in st.get("buckets", []) or []:
            if b.get("bucket_name"):
                name = b["bucket_name"]
                break
    if not name:
        sys.exit("bucket_name not found in amplify_outputs.json")
    return name


def main() -> None:
    if not OUTPUTS.exists():
        sys.exit(f"{OUTPUTS} not found. Run `npm run sandbox:once` first.")
    if not SRC.exists():
        sys.exit(f"{SRC} not found.")

    bucket = bucket_name()
    dest = f"s3://{bucket}/models"
    aws = shutil.which("aws") or "aws"
    cmd = [aws, "s3", "sync", str(SRC), dest,
           "--profile", PROFILE, "--region", REGION]
    print("+", " ".join(cmd))
    subprocess.check_call(cmd)
    print(f"\nseeded models -> {dest}")
    # show what's there now
    subprocess.call([aws, "s3", "ls", f"{dest}/", "--recursive",
                     "--profile", PROFILE, "--region", REGION])


if __name__ == "__main__":
    main()
