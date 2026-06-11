"""Lambda handler (Function URL, AWS_IAM auth) for tire-pressure inference.

Flow (MVP, S3-backed):
  request body = {"audioKey": "audio/<entity>/<id>.wav", "modelId": "exp3v2-cal"}
    1. resolve modelId via S3 models/manifest.json  (default = manifest.default)
    2. download the model's ONNX + meta from S3 -> /tmp (cached while warm)
    3. read the 22050Hz WAV from S3 (audioKey)
    4. audio_infer.run_inference  (preprocess + ONNX ensemble + optional calibration)
    5. store the result JSON to results/<entity>/<id>.json  (accumulate for review)
    6. return the result (+ resultKey) in the response for immediate display

Model switching: a manifest entry carries {path, calibrated}. Both demo entries
point at the same ONNX (models/exp3v2/); `calibrated:false` strips meta.calibration
so the raw vs. calibrated outputs differ — proving the switch end to end.

CORS headers are supplied by the Function URL CORS config (see amplify/backend.ts);
do NOT also set them here or the browser sees duplicate headers and blocks the call.
"""
from __future__ import annotations

import json
import os
import traceback
from datetime import datetime, timezone

import boto3
import onnxruntime as ort

from audio_infer import run_inference

_s3 = boto3.client("s3")
BUCKET = os.environ["BUCKET_NAME"]

# warm-container caches
_SESSIONS: dict[str, tuple] = {}  # model path -> (onnx session, raw meta dict)


def _load_manifest() -> dict:
    """Fetch models/manifest.json fresh each request (tiny file; lets new models
    be added by editing the manifest + uploading files, with no redeploy)."""
    obj = _s3.get_object(Bucket=BUCKET, Key="models/manifest.json")
    return json.loads(obj["Body"].read())


def _resolve_model(manifest: dict, model_id: str | None) -> dict:
    models = manifest.get("models", [])
    if not models:
        raise ValueError("manifest has no models")
    wanted = model_id or manifest.get("default") or models[0]["id"]
    for m in models:
        if m.get("id") == wanted:
            return m
    raise ValueError(f"unknown modelId: {wanted}")


def _get_session(path: str):
    """Download models/<path>/{model.onnx,model.onnx.data,meta.json} to /tmp once,
    build the ONNX session, and cache by path for warm reuse."""
    if path in _SESSIONS:
        return _SESSIONS[path]

    local_dir = os.path.join("/tmp", "models", path)
    os.makedirs(local_dir, exist_ok=True)
    for fname in ("model.onnx", "model.onnx.data", "meta.json"):
        local = os.path.join(local_dir, fname)
        if not os.path.exists(local):
            _s3.download_file(BUCKET, f"models/{path}/{fname}", local)

    with open(os.path.join(local_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)

    so = ort.SessionOptions()
    so.intra_op_num_threads = 0  # use all available vCPUs
    session = ort.InferenceSession(
        os.path.join(local_dir, "model.onnx"),
        sess_options=so,
        providers=["CPUExecutionProvider"],
    )
    _SESSIONS[path] = (session, meta)
    return session, meta


def _parse_audio_key(audio_key: str) -> tuple[str, str]:
    """audio/<entity_id>/<id>.wav -> (entity_id, id). Raises on a bad shape."""
    if not audio_key.startswith("audio/") or not audio_key.endswith(".wav"):
        raise ValueError("audioKey must look like audio/<entity>/<id>.wav")
    parts = audio_key.split("/")
    if len(parts) < 3 or not parts[1] or not parts[-1]:
        raise ValueError("malformed audioKey")
    entity_id = parts[1]
    base_id = parts[-1].rsplit(".", 1)[0]
    return entity_id, base_id


def _resp(status: int, body: dict):
    return {
        "statusCode": status,
        "headers": {"content-type": "application/json"},
        "body": json.dumps(body),
    }


# SnapStart: preload the default model's ONNX session at init so it lands in the
# snapshot — the first request then skips the S3 download + session build.
# Non-fatal: if the manifest/model isn't reachable at init, fall back to lazy load.
try:
    _get_session(_resolve_model(_load_manifest(), None)["path"])
except Exception:  # noqa: BLE001
    pass


def handler(event, context):
    try:
        raw = event.get("body") or ""
        if event.get("isBase64Encoded"):
            import base64
            raw = base64.b64decode(raw).decode("utf-8")
        payload = json.loads(raw) if raw else {}

        audio_key = payload.get("audioKey")
        if not audio_key:
            return _resp(400, {"error": "audioKey is required"})
        entity_id, base_id = _parse_audio_key(audio_key)

        manifest = _load_manifest()
        model = _resolve_model(manifest, payload.get("modelId"))
        session, raw_meta = _get_session(model["path"])

        # per-request meta: honour the manifest's calibrated flag
        meta = dict(raw_meta)
        if not model.get("calibrated", True):
            meta.pop("calibration", None)

        wav = _s3.get_object(Bucket=BUCKET, Key=audio_key)["Body"].read()
        result = run_inference(wav, session, meta)

        # enrich + persist
        result["modelId"] = model["id"]
        result["modelName"] = model.get("name")
        result["audioKey"] = audio_key
        result["createdAt"] = datetime.now(timezone.utc).isoformat()
        result_key = f"results/{entity_id}/{base_id}.json"
        _s3.put_object(
            Bucket=BUCKET,
            Key=result_key,
            Body=json.dumps(result, ensure_ascii=False).encode("utf-8"),
            ContentType="application/json",
        )
        result["resultKey"] = result_key
        result["id"] = base_id

        status = 200 if result.get("hitsUsed", 0) > 0 else 422
        return _resp(status, result)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return _resp(500, {"error": str(exc)})
