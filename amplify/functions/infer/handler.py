"""Lambda handler (Function URL, AWS_IAM auth) for tire-pressure inference.

Flow (MVP, S3-backed):
  request body = {"audioKey": "audio/<entity>/<id>.wav", "modelId": "exp3v2-cal"}
    1. resolve modelId via S3 models/manifest.json  (default = manifest.default)
    2. download the model's ONNX + meta from S3 -> /tmp (cached while warm)
    3. read the WAV from S3 (audioKey; any sample rate — profiles resample)
    4. dispatch to the model's preprocessing profile (manifest "preprocess",
       default logmel_v1) -> features, then ONNX + aggregation (audio_infer)
    5. store the result JSON to results/<entity>/<id>.json  (accumulate for review)
    6. return the result (+ resultKey) in the response for immediate display

Model switching: a manifest entry carries {path, calibrated, preprocess}. The
preprocess field selects a preprocessing profile from preprocessors/ so a model
trained on a different feature pipeline can run with ITS OWN preprocessing —
adding a profile is a code change (PR + redeploy); adding a model that reuses an
existing profile stays S3-only. `calibrated:false` strips meta.calibration.

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

from audio_infer import infer_from_features
from preprocessors import DEFAULT_PROFILE, get_profile

_s3 = boto3.client("s3")
_cognito = boto3.client("cognito-idp")
BUCKET = os.environ["BUCKET_NAME"]
USER_POOL_ID = os.environ.get("USER_POOL_ID", "")
DEVELOPERS_GROUP = "developers"

# warm-container caches
_SESSIONS: dict[str, tuple] = {}  # model path -> (onnx session, raw meta dict)


def _load_manifest() -> dict:
    """Fetch models/manifest.json fresh each request (tiny file; lets new models
    be added by editing the manifest + uploading files, with no redeploy)."""
    obj = _s3.get_object(Bucket=BUCKET, Key="models/manifest.json")
    return json.loads(obj["Body"].read())


def _require_developer(access_token: str | None) -> None:
    """Authorize an admin action: validate the Cognito access token (get_user
    raises on a bad/expired token) then require 'developers' group membership."""
    if not access_token:
        raise PermissionError("missing access token")
    try:
        user = _cognito.get_user(AccessToken=access_token)
    except Exception:  # noqa: BLE001 — invalid/expired token
        raise PermissionError("invalid token")
    groups = _cognito.admin_list_groups_for_user(
        Username=user["Username"], UserPoolId=USER_POOL_ID
    ).get("Groups", [])
    if not any(g.get("GroupName") == DEVELOPERS_GROUP for g in groups):
        raise PermissionError("developer access required")


def _set_default_model(
    model_id: str | None, access_token: str | None, tire_inch: int | str | None = None
) -> dict:
    """developers-only: set the model general users get. With tire_inch (16/15),
    set the per-size default in manifest["defaults"][inch]; without it, set the
    global manifest["default"] (backward compat). Any registered id may be chosen."""
    _require_developer(access_token)
    manifest = _load_manifest()
    ids = [m.get("id") for m in manifest.get("models", [])]
    if model_id not in ids:
        raise ValueError(f"unknown modelId: {model_id}")
    if tire_inch is not None:
        defaults = manifest.setdefault("defaults", {})
        defaults[str(tire_inch)] = model_id
        ret = {"defaults": defaults}
    else:
        manifest["default"] = model_id
        ret = {"default": model_id}
    _s3.put_object(
        Bucket=BUCKET,
        Key="models/manifest.json",
        Body=json.dumps(manifest, ensure_ascii=False).encode("utf-8"),
        ContentType="application/json",
    )
    return ret


def _resolve_model(manifest: dict, model_id: str | None, tire_inch: int | str | None = None) -> dict:
    """Pick a model entry. Priority: explicit model_id > per-size default
    (defaults[inch]) > global default > first model."""
    models = manifest.get("models", [])
    if not models:
        raise ValueError("manifest has no models")
    size_default = None
    if tire_inch is not None:
        size_default = manifest.get("defaults", {}).get(str(tire_inch))
    wanted = model_id or size_default or manifest.get("default") or models[0]["id"]
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

        # admin action: set the default model for general users (developers only)
        if payload.get("action") == "setDefaultModel":
            try:
                return _resp(200, _set_default_model(
                    payload.get("modelId"), payload.get("accessToken"), payload.get("tireInch"),
                ))
            except PermissionError as exc:
                return _resp(403, {"error": str(exc)})

        audio_key = payload.get("audioKey")
        if not audio_key:
            return _resp(400, {"error": "audioKey is required"})
        entity_id, base_id = _parse_audio_key(audio_key)

        manifest = _load_manifest()
        model = _resolve_model(manifest, payload.get("modelId"), payload.get("tireInch"))
        session, raw_meta = _get_session(model["path"])

        # per-request meta: honour the manifest's calibrated flag
        meta = dict(raw_meta)
        if not model.get("calibrated", True):
            meta.pop("calibration", None)

        profile = get_profile(model.get("preprocess"))
        wav = _s3.get_object(Bucket=BUCKET, Key=audio_key)["Body"].read()
        result = infer_from_features(profile.preprocess(wav, meta), session, meta)

        # enrich + persist
        result["preprocess"] = getattr(profile, "PROFILE_ID", DEFAULT_PROFILE)
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
