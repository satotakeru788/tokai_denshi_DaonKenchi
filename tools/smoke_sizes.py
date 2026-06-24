"""Local smoke for the 16in/15in takao_v1 models + manifest size-resolution.

No AWS needed. Runs a synthetic multi-hit WAV through both seed models via the
takao_v1 profile, and checks handler._resolve_model / _set_default_model logic
on the manifest (the S3 put is monkeypatched away).

Run with the research onnx venv:
  C:\\Users\\tk_satou\\DaonKenti-app\\.venv-onnx\\Scripts\\python.exe tools/smoke_sizes.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "amplify" / "functions" / "infer"))
sys.path.insert(0, str(ROOT / "tools"))

import onnxruntime as ort  # noqa: E402

from audio_infer import infer_from_features  # noqa: E402
from preprocessors import get_profile  # noqa: E402
from smoke_local import make_wav  # noqa: E402


def run_model(path: str) -> dict:
    d = ROOT / "seed" / "models" / path
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    session = ort.InferenceSession(str(d / "model.onnx"), providers=["CPUExecutionProvider"])
    profile = get_profile("takao_v1")
    wav = make_wav(sr=48000, gap_s=5.0)  # フロントの原レート相当（takao_v1 min_gap=4.0s）
    return infer_from_features(profile.preprocess(wav, meta), session, meta)


def main() -> None:
    manifest = json.loads((ROOT / "seed" / "models" / "manifest.json").read_text(encoding="utf-8"))

    # 1) manifest が両サイズの代表を持つこと
    assert manifest["defaults"]["16"] == "takao16-hyb", manifest["defaults"]
    assert manifest["defaults"]["15"] == "takao15-hyb", manifest["defaults"]

    # 2) handler の resolve がサイズ別に正しく解決すること
    import handler  # noqa: E402
    assert handler._resolve_model(manifest, None, 16)["id"] == "takao16-hyb"
    assert handler._resolve_model(manifest, None, 15)["id"] == "takao15-hyb"
    # 明示 modelId はサイズより優先
    assert handler._resolve_model(manifest, "exp3v2-cal", 16)["id"] == "exp3v2-cal"
    # tireInch 無し → 全体 default
    assert handler._resolve_model(manifest, None)["id"] == manifest["default"]

    # 3) _set_default_model(tire_inch) が defaults[inch] を更新すること（S3 put はスキップ）
    captured = {}
    handler._require_developer = lambda tok: None  # 認可をバイパス
    handler._load_manifest = lambda: json.loads(json.dumps(manifest))
    handler._s3.put_object = lambda **kw: captured.update(  # type: ignore[attr-defined]
        json.loads(kw["Body"].decode("utf-8"))
    ) or {}
    ret = handler._set_default_model("exp3v2-raw", "tok", 16)
    assert ret["defaults"]["16"] == "exp3v2-raw", ret
    assert captured["defaults"]["16"] == "exp3v2-raw"
    assert captured["defaults"]["15"] == "takao15-hyb"  # 15は不変
    print("resolve/set-default OK:", json.dumps(captured["defaults"], ensure_ascii=False))

    # 4) 実推論（両サイズ）が妥当な kPa を返すこと
    r16 = run_model("takao16hyb")
    r15 = run_model("takao15hyb")
    print(f"16in: kPa={r16['pressureKpa']} hits={r16['hitsUsed']} tag={r16.get('modelTag')}")
    print(f"15in: kPa={r15['pressureKpa']} hits={r15['hitsUsed']} tag={r15.get('modelTag')}")
    for r in (r16, r15):
        assert r["hitsUsed"] >= 4, r
        assert r["pressureKpa"] is not None
        assert 150 <= r["pressureKpa"] <= 320, f"kPa out of sane range: {r['pressureKpa']}"

    print("SIZES SMOKE OK")


if __name__ == "__main__":
    main()
