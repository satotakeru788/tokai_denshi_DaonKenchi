"""Local regression smoke for the profile refactor (no AWS needed).

Runs the synthetic multi-hit WAV through BOTH paths with the seed exp3v2 model:
  A) audio_infer.run_inference            (original composed API)
  B) profile dispatch (preprocessors.get_profile -> infer_from_features)
and asserts the results are identical (and calibration on/off behaves).

Run with a python that has numpy/scipy/onnxruntime, e.g. the research venv:
  C:\\Users\\tk_satou\\DaonKenti-app\\.venv-onnx\\Scripts\\python.exe tools/smoke_local.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "amplify" / "functions" / "infer"))
sys.path.insert(0, str(ROOT / "tools"))

import io  # noqa: E402
import math  # noqa: E402
import random  # noqa: E402
import struct  # noqa: E402
import wave as wave_mod  # noqa: E402

import onnxruntime as ort  # noqa: E402

from audio_infer import infer_from_features, run_inference  # noqa: E402
from preprocessors import get_profile  # noqa: E402


def make_wav(sr=22050, n_hits=6, gap_s=3.5, seed=1) -> bytes:
    """test_endpoint.make_wav と同一（boto3非依存のためコピー）。"""
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
    with wave_mod.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)
    return buf.getvalue()

MODEL_DIR = ROOT / "seed" / "models" / "exp3v2"


def main() -> None:
    meta = json.loads((MODEL_DIR / "meta.json").read_text(encoding="utf-8"))
    session = ort.InferenceSession(str(MODEL_DIR / "model.onnx"),
                                   providers=["CPUExecutionProvider"])
    wav = make_wav()

    a = run_inference(wav, session, meta)
    profile = get_profile(None)  # default = logmel_v1
    b = infer_from_features(profile.preprocess(wav, meta), session, meta)

    print(f"A(run_inference)   : kPa={a['pressureKpa']} raw={a['pressureRawKpa']} hits={a['hitsUsed']}")
    print(f"B(profile dispatch): kPa={b['pressureKpa']} raw={b['pressureRawKpa']} hits={b['hitsUsed']}")
    assert a == b, f"MISMATCH:\nA={a}\nB={b}"

    # calibrated=false 相当（meta から calibration を外す）
    meta_raw = dict(meta)
    meta_raw.pop("calibration", None)
    c = infer_from_features(profile.preprocess(wav, meta_raw), session, meta_raw)
    print(f"raw (no calibration): kPa={c['pressureKpa']} calibrated={c['calibrated']}")
    assert c["calibrated"] is False and c["pressureKpa"] == a["pressureRawKpa"]

    # 48kHz入力（フロントの原レート化相当）でも動くこと
    d = infer_from_features(profile.preprocess(make_wav(sr=48000), meta), session, meta)
    print(f"48kHz input        : kPa={d['pressureKpa']} hits={d['hitsUsed']}")
    assert d["hitsUsed"] >= 4 and d["pressureKpa"] is not None

    print("LOCAL SMOKE OK")


if __name__ == "__main__":
    main()
