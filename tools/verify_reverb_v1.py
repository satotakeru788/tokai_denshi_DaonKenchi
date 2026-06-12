"""reverb_v1 プロファイルの移植一致＋推論一致検証（AWS不要・ローカル）。

検証内容（docs/前処理プロファイル仕様.md §3 の「移植一致テスト」の実例）:
  (a) 学習側前処理 (DaonKenti sagemaker/code/audio_cnn.py: build_hit_wave_samples
      crop_mode=reverb_tail + compute_logmel) と、アプリ側 preprocessors/reverb_v1
      の特徴量が一致すること（打数・ピーク位置・features allclose）
      対照群: 同じ meta を logmel_v1 に通すと一致 **しない** こと（プロファイル差の実在）
  (b) torch チェックポイント（参照値）と ONNX+reverb_v1 経由の
      infer_from_features の per-recording 予測値が ±0.1 kPa で一致すること

実行環境: torch/onnxruntime/soundfile 入りの venv（研究repoの .venv-onnx）
  C:\\Users\\tk_satou\\DaonKenti-app\\.venv-onnx\\Scripts\\python.exe tools/verify_reverb_v1.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

MVP = Path(r"C:\Users\tk_satou\DaonKenchi-mvp")
ANALYSIS_CODE = Path(r"C:\Users\tk_satou\DaonKenti\sagemaker\code")
EXP3_WAV = Path(r"C:\Users\tk_satou\DaonKenti\Data\exp3_wav")
CKPT = Path(r"C:\Users\tk_satou\DaonKenti\outputs\sweep_exp3_ppfull\crop_revtail\extracted\model.pt")
MODEL_DIR = MVP / "seed" / "models" / "revtail120"

sys.path.insert(0, str(ANALYSIS_CODE))
sys.path.insert(0, str(MVP / "amplify" / "functions" / "infer"))
sys.path.insert(0, str(MVP / "tools"))

import torch  # noqa: E402
import onnxruntime as ort  # noqa: E402
import audio_cnn  # noqa: E402  (学習側・移植元)

from audio_infer import infer_from_features  # noqa: E402  (アプリ側共通コア)
from preprocessors import logmel_v1, reverb_v1  # noqa: E402

sys.path.insert(0, str(Path(r"C:\Users\tk_satou\DaonKenti-app\tools")))
from export_onnx import DenormMember, Ensemble, MelSpectrogramCNN  # noqa: E402


def load_reference_ensemble():
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    members = []
    for sd, info in zip(ckpt["ensemble_state_dicts"], ckpt["ensemble_training_info"]):
        m = MelSpectrogramCNN(base_channels=ckpt["channels"], dropout=ckpt["dropout"])
        m.load_state_dict(sd)
        m.eval()
        members.append(DenormMember(m, info["target_mean"], info["target_std"]))
    ens = Ensemble(members)
    ens.eval()
    return ens


def training_side_features(path: Path, meta: dict) -> np.ndarray:
    """学習リポジトリのコードそのもので 特徴量 [N,1,H,W] を作る（参照）。"""
    fs = audio_cnn.FileSample(path=path, pressure_pa=0.0)
    hits, _counts = audio_cnn.build_hit_wave_samples(
        [fs],
        max_hits_per_file=int(meta.get("max_hits_per_file", 0) or 0),
        pre_s=float(meta["pre_seconds"]),
        post_s=float(meta["post_seconds"]),
        target_sr=int(meta["target_sr"]),
        crop_mode="reverb_tail",
    )
    mels = [
        audio_cnn.compute_logmel(
            h.waveform,
            int(meta["target_sr"]),
            n_fft=int(meta["n_fft"]),
            hop=int(meta["hop"]),
            n_mels=int(meta["n_mels"]),
            fmin=float(meta["fmin"]),
            fmax=float(meta["fmax"]),
        )
        for h in hits
    ]
    if not mels:
        return np.zeros((0, 1, int(meta["n_mels"]), 0), dtype=np.float32)
    return np.stack(mels, axis=0)[:, np.newaxis, :, :].astype(np.float32)


def pick_test_wavs() -> list[Path]:
    out = []
    for folder in ["200kPa", "220kPa", "240kPa", "260kPa", "270kPa"]:
        wavs = sorted((EXP3_WAV / folder).glob("*.wav"))
        if wavs:
            out.append(wavs[0])
    return out


def main() -> None:
    meta = json.loads((MODEL_DIR / "meta.json").read_text(encoding="utf-8"))
    sess = ort.InferenceSession(str(MODEL_DIR / "model.onnx"), providers=["CPUExecutionProvider"])
    ens = load_reference_ensemble()

    wavs = pick_test_wavs()
    if not wavs:
        sys.exit(f"no test wavs under {EXP3_WAV}")

    failures = []
    print(f"{'file':38s} {'hits':>4s} {'featDiff':>9s} {'ref kPa':>8s} {'app kPa':>8s} {'logmel kPa':>10s}")
    for p in wavs:
        wav_bytes = p.read_bytes()

        # (a) 移植一致: 学習側 vs reverb_v1
        ft = training_side_features(p, meta)
        fa_dict = reverb_v1.preprocess(wav_bytes, meta)
        fa = fa_dict["features"]
        if fa is None or len(ft) != len(fa):
            failures.append(f"{p.name}: hit count mismatch train={len(ft)} app={0 if fa is None else len(fa)}")
            continue
        feat_diff = float(np.abs(ft - fa).max())
        if feat_diff > 1e-3:
            failures.append(f"{p.name}: feature diff {feat_diff:.2e} > 1e-3")

        # 対照群: logmel_v1 は一致しないこと（プロファイル差の実在証明）
        fl_dict = logmel_v1.preprocess(wav_bytes, meta)
        fl = fl_dict["features"]
        if fl is not None and len(fl) == len(fa):
            contrast = float(np.abs(fl - fa).max())
            if contrast < 1e-2:
                failures.append(f"{p.name}: logmel_v1 and reverb_v1 features identical (contrast {contrast:.2e}) — profile diff not effective")

        # (b) 推論一致: torch参照（学習側特徴量・打平均） vs ONNX+profile
        with torch.no_grad():
            ref = float(ens(torch.from_numpy(ft)).numpy().mean())
        res = infer_from_features(fa_dict, sess, meta)
        app = res["pressureRawKpa"]
        if app is None or abs(ref - app) > 0.11:
            failures.append(f"{p.name}: inference mismatch ref={ref:.2f} app={app}")

        # 参考: logmel_v1 経由の値（切替の効果量）
        res_l = infer_from_features(fl_dict, sess, meta)
        app_l = res_l["pressureRawKpa"]

        print(f"{p.parent.name + '/' + p.name:38s} {len(fa):4d} {feat_diff:9.2e} {ref:8.1f} {app:8.1f} {str(app_l):>10s}")

    print()
    if failures:
        print("FAILURES:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("REVERB_V1 PORT + INFERENCE PARITY OK")


if __name__ == "__main__":
    main()
