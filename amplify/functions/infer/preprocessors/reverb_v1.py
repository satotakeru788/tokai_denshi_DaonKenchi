"""reverb_v1 — crop_mode=reverb_tail 学習モデル用の前処理プロファイル。

研究repo sagemaker/code/audio_cnn.py build_hit_wave_samples の reverb_tail 分岐の移植。
logmel_v1 との差分はクロップのみ:
  logmel_v1:  窓 = [peak - pre_seconds, peak + post_seconds]（合計 = segment_samples）
  reverb_v1:  窓 = [peak, peak + segment_samples]（pre を引かない・打撃後の残響を拾う）
pad/trim → セグメント正規化 → log-mel の順序・定数は学習側と同一
（detect_impact_peaks / compute_logmel は audio_infer の逐語移植関数を再利用）。

meta.json 必須キー: target_sr, n_fft, hop, n_mels, fmin, fmax, segment_samples
（crop_mode があれば reverb_tail であることを検査する — 誤ペアリングを
 サイレント精度劣化ではなく明示的なエラーにする安全弁）
"""
from typing import List

import numpy as np

from audio_infer import (
    compute_logmel,
    detect_impact_peaks,
    load_wav_mono,
    resample_if_needed,
)

PROFILE_ID = "reverb_v1"


def preprocess(wav_bytes: bytes, meta: dict) -> dict:
    crop_mode = meta.get("crop_mode")
    if crop_mode not in (None, "reverb_tail"):
        raise ValueError(f"reverb_v1 expects crop_mode=reverb_tail, got {crop_mode!r}")

    sr_target = int(meta["target_sr"])
    n_fft = int(meta["n_fft"])
    hop = int(meta["hop"])
    n_mels = int(meta["n_mels"])
    fmin = float(meta["fmin"])
    fmax = float(meta["fmax"])
    seg_len = int(meta["segment_samples"])  # 学習側: expected_len = int(post_s * sr)
    min_len = int(sr_target * 0.1)

    mono, sr = load_wav_mono(wav_bytes)
    mono = resample_if_needed(mono, sr, sr_target)
    peaks = detect_impact_peaks(mono, sr_target, max_hits=0)

    mels: List[np.ndarray] = []
    used_peaks: List[int] = []
    for peak in peaks:
        start = int(peak)                       # ← pre を引かない（logmel_v1 との唯一の差分）
        end = min(len(mono), start + seg_len)
        seg = mono[start:end]
        if len(seg) < min_len:
            continue
        if len(seg) < seg_len:
            seg = np.pad(seg, (0, seg_len - len(seg)))
        elif len(seg) > seg_len:
            seg = seg[:seg_len]
        seg = seg - np.mean(seg)
        seg = seg / (np.max(np.abs(seg)) + 1e-12)
        mels.append(compute_logmel(seg, sr_target, n_fft, hop, n_mels, fmin=fmin, fmax=fmax))
        used_peaks.append(int(peak))

    features = np.stack(mels, axis=0)[:, np.newaxis, :, :].astype(np.float32) if mels else None
    return {
        "features": features,
        "peakSeconds": [round(p / sr_target, 3) for p in used_peaks],
        "durationSec": round(len(mono) / sr_target, 2),
    }
