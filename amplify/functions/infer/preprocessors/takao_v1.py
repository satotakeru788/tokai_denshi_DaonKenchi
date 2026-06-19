"""takao_v1 — 髙尾式前処理（al-final7）の numpy/scipy 移植プロファイル。

研究repo の検証済み実装 takao_pp.py（compare/takao_pp.py と同一）を逐語で同梱し、
librosa を使わずに同一特徴量を生成する（librosa版との数値差: 検出0.000s / mel演算0.000dB
/ 全体0.009dB ＝ リサンプラ差のみ）。学習側（train_export_takao.py）も同じ takao_pp を
使うため train == serve。

パイプライン:
  load_wav_mono（全録音 平均除去+ピーク正規化）
  → 打音検出（フレームRMS→find_peaks→オンセット後退）
  → クロップ [onset-0.04, +0.30]s（native sr）
  → bandpass 50-300Hz → 8kHz リサンプル
  → メル(64×38, Slaney) → power_to_db(ref=max)

戻り値の features は logmel_v1 と同じ [N打, 1, H, W] float32。形状は 64×38（exp3v2 の
96×8 とは別物）なので、必ず takao_v1 で学習した ONNX とペアで使うこと（meta は takao の
パラメータを記載。実体は takao_pp の定数を使用）。
"""
from typing import List

import numpy as np

import takao_pp as T
from audio_infer import load_wav_mono

PROFILE_ID = "takao_v1"


def preprocess(wav_bytes: bytes, meta: dict) -> dict:
    mono, sr = load_wav_mono(wav_bytes)  # 平均除去+ピーク正規化済み（学習側と一致）
    mels, used_onsets = T.features_from_mono(mono, sr)
    features = np.stack(mels, axis=0)[:, np.newaxis, :, :].astype(np.float32) if mels else None
    return {
        "features": features,
        "peakSeconds": used_onsets,
        "durationSec": round(len(mono) / sr, 2),
    }
