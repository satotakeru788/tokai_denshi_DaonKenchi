"""logmel_v1 — 既定の前処理プロファイル（exp3_v2 系モデル用）。

研究repoの学習パイプラインを忠実に移植した audio_infer.extract_features を
そのまま使う:
  任意SRのWAV → 22050Hzへリサンプル → 打撃ピーク検出（全打）
  → 各打を切り出し（pre/post秒・正規化）→ log-melスペクトログラム

meta.json 必須キー:
  target_sr, n_fft, hop, n_mels, fmin, fmax,
  pre_seconds, post_seconds, segment_samples
（任意: calibration{x,y} — 集計側で使用。manifest の calibrated=false で無効化）
"""
from audio_infer import extract_features

PROFILE_ID = "logmel_v1"


def preprocess(wav_bytes: bytes, meta: dict) -> dict:
    return extract_features(wav_bytes, meta)
