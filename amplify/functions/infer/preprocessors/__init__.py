"""前処理プロファイルのレジストリ。

manifest.json の各モデルは "preprocess" フィールドでプロファイルを指定する
（未指定は logmel_v1）。「前処理＋モデル」をペアで切り替えるための仕組み。

新しいプロファイルの追加手順（詳細: docs/前処理プロファイル仕様.md）:
  1. このパッケージに <profile_id>.py を作成し、次の2つを実装する
       PROFILE_ID: str
       preprocess(wav_bytes: bytes, meta: dict) -> dict
         戻り値 {"features": np.ndarray [N打,1,H,W] float32 | None,
                 "peakSeconds": list[float], "durationSec": float}
     制約: numpy / scipy のみ使用可。ネットワーク・subprocess・/tmp以外への
     書込は禁止。処理時間は数秒以内（5秒応答目標のため）。
  2. 下の _PROFILES に1行追加する。
  3. PRレビュー（コード安全性・依存・実行時間）を経てデプロイする。
     ※プロファイル追加はコード変更なので再デプロイが必要。同じプロファイルを
       使うモデルの追加は S3 だけで済む（再デプロイ不要）。
"""
from . import logmel_v1, reverb_v1, takao_v1

_PROFILES = {
    logmel_v1.PROFILE_ID: logmel_v1,
    reverb_v1.PROFILE_ID: reverb_v1,
    takao_v1.PROFILE_ID: takao_v1,
}

DEFAULT_PROFILE = logmel_v1.PROFILE_ID


def get_profile(name):
    profile = _PROFILES.get(name or DEFAULT_PROFILE)
    if profile is None:
        raise ValueError(f"unknown preprocess profile: {name}")
    return profile
