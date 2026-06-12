"""デプロイ済みエンドポイントで「前処理＋モデルのセット切替」を検証する。

(c) 切替の証明: 同一 audioKey に modelId だけ変えて投げ、
    - revtail120-raw の値がローカル参照（学習側パイプライン）と一致
    - exp3v2-cal と値が異なる
    - 結果の preprocess フィールドが reverb_v1 / logmel_v1 と切り替わる
(e) アプリ外評価との突合: 学習ジョブが記録した holdout_predictions.csv
    （SageMaker実行時の per-hit 予測）と、同じWAVをアプリに通した値を
    録音ごとに直接比較。実測ラベルとの誤差も holdout MAE と整合するか確認。

実行: .venv-onnx の python（boto3 入り）で
  PYTHONUTF8=1 <venv>/python.exe tools/verify_endpoint_revtail.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

import boto3

MVP = Path(r"C:\Users\tk_satou\DaonKenchi-mvp")
EXP3_WAV = Path(r"C:\Users\tk_satou\DaonKenti\Data\exp3_wav")
HOLDOUT_CSV = Path(r"C:\Users\tk_satou\DaonKenti\outputs\sweep_exp3_ppfull\crop_revtail\output_data\holdout_predictions.csv")

sys.path.insert(0, str(MVP / "tools"))
from test_endpoint import signed_post  # noqa: E402

PROFILE = "daonkenti-dev"
PREFIX = "audio/smoketest"


def load_holdout() -> dict:
    """CSV -> {relpath: {"true": kPa, "preds": [per-hit kPa]}}（relpath 例 200kPa/f009.wav）"""
    per_file: dict[str, dict] = defaultdict(lambda: {"true": None, "preds": []})
    with open(HOLDOUT_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rel = "/".join(row["file_id"].replace("\\", "/").split("/")[-2:])
            per_file[rel]["true"] = float(row["true_pa"])
            per_file[rel]["preds"].append(float(row["pred_pa"]))
    return dict(per_file)


def main() -> None:
    out = json.loads((MVP / "amplify_outputs.json").read_text(encoding="utf-8"))
    url = out["custom"]["inferenceUrl"]
    region = out["custom"].get("inferenceRegion") or out["storage"]["aws_region"]
    bucket = out["storage"]["bucket_name"]
    session = boto3.Session(profile_name=PROFILE, region_name=region)
    creds = session.get_credentials().get_frozen_credentials()
    s3 = session.client("s3")

    holdout = load_holdout()
    print(f"holdout files in CSV: {len(holdout)}  (expected 8)")

    failures: list[str] = []
    uploaded: list[str] = []
    rows = []

    def invoke(audio_key: str, model_id: str) -> dict:
        payload = json.dumps({"audioKey": audio_key, "modelId": model_id}).encode()
        status, body = signed_post(url, payload, creds, region)
        d = json.loads(body) if body else {}
        if status != 200:
            failures.append(f"{model_id} on {audio_key}: status {status} {d}")
        return d

    print(f"\n{'file':22s} {'true':>5s} {'hits(csv/app)':>13s} {'csv mean':>8s} {'app raw':>8s} {'diff':>6s}  preprocess")
    first_key = None
    for rel, info in sorted(holdout.items()):
        local = EXP3_WAV / rel
        if not local.exists():
            failures.append(f"{rel}: local wav missing")
            continue
        audio_key = f"{PREFIX}/revtail-{rel.replace('/', '-')}"
        s3.upload_file(str(local), bucket, audio_key, ExtraArgs={"ContentType": "audio/wav"})
        uploaded.append(audio_key)
        if first_key is None:
            first_key = audio_key

        d = invoke(audio_key, "revtail120-raw")
        csv_mean = mean(info["preds"])
        app_raw = d.get("pressureRawKpa")
        diff = None if app_raw is None else abs(app_raw - csv_mean)
        rows.append((rel, info["true"], len(info["preds"]), d.get("hitsUsed"), csv_mean, app_raw))
        print(f"{rel:22s} {info['true']:5.0f} {len(info['preds']):6d}/{d.get('hitsUsed') or 0:<6d} "
              f"{csv_mean:8.1f} {str(app_raw):>8s} {('%.2f' % diff) if diff is not None else '  -':>6s}  {d.get('preprocess')}")

        if d.get("preprocess") != "reverb_v1":
            failures.append(f"{rel}: preprocess={d.get('preprocess')} (expect reverb_v1)")
        if d.get("hitsUsed") != len(info["preds"]):
            failures.append(f"{rel}: hits app={d.get('hitsUsed')} csv={len(info['preds'])}")
        if diff is None or diff > 0.5:
            failures.append(f"{rel}: app vs SageMaker-recorded mean diff {diff} > 0.5 kPa")

    # (c) 同一キーで exp3v2-cal に切替 → 値が変わり preprocess も変わる
    if first_key:
        d_rev = invoke(first_key, "revtail120-raw")
        d_e3 = invoke(first_key, "exp3v2-cal")
        print(f"\nswitch check on {first_key.split('/')[-1]}:")
        print(f"  revtail120-raw : raw={d_rev.get('pressureRawKpa')} preprocess={d_rev.get('preprocess')} model={d_rev.get('modelId')}")
        print(f"  exp3v2-cal     : kPa={d_e3.get('pressureKpa')} raw={d_e3.get('pressureRawKpa')} preprocess={d_e3.get('preprocess')} model={d_e3.get('modelId')}")
        if d_e3.get("preprocess") != "logmel_v1":
            failures.append(f"switch: exp3v2 preprocess={d_e3.get('preprocess')}")
        if d_rev.get("pressureRawKpa") == d_e3.get("pressureRawKpa"):
            failures.append("switch: revtail and exp3v2 returned identical raw values")

    # (e) 集計: アプリ経由の per-recording MAE と、CSV由来の per-recording MAE
    ok_rows = [r for r in rows if r[5] is not None]
    if ok_rows:
        mae_app = mean(abs(r[5] - r[1]) for r in ok_rows)
        mae_csv = mean(abs(r[4] - r[1]) for r in ok_rows)
        print(f"\nper-recording MAE vs true: app={mae_app:.2f} kPa / SageMaker-recorded={mae_csv:.2f} kPa "
              f"(job per-hit holdout MAE=6.75)")

    # cleanup（このスクリプトが上げたものだけ削除）
    for key in uploaded:
        s3.delete_object(Bucket=bucket, Key=key)
        base = key.split("/")[-1].rsplit(".", 1)[0]
        s3.delete_object(Bucket=bucket, Key=f"results/smoketest/{base}.json")

    print()
    if failures:
        print("FAILURES:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("ENDPOINT SWITCH + EXTERNAL-PARITY OK")


if __name__ == "__main__":
    main()
