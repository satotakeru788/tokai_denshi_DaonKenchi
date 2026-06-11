# DaonKenchi MVP

タイヤを叩いた打音（音声）から空気圧 [kPa] を推定する試用版アプリ。
研究/検証リポジトリ `DaonKenti-app` の推論コア（前処理 + ONNX + 校正）を流用し、
MVP に必要な **ログイン認証 / S3保存 / モデル切替 / S3対応推論Lambda / 再学習データ蓄積** を実装。

## 構成

- **フロント**: React + Vite + TypeScript、Amplify UI `<Authenticator>`（email ログイン必須）。
- **認証**: Amplify `defineAuth`（Cognito User Pool + Identity Pool）。
- **保存**: Amplify `defineStorage`（S3）。プレフィックスは利用者(Cognito identity)単位で分離。
  - `audio/{entity_id}/*` … 原音(m4a/webm) + 解析用 22050Hz WAV（本人 read/write、推論関数 read）
  - `results/{entity_id}/*` … 推定結果 JSON（本人 read、推論関数 write）
  - `labels/{entity_id}/*` … 実測値・タイヤ種別 JSON（再学習用、本人 read/write）
  - `models/*` … モデル + `manifest.json`（全認証ユーザ read、推論関数 read）
- **推論**: Python 3.12 Lambda（numpy/scipy/onnxruntime レイヤ）を **Function URL (authType=AWS_IAM)** で公開。
  フロントはログイン中の Cognito 資格情報で **SigV4 署名**（`aws4fetch`）して同期呼び出し。
- **モデル切替**: `models/manifest.json` のレジストリ（`id/name/desc/path/calibrated`）。
  MVP は同一 ONNX で「校正あり / 校正なし」の2エントリ。追加は manifest + S3 にファイルを足すだけ。

### 処理の流れ（推定ボタン）
1. 録音/ファイルをブラウザで 22050Hz モノラル WAV にデコード（`src/inference.ts`）。
2. 原音 + WAV を `audio/{identityId}/{id}.*` に `uploadData`。
3. `{audioKey, modelId}` を SigV4 署名して Function URL を呼び出し。
4. Lambda が S3 から WAV とモデルを取得 → `audio_infer.run_inference` → 結果を
   `results/{identityId}/{id}.json` に保存しつつレスポンスで即返す。
5. 画面に結果表示。任意で実測値・タイヤ種別を `labels/{identityId}/{id}.json` に保存。

## セットアップ

前提: Node 20+, Python 3.12+, AWS CLI v2、AWS プロファイル `daonkenti-dev`（ap-northeast-1）。

```bash
npm install

# 依存レイヤ（numpy/scipy/onnxruntime）を再生成する場合のみ（既に amplify/layers/pydeps にあれば不要）
python tools/build_layer.py

# バックエンドをデプロイ（auth/storage/function を作成し amplify_outputs.json を生成）
npm run sandbox:once          # = ampx sandbox --once --profile daonkenti-dev

# モデル + manifest を S3 の models/ にアップロード（デプロイ後に1回）
python tools/seed_models.py   # = npm run seed:models

# ローカル開発サーバ
npm run dev
```

> `amplify/layers/pydeps/`（195MB）と `amplify_outputs.json` は `.gitignore` 済み。
> レイヤは `tools/build_layer.py` で、outputs は `ampx sandbox` で再生成できる。

## 動作確認

```bash
npm run build                 # 型チェック + ビルド（要 amplify_outputs.json）
python tools/test_endpoint.py # デプロイ済みエンドポイントの合成WAVスモークテスト
```

`tools/test_endpoint.py` は: 合成打音WAVをS3へ→SigV4署名でFunction URL呼び出し（校正あり/なしで
結果が変わること）→未署名POSTが403で拒否されること→結果JSONがS3に書かれることを確認する。

## モデルの追加

手順の概要（再デプロイ不要）:

1. `seed/models/<id>/` に `model.onnx` / `model.onnx.data` / `meta.json` を置く。
2. `seed/models/manifest.json` の `models` に `{id,name,desc,path,calibrated}` を追記。
3. `python tools/seed_models.py` で S3 に反映。

⚠️ モデルは本アプリの入出力契約（ONNX / 入力=log-mel `[batch,1,n_mels,T]` / 出力=1打あたりkPa）を
満たす必要があります。**他の人がモデルを追加する場合の詳細手順・契約・検証方法・トラブル対処**は
[docs/モデル追加手順.md](docs/モデル追加手順.md) を参照してください。

## 主なファイル

- `amplify/auth/resource.ts` — email 認証
- `amplify/storage/resource.ts` — S3 アクセス定義
- `amplify/functions/infer/{handler.py, audio_infer.py}` — S3対応推論Lambda（モデルはS3から取得）
- `amplify/layers/pydeps/` — Lambda 依存レイヤ（`tools/build_layer.py` 生成）
- `amplify/backend.ts` — auth/storage/function + Function URL(IAM) + 権限 + env
- `src/{main.tsx, App.tsx, inference.ts}` — 認証・録音/選択・モデル選択・結果・再学習フォーム
- `tools/{build_layer.py, seed_models.py, test_endpoint.py}`
- `seed/models/` — S3へ投入するモデルと manifest

## License

MIT-0. See LICENSE.
