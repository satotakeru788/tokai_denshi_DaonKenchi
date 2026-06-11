import { defineStorage } from '@aws-amplify/backend';

/**
 * S3 storage for the DaonKenchi MVP.
 *
 * Prefix layout (entity_id = Cognito identity id, per-user isolation):
 *   audio/{entity_id}/*   原音(m4a/webm) + 解析用 22050Hz WAV。本人 read/write、推論関数 read。
 *   results/{entity_id}/* 推定結果 JSON。本人 read、推論関数 write（蓄積）。
 *   labels/{entity_id}/*  実測値・タイヤ種別 JSON（再学習用）。本人 read/write。
 *   models/*              モデル置き場（model.onnx / .data / meta.json / manifest.json）。
 *                         全認証ユーザ read（フロントは manifest.json のみ取得）。推論関数 read。
 *
 * 推論 Lambda は素の CDK lambda.Function なので、バケットへのアクセスは
 * backend.ts 側の bucket.grantReadWrite(inferFn) で付与する（ここでは利用者=Cognito
 * ユーザのアクセスのみ定義する）。
 */
export const storage = defineStorage({
  name: 'daonkentiMvpStorage',
  access: (allow) => ({
    'audio/{entity_id}/*': [
      allow.entity('identity').to(['read', 'write', 'delete']),
    ],
    'results/{entity_id}/*': [
      allow.entity('identity').to(['read']),
    ],
    'labels/{entity_id}/*': [
      allow.entity('identity').to(['read', 'write', 'delete']),
    ],
    'models/*': [
      allow.authenticated.to(['read']),
    ],
  }),
});
