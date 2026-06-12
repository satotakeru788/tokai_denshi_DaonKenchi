import { fetchAuthSession } from 'aws-amplify/auth';

/** amplify/backend.ts の CfnUserPoolGroup(groupName)と一致させること。 */
export const DEVELOPERS_GROUP = 'developers';

/**
 * ログイン中ユーザーが developers グループ所属かどうか。
 * IDトークンの cognito:groups クレームで判定する(追加のサーバー呼び出しなし)。
 * グループ追加・削除の反映はトークン再発行後(再ログイン、または最長1時間の
 * 期限切れ→自動リフレッシュ後)。
 * UIの出し分け専用 — 実際のアクセス制御は従来どおりIAMが担う。
 */
export async function isDeveloper(): Promise<boolean> {
  try {
    const { tokens } = await fetchAuthSession();
    const groups = tokens?.idToken?.payload['cognito:groups'];
    return Array.isArray(groups) && groups.includes(DEVELOPERS_GROUP);
  } catch {
    return false;
  }
}
