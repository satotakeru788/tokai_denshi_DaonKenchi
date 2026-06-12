import { fetchAuthSession } from 'aws-amplify/auth';
import { getUrl, uploadData } from 'aws-amplify/storage';
import { AwsClient } from 'aws4fetch';
import outputs from '../amplify_outputs.json';

const custom = (outputs as { custom?: { inferenceUrl?: string; inferenceRegion?: string } }).custom ?? {};
const INFERENCE_URL: string = custom.inferenceUrl ?? '';
const REGION: string = custom.inferenceRegion ?? 'ap-northeast-1';

// 音声は S3 経由で Lambda に渡すため Function URL の 6MB 制限は受けない。
// ファイル全体（全打音）を解析対象にする（長さの上限は設けない）。
// 22050Hz への固定ダウンサンプルはやめ、ブラウザのデコードレート
// （通常 44.1k/48kHz）のまま WAV 化して保存する — リサンプルや特徴量化は
// モデルごとの前処理プロファイル（Lambda側）が担う。

export interface ModelInfo {
  id: string;
  name: string;
  desc?: string;
  path?: string;
  calibrated?: boolean;
}

export interface ManifestData {
  models: ModelInfo[];
  default?: string;
}

/** Lambda が返す推定結果（results/{entity}/{id}.json と同形）。 */
export interface InferenceResult {
  pressureKpa: number | null;
  pressureRawKpa?: number | null;
  hitsUsed: number;
  perHitKpa: number[];
  perHitKpaCal?: number[];
  peakSeconds?: number[];
  calibrated?: boolean;
  isMock: boolean;
  modelTag?: string;
  durationSec?: number;
  modelId: string;
  modelName?: string;
  audioKey: string;
  resultKey: string;
  id: string;
  createdAt?: string;
  error?: string;
  /** アップロードしたモノラル波形（デコードレートのまま。波形表示・打音選択用） */
  samples?: Float32Array;
  sampleRate?: number;
}

// ---------------------------------------------------------------------------
// Audio decode -> mono 16-bit PCM WAV（デコードレート維持・リサンプルしない）
// ---------------------------------------------------------------------------
async function decodeToMono(blob: Blob): Promise<{ samples: Float32Array; sampleRate: number }> {
  const arrayBuf = await blob.arrayBuffer();
  const AC: typeof AudioContext =
    window.AudioContext ??
    (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
  const ctx = new AC();
  let decoded: AudioBuffer;
  try {
    decoded = await ctx.decodeAudioData(arrayBuf.slice(0));
  } finally {
    void ctx.close();
  }
  const sampleRate = decoded.sampleRate;
  const length = Math.max(1, Math.ceil(decoded.duration * sampleRate));
  const offline = new OfflineAudioContext(1, length, sampleRate);
  const src = offline.createBufferSource();
  src.buffer = decoded;
  src.connect(offline.destination);
  src.start();
  const rendered = await offline.startRendering();
  return { samples: rendered.getChannelData(0), sampleRate };
}

function encodeWav(samples: Float32Array, sampleRate: number): Uint8Array<ArrayBuffer> {
  const bytes = new Uint8Array(44 + samples.length * 2);
  const view = new DataView(bytes.buffer);
  const writeStr = (off: number, s: string) => {
    for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i));
  };
  const dataLen = samples.length * 2;
  writeStr(0, 'RIFF');
  view.setUint32(4, 36 + dataLen, true);
  writeStr(8, 'WAVE');
  writeStr(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, 'data');
  view.setUint32(40, dataLen, true);
  let off = 44;
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    off += 2;
  }
  return bytes;
}

function extFromBlob(blob: Blob): string {
  if (blob instanceof File && blob.name.includes('.')) {
    return blob.name.split('.').pop()!.toLowerCase();
  }
  const sub = (blob.type.split('/')[1] ?? '').split(';')[0];
  if (sub === 'mp4' || sub === 'x-m4a') return 'm4a';
  if (sub === 'mpeg') return 'mp3';
  if (sub === 'webm') return 'webm';
  if (sub === 'wav' || sub === 'x-wav') return 'wav';
  if (sub === 'ogg') return 'ogg';
  return 'bin';
}

// ---------------------------------------------------------------------------
// Storage helpers
// ---------------------------------------------------------------------------

/** models/manifest.json を Storage から取得（モデル選択プルダウン用）。
 * downloadData は「固定URL＋署名ヘッダ」の GET になりブラウザの HTTP
 * ヒューリスティックキャッシュに当たるため、モデル追加が数時間反映されない
 * ことがある。毎回変わる署名付きURL＋ cache:no-store で常に最新を取得する。 */
export async function loadModels(): Promise<ManifestData> {
  const { url } = await getUrl({ path: 'models/manifest.json', options: { expiresIn: 60 } });
  const res = await fetch(url.toString(), { cache: 'no-store' });
  if (!res.ok) {
    throw new Error(`manifest取得に失敗しました (HTTP ${res.status})`);
  }
  const data = (await res.json()) as ManifestData;
  return { models: data.models ?? [], default: data.default };
}

interface LabelInput {
  audioKey: string;
  resultKey: string;
  id: string;
  modelId: string;
  estimatedKpa: number | null;
  actualKpa: number;
}

/** 再学習用ラベル（実測値）を labels/{identityId}/{id}.json に保存。 */
export async function saveLabel(input: LabelInput): Promise<string> {
  const { identityId } = await fetchAuthSession();
  const key = `labels/${identityId}/${input.id}.json`;
  const payload = { ...input, identityId, createdAt: new Date().toISOString() };
  await uploadData({
    path: key,
    data: JSON.stringify(payload),
    options: { contentType: 'application/json' },
  }).result;
  return key;
}

// ---------------------------------------------------------------------------
// 推定: decode -> WAV化 -> S3アップロード(原音+WAV) -> SigV4署名でLambda呼び出し
// ---------------------------------------------------------------------------
export async function estimatePressure(audio: Blob, modelId: string): Promise<InferenceResult> {
  if (!INFERENCE_URL) {
    throw new Error('推論エンドポイント未設定です（バックエンド未デプロイ）。');
  }

  const session = await fetchAuthSession();
  const { identityId, credentials } = session;
  if (!identityId || !credentials) {
    throw new Error('ログインセッションが無効です。再ログインしてください。');
  }

  const { samples: mono, sampleRate } = await decodeToMono(audio);
  if (mono.length < sampleRate * 0.2) {
    throw new Error('音声が短すぎます。タイヤを叩いた音を録音してください。');
  }
  const wav = encodeWav(mono, sampleRate);

  const id = crypto.randomUUID();
  const audioKey = `audio/${identityId}/${id}.wav`;
  const origKey = `audio/${identityId}/${id}.orig.${extFromBlob(audio)}`;

  // 原音（将来の再学習で別環境decode用）と解析用WAVの両方を保存。
  await Promise.all([
    uploadData({
      path: audioKey,
      data: new Blob([wav], { type: 'audio/wav' }),
      options: { contentType: 'audio/wav' },
    }).result,
    uploadData({
      path: origKey,
      data: audio,
      options: { contentType: audio.type || 'application/octet-stream' },
    }).result,
  ]);

  // ログイン中のCognito資格情報でSigV4署名し、IAM認証のFunction URLを呼ぶ。
  const aws = new AwsClient({
    accessKeyId: credentials.accessKeyId,
    secretAccessKey: credentials.secretAccessKey,
    sessionToken: credentials.sessionToken,
    region: REGION,
    service: 'lambda',
  });
  const res = await aws.fetch(INFERENCE_URL, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ audioKey, modelId }),
  });

  const data = (await res.json().catch(() => ({}))) as Partial<InferenceResult> & { error?: string };
  if (!res.ok || data.pressureKpa == null) {
    if (data.error === 'no_hits_detected') {
      throw new Error('打音が検出できませんでした。タイヤをはっきり叩いて録音してください。');
    }
    throw new Error(data.error ? `推定エラー: ${data.error}` : '推定に失敗しました。');
  }
  return {
    ...(data as InferenceResult),
    audioKey: data.audioKey ?? audioKey,
    id: data.id ?? id,
    isMock: false,
    samples: mono,
    sampleRate,
  };
}
