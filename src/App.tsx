import { useEffect, useRef, useState } from "react";
import {
  estimatePressure,
  loadModels,
  saveLabel,
  type InferenceResult,
  type ModelInfo,
} from "./inference";

interface AppProps {
  signOut?: () => void;
  username?: string;
}

function App({ signOut, username }: AppProps) {
  // --- モデル一覧 ---
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [modelId, setModelId] = useState("");
  const [modelsError, setModelsError] = useState("");

  // --- 音声 ---
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [audioLabel, setAudioLabel] = useState("");
  const [recording, setRecording] = useState(false);
  const recRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  // --- 推定 ---
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<InferenceResult | null>(null);

  // --- 再学習ラベル ---
  const [actualKpa, setActualKpa] = useState("");
  const [labelSaving, setLabelSaving] = useState(false);
  const [labelSaved, setLabelSaved] = useState(false);
  const [labelError, setLabelError] = useState("");

  useEffect(() => {
    loadModels()
      .then((m) => {
        setModels(m.models);
        setModelId(m.default ?? m.models[0]?.id ?? "");
      })
      .catch((e) => setModelsError(e instanceof Error ? e.message : String(e)));
  }, []);

  async function startRec() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const rec = new MediaRecorder(stream);
      chunksRef.current = [];
      rec.ondataavailable = (e) => {
        if (e.data.size) chunksRef.current.push(e.data);
      };
      rec.onstop = () => {
        const type = rec.mimeType || "audio/webm";
        setAudioBlob(new Blob(chunksRef.current, { type }));
        setAudioLabel("録音した音声");
        stream.getTracks().forEach((t) => t.stop());
      };
      rec.start();
      recRef.current = rec;
      setRecording(true);
      resetResult();
    } catch {
      setError("マイクにアクセスできませんでした。ブラウザの権限を確認してください。");
    }
  }

  function stopRec() {
    recRef.current?.stop();
    setRecording(false);
  }

  function onPickFile(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (f) {
      setAudioBlob(f);
      setAudioLabel(f.name);
      resetResult();
    }
  }

  function resetResult() {
    setResult(null);
    setError("");
    setLabelSaved(false);
    setLabelError("");
    setActualKpa("");
  }

  async function onEstimate() {
    if (!audioBlob || !modelId) return;
    setLoading(true);
    setError("");
    setResult(null);
    setLabelSaved(false);
    try {
      const r = await estimatePressure(audioBlob, modelId);
      setResult(r);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  async function onSaveLabel() {
    if (!result) return;
    const kpa = parseFloat(actualKpa);
    if (Number.isNaN(kpa)) {
      setLabelError("実測値(kPa)を数値で入力してください。");
      return;
    }
    setLabelSaving(true);
    setLabelError("");
    try {
      await saveLabel({
        audioKey: result.audioKey,
        resultKey: result.resultKey,
        id: result.id,
        modelId: result.modelId,
        estimatedKpa: result.pressureKpa,
        actualKpa: kpa,
      });
      setLabelSaved(true);
    } catch (e) {
      setLabelError(e instanceof Error ? e.message : String(e));
    } finally {
      setLabelSaving(false);
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <div>
          <h1>DaonKenchi</h1>
          <p className="subtitle">タイヤ打音から空気圧を推定</p>
        </div>
        <div className="user-box">
          {username && <span className="user">{username}</span>}
          <button className="link" onClick={signOut}>
            ログアウト
          </button>
        </div>
      </header>

      <main className="card">
        {/* 1. 音声入力 */}
        <section>
          <h2>1. タイヤ打音を入力</h2>
          <div className="row">
            {recording ? (
              <button className="btn danger" onClick={stopRec}>
                ■ 録音停止
              </button>
            ) : (
              <button className="btn" onClick={startRec} disabled={loading}>
                ● 録音開始
              </button>
            )}
            <label className="btn ghost file">
              ファイル選択
              <input type="file" accept="audio/*" onChange={onPickFile} hidden />
            </label>
          </div>
          {audioLabel && <p className="hint">選択中: {audioLabel}</p>}
          <p className="hint">タイヤを数回（3回以上）はっきり叩いた音を入力してください。</p>
        </section>

        {/* 2. モデル選択 */}
        <section>
          <h2>2. モデルを選択</h2>
          {modelsError ? (
            <p className="error">モデル一覧の取得に失敗しました: {modelsError}</p>
          ) : (
            <select value={modelId} onChange={(e) => setModelId(e.target.value)} disabled={loading}>
              {models.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name}
                </option>
              ))}
            </select>
          )}
          {models.find((m) => m.id === modelId)?.desc && (
            <p className="hint">{models.find((m) => m.id === modelId)?.desc}</p>
          )}
        </section>

        {/* 3. 推定 */}
        <section>
          <button className="btn primary wide" onClick={onEstimate} disabled={!audioBlob || !modelId || loading}>
            {loading ? "推定中…" : "空気圧を推定する"}
          </button>
          {error && <p className="error">{error}</p>}
        </section>

        {/* 結果 */}
        {result && (
          <section className="result">
            <h2>推定結果</h2>
            <div className="pressure">
              {result.pressureKpa}
              <span className="unit"> kPa</span>
            </div>
            <dl className="meta">
              <div>
                <dt>使用モデル</dt>
                <dd>{result.modelName ?? result.modelId}</dd>
              </div>
              <div>
                <dt>使用打数</dt>
                <dd>{result.hitsUsed} 打</dd>
              </div>
              {result.pressureRawKpa != null && (
                <div>
                  <dt>校正前</dt>
                  <dd>{result.pressureRawKpa} kPa</dd>
                </div>
              )}
              <div>
                <dt>校正</dt>
                <dd>{result.calibrated ? "あり" : "なし"}</dd>
              </div>
              {result.durationSec != null && (
                <div>
                  <dt>解析長</dt>
                  <dd>{result.durationSec} 秒</dd>
                </div>
              )}
            </dl>
            {result.perHitKpa?.length > 0 && (
              <p className="hint">各打: {result.perHitKpa.join(" / ")} kPa</p>
            )}

            {/* 再学習フォーム */}
            <div className="relearn">
              <h3>実測値の登録（再学習用・任意）</h3>
              <div className="form-grid">
                <label className="full">
                  実測空気圧 [kPa]
                  <input
                    type="number"
                    inputMode="decimal"
                    value={actualKpa}
                    onChange={(e) => setActualKpa(e.target.value)}
                    placeholder="例: 240"
                  />
                </label>
              </div>
              <button className="btn" onClick={onSaveLabel} disabled={labelSaving || labelSaved}>
                {labelSaved ? "保存しました ✓" : labelSaving ? "保存中…" : "実測値を保存"}
              </button>
              {labelError && <p className="error">{labelError}</p>}
            </div>
          </section>
        )}
      </main>

      <footer className="app-footer">DaonKenchi MVP — 試用版</footer>
    </div>
  );
}

export default App;
