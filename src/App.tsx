import { type ChangeEvent, useEffect, useRef, useState } from "react";
import {
  estimatePressure,
  loadModels,
  saveLabel,
  type InferenceResult,
  type ModelInfo,
} from "./inference";
import { Waveform } from "./Waveform";
import { isDeveloper } from "./userGroups";

interface AppProps {
  signOut?: () => void;
  username?: string;
}

type Source = "record" | "file" | null;
type FileMode = "single" | "all" | "pick";

function App({ signOut, username }: AppProps) {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [modelId, setModelId] = useState("");
  const [modelsError, setModelsError] = useState("");
  const [isDev, setIsDev] = useState(false);

  const [source, setSource] = useState<Source>(null);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [audioLabel, setAudioLabel] = useState("");
  const [recording, setRecording] = useState(false);
  const recRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<InferenceResult | null>(null);

  const [mode, setMode] = useState<FileMode>("all");
  const [picked, setPicked] = useState(0);

  const [actualKpa, setActualKpa] = useState("");
  const [labelSaving, setLabelSaving] = useState(false);
  const [labelSaved, setLabelSaved] = useState(false);
  const [labelError, setLabelError] = useState("");

  useEffect(() => {
    // モデル一覧は全ユーザーで取得する(一般ユーザーも default モデルの
    // modelId をセットしないと推定ボタンが有効にならない)
    loadModels()
      .then((m) => {
        setModels(m.models);
        setModelId(m.default ?? m.models[0]?.id ?? "");
      })
      .catch((e) => setModelsError(e instanceof Error ? e.message : String(e)));
    isDeveloper().then(setIsDev);
  }, []);

  // --- 表示する値（結果ボックスは常時表示、推定後に値が入る） ---
  const perHit = result?.perHitKpaCal ?? result?.perHitKpa ?? [];
  let shownValue: number | null = null;
  let shownSub = "";
  if (loading) {
    shownSub = "推定中…";
  } else if (result) {
    if (source === "record") {
      shownValue = result.pressureKpa;
    } else if (mode === "all") {
      shownValue = result.pressureKpa;
      shownSub = `ファイル全${result.hitsUsed}打の平均`;
    } else if (mode === "single") {
      shownValue = perHit[0] ?? result.pressureKpa;
      shownSub = "1打目の推定";
    } else {
      shownValue = perHit[picked] ?? null;
      shownSub = `${picked + 1}打目`;
    }
  }

  function resetForNewInput() {
    setResult(null);
    setError("");
    setLabelSaved(false);
    setLabelError("");
    setActualKpa("");
    setPicked(0);
    setMode("all");
  }

  async function startRec() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const rec = new MediaRecorder(stream);
      chunksRef.current = [];
      rec.ondataavailable = (e) => {
        if (e.data.size) chunksRef.current.push(e.data);
      };
      rec.onstop = () => {
        setAudioBlob(new Blob(chunksRef.current, { type: rec.mimeType || "audio/webm" }));
        setSource("record");
        setAudioLabel("録音した音声");
        stream.getTracks().forEach((t) => t.stop());
      };
      rec.start();
      recRef.current = rec;
      setRecording(true);
      resetForNewInput();
    } catch {
      setError("マイクにアクセスできませんでした。ブラウザの権限を確認してください。");
    }
  }

  function stopRec() {
    recRef.current?.stop();
    setRecording(false);
  }

  function onPickFile(e: ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (f) {
      setAudioBlob(f);
      setSource("file");
      setAudioLabel(f.name);
      resetForNewInput();
    }
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
      setPicked(0);
      setMode("all");
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
        estimatedKpa: shownValue,
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
      <header className="topbar">
        <h1>打音検知</h1>
        <div className="user-box">
          {username && <span className="user">{username}</span>}
          <button className="link" onClick={signOut}>
            ログアウト
          </button>
        </div>
      </header>

      {/* モデル選択(developers グループのみ表示。一般ユーザーは manifest の
          default モデルで推定。エラーは全ユーザーに見せる — 取得失敗時に推定
          ボタンが無効のまま無言になるのを防ぐ) */}
      {modelsError ? (
        <div className="error">モデル一覧の取得に失敗: {modelsError}</div>
      ) : isDev ? (
        <select
          className="model-select"
          value={modelId}
          onChange={(e) => setModelId(e.target.value)}
          disabled={loading}
        >
          {models.map((m) => (
            <option key={m.id} value={m.id}>
              {m.name}
            </option>
          ))}
        </select>
      ) : null}

      {/* 入力：録音は大きく、ファイル選択は小さくサブ的に */}
      <div className="input-row">
        {recording ? (
          <button className="rec-btn recording" onClick={stopRec}>
            ■ 録音を停止
          </button>
        ) : (
          <button className="rec-btn" onClick={startRec} disabled={loading}>
            ● 録音する
          </button>
        )}
        <label className="file-link">
          ファイル選択
          <input type="file" accept="audio/*" onChange={onPickFile} hidden />
        </label>
      </div>
      {recording ? (
        <div className="status-chip rec">● 録音中… タイヤを1回はっきり叩いてください</div>
      ) : audioBlob && source === "record" ? (
        <div className="status-chip ok">✓ 録音完了</div>
      ) : audioBlob && source === "file" ? (
        <div className="status-chip ok">✓ ファイル選択済み: {audioLabel}</div>
      ) : null}

      {/* 推定ボタン */}
      <button
        className="estimate-btn"
        onClick={onEstimate}
        disabled={!audioBlob || !modelId || loading}
      >
        {loading ? "推定中…" : "空気圧を推定する"}
      </button>

      {error && <div className="error">{error}</div>}

      {/* ファイルの場合：3種類の推定モード（結果ボックスの上に配置） */}
      {result && source === "file" && (
        <div className="file-modes">
          <div className="mode-toggle">
            <button className={mode === "single" ? "on" : ""} onClick={() => setMode("single")}>
              1打音
            </button>
            <button className={mode === "all" ? "on" : ""} onClick={() => setMode("all")}>
              全打音
            </button>
            <button className={mode === "pick" ? "on" : ""} onClick={() => setMode("pick")}>
              波形から選択
            </button>
          </div>
          {mode === "pick" && result.samples && (
            <Waveform
              samples={result.samples}
              sampleRate={result.sampleRate ?? 22050}
              peakSeconds={result.peakSeconds ?? []}
              picked={picked}
              onPick={setPicked}
            />
          )}
        </div>
      )}

      {/* 推定結果ボックス（常時表示） */}
      <div className="result-box">
        <div className={shownValue != null ? "pressure" : "pressure empty"}>
          {shownValue != null ? shownValue : "—"}
          <span className="unit"> kPa</span>
        </div>
        {shownSub && <div className="result-sub">{shownSub}</div>}
      </div>

      {/* 実測値の登録（結果ボックスの下・再学習用） */}
      {result && shownValue != null && (
        <div className="label-row">
          <div className="kpa-field">
            <input
              type="number"
              inputMode="decimal"
              placeholder="実測値（例: 240）"
              value={actualKpa}
              onChange={(e) => setActualKpa(e.target.value)}
            />
            <span className="kpa-unit">kPa</span>
          </div>
          <button onClick={onSaveLabel} disabled={labelSaving || labelSaved}>
            {labelSaved ? "保存済 ✓" : labelSaving ? "保存中…" : "保存"}
          </button>
        </div>
      )}
      {labelError && <div className="error">{labelError}</div>}
    </div>
  );
}

export default App;
