import { type ChangeEvent, useRef, useState, useEffect } from "react";
import {
  estimatePressure,
  loadModels,
  saveLabel,
  setDefaultModel,
  TIRE_INCHES,
  type InferenceResult,
  type ModelInfo,
  type TireInch,
} from "./inference";
import { Waveform } from "./Waveform";
import { isDeveloper } from "./userGroups";

interface AppProps {
  signOut?: () => void;
  username?: string;
}

type Source = "record" | "file" | null;
type FileMode = "single" | "all" | "pick";

/** tireInch 未宣言のモデルは 16in として扱う（後方互換）。 */
function modelsOfInch(ms: ModelInfo[], inch: TireInch): ModelInfo[] {
  return ms.filter((m) => (m.tireInch ?? 16) === inch);
}

/** あるサイズで使うモデルidを決める。サイズ別代表 → そのサイズ先頭 → 全体既定 の順。 */
function pickModelForInch(
  ms: ModelInfo[],
  defaults: Record<string, string>,
  fallback: string | undefined,
  inch: TireInch,
): string {
  const ofSize = modelsOfInch(ms, inch);
  const d = defaults[String(inch)];
  if (d && ofSize.some((m) => m.id === d)) return d;
  if (ofSize[0]) return ofSize[0].id;
  return fallback ?? ms[0]?.id ?? "";
}

const INCH_LABEL: Record<TireInch, string> = {
  16: "16インチ（普通車）",
  15: "15インチ（軽自動車）",
};

// --- アイコン（依存追加なしのインラインSVG） ---
function IconGear() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <circle cx="12" cy="12" r="3" />
      <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
    </svg>
  );
}

function IconMic() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <rect x="9" y="2" width="6" height="12" rx="3" />
      <path d="M5 10a7 7 0 0 0 14 0" />
      <line x1="12" y1="19" x2="12" y2="22" />
    </svg>
  );
}

function IconStop() {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
      <rect x="6" y="6" width="12" height="12" rx="2.5" />
    </svg>
  );
}

function IconFile() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <polyline points="14 2 14 8 20 8" />
    </svg>
  );
}

function App({ signOut, username }: AppProps) {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [modelId, setModelId] = useState("");
  const [modelsError, setModelsError] = useState("");
  const [isDev, setIsDev] = useState(false);
  // 管理者だけが切り替えられる: true=管理者画面 / false=一般ユーザー画面プレビュー
  const [adminView, setAdminView] = useState(true);

  // タイヤサイズ選択（全ユーザー）。選んだサイズの代表モデルで推論する。
  const [tireInch, setTireInch] = useState<TireInch>(16);
  const [defaults, setDefaults] = useState<Record<string, string>>({});
  const [manifestDefault, setManifestDefault] = useState<string | undefined>(undefined);

  // 管理者: サイズ別の代表モデル設定（16in / 15in 個別）
  const [adminDefault, setAdminDefault] = useState<Record<TireInch, string>>({ 16: "", 15: "" });
  const [savingInch, setSavingInch] = useState<TireInch | null>(null);
  const [defaultMsg, setDefaultMsg] = useState<{ inch: TireInch; text: string; ok: boolean } | null>(
    null,
  );

  const [source, setSource] = useState<Source>(null);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [audioLabel, setAudioLabel] = useState("");
  const [recording, setRecording] = useState(false);
  const recRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  // 一般ユーザー画面: 録音停止後に自動推定するためのフラグ
  const autoRef = useRef(false);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<InferenceResult | null>(null);

  const [mode, setMode] = useState<FileMode>("all");
  const [picked, setPicked] = useState(0);

  const [actualKpa, setActualKpa] = useState("");
  const [labelSaving, setLabelSaving] = useState(false);
  const [labelSaved, setLabelSaved] = useState(false);
  const [labelError, setLabelError] = useState("");

  // 設定ポップアップ（一般ユーザー画面）
  const [settingsOpen, setSettingsOpen] = useState(false);

  // 自動推定はコールバック（rec.onstop）から呼ぶため、最新の modelId/tireInch を
  // ref で参照してクロージャの古い値を避ける。
  const modelIdRef = useRef(modelId);
  const tireInchRef = useRef(tireInch);
  useEffect(() => {
    modelIdRef.current = modelId;
  }, [modelId]);
  useEffect(() => {
    tireInchRef.current = tireInch;
  }, [tireInch]);

  useEffect(() => {
    // モデル一覧は全ユーザーで取得する(一般ユーザーも default モデルの
    // modelId をセットしないと推定ボタンが有効にならない)
    loadModels()
      .then((m) => {
        const dfs = m.defaults ?? {};
        setModels(m.models);
        setDefaults(dfs);
        setManifestDefault(m.default);
        const inch: TireInch = 16;
        setTireInch(inch);
        setModelId(pickModelForInch(m.models, dfs, m.default, inch));
        setAdminDefault({
          16: pickModelForInch(m.models, dfs, m.default, 16),
          15: pickModelForInch(m.models, dfs, m.default, 15),
        });
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
    } else if (mode === "single") {
      shownValue = perHit[0] ?? result.pressureKpa;
    } else {
      shownValue = perHit[picked] ?? null;
    }
  }

  function setView(admin: boolean) {
    if (admin === adminView) return;
    setAdminView(admin);
    setDefaultMsg(null);
    if (!admin) {
      // 一般ユーザー画面: モデル上書きを解除し、サイズの代表モデルで推論
      setModelId(pickModelForInch(models, defaults, manifestDefault, tireInch));
    }
  }

  function onSelectInch(inch: TireInch) {
    if (inch === tireInch) return;
    setTireInch(inch);
    // サイズを変えたら、そのサイズの代表モデルに切り替える（dev はこの後上書き可）
    setModelId(pickModelForInch(models, defaults, manifestDefault, inch));
    resetForNewInput();
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
        const blob = new Blob(chunksRef.current, { type: rec.mimeType || "audio/webm" });
        setAudioBlob(blob);
        setSource("record");
        setAudioLabel("録音した音声");
        stream.getTracks().forEach((t) => t.stop());
        // 一般ユーザー画面: 録音を止めたら瞬時に推定を開始する
        if (autoRef.current) {
          autoRef.current = false;
          void runEstimate(blob, "record");
        }
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

  /** 一般ユーザー画面: 録音ボタン1つで 録音開始 / 停止+即推定 をトグル。 */
  function toggleRecord() {
    if (loading) return;
    if (recording) {
      autoRef.current = true; // 停止 → onstop で自動推定
      recRef.current?.stop();
      setRecording(false);
    } else {
      void startRec();
    }
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

  /** 一般ユーザー画面: ファイルを選んだら設定を閉じて即推定する。 */
  function onPickFileUser(e: ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (!f) return;
    setSettingsOpen(false);
    setAudioBlob(f);
    setSource("file");
    setAudioLabel(f.name);
    resetForNewInput();
    void runEstimate(f, "file");
    e.target.value = ""; // 同じファイルを連続で選べるように
  }

  /** 推定本体。blob と source を引数で受け、modelId/tireInch は最新値(ref)を使う。 */
  async function runEstimate(blob: Blob, src: Exclude<Source, null>) {
    const mid = modelIdRef.current;
    if (!blob || !mid) return;
    setLoading(true);
    setError("");
    setResult(null);
    setLabelSaved(false);
    try {
      const r = await estimatePressure(blob, mid, tireInchRef.current);
      setSource(src);
      setResult(r);
      setPicked(0);
      setMode("all");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  async function onEstimate() {
    if (!audioBlob || !source) return;
    await runEstimate(audioBlob, source);
  }

  async function onSetDefault(inch: TireInch) {
    const mid = adminDefault[inch];
    if (!mid) return;
    setSavingInch(inch);
    setDefaultMsg(null);
    try {
      await setDefaultModel(mid, inch);
      setDefaults((d) => ({ ...d, [String(inch)]: mid }));
      setDefaultMsg({ inch, text: "✓ 一般ユーザーの使用モデルを更新しました", ok: true });
    } catch (e) {
      setDefaultMsg({ inch, text: e instanceof Error ? e.message : String(e), ok: false });
    } finally {
      setSavingInch(null);
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

  // =========================================================================
  // 管理者画面（developers かつ adminView のとき）。従来UIのまま。
  // =========================================================================
  if (isDev && adminView) {
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

        {/* 画面切り替え（developers のみ）。選んだ方の画面になる。 */}
        <div className="mode-toggle view-switch">
          <button className={!adminView ? "on" : ""} onClick={() => setView(false)}>
            一般ユーザー
          </button>
          <button className={adminView ? "on" : ""} onClick={() => setView(true)}>
            管理者
          </button>
        </div>

        {/* タイヤサイズ選択（全ユーザー）。選んだサイズの代表モデルで推論する。 */}
        <div className="size-row">
          <span className="size-label">タイヤサイズ</span>
          <div className="mode-toggle size-toggle">
            {TIRE_INCHES.map((inch) => (
              <button
                key={inch}
                className={tireInch === inch ? "on" : ""}
                onClick={() => onSelectInch(inch)}
                disabled={loading}
              >
                {INCH_LABEL[inch]}
              </button>
            ))}
          </div>
        </div>

        {modelsError && <div className="error">モデル一覧の取得に失敗: {modelsError}</div>}

        {!modelsError && (
          <select
            className="model-select"
            value={modelId}
            onChange={(e) => setModelId(e.target.value)}
            disabled={loading}
          >
            {modelsOfInch(models, tireInch).map((m) => (
              <option key={m.id} value={m.id}>
                {m.name}
              </option>
            ))}
          </select>
        )}

        {/* 代表モデル設定（管理者画面のみ）。一般ユーザーが使うモデルを設定。 */}
        {!modelsError && (
          <div className="dev-defaults">
            <div className="dev-default">
              <span className="dev-default-label">一般ユーザー</span>
              <select
                value={adminDefault[tireInch]}
                onChange={(e) => {
                  setAdminDefault((a) => ({ ...a, [tireInch]: e.target.value }));
                  setDefaultMsg(null);
                }}
                disabled={savingInch !== null}
              >
                {modelsOfInch(models, tireInch).map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.name}
                  </option>
                ))}
              </select>
              <button onClick={() => onSetDefault(tireInch)} disabled={savingInch !== null}>
                {savingInch === tireInch ? "設定中…" : "設定"}
              </button>
              {defaultMsg?.inch === tireInch && (
                <span className={"dev-default-msg" + (defaultMsg.ok ? " ok" : " err")}>
                  {defaultMsg.text}
                </span>
              )}
            </div>
          </div>
        )}

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

        <button
          className="estimate-btn"
          onClick={onEstimate}
          disabled={!audioBlob || !modelId || loading}
        >
          {loading ? "推定中…" : "空気圧を推定する"}
        </button>

        {error && <div className="error">{error}</div>}

        {result && source === "file" && (
          <div className="file-modes">
            <div className="mode-toggle">
              <button className={mode === "single" ? "on" : ""} onClick={() => setMode("single")}>
                1打音
              </button>
              <button className={mode === "all" ? "on" : ""} onClick={() => setMode("all")}>
                全打音平均
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

        <div className="result-box">
          <div className={shownValue != null ? "pressure" : "pressure empty"}>
            {shownValue != null ? shownValue : "—"}
            <span className="unit"> kPa</span>
          </div>
          {shownSub && <div className="result-sub">{shownSub}</div>}
        </div>

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

  // =========================================================================
  // 一般ユーザー画面（ガラス調 / 録音中心）。
  // =========================================================================
  return (
    <div className="user-screen">
      {/* 背景オーロラ（ぼかしたカラーブロブ）。ガラス越しに色が透けるための“背後の色変化”。 */}
      <div className="bg-orbs" aria-hidden="true" />

      {/* 右上: 設定 */}
      <button className="gear-btn" onClick={() => setSettingsOpen(true)} aria-label="設定">
        <IconGear />
      </button>

      {/* 中央: 推定結果 */}
      <div className="user-main">
        {modelsError && <div className="glass-note err">モデル取得に失敗: {modelsError}</div>}

        <div className="result-glass">
          {loading ? (
            <div className="estimating">
              <span className="spinner" aria-hidden="true" />
            </div>
          ) : recording ? (
            <div className="rec-indicator" role="status" aria-label="録音中">
              <span />
              <span />
              <span />
              <span />
              <span />
            </div>
          ) : shownValue != null ? (
            <>
              <div className="pressure">
                <span className="pressure-num">{shownValue}</span>
                <span className="unit">kPa</span>
              </div>
              {shownSub && <div className="result-sub">{shownSub}</div>}
            </>
          ) : null}
        </div>

        {error && <div className="glass-note err">{error}</div>}

        {/* ファイル選択時のみ: 打音モード切替 + 波形（控えめ） */}
        {result && source === "file" && (
          <div className="file-modes-glass">
            <div className="glass-seg">
              <button className={mode === "single" ? "on" : ""} onClick={() => setMode("single")}>
                1打音
              </button>
              <button className={mode === "all" ? "on" : ""} onClick={() => setMode("all")}>
                全打音平均
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

        {/* 実測値の登録（控えめ・再学習用） */}
        {result && shownValue != null && (
          <div className="label-glass">
            <div className="kpa-field-glass">
              <input
                type="number"
                inputMode="decimal"
                placeholder="実測値"
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
        {labelError && <div className="glass-note err">{labelError}</div>}
      </div>

      {/* 下部中央: 録音ボタン（最も目立つ） */}
      <div className="user-dock">
        <button
          className={"record-fab" + (recording ? " recording" : "")}
          onClick={toggleRecord}
          disabled={loading}
          aria-label={recording ? "録音を停止" : "録音する"}
        >
          {recording ? <IconStop /> : <IconMic />}
        </button>
      </div>

      {/* 設定ポップアップ（透明＋ぼかし、背景もぼかし） */}
      {settingsOpen && (
        <div className="modal-overlay" onClick={() => setSettingsOpen(false)}>
          <div
            className="modal-glass"
            role="dialog"
            aria-modal="true"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="modal-section">
              <div className="glass-seg">
                {TIRE_INCHES.map((inch) => (
                  <button
                    key={inch}
                    className={tireInch === inch ? "on" : ""}
                    onClick={() => onSelectInch(inch)}
                    disabled={loading}
                  >
                    {INCH_LABEL[inch]}
                  </button>
                ))}
              </div>
            </div>

            <label className={"ghost-btn modal-file" + (loading ? " disabled" : "")}>
              <IconFile />
              ファイルから選択
              <input
                type="file"
                accept="audio/*"
                onChange={onPickFileUser}
                disabled={loading}
                hidden
              />
            </label>

            {isDev && (
              <button
                className="ghost-btn"
                onClick={() => {
                  setSettingsOpen(false);
                  setView(true);
                }}
              >
                管理者画面へ
              </button>
            )}

            <button className="logout-btn" onClick={signOut}>
              ログアウト
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
