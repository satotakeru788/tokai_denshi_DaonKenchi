import { type MouseEvent as ReactMouseEvent, useEffect, useRef } from "react";

interface Props {
  samples: Float32Array;
  sampleRate: number;
  peakSeconds: number[];
  picked: number;
  onPick: (i: number) => void;
}

/** 全打音の波形を描画し、検出された打音(ピーク)を縦線で表示。クリックで最寄りの打音を選択。 */
export function Waveform({ samples, sampleRate, peakSeconds, picked, onPick }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    canvas.width = Math.max(1, Math.floor(w * dpr));
    canvas.height = Math.max(1, Math.floor(h * dpr));
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);
    const dur = samples.length / sampleRate || 1;

    // waveform: min/max per pixel column
    ctx.strokeStyle = "#cbd5e1";
    ctx.lineWidth = 1;
    ctx.beginPath();
    const step = Math.max(1, Math.floor(samples.length / w));
    for (let x = 0; x < w; x++) {
      let min = 1;
      let max = -1;
      const start = x * step;
      for (let i = 0; i < step; i++) {
        const v = samples[start + i] || 0;
        if (v < min) min = v;
        if (v > max) max = v;
      }
      const y1 = (1 - (max + 1) / 2) * h;
      const y2 = (1 - (min + 1) / 2) * h;
      ctx.moveTo(x + 0.5, y1);
      ctx.lineTo(x + 0.5, y2);
    }
    ctx.stroke();

    // detected hits
    peakSeconds.forEach((t, i) => {
      const x = (t / dur) * w;
      const on = i === picked;
      ctx.strokeStyle = on ? "#2563eb" : "#fb7185";
      ctx.lineWidth = on ? 2.5 : 1;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
      if (on) {
        ctx.fillStyle = "#2563eb";
        ctx.beginPath();
        ctx.arc(x, 6, 4, 0, Math.PI * 2);
        ctx.fill();
      }
    });
  }, [samples, sampleRate, peakSeconds, picked]);

  function handleClick(e: ReactMouseEvent<HTMLCanvasElement>) {
    if (!peakSeconds.length) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const dur = samples.length / sampleRate || 1;
    const t = (x / rect.width) * dur;
    let best = 0;
    let bd = Infinity;
    peakSeconds.forEach((pt, i) => {
      const d = Math.abs(pt - t);
      if (d < bd) {
        bd = d;
        best = i;
      }
    });
    onPick(best);
  }

  return <canvas ref={canvasRef} className="waveform" onClick={handleClick} />;
}
