"""Audio preprocessing + ONNX inference for tire-pressure estimation.

Ported verbatim (spectral-subtraction-OFF path) from the research repo
`sagemaker/code/audio_cnn.py`. Keep numerically identical to training:
  load mono -> resample 22050 -> detect_impact_peaks (all hits)
  -> per-peak crop [peak-pre, peak+post] -> log-mel (n_mels=96, fmin20, fmax3000)
  -> ONNX ensemble (output already de-normalized kPa per hit) -> mean over hits.
"""
from __future__ import annotations

import io
import wave
from math import gcd
from typing import List, Tuple

import numpy as np
from scipy import signal


# ---------------------------------------------------------------------------
# Audio loading (16/32/8-bit PCM WAV) -> mono float, matching load_audio_mono
# ---------------------------------------------------------------------------
def load_wav_mono(wav_bytes: bytes) -> Tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        n_channels = w.getnchannels()
        sampwidth = w.getsampwidth()
        sr = w.getframerate()
        raw = w.readframes(w.getnframes())
    if sampwidth == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sampwidth == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif sampwidth == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"unsupported WAV sample width: {sampwidth}")
    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)
    data = data - np.mean(data)
    data = data / (np.max(np.abs(data)) + 1e-12)
    return data.astype(np.float32), int(sr)


def resample_if_needed(wave_arr: np.ndarray, src_sr: int, target_sr: int) -> np.ndarray:
    if src_sr == target_sr:
        return wave_arr.astype(np.float32)
    g = gcd(src_sr, target_sr)
    return signal.resample_poly(wave_arr, target_sr // g, src_sr // g).astype(np.float32)


# ---------------------------------------------------------------------------
# Impact peak detection (robust version) -- verbatim from audio_cnn.py
# ---------------------------------------------------------------------------
def moving_average(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    kernel = np.ones(win, dtype=np.float32) / win
    return np.convolve(x, kernel, mode="same")


def _otsu_threshold(vals: np.ndarray) -> Tuple[float, float]:
    v = np.sort(np.asarray(vals, dtype=np.float64))
    n = len(v)
    if n < 3:
        return (float(v[0]) if n else 0.0), 1.0
    best_t, best_var = float(v[0]), -1.0
    csum = np.cumsum(v)
    total = csum[-1]
    for i in range(1, n):
        w0 = i / n
        m0 = csum[i - 1] / i
        m1 = (total - csum[i - 1]) / (n - i)
        var = w0 * (1.0 - w0) * (m0 - m1) ** 2
        if var > best_var:
            best_var, best_t = var, 0.5 * (v[i - 1] + v[i])
    lo = v[v < best_t]
    hi = v[v >= best_t]
    sep = float(hi.mean() / (lo.mean() + 1e-9)) if len(lo) and len(hi) else 1.0
    return best_t, sep


def _refractory_select(peaks: np.ndarray, amp: np.ndarray, sr: int, refractory_s: float) -> np.ndarray:
    rn = refractory_s * sr
    kept: List[int] = []
    for i in np.argsort(amp)[::-1]:
        pi = int(peaks[i])
        if all(abs(pi - int(peaks[j])) >= rn for j in kept):
            kept.append(int(i))
    return peaks[np.array(sorted(kept), dtype=np.int64)]


def detect_impact_peaks(mono: np.ndarray, sr: int, max_hits: int | None = None,
                        bimodal_sep: float = 2.0, refractory_s: float = 3.0) -> np.ndarray:
    env = np.abs(signal.hilbert(mono))
    env = moving_average(env, max(3, int(sr * 0.008)))
    min_distance = max(1, int(sr * 0.25))

    no_cap = max_hits is None or max_hits <= 0
    target = 1 if no_cap else max_hits

    peaks = np.array([], dtype=np.int64)
    for percentile, prom_ratio in [(92, 0.12), (88, 0.08), (82, 0.05), (75, 0.03)]:
        height = np.percentile(env, percentile)
        prominence = max(np.max(env) * prom_ratio, np.std(env) * 0.4)
        p, _ = signal.find_peaks(env, distance=min_distance, height=height, prominence=prominence)
        if len(p) > 0:
            peaks = p.astype(np.int64)
        if len(peaks) >= target:
            break

    if len(peaks) == 0:
        p, _ = signal.find_peaks(env, distance=min_distance)
        peaks = p.astype(np.int64)
        if len(peaks) == 0:
            return peaks
    if len(peaks) >= 3:
        amp = env[peaks]
        thr, sep = _otsu_threshold(amp)
        if sep > bimodal_sep:
            keep = amp >= thr
            if keep.any():
                peaks = peaks[keep]
    if refractory_s > 0 and len(peaks) > 1:
        peaks = _refractory_select(peaks, env[peaks], sr, refractory_s)
    if not no_cap and len(peaks) > max_hits:
        keep = np.argsort(env[peaks])[-max_hits:]
        peaks = peaks[keep]
    return np.sort(peaks).astype(np.int64)


# ---------------------------------------------------------------------------
# Log-mel spectrogram -- verbatim from audio_cnn.py (SS off, no crop)
# ---------------------------------------------------------------------------
def hz_to_mel(freq_hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + freq_hz / 700.0)


def mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10 ** (mel / 2595.0) - 1.0)


def build_mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float = 20.0, fmax: float = 10000.0) -> np.ndarray:
    fmax = min(fmax, sr / 2)
    mel_points = np.linspace(hz_to_mel(np.array([fmin]))[0], hz_to_mel(np.array([fmax]))[0], n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)

    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(1, n_mels + 1):
        left, center, right = bins[i - 1], bins[i], bins[i + 1]
        if center <= left:
            center = left + 1
        if right <= center:
            right = center + 1
        for j in range(left, center):
            fb[i - 1, j] = (j - left) / max(center - left, 1)
        for j in range(center, right):
            fb[i - 1, j] = (right - j) / max(right - center, 1)
    return fb


def compute_stft_power(wave_arr: np.ndarray, sr: int, n_fft: int, hop: int) -> np.ndarray:
    _, _, zxx = signal.stft(
        wave_arr,
        fs=sr,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop,
        nfft=n_fft,
        boundary=None,
        padded=False,
    )
    return (np.abs(zxx) ** 2).astype(np.float32)


def compute_logmel(wave_arr: np.ndarray, sr: int, n_fft: int, hop: int, n_mels: int,
                   fmin: float = 20.0, fmax: float = 3000.0) -> np.ndarray:
    power = compute_stft_power(wave_arr, sr, n_fft, hop)
    fb = build_mel_filterbank(sr, n_fft, n_mels, fmin=fmin, fmax=fmax)
    mel = np.dot(fb, power)
    mel = np.log(mel + 1e-8).astype(np.float32)
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    return mel


# ---------------------------------------------------------------------------
# Feature extraction (= the "logmel_v1" preprocessing profile)
# ---------------------------------------------------------------------------
def extract_features(wav_bytes: bytes, meta: dict) -> dict:
    """WAV -> per-hit log-mel features. Numerically identical to the original
    run_inference front half (resample 22050 -> detect peaks -> crop -> log-mel)."""
    sr_target = int(meta["target_sr"])
    n_fft = int(meta["n_fft"])
    hop = int(meta["hop"])
    n_mels = int(meta["n_mels"])
    fmin = float(meta["fmin"])
    fmax = float(meta["fmax"])
    pre_n = int(meta["pre_seconds"] * sr_target)
    post_n = int(meta["post_seconds"] * sr_target)
    seg_len = int(meta["segment_samples"])
    min_len = int(sr_target * 0.1)

    mono, sr = load_wav_mono(wav_bytes)
    mono = resample_if_needed(mono, sr, sr_target)
    peaks = detect_impact_peaks(mono, sr_target, max_hits=0)

    mels: List[np.ndarray] = []
    used_peaks: List[int] = []
    for peak in peaks:
        start = max(0, int(peak) - pre_n)
        end = min(len(mono), int(peak) + post_n)
        seg = mono[start:end]
        if len(seg) < min_len:
            continue
        if len(seg) < seg_len:
            seg = np.pad(seg, (0, seg_len - len(seg)))
        elif len(seg) > seg_len:
            seg = seg[:seg_len]
        seg = seg - np.mean(seg)
        seg = seg / (np.max(np.abs(seg)) + 1e-12)
        mels.append(compute_logmel(seg, sr_target, n_fft, hop, n_mels, fmin=fmin, fmax=fmax))
        used_peaks.append(int(peak))

    features = np.stack(mels, axis=0)[:, np.newaxis, :, :].astype(np.float32) if mels else None
    return {
        "features": features,
        "peakSeconds": [round(p / sr_target, 3) for p in used_peaks],
        "durationSec": round(len(mono) / sr_target, 2),
    }


# ---------------------------------------------------------------------------
# Model run + aggregation (shared by all preprocessing profiles)
# ---------------------------------------------------------------------------
def infer_from_features(feats: dict, session, meta: dict) -> dict:
    """features [N,1,H,W] -> ONNX (per-hit kPa) -> mean over hits + calibration."""
    cal = meta.get("calibration")
    if cal:
        cx = np.asarray(cal["x"], dtype=np.float64)
        cy = np.asarray(cal["y"], dtype=np.float64)
        apply_cal = lambda v: float(np.interp(v, cx, cy))
    else:
        apply_cal = lambda v: float(v)

    x = feats.get("features")
    if x is None or len(x) == 0:
        return {"pressureKpa": None, "pressureRawKpa": None, "hitsUsed": 0,
                "perHitKpa": [], "perHitKpaCal": [], "peakSeconds": [],
                "calibrated": bool(cal), "isMock": False, "error": "no_hits_detected"}

    in_name = session.get_inputs()[0].name
    out = session.run(None, {in_name: x})[0].reshape(-1)

    raw_mean = float(np.mean(out))
    per_hit_raw = [round(float(v), 1) for v in out.tolist()]
    per_hit_cal = [round(apply_cal(v), 1) for v in out.tolist()]
    return {
        # 録音単位の校正（全打平均→曲線）が最良なので pressureKpa はこれ
        "pressureKpa": round(apply_cal(raw_mean), 1),
        "pressureRawKpa": round(raw_mean, 1),
        "hitsUsed": len(per_hit_raw),
        "perHitKpa": per_hit_raw,
        "perHitKpaCal": per_hit_cal,
        "calibrated": bool(cal),
        "peakSeconds": feats.get("peakSeconds", []),
        "isMock": False,
        "modelTag": meta.get("seed_tag"),
        "durationSec": feats.get("durationSec"),
    }


# ---------------------------------------------------------------------------
# End-to-end inference (= extract_features + infer_from_features)
# ---------------------------------------------------------------------------
def run_inference(wav_bytes: bytes, session, meta: dict) -> dict:
    return infer_from_features(extract_features(wav_bytes, meta), session, meta)
