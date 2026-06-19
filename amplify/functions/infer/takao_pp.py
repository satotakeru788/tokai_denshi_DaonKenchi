"""Takao preprocessing in pure numpy/scipy (Lambda-compatible, no librosa).

Faithful port of the al-final7 pipeline (cut_taps_v2 + preprocess_v3):
  detect taps (framed RMS -> find_peaks -> onset retreat)
  -> crop [onset + start_offset, +clip_len]  (native sr)
  -> bandpass 50-300 Hz (butterworth 4, sosfiltfilt)
  -> resample to 8000 Hz
  -> mel spectrogram (librosa-equivalent: Slaney mel + slaney norm, hann, center)
  -> power_to_db(ref=max, top_db=80)  -> pad/crop to t_frames

This module is the single source of truth: it is used for BOTH training (research)
and serving (copied verbatim into the Lambda profile preprocessors/takao_v1.py),
so train == serve. Mel math is validated to match librosa in test_match.py.
"""
from __future__ import annotations

from math import gcd
from typing import List

import numpy as np
from scipy import signal as sps

# al-final7 production config
TARGET_SR = 8000
BP_LO, BP_HI = 50.0, 300.0
N_FFT, HOP, N_MELS, FMIN, FMAX = 2048, 64, 64, 50.0, 300.0
T_SEC = 0.30
CLIP_LEN, START_OFFSET = 0.30, -0.04
RMS_FRAME, RMS_HOP, MIN_GAP_S = 512, 128, 4.0


# ---------------------------------------------------------------- detection
def _frame_rms(y: np.ndarray, frame: int, hop: int) -> np.ndarray:
    """librosa.feature.rms equivalent (center=True, constant pad)."""
    yp = np.pad(y, frame // 2, mode="constant")
    n = 1 + (len(yp) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    return np.sqrt(np.mean(yp[idx].astype(np.float64) ** 2, axis=1))


def detect_onsets(y: np.ndarray, sr: int) -> np.ndarray:
    rms = _frame_rms(y, RMS_FRAME, RMS_HOP)
    thr = max(0.25 * rms.max(), np.median(rms) + 5 * rms.std())
    dist = int(MIN_GAP_S * sr / RMS_HOP)
    peaks, _ = sps.find_peaks(rms, height=thr, distance=dist, prominence=0.10 * rms.max())
    if len(peaks):
        ph = rms[peaks]
        peaks = peaks[ph >= 0.4 * np.median(ph)]
    onsets = []
    for pk in peaks:
        thr_on = 0.15 * rms[pk]
        j = pk
        while j > 0 and rms[j] > thr_on:
            j -= 1
        onsets.append(j * RMS_HOP / sr)
    return np.array(onsets)


# ---------------------------------------------------------------- dsp
def bandpass(y: np.ndarray, sr: int, lo: float, hi: float) -> np.ndarray:
    hi = min(hi, sr / 2 - 1)
    sos = sps.butter(4, [lo, hi], btype="band", fs=sr, output="sos")
    return sps.sosfiltfilt(sos, y).astype(np.float32)


def resample_to(y: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return y.astype(np.float32)
    g = gcd(int(sr), int(target_sr))
    return sps.resample_poly(y, target_sr // g, sr // g).astype(np.float32)


# ---------------------------------------------------------------- mel (librosa-equivalent)
def _hz_to_mel_slaney(f):
    f = np.asarray(f, dtype=np.float64)
    f_sp, min_log_hz = 200.0 / 3, 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    mel = f / f_sp
    t = f >= min_log_hz
    mel = np.where(t, min_log_mel + np.log(np.where(t, f, min_log_hz) / min_log_hz) / logstep, mel)
    return mel


def _mel_to_hz_slaney(m):
    m = np.asarray(m, dtype=np.float64)
    f_sp, min_log_hz = 200.0 / 3, 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    f = f_sp * m
    t = m >= min_log_mel
    f = np.where(t, min_log_hz * np.exp(logstep * (m - min_log_mel)), f)
    return f


def _mel_filterbank(sr, n_fft, n_mels, fmin, fmax) -> np.ndarray:
    nbin = n_fft // 2 + 1
    fftfreqs = np.linspace(0.0, sr / 2.0, nbin)
    mpts = np.linspace(_hz_to_mel_slaney(fmin), _hz_to_mel_slaney(fmax), n_mels + 2)
    freqs = _mel_to_hz_slaney(mpts)
    fdiff = np.diff(freqs)
    ramps = freqs[:, None] - fftfreqs[None, :]
    fb = np.zeros((n_mels, nbin), dtype=np.float64)
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        fb[i] = np.maximum(0.0, np.minimum(lower, upper))
    enorm = 2.0 / (freqs[2 : n_mels + 2] - freqs[:n_mels])
    fb *= enorm[:, None]
    return fb.astype(np.float32)


def _stft_power(y: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    yp = np.pad(y.astype(np.float64), n_fft // 2, mode="constant")
    win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)  # periodic hann
    n = 1 + (len(yp) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n)[:, None]
    spec = np.fft.rfft(yp[idx] * win[None, :], n=n_fft, axis=1)
    return (np.abs(spec) ** 2).T.astype(np.float32)  # [freq, time]


def _power_to_db(S: np.ndarray, top_db: float = 80.0, amin: float = 1e-10) -> np.ndarray:
    ld = 10.0 * np.log10(np.maximum(amin, S))
    ld -= 10.0 * np.log10(np.maximum(amin, S.max()))
    return np.maximum(ld, ld.max() - top_db).astype(np.float32)


def mel_db(y: np.ndarray, sr: int) -> np.ndarray:
    """8kHz mono clip -> [N_MELS, t_frames] log-mel (power_to_db ref=max)."""
    P = _stft_power(y, N_FFT, HOP)
    fb = _mel_filterbank(sr, N_FFT, N_MELS, FMIN, FMAX)
    Sdb = _power_to_db(fb @ P)
    T = max(8, int(round(T_SEC * TARGET_SR / HOP)))
    if Sdb.shape[1] < T:
        Sdb = np.pad(Sdb, ((0, 0), (0, T - Sdb.shape[1])), mode="constant", constant_values=Sdb.min())
    return Sdb[:, :T]


# ---------------------------------------------------------------- end to end
def features_from_mono(mono: np.ndarray, sr: int) -> List[np.ndarray]:
    """mono (native sr) -> list of per-hit mel [N_MELS, t_frames]."""
    ons = detect_onsets(mono, sr)
    clen = int(CLIP_LEN * sr)
    mels: List[np.ndarray] = []
    used: List[float] = []
    for t in ons:
        s = max(0, min(int((t + START_OFFSET) * sr), len(mono) - clen))
        clip = mono[s : s + clen]
        if len(clip) < clen:
            clip = np.pad(clip, (0, clen - len(clip)))
        clip = bandpass(clip, sr, BP_LO, BP_HI)
        clip = resample_to(clip, sr, TARGET_SR)
        mels.append(mel_db(clip, TARGET_SR))
        used.append(round(float(t), 3))
    return mels, used
