"""Audio loading and log-mel feature extraction.

Implemented with numpy/scipy only, so the feature pipeline is identical on a
laptop, a server and a Raspberry Pi (no librosa dependency at inference time).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly


@dataclass(frozen=True)
class FeatureConfig:
    sr: int = 16000
    n_fft: int = 1024
    hop: int = 512
    n_mels: int = 128
    fmin: float = 0.0
    fmax: float | None = None
    frames: int = 5  # context frames stacked for frame-level models


def load_wav(path: str, sr: int = 16000) -> np.ndarray:
    """Load a wav as mono float32 in [-1, 1], resampled to `sr`.

    MIMII files are 8-channel; we keep channel 0 (as in the DCASE baseline).
    """
    file_sr, x = wavfile.read(path)
    if x.ndim > 1:
        x = x[:, 0]
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float32) / float(np.iinfo(x.dtype).max)
    else:
        x = x.astype(np.float32)
    if file_sr != sr:
        g = np.gcd(int(file_sr), int(sr))
        x = resample_poly(x, sr // g, file_sr // g).astype(np.float32)
    return x


def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10 ** (np.asarray(m) / 2595.0) - 1.0)


@lru_cache(maxsize=8)
def mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float | None) -> np.ndarray:
    fmax = fmax or sr / 2
    mels = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    hz = _mel_to_hz(mels)
    bins = np.fft.rfftfreq(n_fft, 1.0 / sr)
    fb = np.zeros((n_mels, len(bins)), dtype=np.float32)
    for i in range(n_mels):
        lo, ce, hi = hz[i], hz[i + 1], hz[i + 2]
        up = (bins - lo) / max(ce - lo, 1e-9)
        down = (hi - bins) / max(hi - ce, 1e-9)
        fb[i] = np.maximum(0.0, np.minimum(up, down))
    # Slaney-style area normalisation
    fb *= (2.0 / (hz[2:] - hz[:-2]))[:, None]
    return fb


def mel_band_centers(cfg: FeatureConfig = FeatureConfig()) -> np.ndarray:
    """Centre frequency (Hz) of each mel band; used to explain anomalies."""
    fmax = cfg.fmax or cfg.sr / 2
    mels = np.linspace(_hz_to_mel(cfg.fmin), _hz_to_mel(fmax), cfg.n_mels + 2)
    return _mel_to_hz(mels[1:-1])


def log_mel(x: np.ndarray, cfg: FeatureConfig = FeatureConfig()) -> np.ndarray:
    """Return log-mel spectrogram, shape (n_mels, n_frames), in dB-like units."""
    if len(x) < cfg.n_fft:
        x = np.pad(x, (0, cfg.n_fft - len(x)))
    n_frames = 1 + (len(x) - cfg.n_fft) // cfg.hop
    idx = np.arange(cfg.n_fft)[None, :] + cfg.hop * np.arange(n_frames)[:, None]
    frames = x[idx] * np.hanning(cfg.n_fft).astype(np.float32)
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2  # (T, F)
    fb = mel_filterbank(cfg.sr, cfg.n_fft, cfg.n_mels, cfg.fmin, cfg.fmax)
    mel = power @ fb.T  # (T, M)
    return (10.0 * np.log10(mel + 1e-10)).T.astype(np.float32)


def stack_frames(S: np.ndarray, frames: int) -> np.ndarray:
    """(n_mels, T) -> (T - frames + 1, n_mels * frames) context vectors."""
    n_mels, T = S.shape
    n = T - frames + 1
    if n <= 0:
        raise ValueError("clip too short for the requested context")
    out = np.empty((n, n_mels * frames), dtype=np.float32)
    for t in range(frames):
        out[:, n_mels * t:n_mels * (t + 1)] = S[:, t:t + n].T
    return out


def clip_embedding(S: np.ndarray) -> np.ndarray:
    """Fixed-length clip descriptor: per-band mean, std and 95th percentile."""
    return np.concatenate([S.mean(1), S.std(1), np.percentile(S, 95, axis=1)]).astype(np.float32)
