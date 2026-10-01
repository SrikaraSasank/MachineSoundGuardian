"""Physics-based evidence used to explain an anomaly.

These numbers are what the ticket agent is allowed to talk about. Keeping the
evidence explicit (instead of letting an LLM look at raw audio) is what keeps
the generated tickets grounded.

Core idea: classic *envelope analysis* from vibration engineering.
  1. Estimate shaft speed from the strongest low-frequency tone.
  2. High-pass the signal (>2 kHz), take its amplitude envelope (Hilbert) and
     find the dominant repetition frequency of that envelope.
  3. Compare it with shaft speed:
       ~1x (or 2x) shaft speed      -> once/twice-per-rev: imbalance, misalignment
       non-integer multiple (e.g. 3.6x) -> bearing defect frequency (BPFO/BPFI)
       integer multiple >= 3        -> blade-pass / vane-pass / gear-mesh
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt, welch
from scipy.stats import kurtosis

from .audio_features import FeatureConfig, mel_band_centers

Z_KEYS = ("kurtosis_hp", "envelope_prominence")


def signal_stats(x: np.ndarray, cfg: FeatureConfig = FeatureConfig()) -> dict:
    sr = cfg.sr
    nper = min(16384, len(x))
    f, P = welch(x, sr, nperseg=nper)
    m = (f >= 20) & (f <= 300)
    shaft = float(f[m][np.argmax(P[m])])
    hp = sosfiltfilt(butter(4, 2000, "highpass", fs=sr, output="sos"), x)
    env = np.abs(hilbert(hp))
    env -= env.mean()
    fe, Pe = welch(env, sr, nperseg=nper)
    m = (fe >= 10) & (fe <= 1000)
    peak = float(fe[m][np.argmax(Pe[m])])
    prom = float(Pe[m].max() / (np.median(Pe[m]) + 1e-20))
    return {
        "kurtosis_hp": float(kurtosis(hp)),
        "envelope_prominence": float(np.log10(prom + 1e-12)),
        "shaft_hz": round(shaft, 1),
        "envelope_peak_hz": round(peak, 1),
        "rms_db": float(20 * np.log10(np.sqrt((x ** 2).mean()) + 1e-12)),
    }


class NormalProfile:
    """Statistics of a machine's normal sound, learned from training clips."""

    def fit(self, mels: list[np.ndarray], stats: list[dict], cfg: FeatureConfig = FeatureConfig()):
        band_means = np.stack([S.mean(1) for S in mels])
        self.band_mu = band_means.mean(0)
        self.band_sd = band_means.std(0) + 1e-3
        self.centers = mel_band_centers(cfg)
        self.stat_mu = {k: float(np.mean([s[k] for s in stats])) for k in Z_KEYS}
        self.stat_sd = {k: float(np.std([s[k] for s in stats]) + 1e-3) for k in Z_KEYS}
        self.rms_db = float(np.mean([s["rms_db"] for s in stats]))
        return self

    def evidence(self, S: np.ndarray, stats: dict, top_k: int = 5) -> dict:
        z = (S.mean(1) - self.band_mu) / self.band_sd
        order = np.argsort(-np.abs(z))[:top_k]
        bands = [
            {"center_hz": int(self.centers[i]), "z": round(float(z[i]), 2),
             "direction": "louder" if z[i] > 0 else "quieter"}
            for i in order
        ]
        stat_z = {k: round((stats[k] - self.stat_mu[k]) / self.stat_sd[k], 2) for k in Z_KEYS}
        return {
            "top_bands": bands,
            "indicator_z": stat_z,
            "shaft_hz": stats["shaft_hz"],
            "envelope_peak_hz": stats["envelope_peak_hz"],
            "envelope_to_shaft_ratio": round(stats["envelope_peak_hz"] / max(stats["shaft_hz"], 1e-6), 2),
            "loudness_change_db": round(stats["rms_db"] - self.rms_db, 1),
        }


def fault_hypotheses(ev: dict) -> list[str]:
    """Rule-based hints, phrased as things to check, not diagnoses."""
    h = []
    r = ev["envelope_to_shaft_ratio"]
    shaft, peak = ev["shaft_hz"], ev["envelope_peak_hz"]
    if ev["indicator_z"]["envelope_prominence"] > 4:
        near = round(r)
        if near in (1, 2) and abs(r - near) < 0.1:
            h.append(f"High-frequency noise is modulated at {near}x shaft speed ({peak} Hz vs shaft {shaft} Hz): "
                     "check for rotor imbalance, misalignment or looseness.")
        elif abs(r - near) >= 0.1 and 1.5 < r < 12:
            h.append(f"Repetitive impacts at {peak} Hz, a non-integer {r}x multiple of shaft speed ({shaft} Hz): "
                     "matches a rolling-element bearing defect frequency; check bearing condition and lubrication.")
        else:
            h.append(f"Strong periodic modulation at {peak} Hz ({r}x shaft speed): "
                     "check blade-pass, vane-pass or gear-mesh related components.")
    tonal = [b for b in ev["top_bands"] if b["center_hz"] >= 1500 and b["z"] > 4]
    if tonal and not h:
        h.append(f"New narrow-band tone near {tonal[0]['center_hz']} Hz without periodic impacts: "
                 "check for rubbing, a worn belt or changed gear mesh.")
    if not h:
        h.append("Sound differs from the learned normal profile, but no single fault signature dominates.")
    return h
