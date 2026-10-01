"""Synthetic machine sounds in DCASE 2022+ layout.

Used for smoke tests and the dashboard demo when real data is not downloaded.
It is NOT a substitute for the real benchmark: report real-data numbers only.

Normal sound: rotating-machine hum (fundamental + harmonics) over coloured noise.
Fault types injected into anomalies:
  bearing   - periodic high-frequency impulses (outer-race defect style)
  imbalance - strong once-per-revolution amplitude modulation, raised 1x tone
  whine     - narrow-band tonal component in the 2-4 kHz range (rub / gear mesh)
Target domain = different speed (f0 shifted) and different background noise,
to exercise domain-shift handling.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import lfilter

SR = 16000
FAULTS = ("bearing", "imbalance", "whine")


def _noise(rng, n, color):
    w = rng.standard_normal(n)
    return lfilter([1.0], [1.0, -color], w)  # AR(1): higher color = more low-frequency


def machine_clip(rng, f0, noise_color, noise_gain, dur=4.0, fault=None, severity=1.0):
    n = int(SR * dur)
    t = np.arange(n) / SR
    f = f0 * (1 + 0.004 * rng.standard_normal())  # small speed jitter
    x = np.zeros(n)
    for h, a in zip(range(1, 9), [1.0, 0.6, 0.45, 0.3, 0.2, 0.15, 0.1, 0.08]):
        x += a * np.sin(2 * np.pi * h * f * t + rng.uniform(0, 2 * np.pi))
    x *= 1 + 0.03 * rng.standard_normal()
    noise = _noise(rng, n, noise_color)
    noise /= np.std(noise) + 1e-9
    x = 0.25 * x / np.std(x) + noise_gain * noise

    if fault == "bearing":
        period = int(SR / (f * 3.57))  # ~BPFO for a typical bearing
        imp = np.zeros(n)
        imp[rng.integers(0, period)::period] = 1.0
        ring = np.exp(-np.arange(200) / 25.0) * np.sin(2 * np.pi * rng.uniform(3000, 5500) * np.arange(200) / SR)
        x += 0.5 * severity * np.convolve(imp, ring, mode="same")
    elif fault == "imbalance":
        x *= 1 + 0.5 * severity * np.sin(2 * np.pi * f * t)
        x += 0.2 * severity * np.sin(2 * np.pi * f * t)
    elif fault == "whine":
        fw = rng.uniform(2000, 4000)
        x += 0.12 * severity * np.sin(2 * np.pi * fw * t) * (1 + 0.2 * np.sin(2 * np.pi * 3 * t))
    x = x / (np.max(np.abs(x)) + 1e-9) * 0.8
    return (x * 32767).astype(np.int16)


def generate(out_dir: str | Path, machines=("fan", "pump"), sections=3, n_train_source=60,
             n_train_target=6, n_test=30, seed=7) -> pd.DataFrame:
    """Write a small DCASE-2022-style dataset and return the fault ground truth."""
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    truth = []
    for mi, machine in enumerate(machines):
        for s in range(sections):
            base_f0 = 50 + 17 * s + 30 * mi
            domains = {
                "source": dict(f0=base_f0, noise_color=0.95, noise_gain=0.10),
                "target": dict(f0=base_f0 * 1.18, noise_color=0.70, noise_gain=0.16),
            }
            for dom, params in domains.items():
                n_train = n_train_source if dom == "source" else n_train_target
                plan = [("train", "normal", None)] * n_train
                plan += [("test", "normal", None)] * n_test
                plan += [("test", "anomaly", FAULTS[i % 3]) for i in range(n_test)]
                for i, (split, lab, fault) in enumerate(plan):
                    d = out_dir / machine / split
                    d.mkdir(parents=True, exist_ok=True)
                    name = f"section_{s:02d}_{dom}_{split}_{lab}_{i:04d}.wav"
                    sev = rng.uniform(0.35, 1.0)
                    wavfile.write(d / name, SR, machine_clip(rng, fault=fault, severity=sev, **params))
                    truth.append(dict(file=name, machine=machine, fault=fault or "none", severity=round(sev, 2)))
    df = pd.DataFrame(truth)
    df.to_csv(out_dir / "fault_ground_truth.csv", index=False)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate a synthetic DCASE-style dataset")
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--sections", type=int, default=3)
    args = ap.parse_args()
    df = generate(args.out, sections=args.sections)
    print(f"Wrote {len(df)} clips to {args.out}")
