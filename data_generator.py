"""
CWRU Bearing Dataset Simulator  (v2 — Physics-accurate)
========================================================
Generates realistic vibration signals for 4 bearing health states.
Each impulse train is generated PER SEGMENT with amplitude/frequency
jitter so features are robustly class-separable and generalise to
new RNG draws (i.e. train/inference use same distribution).

Classes
  0 → Normal            shaft harmonics + white noise
  1 → Inner Race Fault  BPFI impulses, shaft-modulated
  2 → Outer Race Fault  BPFO impulses, no shaft modulation
  3 → Ball Fault        BSF impulses, cage-modulated
"""

import numpy as np
import pandas as pd
import os

FS        = 12_000
SHAFT_HZ  = 1_750 / 60      # 29.17 Hz
BPFI_HZ   = 162.2
BPFO_HZ   = 107.4
BSF_HZ    = 141.1
CAGE_HZ   = 13.6

N_SAMPLES  = 1_024
N_SEGMENTS = 300
SEED       = 42

CLASS_MAP = {0: "Normal", 1: "Inner Race Fault",
             2: "Outer Race Fault", 3: "Ball Fault"}

_rng = np.random.default_rng(SEED)


def _t():
    return np.linspace(0, N_SAMPLES / FS, N_SAMPLES, endpoint=False)


def _noise(sigma=0.05):
    return _rng.normal(0, sigma, N_SAMPLES)


def _impulse_train(freq: float, tau: float = 0.0008) -> np.ndarray:
    """Per-segment decaying cosine impulse train with amplitude jitter."""
    t   = _t()
    sig = np.zeros(N_SAMPLES)
    jitter = _rng.uniform(0.99, 1.01)       # ±1 % freq randomness
    period = 1.0 / (freq * jitter)
    for ti in np.arange(0, t[-1] + period, period):
        amp = _rng.uniform(0.8, 1.4)
        sig += amp * np.exp(-np.abs(t - ti) / tau) * \
               np.cos(2 * np.pi * 5_000 * (t - ti)) * (t >= ti)
    return sig


def gen_normal(n: int = N_SEGMENTS) -> np.ndarray:
    t = _t()
    segs = []
    for _ in range(n):
        sig  = 0.40 * np.sin(2*np.pi*SHAFT_HZ*t)
        sig += 0.15 * np.sin(2*np.pi*2*SHAFT_HZ*t)
        sig += 0.08 * np.sin(2*np.pi*3*SHAFT_HZ*t)
        sig += _noise(sigma=_rng.uniform(0.06, 0.10))
        segs.append(sig)
    return np.array(segs)


def gen_inner_race(n: int = N_SEGMENTS) -> np.ndarray:
    t = _t()
    segs = []
    for _ in range(n):
        imp  = _impulse_train(BPFI_HZ)
        mod  = 0.5 * (1 + np.sin(2*np.pi*SHAFT_HZ*t))
        sig  = _rng.uniform(0.5, 0.9) * imp * mod
        sig += 0.20 * np.sin(2*np.pi*SHAFT_HZ*t)
        sig += 0.12 * np.sin(2*np.pi*(BPFI_HZ - SHAFT_HZ)*t)
        sig += _noise(sigma=_rng.uniform(0.03, 0.07))
        segs.append(sig)
    return np.array(segs)


def gen_outer_race(n: int = N_SEGMENTS) -> np.ndarray:
    t = _t()
    segs = []
    for _ in range(n):
        imp  = _impulse_train(BPFO_HZ)
        sig  = _rng.uniform(0.5, 0.9) * imp
        sig += 0.18 * np.sin(2*np.pi*SHAFT_HZ*t)
        sig += _noise(sigma=_rng.uniform(0.03, 0.07))
        segs.append(sig)
    return np.array(segs)


def gen_ball_fault(n: int = N_SEGMENTS) -> np.ndarray:
    t = _t()
    segs = []
    for _ in range(n):
        imp  = _impulse_train(BSF_HZ, tau=0.001)
        mod  = 0.5 * (1 + np.sin(2*np.pi*CAGE_HZ*t))
        sig  = _rng.uniform(0.4, 0.8) * imp * mod
        sig += 0.18 * np.sin(2*np.pi*SHAFT_HZ*t)
        sig += _noise(sigma=_rng.uniform(0.03, 0.07))
        segs.append(sig)
    return np.array(segs)


def generate_dataset(out_dir: str = "data"):
    os.makedirs(out_dir, exist_ok=True)
    generators = [gen_normal, gen_inner_race, gen_outer_race, gen_ball_fault]
    all_segs, all_labels = [], []
    for label, fn in enumerate(generators):
        segs = fn(N_SEGMENTS)
        all_segs.append(segs)
        all_labels.extend([label] * N_SEGMENTS)
        print(f"  Generated {N_SEGMENTS} segments → class {label}: {CLASS_MAP[label]}")
    X = np.vstack(all_segs)
    y = np.array(all_labels)
    cols   = [f"t_{i}" for i in range(N_SAMPLES)]
    df_raw = pd.DataFrame(X[:400], columns=cols)
    df_raw.insert(0, "label",      y[:400])
    df_raw.insert(1, "class_name", [CLASS_MAP[int(l)] for l in y[:400]])
    df_raw.to_csv(f"{out_dir}/raw_vibration_samples.csv", index=False)
    print(f"\n  Raw CSV saved → {out_dir}/raw_vibration_samples.csv")
    return X, y, _t()


if __name__ == "__main__":
    from scipy import stats
    print("Generating dataset …")
    X, y, _ = generate_dataset()
    print("\nKurtosis (healthy→low, faulty→high):")
    for lbl, name in CLASS_MAP.items():
        segs = X[y == lbl]
        kts  = [stats.kurtosis(s) for s in segs]
        print(f"  {name:<22} {np.mean(kts):>6.2f} ± {np.std(kts):.2f}")
