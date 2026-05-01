"""
Feature Extraction Pipeline
============================
Extracts time-domain and frequency-domain features from raw vibration signals.

Features extracted per segment:
  Time-domain  : RMS, Peak, Crest Factor, Kurtosis, Skewness, Shape Factor,
                 Impulse Factor, Variance, Mean Abs
  Frequency-domain (FFT): Dominant frequency, Spectral centroid, Spectral
                 entropy, Band-power at shaft/inner-race/outer-race/ball harmonics,
                 Top-5 FFT magnitudes
"""

import numpy as np
from scipy import stats
from scipy.fft import rfft, rfftfreq
import pandas as pd
import os


# ── Constants ─────────────────────────────────────────────────────────────────
FS       = 12_000
SHAFT_HZ = 1_750 / 60
BPFI_HZ  = 162.2
BPFO_HZ  = 107.4
BSF_HZ   = 141.1

FREQ_BANDS = {
    "shaft":       (SHAFT_HZ * 0.8,  SHAFT_HZ * 3.5),
    "bpfo":        (BPFO_HZ  * 0.8,  BPFO_HZ  * 2.5),
    "bpfi":        (BPFI_HZ  * 0.8,  BPFI_HZ  * 2.5),
    "bsf":         (BSF_HZ   * 0.8,  BSF_HZ   * 2.5),
    "resonance":   (4_000,            6_000),
}


# ── Feature functions ──────────────────────────────────────────────────────────

def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)))

def peak(x: np.ndarray) -> float:
    return float(np.max(np.abs(x)))

def crest_factor(x: np.ndarray) -> float:
    r = rms(x)
    return float(peak(x) / r) if r > 0 else 0.0

def kurtosis(x: np.ndarray) -> float:
    return float(stats.kurtosis(x, fisher=True))   # excess kurtosis

def skewness(x: np.ndarray) -> float:
    return float(stats.skew(x))

def shape_factor(x: np.ndarray) -> float:
    r = rms(x)
    maa = np.mean(np.abs(x))
    return float(r / maa) if maa > 0 else 0.0

def impulse_factor(x: np.ndarray) -> float:
    maa = np.mean(np.abs(x))
    return float(peak(x) / maa) if maa > 0 else 0.0

def variance(x: np.ndarray) -> float:
    return float(np.var(x))

def mean_abs(x: np.ndarray) -> float:
    return float(np.mean(np.abs(x)))


def spectral_centroid(mags: np.ndarray, freqs: np.ndarray) -> float:
    total = np.sum(mags)
    return float(np.sum(freqs * mags) / total) if total > 0 else 0.0

def spectral_entropy(mags: np.ndarray) -> float:
    p = mags / (np.sum(mags) + 1e-12)
    return float(-np.sum(p * np.log(p + 1e-12)))

def band_power(mags: np.ndarray, freqs: np.ndarray, flo: float, fhi: float) -> float:
    mask = (freqs >= flo) & (freqs <= fhi)
    return float(np.sum(mags[mask] ** 2))

def dominant_freq(mags: np.ndarray, freqs: np.ndarray) -> float:
    return float(freqs[np.argmax(mags)])


# ── Main extractor ─────────────────────────────────────────────────────────────

def extract_features(segment: np.ndarray) -> dict:
    """Extract all features from a single vibration segment (1-D array)."""
    feats = {}

    # ─ Time-domain features ─
    feats["rms"]           = rms(segment)
    feats["peak"]          = peak(segment)
    feats["crest_factor"]  = crest_factor(segment)
    feats["kurtosis"]      = kurtosis(segment)
    feats["skewness"]      = skewness(segment)
    feats["shape_factor"]  = shape_factor(segment)
    feats["impulse_factor"]= impulse_factor(segment)
    feats["variance"]      = variance(segment)
    feats["mean_abs"]      = mean_abs(segment)

    # ─ Frequency-domain features ─
    n     = len(segment)
    fft_c = rfft(segment * np.hanning(n))       # Hann window to reduce leakage
    mags  = np.abs(fft_c) / n
    freqs = rfftfreq(n, d=1.0 / FS)

    feats["dominant_freq"]      = dominant_freq(mags, freqs)
    feats["spectral_centroid"]  = spectral_centroid(mags, freqs)
    feats["spectral_entropy"]   = spectral_entropy(mags)

    for band_name, (flo, fhi) in FREQ_BANDS.items():
        feats[f"band_power_{band_name}"] = band_power(mags, freqs, flo, fhi)

    # ─ Top-5 FFT magnitude peaks ─
    top5_idx = np.argsort(mags)[-5:][::-1]
    for k, idx in enumerate(top5_idx, 1):
        feats[f"fft_peak{k}_freq"] = float(freqs[idx])
        feats[f"fft_peak{k}_mag"]  = float(mags[idx])

    return feats


def build_feature_matrix(X: np.ndarray, y: np.ndarray,
                          class_map: dict, out_dir: str = "data") -> pd.DataFrame:
    """Extract features from all segments and return a tidy DataFrame."""
    print(f"Extracting features from {len(X)} segments …")
    rows = []
    for i, seg in enumerate(X):
        feats = extract_features(seg)
        feats["label"]      = int(y[i])
        feats["class_name"] = class_map[int(y[i])]
        rows.append(feats)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(X)} segments processed")

    df = pd.DataFrame(rows)
    # re-order: label columns first
    meta_cols = ["label", "class_name"]
    feat_cols = [c for c in df.columns if c not in meta_cols]
    df = df[meta_cols + feat_cols]

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "bearing_features.csv")
    df.to_csv(out_path, index=False)
    print(f"\nFeature matrix saved → {out_path}")
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Feature columns ({len(feat_cols)}): {feat_cols}")
    return df


if __name__ == "__main__":
    from data_generator import generate_dataset, CLASS_MAP
    X, y, _ = generate_dataset()
    df = build_feature_matrix(X, y, CLASS_MAP)
    print("\nClass distribution:")
    print(df["class_name"].value_counts())
    print("\nFeature stats (first 5 features):")
    print(df.iloc[:, 2:7].describe().round(4))
