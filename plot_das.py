"""Render the coherent real-RF DAS output as a log-compressed envelope image."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal


def analytic_envelope(rf: np.ndarray, apply_bandpass: bool = False) -> np.ndarray:
    if apply_bandpass:
        b, a = signal.butter(4, [0.08, 0.75], btype="bandpass")
        rf = signal.filtfilt(b, a, rf, axis=0)
    n = rf.shape[0]
    spectrum = np.fft.fft(rf, axis=0)
    multiplier = np.zeros(n)
    multiplier[0] = 1.0
    if n % 2 == 0:
        multiplier[n // 2] = 1.0
        multiplier[1 : n // 2] = 2.0
    else:
        multiplier[1 : (n + 1) // 2] = 2.0
    return np.abs(np.fft.ifft(spectrum * multiplier[:, None], axis=0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="CSV written by das_reference or dmas")
    parser.add_argument("output", type=Path, help="Output PNG path")
    parser.add_argument("--bandpass", action="store_true", help="Apply F-DMAS bandpass filter")
    args = parser.parse_args()

    is_dmas = "dmas" in args.input.name.lower() or args.bandpass

    with args.input.open(newline="") as f:
        rows = list(csv.DictReader(f))

    x_all = np.array([float(row["x_mm"]) for row in rows])
    z_all = np.array([float(row["z_mm"]) for row in rows])
    x_values = np.unique(np.round(x_all, 4))
    z_values = np.unique(np.round(z_all, 4))
    expected = len(x_values) * len(z_values)
    if len(rows) != expected:
        raise ValueError(f"CSV has {len(rows)} rows, expected {expected} ({len(x_values)} x {len(z_values)})")

    rf = np.array([float(row["rf_sum"]) for row in rows]).reshape(len(z_values), len(x_values))
    envelope = analytic_envelope(rf, apply_bandpass=is_dmas)
    peak = float(envelope.max())
    if not np.isfinite(peak) or peak == 0.0:
        raise ValueError("Beamforming output has no finite nonzero envelope")
    db = 20.0 * np.log10(np.maximum(envelope / peak, 1e-6))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    image = ax.imshow(db, extent=[x_values[0], x_values[-1], z_values[-1], z_values[0]],
                      cmap="gray", vmin=-60, vmax=0, aspect="auto", interpolation="nearest")
    ax.set_xlabel("Lateral position x (mm)")
    ax.set_ylabel("Depth z (mm)")
    title_str = "CPWC Filtered DMAS (F-DMAS) envelope (−60 dB display range)" if is_dmas else "CPWC DAS envelope (−60 dB display range)"
    ax.set_title(title_str)
    fig.colorbar(image, ax=ax, label="Relative amplitude (dB)")
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    print(f"Wrote {args.output}; envelope peak={peak:.6g}, grid={len(x_values)} x {len(z_values)}")


if __name__ == "__main__":
    main()
