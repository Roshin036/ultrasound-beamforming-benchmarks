"""Evaluate beamformed RF output against official zea ground truth B-mode.

Computes:
- SSIM (Structural Similarity Index)
- PSNR (Peak Signal-to-Noise Ratio in dB)
- RMSE (Root Mean Square Error)
- NRMSE (Normalized Root Mean Square Error)
- Pearson Correlation (r)
- Max Absolute Error
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert


def analytic_envelope(rf: np.ndarray) -> np.ndarray:
    """Hilbert envelope along the axial depth axis (axis 0)."""
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


def compute_metrics(bmode_test: np.ndarray, bmode_ref: np.ndarray) -> dict[str, float]:
    """Compute SSIM, PSNR, RMSE, NRMSE, and Pearson Correlation."""
    diff = bmode_test - bmode_ref
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    ref_range = float(bmode_ref.max() - bmode_ref.min())
    if ref_range == 0.0:
        ref_range = 60.0
    nrmse = rmse / ref_range

    # PSNR based on dynamic range (60 dB)
    max_val = 0.0
    min_val = -60.0
    dr = max_val - min_val  # 60 dB
    psnr = 10.0 * np.log10((dr ** 2) / (mse + 1e-12)) if mse > 0 else float("inf")

    # Pearson Correlation
    flat_test = bmode_test.ravel()
    flat_ref = bmode_ref.ravel()
    corr = float(np.corrcoef(flat_test, flat_ref)[0, 1])
    max_abs = float(np.max(np.abs(diff)))

    # Simplified 2D SSIM
    c1 = (0.01 * dr) ** 2
    c2 = (0.03 * dr) ** 2
    mu_x = np.mean(bmode_test)
    mu_y = np.mean(bmode_ref)
    sigma_x_sq = np.var(bmode_test)
    sigma_y_sq = np.var(bmode_ref)
    sigma_xy = np.mean((bmode_test - mu_x) * (bmode_ref - mu_y))

    ssim = float(((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) /
                 ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x_sq + sigma_y_sq + c2)))

    return {
        "ssim": ssim,
        "psnr_db": psnr,
        "rmse_db": rmse,
        "nrmse": nrmse,
        "correlation": corr,
        "max_abs_error_db": max_abs,
    }


def load_csv_bmode(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    x = np.array([float(r["x_mm"]) for r in rows])
    z = np.array([float(r["z_mm"]) for r in rows])
    rf = np.array([float(r["rf_sum"]) for r in rows])

    x_unique = np.unique(x)
    z_unique = np.unique(z)
    nx = len(x_unique)
    nz = len(z_unique)

    rf_2d = rf.reshape(nz, nx)
    env = analytic_envelope(rf_2d)
    peak = float(np.max(env))
    bmode = 20.0 * np.log10(np.maximum(env / (peak + 1e-12), 1e-3))  # [-60, 0] dB
    return x_unique, z_unique, rf_2d, bmode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path, help="CSV output from beamformer")
    parser.add_argument("ground_truth_npy", type=Path, help="zea ground truth bmode npy")
    parser.add_argument("--output_plot", type=Path, default=None, help="Path for comparison plot")
    args = parser.parse_args()

    ref_bmode = np.load(args.ground_truth_npy)
    x_u, z_u, rf, test_bmode = load_csv_bmode(args.input_csv)

    if test_bmode.shape != ref_bmode.shape:
        print(f"Warning: shape mismatch (test: {test_bmode.shape}, ref: {ref_bmode.shape})")
        # Resize or interpolate if needed

    metrics = compute_metrics(test_bmode, ref_bmode)
    print("--- Evaluation Metrics vs Ground Truth ---")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6g}")

    if args.output_plot:
        args.output_plot.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), constrained_layout=True)
        extent = [x_u[0], x_u[-1], z_u[-1], z_u[0]]

        im0 = axes[0].imshow(test_bmode, extent=extent, cmap="gray", vmin=-60, vmax=0, aspect="auto")
        axes[0].set_title("Beamformed B-mode (Ours)")
        axes[0].set_xlabel("x (mm)")
        axes[0].set_ylabel("z (mm)")

        im1 = axes[1].imshow(ref_bmode, extent=extent, cmap="gray", vmin=-60, vmax=0, aspect="auto")
        axes[1].set_title("Ground Truth (official zea)")
        axes[1].set_xlabel("x (mm)")
        axes[1].set_ylabel("z (mm)")

        diff = np.abs(test_bmode - ref_bmode)
        im2 = axes[2].imshow(diff, extent=extent, cmap="magma", vmin=0, vmax=20, aspect="auto")
        axes[2].set_title(f"Absolute Difference (|Δ| dB)\nSSIM: {metrics['ssim']:.4f}, PSNR: {metrics['psnr_db']:.2f} dB")
        axes[2].set_xlabel("x (mm)")
        axes[2].set_ylabel("z (mm)")

        fig.colorbar(im0, ax=axes[:2], label="Amplitude (dB)")
        fig.colorbar(im2, ax=axes[2], label="|Error| (dB)")

        fig.savefig(args.output_plot, dpi=160)
        plt.close(fig)
        print(f"Saved comparison plot to {args.output_plot}")


if __name__ == "__main__":
    main()
