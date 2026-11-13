"""Create a side-by-side CPU/CUDA DAS envelope and RF-error comparison."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    x = np.array([float(row["x_mm"]) for row in rows])
    z = np.array([float(row["z_mm"]) for row in rows])
    values = np.array([float(row["rf_sum"]) for row in rows])
    x_values = np.unique(x)
    z_values = np.unique(z)
    if len(rows) != x_values.size * z_values.size:
        raise ValueError(f"{path} is not a complete rectangular grid")
    return x_values, z_values, values.reshape(z_values.size, x_values.size)


def analytic_envelope(rf: np.ndarray) -> np.ndarray:
    n = rf.shape[0]
    multiplier = np.zeros(n)
    multiplier[0] = 1.0
    if n % 2 == 0:
        multiplier[n // 2] = 1.0
        multiplier[1:n // 2] = 2.0
    else:
        multiplier[1:(n + 1) // 2] = 2.0
    return np.abs(np.fft.ifft(np.fft.fft(rf, axis=0) * multiplier[:, None], axis=0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cpu", type=Path, help="CPU DAS CSV")
    parser.add_argument("cuda", type=Path, help="CUDA DAS CSV")
    parser.add_argument("output", type=Path, help="Output PNG")
    args = parser.parse_args()

    x_cpu, z_cpu, cpu = load_csv(args.cpu)
    x_cuda, z_cuda, cuda = load_csv(args.cuda)
    if not (np.allclose(x_cpu, x_cuda) and np.allclose(z_cpu, z_cuda)):
        raise ValueError("CPU and CUDA outputs use different image grids")
    if cpu.shape != cuda.shape:
        raise ValueError("CPU and CUDA output dimensions differ")

    cpu_env = analytic_envelope(cpu)
    cuda_env = analytic_envelope(cuda)
    scale = float(max(cpu_env.max(), cuda_env.max()))
    if not np.isfinite(scale) or scale == 0.0:
        raise ValueError("Outputs have no finite nonzero envelope")
    floor = 1e-6
    cpu_db = 20.0 * np.log10(np.maximum(cpu_env / scale, floor))
    cuda_db = 20.0 * np.log10(np.maximum(cuda_env / scale, floor))
    abs_error = np.abs(cuda - cpu)
    error_db = 20.0 * np.log10(np.maximum(abs_error / scale, floor))
    nrmse = np.sqrt(np.mean((cuda - cpu) ** 2)) / np.sqrt(np.mean(cpu ** 2))
    corr = np.corrcoef(cpu.ravel(), cuda.ravel())[0, 1]

    extent = [x_cpu[0], x_cpu[-1], z_cpu[-1], z_cpu[0]]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.8), constrained_layout=True)
    for ax, image_data, title in (
        (axes[0], cpu_db, "CPU DAS envelope"),
        (axes[1], cuda_db, "CUDA DAS envelope"),
    ):
        im = ax.imshow(image_data, extent=extent, cmap="gray", vmin=-60, vmax=0,
                       aspect="auto", interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("Lateral position x (mm)")
        ax.set_ylabel("Depth z (mm)")
    fig.colorbar(im, ax=axes[:2], label="Envelope level (dB; shared scale)")

    diff = axes[2].imshow(error_db, extent=extent, cmap="magma", vmin=-120, vmax=-60,
                          aspect="auto", interpolation="nearest")
    axes[2].set_title("Absolute RF difference")
    axes[2].set_xlabel("Lateral position x (mm)")
    axes[2].set_ylabel("Depth z (mm)")
    fig.colorbar(diff, ax=axes[2], label="|CUDA − CPU| relative to shared envelope peak (dB)")
    fig.suptitle(f"CPU vs CUDA DAS · NRMSE {nrmse:.3g} · correlation {corr:.12f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(f"Wrote {args.output}; max |RF difference|={abs_error.max():.6g}")


if __name__ == "__main__":
    main()
