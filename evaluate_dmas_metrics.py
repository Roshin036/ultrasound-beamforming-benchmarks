"""Compare DAS vs DMAS beamforming in terms of Contrast Ratio (CR),
Contrast-to-Noise Ratio (CNR), and lateral beam profile.
Generates a 3-panel comparative figure:
- Panel 1: DAS B-mode with Cyst and Background ROIs
- Panel 2: DMAS B-mode with Cyst and Background ROIs
- Panel 3: Lateral Profile through Cyst center (z = 40.0 mm)
"""

import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

WORK_DIR = Path("C:/Users/arjun/Desktop/Sem 7/Ultrasound Project/ultrasound-beamforming-benchmarks")
BUILD_DIR = WORK_DIR / "build"
RESULTS_DIR = WORK_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

DAS_CSV = BUILD_DIR / "cuda_template_O3_FP32_hypoechoic.csv"
DMAS_CSV = BUILD_DIR / "dmas_cuda_template_O3_FP32_hypoechoic.csv"

def analytic_envelope(rf: np.ndarray) -> np.ndarray:
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

def load_bmode(csv_path: Path, nx=507, nz=600, apply_bandpass: bool = False):
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rf_flat = np.array([float(r["rf_sum"]) for r in reader], dtype=np.float32)
    rf_2d = rf_flat.reshape(nz, nx)
    if apply_bandpass:
        import scipy.signal as signal
        b, a = signal.butter(4, [0.08, 0.75], btype="bandpass")
        rf_2d = signal.filtfilt(b, a, rf_2d, axis=0)
    env = analytic_envelope(rf_2d)
    peak = float(env.max())
    db = 20.0 * np.log10(np.maximum(env / peak, 1e-6))
    return env, db

def main():
    if not DAS_CSV.exists() or not DMAS_CSV.exists():
        print(f"Missing input CSVs: DAS={DAS_CSV.exists()}, DMAS={DMAS_CSV.exists()}")
        return

    print("Loading DAS and DMAS outputs...")
    nx, nz = 507, 600
    xmin, xmax = -19.0, 19.0
    zmin, zmax = 5.0, 50.0

    x_coords = np.linspace(xmin, xmax, nx)
    z_coords = np.linspace(zmin, zmax, nz)
    X, Z = np.meshgrid(x_coords, z_coords)

    das_env, das_db = load_bmode(DAS_CSV, nx, nz, apply_bandpass=False)
    dmas_env, dmas_db = load_bmode(DMAS_CSV, nx, nz, apply_bandpass=True)

    # Define ROIs for hypoechoic cyst
    cyst_x, cyst_z = -8.4, 40.0
    dist = np.hypot(X - cyst_x, Z - cyst_z)

    cyst_roi = dist <= 3.5
    bg_roi = (dist >= 5.5) & (dist <= 8.5)

    # Compute Contrast Metrics
    # Contrast Ratio: 20 * log10(mu_cyst / mu_bg)
    das_mu_cyst = np.mean(das_env[cyst_roi])
    das_mu_bg = np.mean(das_env[bg_roi])
    das_cr = 20.0 * np.log10(max(das_mu_cyst / das_mu_bg, 1e-12))
    das_cnr = abs(das_mu_bg - das_mu_cyst) / np.sqrt(np.var(das_env[bg_roi]) + np.var(das_env[cyst_roi]))

    dmas_mu_cyst = np.mean(dmas_env[cyst_roi])
    dmas_mu_bg = np.mean(dmas_env[bg_roi])
    dmas_cr = 20.0 * np.log10(max(dmas_mu_cyst / dmas_mu_bg, 1e-12))
    dmas_cnr = abs(dmas_mu_bg - dmas_mu_cyst) / np.sqrt(np.var(dmas_env[bg_roi]) + np.var(dmas_env[cyst_roi]))

    print(f"\n--- Contrast & Sidelobe Enhancement ---")
    print(f"DAS : Contrast Ratio (CR) = {das_cr:.2f} dB, CNR = {das_cnr:.2f}")
    print(f"DMAS: Contrast Ratio (CR) = {dmas_cr:.2f} dB, CNR = {dmas_cnr:.2f}")
    print(f"Delta CR: {dmas_cr - das_cr:.2f} dB improvement in cyst clearing")

    # Plot 3-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), constrained_layout=True)

    # Panel 1: DAS
    im0 = axes[0].imshow(das_db, extent=[xmin, xmax, zmax, zmin], cmap="gray", vmin=-60, vmax=0, aspect="auto")
    axes[0].set_title(f"Standard DAS (Delay-and-Sum)\nCR = {das_cr:.1f} dB · CNR = {das_cnr:.2f}", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Lateral position x (mm)")
    axes[0].set_ylabel("Depth z (mm)")
    circle_cyst = plt.Circle((cyst_x, cyst_z), 3.5, color="cyan", fill=False, linewidth=1.5, linestyle="--", label="Cyst ROI")
    circle_bg1 = plt.Circle((cyst_x, cyst_z), 5.5, color="yellow", fill=False, linewidth=1.2, linestyle=":", label="Background ROI")
    circle_bg2 = plt.Circle((cyst_x, cyst_z), 8.5, color="yellow", fill=False, linewidth=1.2, linestyle=":")
    axes[0].add_patch(circle_cyst)
    axes[0].add_patch(circle_bg1)
    axes[0].add_patch(circle_bg2)
    axes[0].axhline(cyst_z, color="red", linestyle="--", alpha=0.5, label=f"Profile (z={cyst_z} mm)")
    axes[0].legend(loc="lower left", fontsize=8)

    # Panel 2: Filtered DMAS
    im1 = axes[1].imshow(dmas_db, extent=[xmin, xmax, zmax, zmin], cmap="gray", vmin=-60, vmax=0, aspect="auto")
    axes[1].set_title(f"Filtered DMAS (F-DMAS)\nCR = {dmas_cr:.1f} dB · CNR = {dmas_cnr:.2f}", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Lateral position x (mm)")
    axes[1].set_ylabel("Depth z (mm)")
    circle_cyst2 = plt.Circle((cyst_x, cyst_z), 3.5, color="cyan", fill=False, linewidth=1.5, linestyle="--")
    circle_bg1_2 = plt.Circle((cyst_x, cyst_z), 5.5, color="yellow", fill=False, linewidth=1.2, linestyle=":")
    circle_bg2_2 = plt.Circle((cyst_x, cyst_z), 8.5, color="yellow", fill=False, linewidth=1.2, linestyle=":")
    axes[1].add_patch(circle_cyst2)
    axes[1].add_patch(circle_bg1_2)
    axes[1].add_patch(circle_bg2_2)
    axes[1].axhline(cyst_z, color="red", linestyle="--", alpha=0.5)

    # Colorbar
    fig.colorbar(im1, ax=[axes[0], axes[1]], shrink=0.85, label="Log Envelope (dB)")

    # Panel 3: Lateral Line Profile through cyst center
    iz_target = int(np.argmin(np.abs(z_coords - cyst_z)))
    das_profile = das_db[iz_target, :]
    dmas_profile = dmas_db[iz_target, :]

    axes[2].plot(x_coords, das_profile, label="Standard DAS", color="tab:blue", linewidth=1.8)
    axes[2].plot(x_coords, dmas_profile, label="Filtered DMAS (F-DMAS)", color="tab:orange", linewidth=1.8)
    axes[2].axvspan(cyst_x - 3.5, cyst_x + 3.5, color="cyan", alpha=0.15, label="Cyst Region")
    axes[2].set_xlim([xmin, xmax])
    axes[2].set_ylim([-60, 0])
    axes[2].set_xlabel("Lateral position x (mm)")
    axes[2].set_ylabel("Amplitude (dB)")
    axes[2].set_title(f"Lateral Profile across Cyst (z = {cyst_z} mm)\nDemonstrating Side-Lobe Suppression", fontsize=11, fontweight="bold")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="lower right", fontsize=8)

    out_plot = PLOTS_DIR / "das_vs_dmas_contrast_comparison.png"
    fig.savefig(out_plot, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved DAS vs DMAS comparison plot to {out_plot}")

if __name__ == "__main__":
    main()
