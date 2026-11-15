"""Evaluate CPU and GPU single precision outputs against the FP64 reference ground truth."""

import csv
from pathlib import Path
import numpy as np

WORK_DIR = Path("C:/Users/arjun/Desktop/Sem 7/Ultrasound Project/ultrasound-beamforming-benchmarks")
BUILD_DIR = WORK_DIR / "build"
RESULTS_DIR = WORK_DIR / "results"
BENCH_DIR = RESULTS_DIR / "benchmarks"

FP64_REF = BUILD_DIR / "cuda_template_O3_FP64_hypoechoic.csv"

def load_rf(csv_path):
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        return np.array([float(r["rf_sum"]) for r in reader], dtype=np.float64)

def main():
    if not FP64_REF.exists():
        print(f"FP64 reference {FP64_REF} not found")
        return

    print("Loading FP64 Ground Truth Reference...")
    ref = load_rf(FP64_REF)
    peak = float(np.max(ref) - np.min(ref))

    targets = [
        ("cpu_naive_O3", BUILD_DIR / "cpu_naive_O3_hypoechoic.csv"),
        ("cpu_template_O3", BUILD_DIR / "cpu_template_O3_hypoechoic.csv"),
        ("cuda_naive_O3_FP32", BUILD_DIR / "cuda_naive_O3_FP32_hypoechoic.csv"),
        ("cuda_template_O3_FP32", BUILD_DIR / "cuda_template_O3_FP32_hypoechoic.csv"),
    ]

    out_csv = BENCH_DIR / "fp64_ground_truth_metrics.csv"
    records = []

    print(f"\n{'Target':<25} | {'Max Abs Err':<12} | {'RMSE':<10} | {'NRMSE':<12} | {'PSNR (dB)':<10} | {'Correlation':<12}")
    print("-" * 92)

    for name, path in targets:
        if not path.exists():
            print(f"Skipping {name}: file does not exist")
            continue
        test = load_rf(path)
        diff = test - ref
        max_abs = float(np.max(np.abs(diff)))
        mse = float(np.mean(diff ** 2))
        rmse = float(np.sqrt(mse))
        nrmse = rmse / peak
        psnr = 10.0 * np.log10((peak ** 2) / (mse + 1e-15)) if mse > 0 else float("inf")
        corr = float(np.corrcoef(test, ref)[0, 1])

        records.append({
            "target": name,
            "max_abs_error": max_abs,
            "rmse": rmse,
            "nrmse": nrmse,
            "psnr_db": psnr,
            "correlation": corr
        })
        print(f"{name:<25} | {max_abs:<12.5g} | {rmse:<10.5g} | {nrmse:<12.6g} | {psnr:<10.2f} | {corr:<12.8f}")

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"\nSaved FP64 ground truth metrics to {out_csv}")

if __name__ == "__main__":
    main()
