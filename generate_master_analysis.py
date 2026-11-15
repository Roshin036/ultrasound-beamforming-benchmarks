"""Generate master visual analysis, latency comparison plots, and summary table."""

import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

WORK_DIR = Path("C:/Users/arjun/Desktop/Sem 7/Ultrasound Project/ultrasound-beamforming-benchmarks")
RESULTS_DIR = WORK_DIR / "results"
BENCH_DIR = RESULTS_DIR / "benchmarks"
PLOTS_DIR = RESULTS_DIR / "plots"

def main():
    cpu_csv = BENCH_DIR / "das_full_benchmark_results.csv"
    cuda_csv = BENCH_DIR / "cuda_matrix_results.csv"
    master_csv = BENCH_DIR / "master_benchmarks.csv"

    all_rows = []
    if cpu_csv.exists():
        with open(cpu_csv, newline="") as f:
            all_rows.extend(list(csv.DictReader(f)))
    if cuda_csv.exists():
        with open(cuda_csv, newline="") as f:
            all_rows.extend(list(csv.DictReader(f)))

    if not all_rows:
        print("No benchmark records found yet.")
        return

    # Write combined master CSV
    fieldnames = []
    for r in all_rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    with open(master_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Saved master benchmark records to {master_csv}")

    # Plot 1: Latency Comparison across Optimization Flags (-O1, -O2, -O3)
    # Filter for hypoechoic dataset
    hypo_rows = [r for r in all_rows if r.get("dataset") == "hypoechoic" and r.get("median_ms")]
    if not hypo_rows:
        return

    fig, ax = plt.subplots(figsize=(12, 6.5), constrained_layout=True)
    
    variants = sorted(list({f"{r['target']}_{r.get('precision', '')}" for r in hypo_rows}))
    opt_flags = ["-O1", "-O2", "-O3"]
    x = np.arange(len(variants))
    width = 0.25

    for idx, opt in enumerate(opt_flags):
        medians = []
        for v in variants:
            match = [r for r in hypo_rows if f"{r['target']}_{r.get('precision', '')}" == v and r["optimization"] == opt]
            if match and match[0]["median_ms"]:
                medians.append(float(match[0]["median_ms"]))
            else:
                medians.append(0.0)
        ax.bar(x + idx * width, medians, width, label=opt)

    ax.set_yscale("log")
    ax.set_ylabel("Execution Time per Frame (ms, log scale)")
    ax.set_title("Ultrasound DAS Execution Latency · 507 × 600 Grid (304,200 Pixels)\nCPU vs GPU across Compiler Optimizations (-O1, -O2, -O3)")
    ax.set_xticks(x + width, [v.replace("_", " ").strip() for v in variants], rotation=25, ha="right")
    ax.legend(title="Optimization Level")
    ax.grid(axis="y", which="both", alpha=0.3)

    latency_plot = PLOTS_DIR / "master_latency_comparison.png"
    fig.savefig(latency_plot, dpi=160)
    plt.close(fig)
    print(f"Saved latency plot to {latency_plot}")

    # Plot 2: Image Quality Metrics (SSIM & PSNR) vs Latency (Pareto Frontier)
    valid_metrics = [r for r in hypo_rows if r.get("ssim") and r.get("median_ms")]
    if valid_metrics:
        fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
        for r in valid_metrics:
            lat = float(r["median_ms"])
            ssim = float(r["ssim"])
            label = f"{r['target']} {r.get('precision', '')} {r['optimization']}"
            ax.scatter(lat, ssim, s=90, alpha=0.85, label=label)
            ax.annotate(r['optimization'], (lat, ssim), textcoords="offset points", xytext=(5, 3), fontsize=8)

        ax.set_xscale("log")
        ax.set_xlabel("Latency per Frame (ms, log scale)")
        ax.set_ylabel("SSIM (vs Official zea Ground Truth)")
        ax.set_title("Pareto Analysis: Execution Speed vs. Image Fidelity (SSIM)")
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=8)

        pareto_plot = PLOTS_DIR / "pareto_latency_vs_ssim.png"
        fig.savefig(pareto_plot, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved Pareto plot to {pareto_plot}")

if __name__ == "__main__":
    main()
