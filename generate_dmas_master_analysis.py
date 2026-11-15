"""Generate master visual analysis and comparison plots for DMAS benchmark suite."""

import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

WORK_DIR = Path("C:/Users/arjun/Desktop/Sem 7/Ultrasound Project/ultrasound-beamforming-benchmarks")
RESULTS_DIR = WORK_DIR / "results"
BENCH_DIR = RESULTS_DIR / "benchmarks"
PLOTS_DIR = RESULTS_DIR / "plots"

def main():
    cpu_csv = BENCH_DIR / "dmas_cpu_benchmark_results.csv"
    cuda_csv = BENCH_DIR / "dmas_matrix_results.csv"
    dmas_master_csv = BENCH_DIR / "dmas_master_benchmarks.csv"
    das_master_csv = BENCH_DIR / "master_benchmarks.csv"

    all_rows = []
    if cpu_csv.exists():
        with open(cpu_csv, newline="") as f:
            all_rows.extend(list(csv.DictReader(f)))
    if cuda_csv.exists():
        with open(cuda_csv, newline="") as f:
            all_rows.extend(list(csv.DictReader(f)))

    if not all_rows:
        print("No DMAS benchmark records found.")
        return

    # 1. Write combined master DMAS CSV
    fieldnames = []
    for r in all_rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    with open(dmas_master_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Saved master DMAS benchmark records to {dmas_master_csv}")

    # 2. DMAS Latency Comparison Plot across Optimization Flags (-O1, -O2, -O3)
    hypo_rows = [r for r in all_rows if r.get("dataset") == "hypoechoic" and r.get("median_ms")]
    if hypo_rows:
        fig, ax = plt.subplots(figsize=(12, 6.5), constrained_layout=True)
        variants = sorted(list({f"{r['target']}_{r.get('precision', '')}" for r in hypo_rows}))
        opt_flags = ["-O1", "-O2", "-O3"]
        x = np.arange(len(variants))
        width = 0.25

        colors = ["#4C72B0", "#55A868", "#C44E52"]
        for idx, opt in enumerate(opt_flags):
            medians = []
            for v in variants:
                match = [r for r in hypo_rows if f"{r['target']}_{r.get('precision', '')}" == v and r["optimization"] == opt]
                if match and match[0]["median_ms"]:
                    medians.append(float(match[0]["median_ms"]))
                else:
                    medians.append(0.0)
            bars = ax.bar(x + idx * width, medians, width, label=opt, color=colors[idx], alpha=0.85)
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    val_str = f"{height:.1f} ms" if height < 1000 else f"{height/1000:.2f} s"
                    ax.annotate(val_str,
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3), textcoords="offset points",
                                ha="center", va="bottom", fontsize=7.5, rotation=45)

        ax.set_yscale("log")
        ax.set_ylabel("Execution Time per Frame (ms, log scale)", fontsize=11)
        ax.set_title("Filtered DMAS Execution Latency · 507 × 600 Grid (304,200 Pixels)\nCPU vs GPU across Compiler Optimizations (-O1, -O2, -O3)", fontsize=12, fontweight="bold")
        ax.set_xticks(x + width, [v.replace("_", " ").replace("float32", "FP32").strip() for v in variants], rotation=25, ha="right", fontsize=9.5)
        ax.legend(title="Optimization Level", fontsize=10)
        ax.grid(axis="y", which="both", alpha=0.3)

        dmas_latency_plot = PLOTS_DIR / "dmas_latency_comparison.png"
        fig.savefig(dmas_latency_plot, dpi=160)
        plt.close(fig)
        print(f"Saved DMAS latency plot to {dmas_latency_plot}")

    # 3. DAS vs DMAS Latency Comparison Overlay at -O3
    if das_master_csv.exists() and dmas_master_csv.exists():
        with open(das_master_csv, newline="") as f:
            das_rows = list(csv.DictReader(f))

        # Filter for -O3 and hypoechoic
        das_o3 = {f"{r['target']}_{r.get('precision', '')}": float(r['median_ms'])
                  for r in das_rows if r.get("optimization") == "-O3" and r.get("dataset") == "hypoechoic" and r.get("median_ms")}
        dmas_o3 = {f"{r['target']}_{r.get('precision', '')}": float(r['median_ms'])
                   for r in all_rows if r.get("optimization") == "-O3" and r.get("dataset") == "hypoechoic" and r.get("median_ms")}

        # Standardize keys for clean 1-to-1 comparison
        key_pairs = [
            ("cpu_naive_float32", "dmas_cpu_naive_float32", "CPU Naive"),
            ("cpu_template_float32", "dmas_cpu_template_float32", "CPU Template"),
            ("cuda_naive_FP32", "dmas_cuda_naive_FP32", "GPU Naive FP32"),
            ("cuda_template_FP32", "dmas_cuda_template_FP32", "GPU Template FP32"),
            ("cuda_naive_FP64", "dmas_cuda_naive_FP64", "GPU Naive FP64"),
            ("cuda_template_FP64", "dmas_cuda_template_FP64", "GPU Template FP64"),
        ]

        labels = []
        das_vals = []
        dmas_vals = []
        for das_k, dmas_k, lbl in key_pairs:
            if das_k in das_o3 and dmas_k in dmas_o3:
                labels.append(lbl)
                das_vals.append(das_o3[das_k])
                dmas_vals.append(dmas_o3[dmas_k])

        if labels:
            fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
            x = np.arange(len(labels))
            w = 0.35

            bars1 = ax.bar(x - w/2, das_vals, w, label="Standard DAS", color="#2b5c8f", alpha=0.85)
            bars2 = ax.bar(x + w/2, dmas_vals, w, label="Filtered DMAS (O(M))", color="#d95f02", alpha=0.85)

            for bar in bars1:
                h = bar.get_height()
                ax.annotate(f"{h:.1f}ms" if h < 1000 else f"{h/1000:.2f}s",
                            xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=7.5, rotation=35)
            for bar in bars2:
                h = bar.get_height()
                ax.annotate(f"{h:.1f}ms" if h < 1000 else f"{h/1000:.2f}s",
                            xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=7.5, rotation=35)

            ax.set_yscale("log")
            ax.set_ylabel("Execution Time per Frame (ms, log scale)", fontsize=11)
            ax.set_title("Standard DAS vs. Filtered DMAS Execution Latency (-O3 Optimization)\n507 × 600 Grid (304,200 Pixels) · GTX 1650 vs Multi-core Host", fontsize=12, fontweight="bold")
            ax.set_xticks(x, labels, fontsize=10)
            ax.legend(fontsize=10)
            ax.grid(axis="y", which="both", alpha=0.3)

            comp_plot = PLOTS_DIR / "das_vs_dmas_latency_comparison.png"
            fig.savefig(comp_plot, dpi=160)
            plt.close(fig)
            print(f"Saved DAS vs DMAS comparison plot to {comp_plot}")

if __name__ == "__main__":
    main()
