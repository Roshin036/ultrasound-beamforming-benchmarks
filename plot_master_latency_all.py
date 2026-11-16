"""Generate comprehensive latency plot for all DAS and DMAS variants across CPU, GPU, and compilers."""

import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

WORK_DIR = Path("C:/Users/arjun/Desktop/Sem 7/Ultrasound Project/ultrasound-beamforming-benchmarks")
RESULTS_DIR = WORK_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

# Load baseline data
das_master_csv = RESULTS_DIR / "benchmarks/master_benchmarks.csv"
dmas_master_csv = RESULTS_DIR / "benchmarks/dmas_master_benchmarks.csv"
cpu_upgraded_csv = RESULTS_DIR / "benchmarks/upgraded_template_benchmarks.csv"
gpu_upgraded_csv = RESULTS_DIR / "benchmarks/gpu_template_all_benchmarks.csv"

def norm_prec(p):
    if not p:
        return ""
    p_up = str(p).upper()
    if "32" in p_up or "FLOAT" in p_up:
        return "FP32"
    if "64" in p_up or "DOUBLE" in p_up:
        return "FP64"
    return p_up

# Load baseline rows
with open(das_master_csv, newline="") as f:
    das_rows = [r for r in csv.DictReader(f) if r.get("dataset") == "hypoechoic" and r.get("median_ms")]

with open(dmas_master_csv, newline="") as f:
    dmas_rows = [r for r in csv.DictReader(f) if r.get("dataset") == "hypoechoic" and r.get("median_ms")]

# Load newly measured CPU and GPU template benchmarks
upgraded_map = {}

# CPU upgraded template
if cpu_upgraded_csv.exists():
    with open(cpu_upgraded_csv, newline="") as f:
        for r in csv.DictReader(f):
            key = (r["algorithm"], r["target"], r["optimization"], "FP32")
            upgraded_map[key] = float(r["median_ms"])

# GPU upgraded template (both FP32 and FP64!)
if gpu_upgraded_csv.exists():
    with open(gpu_upgraded_csv, newline="") as f:
        for r in csv.DictReader(f):
            key = (r["algorithm"], r["target"], r["optimization"], norm_prec(r["precision"]))
            upgraded_map[key] = float(r["median_ms"])

print("Loaded upgraded map keys:")
for k, v in sorted(upgraded_map.items()):
    print(f"  {k}: {v:.2f} ms")

# Standardized targets for plotting
targets_info = [
    ("CPU Naive", "cpu_naive", "dmas_cpu_naive", "FP32", False),
    ("CPU Template (TMP)", "cpu_template", "dmas_cpu_template", "FP32", True),
    ("GPU Naive FP32", "cuda_naive", "dmas_cuda_naive", "FP32", False),
    ("GPU Template FP32 (TMP)", "cuda_template", "dmas_cuda_template", "FP32", True),
    ("GPU Naive FP64", "cuda_naive", "dmas_cuda_naive", "FP64", False),
    ("GPU Template FP64 (TMP)", "cuda_template", "dmas_cuda_template", "FP64", True),
]

def get_latency(rows, target_sub, opt, prec, algo, is_template):
    # Check if upgraded template exists
    if is_template and (algo, target_sub, opt, prec) in upgraded_map:
        return upgraded_map[(algo, target_sub, opt, prec)]
    
    target_clean = target_sub.replace("dmas_", "")
    # Search baseline rows
    for r in rows:
        if r.get("optimization") == opt and (target_sub == r.get("target") or target_clean == r.get("target")):
            row_prec = norm_prec(r.get("precision", ""))
            if not prec or row_prec == prec:
                return float(r["median_ms"])
    return None

def main():
    fig, axes = plt.subplots(2, 1, figsize=(14, 11), constrained_layout=True)

    # -------------------------------------------------------------
    # Panel 1: Head-to-Head DAS vs DMAS at -O3 (Log Scale)
    # -------------------------------------------------------------
    labels = [t[0] for t in targets_info]
    das_vals_o3 = []
    dmas_vals_o3 = []

    for label, das_t, dmas_t, prec, is_tmp in targets_info:
        das_lat = get_latency(das_rows, das_t, "-O3", prec, "DAS", is_tmp)
        dmas_lat = get_latency(dmas_rows, dmas_t, "-O3", prec, "DMAS", is_tmp)
        print(f"Panel 1 -> {label}: DAS={das_lat} ms, DMAS={dmas_lat} ms")
        das_vals_o3.append(das_lat if das_lat else 0.0)
        dmas_vals_o3.append(dmas_lat if dmas_lat else 0.0)

    x = np.arange(len(labels))
    w = 0.36

    bars1 = axes[0].bar(x - w/2, das_vals_o3, w, label="Standard DAS", color="#1f77b4", alpha=0.9, edgecolor="black", linewidth=0.8)
    bars2 = axes[0].bar(x + w/2, dmas_vals_o3, w, label="Filtered DMAS (F-DMAS)", color="#ff7f0e", alpha=0.9, edgecolor="black", linewidth=0.8)

    for bar in bars1:
        h = bar.get_height()
        if h > 0:
            txt = f"{h:.1f} ms" if h < 1000 else f"{h/1000:.2f} s"
            axes[0].annotate(txt, xy=(bar.get_x() + bar.get_width() / 2, h),
                             xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                             fontsize=8.5, fontweight="bold", color="#1f77b4")

    for bar in bars2:
        h = bar.get_height()
        if h > 0:
            txt = f"{h:.1f} ms" if h < 1000 else f"{h/1000:.2f} s"
            axes[0].annotate(txt, xy=(bar.get_x() + bar.get_width() / 2, h),
                             xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                             fontsize=8.5, fontweight="bold", color="#d95f02")

    axes[0].set_yscale("log")
    axes[0].set_ylim([10, 50000])
    axes[0].set_ylabel("Execution Time per Frame (ms, log scale)", fontsize=11, fontweight="bold")
    axes[0].set_title("Ultrasound Beamforming Execution Latency: DAS vs. DMAS (-O3 Optimization)\n507 × 600 Grid (304,200 Pixels) · Intel Core i7-10750H vs NVIDIA GeForce GTX 1650", fontsize=12, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=10.5, fontweight="bold")
    axes[0].legend(fontsize=11, loc="upper right")
    axes[0].grid(axis="y", which="both", alpha=0.3, linestyle="--")

    # Add speedup callout annotations for TMP upgrades
    axes[0].annotate("8.4× CPU Speedup\n1.78s via Precomputation & TMP!",
                     xy=(1 - w/2, das_vals_o3[1]), xytext=(1 - 0.35, 7500),
                     arrowprops=dict(arrowstyle="->", color="purple", lw=1.8),
                     fontsize=9, fontweight="bold", color="purple",
                     bbox=dict(boxstyle="round,pad=0.3", fc="#f3e5f5", ec="purple", lw=1))

    axes[0].annotate("2.2× GPU Speedup (FP32)\n20.2 ms (49.5 FPS)!",
                     xy=(3 - w/2, das_vals_o3[3]), xytext=(3 - 0.35, 120),
                     arrowprops=dict(arrowstyle="->", color="#006600", lw=1.8),
                     fontsize=9, fontweight="bold", color="#006600",
                     bbox=dict(boxstyle="round,pad=0.3", fc="#e8f5e9", ec="#006600", lw=1))

    axes[0].annotate("4.2× GPU Speedup (FP64)\n315 ms vs 1,320 ms!",
                     xy=(5 - w/2, das_vals_o3[5]), xytext=(5 - 0.45, 1200),
                     arrowprops=dict(arrowstyle="->", color="#b71c1c", lw=1.8),
                     fontsize=9, fontweight="bold", color="#b71c1c",
                     bbox=dict(boxstyle="round,pad=0.3", fc="#ffebee", ec="#b71c1c", lw=1))

    # -------------------------------------------------------------
    # Panel 2: Full Compiler Matrix across -O1, -O2, -O3 (Grouped)
    # -------------------------------------------------------------
    opts = ["-O1", "-O2", "-O3"]
    colors_das = ["#9ecae1", "#4292c6", "#08519c"]
    colors_dmas = ["#fdd0a2", "#fdae6b", "#d94801"]

    sub_w = 0.13
    offsets = [-2.5 * sub_w, -1.5 * sub_w, -0.5 * sub_w, 0.5 * sub_w, 1.5 * sub_w, 2.5 * sub_w]

    for i_opt, opt in enumerate(opts):
        das_vals = []
        dmas_vals = []
        for label, das_t, dmas_t, prec, is_tmp in targets_info:
            d_val = get_latency(das_rows, das_t, opt, prec, "DAS", is_tmp) or 0.0
            dm_val = get_latency(dmas_rows, dmas_t, opt, prec, "DMAS", is_tmp) or 0.0
            das_vals.append(d_val)
            dmas_vals.append(dm_val)

        axes[1].bar(x + offsets[i_opt], das_vals, sub_w, label=f"DAS {opt}", color=colors_das[i_opt], edgecolor="black", linewidth=0.5)
        axes[1].bar(x + offsets[i_opt + 3], dmas_vals, sub_w, label=f"DMAS {opt}", color=colors_dmas[i_opt], edgecolor="black", linewidth=0.5)

    axes[1].set_yscale("log")
    axes[1].set_ylim([10, 60000])
    axes[1].set_ylabel("Execution Time per Frame (ms, log scale)", fontsize=11, fontweight="bold")
    axes[1].set_title("Compiler Optimization Sensitivity across -O1, -O2, -O3 for All Platforms & Precisions", fontsize=12, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=10.5, fontweight="bold")
    axes[1].legend(ncol=6, fontsize=9.5, loc="upper right")
    axes[1].grid(axis="y", which="both", alpha=0.3, linestyle="--")

    out_file = PLOTS_DIR / "master_latency_all_das_dmas.png"
    fig.savefig(out_file, dpi=160)
    plt.close(fig)
    print("Saved fresh comprehensive latency plot to", out_file)

if __name__ == "__main__":
    main()
