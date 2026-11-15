"""Orchestrate compilation across -O1, -O2, -O3, execute benchmarks, and collect metrics."""

import csv
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np

WORK_DIR = Path("C:/Users/arjun/Desktop/Sem 7/Ultrasound Project/ultrasound-beamforming-benchmarks")
RESULTS_DIR = WORK_DIR / "results"
BENCH_DIR = RESULTS_DIR / "benchmarks"
PLOTS_DIR = RESULTS_DIR / "plots"
BUILD_DIR = WORK_DIR / "build"

BENCH_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

GROUND_TRUTH_NPY = Path("C:/Users/arjun/Desktop/Sem 7/Project/data/official_zea_reference/zea_bmode.npy")
DATASETS = {
    "hypoechoic": BUILD_DIR / "cpwc_hypoechoic.bundle",
    "hyperechoic": BUILD_DIR / "cpwc_hyperechoic.bundle",
}

# Grid parameters matching official zea grid
XMIN = -19.0
XMAX = 19.0
NX = 507
ZMIN = 5.0
ZMAX = 50.0
NZ = 600

def run_cmd(cmd, cwd=str(WORK_DIR)):
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
    return res.returncode, res.stdout, res.stderr

def run_wsl(cmd):
    wsl_cmd = f"wsl.exe -e bash -c \"cd '/mnt/c/Users/arjun/Desktop/Sem 7/Ultrasound Project/ultrasound-beamforming-benchmarks' && {cmd}\""
    return subprocess.run(wsl_cmd, shell=True, capture_output=True, text=True)

def main():
    print("=== Starting Ultrasound Beamforming Benchmark Suite ===")
    records = []

    # 1. Compile CPU variants under -O1, -O2, -O3
    cpu_targets = [
        ("cpu_naive", "das_cpu_naive.cpp"),
        ("cpu_template", "das_cpu_template.cpp"),
    ]
    opt_flags = ["-O1", "-O2", "-O3"]

    for target_name, src_file in cpu_targets:
        for opt in opt_flags:
            exe_name = f"{target_name}_{opt.replace('-', '')}"
            wsl_exe = f"./{exe_name}"
            print(f"\n--- Compiling {target_name} with {opt} ---")
            t0 = time.time()
            compile_cmd = f"g++ -std=c++20 {opt} {src_file} -o {exe_name}"
            res = run_wsl(compile_cmd)
            compile_time = time.time() - t0
            if res.returncode != 0:
                print(f"Compilation failed for {exe_name}:\n{res.stderr}")
                continue
            print(f"Compiled {exe_name} in {compile_time:.2f}s")

            # Check binary size
            exe_path = WORK_DIR / exe_name
            bin_size = exe_path.stat().st_size if exe_path.exists() else 0

            # Run on each dataset
            for ds_name, bundle_path in DATASETS.items():
                print(f"Running {exe_name} on {ds_name}...")
                out_csv = BUILD_DIR / f"{exe_name}_{ds_name}.csv"
                wsl_bundle = f"build/{bundle_path.name}"
                wsl_out = f"build/{out_csv.name}"

                # 1. Image generation run
                cmd_run = f"{wsl_exe} {wsl_bundle} {wsl_out} {XMIN} {XMAX} {NX} {ZMIN} {ZMAX} {NZ}"
                res_run = run_wsl(cmd_run)
                if res_run.returncode != 0:
                    print(f"Run error:\n{res_run.stderr}")
                    continue

                # 2. Benchmark run
                cmd_bench = f"{wsl_exe} {wsl_bundle} {wsl_out} {XMIN} {XMAX} {NX} {ZMIN} {ZMAX} {NZ} --benchmark"
                res_bench = run_wsl(cmd_bench)
                bench_out = res_bench.stdout
                print(f"Benchmark: {bench_out.strip()}")

                # Parse benchmark stdout
                # e.g.: BENCHMARK variant=... mean_ms=... median_ms=... p95_ms=... min_ms=... max_ms=...
                bench_stats = {}
                for token in bench_out.strip().split():
                    if "=" in token:
                        k, v = token.split("=", 1)
                        try:
                            bench_stats[k] = float(v)
                        except ValueError:
                            bench_stats[k] = v

                # 3. Evaluate Metrics vs Ground Truth (for hypoechoic)
                metrics = {}
                if ds_name == "hypoechoic" and GROUND_TRUTH_NPY.exists():
                    comp_plot = PLOTS_DIR / f"{exe_name}_{ds_name}_comparison.png"
                    eval_cmd = f"python evaluate_metrics.py \"{out_csv}\" \"{GROUND_TRUTH_NPY}\" --output_plot \"{comp_plot}\""
                    res_eval = subprocess.run(eval_cmd, shell=True, capture_output=True, text=True, cwd=str(WORK_DIR))
                    for line in res_eval.stdout.splitlines():
                        if ":" in line and not line.startswith("---"):
                            k, v = line.strip().split(":", 1)
                            try:
                                metrics[k.strip()] = float(v.strip())
                            except ValueError:
                                pass

                record = {
                    "target": target_name,
                    "optimization": opt,
                    "dataset": ds_name,
                    "precision": "float32",
                    "compile_time_s": compile_time,
                    "binary_bytes": bin_size,
                    "mean_ms": bench_stats.get("mean_ms", None),
                    "median_ms": bench_stats.get("median_ms", None),
                    "p95_ms": bench_stats.get("p95_ms", None),
                    "min_ms": bench_stats.get("min_ms", None),
                    "max_ms": bench_stats.get("max_ms", None),
                    "ssim": metrics.get("ssim", None),
                    "psnr_db": metrics.get("psnr_db", None),
                    "rmse_db": metrics.get("rmse_db", None),
                    "nrmse": metrics.get("nrmse", None),
                    "correlation": metrics.get("correlation", None),
                }
                records.append(record)

    # Save summary CSV
    summary_csv = BENCH_DIR / "das_full_benchmark_results.csv"
    if records:
        with open(summary_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
        print(f"\nSaved benchmark summary to {summary_csv}")

if __name__ == "__main__":
    main()
