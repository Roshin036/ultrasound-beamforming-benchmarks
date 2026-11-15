"""Run CUDA benchmarks across optimization levels -O1, -O2, -O3, both FP32 and FP64, naive and template."""

import csv
import os
from pathlib import Path
import subprocess
import time

WORK_DIR = Path("C:/Users/arjun/Desktop/Sem 7/Ultrasound Project/ultrasound-beamforming-benchmarks")
BUILD_DIR = WORK_DIR / "build"
RESULTS_DIR = WORK_DIR / "results"
BENCH_DIR = RESULTS_DIR / "benchmarks"
PLOTS_DIR = RESULTS_DIR / "plots"

GROUND_TRUTH_NPY = Path("C:/Users/arjun/Desktop/Sem 7/Project/data/official_zea_reference/zea_bmode.npy")
DATASETS = {
    "hypoechoic": BUILD_DIR / "cpwc_hypoechoic.bundle",
    "hyperechoic": BUILD_DIR / "cpwc_hyperechoic.bundle",
}

XMIN, XMAX, NX = -19.0, 19.0, 507
ZMIN, ZMAX, NZ = 5.0, 50.0, 600

def run_wsl(cmd):
    wsl_cmd = f"wsl.exe -e bash -c \"cd '/mnt/c/Users/arjun/Desktop/Sem 7/Ultrasound Project/ultrasound-beamforming-benchmarks' && {cmd}\""
    return subprocess.run(wsl_cmd, shell=True, capture_output=True, text=True)

def main():
    print("=== Running CUDA Benchmark Matrix ===")
    records = []

    targets = [
        ("cuda_naive", "das_cuda_naive.cu"),
        ("cuda_template", "das_cuda_template.cu"),
    ]
    opt_flags = ["-O1", "-O2", "-O3"]
    precisions = [("FP32", ""), ("FP64", "--double")]

    for target_name, src_file in targets:
        for opt in opt_flags:
            exe_name = f"{target_name}_{opt.replace('-', '')}"
            wsl_exe = f"./{exe_name}"
            print(f"\n--- Compiling {target_name} with {opt} ---")
            t0 = time.time()
            compile_cmd = f"nvcc -arch=sm_75 -std=c++20 {opt} {src_file} -o {exe_name}"
            res = run_wsl(compile_cmd)
            compile_time = time.time() - t0
            if res.returncode != 0:
                print(f"Compilation error:\n{res.stderr}")
                continue
            print(f"Compiled {exe_name} in {compile_time:.2f}s")

            exe_path = WORK_DIR / exe_name
            bin_size = exe_path.stat().st_size if exe_path.exists() else 0

            for prec_name, prec_flag in precisions:
                for ds_name, bundle_path in DATASETS.items():
                    tag = f"{exe_name}_{prec_name}_{ds_name}"
                    print(f"Running {tag}...")
                    out_csv = BUILD_DIR / f"{tag}.csv"
                    wsl_bundle = f"build/{bundle_path.name}"
                    wsl_out = f"build/{out_csv.name}"

                    # 1. Image generation run
                    cmd_run = f"{wsl_exe} {wsl_bundle} {wsl_out} {XMIN} {XMAX} {NX} {ZMIN} {ZMAX} {NZ} {prec_flag}"
                    res_run = run_wsl(cmd_run)
                    if res_run.returncode != 0:
                        print(f"Run error:\n{res_run.stderr}")
                        continue

                    # 2. Benchmark run
                    cmd_bench = f"{wsl_exe} {wsl_bundle} {wsl_out} {XMIN} {XMAX} {NX} {ZMIN} {ZMAX} {NZ} {prec_flag} --benchmark"
                    res_bench = run_wsl(cmd_bench)
                    bench_out = res_bench.stdout
                    print(f"Benchmark: {bench_out.strip()}")

                    bench_stats = {}
                    for token in bench_out.strip().split():
                        if "=" in token:
                            k, v = token.split("=", 1)
                            try:
                                bench_stats[k] = float(v)
                            except ValueError:
                                bench_stats[k] = v

                    # 3. Evaluate Metrics vs Ground Truth (hypoechoic)
                    metrics = {}
                    if ds_name == "hypoechoic" and GROUND_TRUTH_NPY.exists():
                        comp_plot = PLOTS_DIR / f"{tag}_comparison.png"
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
                        "precision": prec_name,
                        "compile_time_s": compile_time,
                        "binary_bytes": bin_size,
                        "mean_ms": bench_stats.get("mean_ms", None),
                        "median_ms": bench_stats.get("median_ms", None),
                        "p95_ms": bench_stats.get("p95_ms", None),
                        "min_ms": bench_stats.get("min_ms", None),
                        "max_ms": bench_stats.get("max_ms", None),
                        "device_buffer_bytes": bench_stats.get("device_buffer_bytes", None),
                        "ssim": metrics.get("ssim", None),
                        "psnr_db": metrics.get("psnr_db", None),
                        "rmse_db": metrics.get("rmse_db", None),
                        "nrmse": metrics.get("nrmse", None),
                        "correlation": metrics.get("correlation", None),
                    }
                    records.append(record)

    out_csv_path = BENCH_DIR / "cuda_matrix_results.csv"
    if records:
        with open(out_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
        print(f"\nWrote CUDA matrix results to {out_csv_path}")

if __name__ == "__main__":
    main()
