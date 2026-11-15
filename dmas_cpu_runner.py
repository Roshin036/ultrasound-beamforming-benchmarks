"""Run CPU DMAS benchmarks across -O1, -O2, -O3, naive and template."""

import csv
from pathlib import Path
import subprocess
import time

WORK_DIR = Path("C:/Users/arjun/Desktop/Sem 7/Ultrasound Project/ultrasound-beamforming-benchmarks")
BUILD_DIR = WORK_DIR / "build"
RESULTS_DIR = WORK_DIR / "results"
BENCH_DIR = RESULTS_DIR / "benchmarks"

BENCH_DIR.mkdir(parents=True, exist_ok=True)

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
    print("=== Running CPU DMAS Benchmark Suite ===")
    records = []

    cpu_targets = [
        ("dmas_cpu_naive", "dmas_cpu_naive.cpp"),
        ("dmas_cpu_template", "dmas_cpu_template.cpp"),
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
                print(f"Compilation error:\n{res.stderr}")
                continue
            print(f"Compiled {exe_name} in {compile_time:.2f}s")

            for ds_name, bundle_path in DATASETS.items():
                tag = f"{exe_name}_{ds_name}"
                print(f"Running {tag}...")
                out_csv = BUILD_DIR / f"{tag}.csv"
                wsl_bundle = f"build/{bundle_path.name}"
                wsl_out = f"build/{out_csv.name}"

                # 1. Run generation
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

                bench_stats = {}
                for token in bench_out.strip().split():
                    if "=" in token:
                        k, v = token.split("=", 1)
                        try:
                            bench_stats[k] = float(v)
                        except ValueError:
                            bench_stats[k] = v

                records.append({
                    "algorithm": "DMAS",
                    "target": target_name,
                    "optimization": opt,
                    "dataset": ds_name,
                    "precision": "float32",
                    "compile_time_s": compile_time,
                    "mean_ms": bench_stats.get("mean_ms", None),
                    "median_ms": bench_stats.get("median_ms", None),
                    "p95_ms": bench_stats.get("p95_ms", None),
                    "min_ms": bench_stats.get("min_ms", None),
                    "max_ms": bench_stats.get("max_ms", None),
                })

    out_csv_path = BENCH_DIR / "dmas_cpu_benchmark_results.csv"
    if records:
        with open(out_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
        print(f"\nWrote CPU DMAS benchmark results to {out_csv_path}")

if __name__ == "__main__":
    main()
