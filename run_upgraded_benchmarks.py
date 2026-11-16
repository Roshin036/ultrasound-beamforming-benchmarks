import csv
from pathlib import Path
import subprocess

WORK_DIR = Path("C:/Users/arjun/Desktop/Sem 7/Ultrasound Project/ultrasound-beamforming-benchmarks")
BUILD_DIR = WORK_DIR / "build"

opts = ["-O1", "-O2", "-O3"]
targets = [
    ("das_cpu_template", "cpu_template", "DAS"),
    ("dmas_cpu_template", "dmas_cpu_template", "DMAS"),
]

def run_wsl(cmd):
    wsl_cmd = f"wsl.exe -e bash -c \"cd '/mnt/c/Users/arjun/Desktop/Sem 7/Ultrasound Project/ultrasound-beamforming-benchmarks' && {cmd}\""
    return subprocess.run(wsl_cmd, shell=True, capture_output=True, text=True)

results = []
for src_name, target_name, algo in targets:
    for opt in opts:
        exe = f"build/{src_name}_{opt.replace('-', '')}"
        print(f"--- Compiling {src_name} {opt} ---")
        res_c = run_wsl(f"g++ -std=c++20 {opt} -mavx2 -mfma {src_name}.cpp -o {exe}")
        if res_c.returncode != 0:
            print("Compile error:", res_c.stderr)
            continue

        print(f"--- Benchmarking {src_name} {opt} ---")
        res_b = run_wsl(f"./{exe} build/cpwc_hypoechoic.bundle /dev/null -19.0 19.0 507 5.0 50.0 600 --benchmark")
        out = res_b.stdout.strip()
        print(out)
        stats = {}
        for token in out.split():
            if "=" in token:
                k, v = token.split("=", 1)
                try:
                    stats[k] = float(v)
                except ValueError:
                    stats[k] = v
        results.append({
            "algorithm": algo,
            "target": target_name,
            "optimization": opt,
            "dataset": "hypoechoic",
            "precision": "float32",
            "median_ms": stats.get("median_ms"),
            "mean_ms": stats.get("mean_ms"),
            "min_ms": stats.get("min_ms"),
            "max_ms": stats.get("max_ms"),
        })

out_csv = WORK_DIR / "results/benchmarks/upgraded_template_benchmarks.csv"
if results:
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved upgraded template benchmarks to {out_csv}")
