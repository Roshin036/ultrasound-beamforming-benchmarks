import subprocess
import re
from pathlib import Path
import csv

WORK_DIR = Path("C:/Users/arjun/Desktop/Sem 7/Ultrasound Project/ultrasound-beamforming-benchmarks")
WSL_WORK_DIR = "/mnt/c/Users/arjun/Desktop/Sem 7/Ultrasound Project/ultrasound-beamforming-benchmarks"
DATASET = f"{WSL_WORK_DIR}/build/cpwc_hypoechoic.bundle"
# If bundle doesn't exist, check archive or data
GRID_ARGS = "-19.0 19.0 507 5.0 50.0 600"

results = []

def run_wsl(bash_cmd):
    cmd = ["wsl.exe", "bash", "-c", bash_cmd]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        print("Error executing:", bash_cmd)
        print("Stderr:", p.stderr)
        raise RuntimeError(f"Command failed: {p.stderr}")
    return p.stdout

# Check where cpwc_hypoechoic.bundle is
check_bundle = run_wsl(f"ls -la '{WSL_WORK_DIR}/build/cpwc_hypoechoic.bundle' 2>/dev/null || ls -la '{WSL_WORK_DIR}/data/archive/cpwc_hypoechoic.bundle' 2>/dev/null")
print("Found bundle:", check_bundle.strip())
bundle_path = f"{WSL_WORK_DIR}/build/cpwc_hypoechoic.bundle"

targets = [
    ("DAS", "das_cuda_template.cu", "bin/test_das_cuda_tmp"),
    ("DMAS", "dmas_cuda_template.cu", "bin/test_dmas_cuda_tmp")
]
opts = ["-O1", "-O2", "-O3"]
precisions = ["FP32", "FP64"]

run_wsl(f"mkdir -p '{WSL_WORK_DIR}/bin' '{WSL_WORK_DIR}/results'")

for algo, src, bin_rel in targets:
    for opt in opts:
        bin_path = f"{WSL_WORK_DIR}/{bin_rel}_{opt}"
        src_path = f"{WSL_WORK_DIR}/{src}"
        compile_cmd = f"nvcc {opt} -arch=sm_75 -std=c++17 '{src_path}' -o '{bin_path}'"
        print(f"\nCompiling {algo} {opt}...")
        run_wsl(compile_cmd)
        
        for prec in precisions:
            flag_str = "--benchmark" + (" --double" if prec == "FP64" else "")
            out_csv = f"{WSL_WORK_DIR}/results/bench_tmp_{algo}_{opt}_{prec}.csv"
            run_cmd_str = f"'{bin_path}' '{bundle_path}' '{out_csv}' {GRID_ARGS} {flag_str}"
            
            print(f"Running {algo} {opt} {prec}...")
            out = run_wsl(run_cmd_str)
            
            # Match benchmark output: e.g. "Median execution time: 19.73 ms"
            med_m = re.search(r"Median execution time:\s+([\d\.]+)\s+ms", out)
            mean_m = re.search(r"Mean execution time:\s+([\d\.]+)\s+ms", out)
            min_m = re.search(r"Min execution time:\s+([\d\.]+)\s+ms", out)
            max_m = re.search(r"Max execution time:\s+([\d\.]+)\s+ms", out)
            
            if med_m:
                med = float(med_m.group(1))
                mean = float(mean_m.group(1)) if mean_m else med
                mn = float(min_m.group(1)) if min_m else med
                mx = float(max_m.group(1)) if max_m else med
                print(f"  ==> {algo} {opt} {prec}: Median = {med:.2f} ms | Min = {mn:.2f} ms | Mean = {mean:.2f} ms")
                results.append({
                    "algorithm": algo,
                    "target": "cuda_template" if algo == "DAS" else "dmas_cuda_template",
                    "optimization": opt,
                    "precision": prec,
                    "median_ms": med,
                    "mean_ms": mean,
                    "min_ms": mn,
                    "max_ms": mx
                })
            else:
                print("Could not parse output:", out)

out_summary = WORK_DIR / "results/benchmarks/gpu_template_all_benchmarks.csv"
with open(out_summary, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["algorithm", "target", "optimization", "precision", "median_ms", "mean_ms", "min_ms", "max_ms"])
    writer.writeheader()
    writer.writerows(results)

print("\nAll GPU template benchmarks completed successfully!")
print("Saved to:", out_summary)
