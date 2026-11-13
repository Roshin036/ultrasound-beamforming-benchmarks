"""Measure clean CMake target build wall times using the installed Windows toolchain."""

from __future__ import annotations

import argparse
import csv
import subprocess
import time
from pathlib import Path


def run_command(command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        f"cmd.exe /d /c {command}",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=Path("build/cmake_nmake"))
    parser.add_argument("--output", type=Path, default=Path("build/das_compile_times.csv"))
    parser.add_argument(
        "--cmake", default=r"C:\Program\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
    )
    parser.add_argument("--vsdevcmd", default=r"C:\Program\Common7\Tools\VsDevCmd.bat")
    parser.add_argument("--targets", nargs="+", default=[
        "das_cpu_naive", "das_cpu_template", "das_cuda_naive", "das_cuda_template"
    ])
    args = parser.parse_args()

    def invoke(cmake_args: str) -> subprocess.CompletedProcess[str]:
        command = (
            f'call "{args.vsdevcmd}" -no_logo -arch=x64 && '
            f'"{args.cmake}" --build "{args.build_dir.resolve()}" {cmake_args}'
        )
        return run_command(command)

    results: list[tuple[str, float]] = []
    for target in args.targets:
        clean = invoke("--target clean")
        if clean.returncode != 0:
            raise RuntimeError(f"CMake clean failed before {target}:\n{clean.stdout}")
        start = time.perf_counter()
        build = invoke(f"--target {target}")
        elapsed = time.perf_counter() - start
        print(build.stdout, end="")
        if build.returncode != 0:
            raise RuntimeError(f"Build failed for {target} after {elapsed:.3f}s")
        results.append((target, elapsed))
        print(f"BUILD_TIME target={target} clean_wall_seconds={elapsed:.3f}")

    # Restore all benchmark executables after the per-target clean builds.
    restore = invoke("--target das_cpu_naive das_cpu_template das_cuda_naive das_cuda_template")
    print(restore.stdout, end="")
    if restore.returncode != 0:
        raise RuntimeError("Could not restore all benchmark executables")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["target", "clean_build_wall_seconds", "build_type", "notes"])
        for target, elapsed in results:
            writer.writerow([target, f"{elapsed:.3f}", "Release", "includes VS environment, CMake/NMake, compile, and link"])
    print(f"Wrote build times to {args.output}")


if __name__ == "__main__":
    main()
