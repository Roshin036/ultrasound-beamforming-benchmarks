"""Plot recorded DAS benchmark latency summaries (median with min/max whiskers)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Benchmark summary CSV")
    parser.add_argument("output", type=Path, help="Output PNG")
    args = parser.parse_args()

    with args.input.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("Benchmark summary is empty")
    names = [row["variant"].replace("_", " ") for row in rows]
    median = np.array([float(row["median_ms"]) for row in rows])
    low = np.array([float(row["min_ms"]) for row in rows])
    high = np.array([float(row["max_ms"]) for row in rows])
    means = [float(row["mean_ms"]) for row in rows]
    p95s = [float(row["p95_ms"]) for row in rows]
    colors = ["#4C78A8", "#72A0C1", "#F58518", "#E2A04A"]

    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    x = np.arange(len(rows))
    ax.bar(x, median, color=colors, width=0.68, zorder=2)
    ax.errorbar(x, median, yerr=np.vstack((median - low, high - median)),
                fmt="none", ecolor="#222222", capsize=5, linewidth=1.2, zorder=3)
    ax.set_yscale("log")
    ax.set_xticks(x, names)
    ax.set_ylabel("Latency per image (ms, log scale)")
    ax.set_title("DAS compute latency · 128 × 191 pixels · 30 runs after 2 warmups\n"
                 "CPU: steady_clock compute loop · CUDA: event-timed kernel · bars: median, whiskers: min–max")
    ax.grid(axis="y", which="both", alpha=0.25, zorder=0)
    for i, (mean, p95) in enumerate(zip(means, p95s)):
        ax.annotate(f"mean {mean:.3g}\np95 {p95:.3g}",
                    (i, median[i]), xytext=(7, 2), textcoords="offset points",
                    ha="left", va="bottom", fontsize=8)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
