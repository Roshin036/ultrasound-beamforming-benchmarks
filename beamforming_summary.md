# Ultrasound Beamforming Master Benchmark Report

## 1. Executive Summary & Hardware Context

This report documents the comprehensive benchmarking of Delay-and-Sum (DAS) beamforming across **24 distinct configuration variants**:
- **Platforms**: CPU Naive, CPU Compile-Time Specialized (Template / TMP), GPU Naive, GPU Template.
- **Precisions**: Single Precision (FP32) and Double Precision (FP64) on GPU.
- **Compiler Optimization Levels**: `-O1`, `-O2`, `-O3` across all targets.
- **Evaluation Datasets**: CPWC Hypoechoic Cyst Phantom and CPWC Hyperechoic Scatterers Phantom.
- **Standardized Grid**: Official clinical grid **$507 \times 600$ (304,200 pixels)** spanning $x \in [-19, 19]$ mm and $z \in [5, 50]$ mm.
- **Hardware Architecture**: **NVIDIA GeForce GTX 1650** (Turing architecture, `sm_75`, 4 GB VRAM) running CUDA 12.0 / GCC 13.3.0.

```mermaid
flowchart TD
    Raw["Raw CPWC RF Data (21 TX × 128 RX × 4352 Samples)"] --> CPU["CPU Pipeline (GCC 13.3.0)"]
    Raw --> GPU["GPU Pipeline (CUDA 12.0, sm_75)"]
    
    CPU --> C1["CPU Naive (-O1, -O2, -O3) ~15,000–19,500 ms"]
    CPU --> C2["CPU Template (-O1, -O2, -O3) ~15,000–15,700 ms"]
    
    GPU --> G1["GPU FP32 (Naive / Template) ~42.7–51.5 ms (23.4 FPS)"]
    GPU --> G2["GPU FP64 (Naive / Template) ~1,319–1,414 ms (0.76 FPS)"]
    
    G1 --> Out["Coherent 507 × 600 Beamformed B-Mode Image"]
    C2 --> Out
```

---

## 2. Master Benchmark Results (507 × 600 Grid, 304,200 Pixels)

Measured over 30 timed kernel runs following 2 warm-ups (CPU compute loop timed via `std::chrono::steady_clock`; GPU kernel timed via `cudaEvent_t`):

### Hypoechoic Cyst Dataset
| Target | Opt | Precision | Binary Size | Median (ms) | Mean (ms) | p95 (ms) | Min (ms) | Throughput (FPS) | Speedup vs CPU Naive |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CPU Naive** | `-O1` | FP32 | 37.4 KB | 15,384.3 | 15,527.1 | 15,998.9 | 14,909.3 | 0.065 FPS | 1.00× |
| **CPU Naive** | `-O2` | FP32 | 38.0 KB | 15,189.1 | 15,150.0 | 15,402.6 | 14,834.3 | 0.066 FPS | 1.01× |
| **CPU Naive** | `-O3` | FP32 | 37.9 KB | 19,557.8 | 19,281.3 | 20,027.5 | 17,496.6 | 0.051 FPS | 0.79× |
| **CPU Template** | `-O1` | FP32 | 37.5 KB | 23,139.3 | 30,379.5 | 59,275.3 | 18,865.7 | 0.043 FPS | 0.66× |
| **CPU Template** | `-O2` | FP32 | 42.2 KB | 15,695.1 | 15,847.8 | 16,046.6 | 15,494.0 | 0.064 FPS | 0.98× |
| **CPU Template** | `-O3` | FP32 | 37.9 KB | 15,001.1 | 15,146.1 | 15,494.5 | 14,808.2 | 0.067 FPS | 1.03× |
| **CUDA Naive** | `-O1` | **FP32** | 914 KB | 43.67 | 44.04 | 44.04 | 43.17 | 22.90 FPS | **352.3×** |
| **CUDA Naive** | `-O2` | **FP32** | 914 KB | 48.08 | 48.68 | 51.67 | 43.57 | 20.80 FPS | **320.0×** |
| **CUDA Naive** | `-O3` | **FP32** | 922 KB | 51.54 | 50.44 | 52.07 | 43.71 | 19.40 FPS | **298.5×** |
| **CUDA Naive** | `-O1` | FP64 | 914 KB | 1,319.21 | 1,320.58 | 1,321.78 | 1,317.93 | 0.76 FPS | 11.66× |
| **CUDA Naive** | `-O2` | FP64 | 914 KB | 1,319.35 | 1,329.89 | 1,349.47 | 1,318.15 | 0.76 FPS | 11.66× |
| **CUDA Naive** | `-O3` | FP64 | 922 KB | 1,320.55 | 1,342.74 | 1,385.61 | 1,318.14 | 0.76 FPS | 11.65× |
| **CUDA Template** | `-O1` | **FP32** | 906 KB | 51.01 | 49.49 | 51.46 | 43.03 | 19.60 FPS | **301.6×** |
| **CUDA Template** | `-O2` | **FP32** | 906 KB | 47.44 | 48.02 | 51.53 | 43.13 | 21.08 FPS | **324.3×** |
| **CUDA Template** | `-O3` | **FP32** | 914 KB | **42.75** | **43.31** | **44.51** | **41.89** | **23.39 FPS** | **359.9×** |
| **CUDA Template** | `-O1` | FP64 | 906 KB | 1,406.73 | 1,388.00 | 1,440.72 | 1,318.11 | 0.71 FPS | 10.94× |
| **CUDA Template** | `-O2` | FP64 | 906 KB | 1,319.67 | 1,320.37 | 1,320.32 | 1,318.22 | 0.76 FPS | 11.66× |
| **CUDA Template** | `-O3` | FP64 | 914 KB | 1,319.47 | 1,319.46 | 1,320.38 | 1,318.23 | 0.76 FPS | 11.66× |

---

## 3. Visualizations

### Master Execution Latency (CPU vs GPU across -O1, -O2, -O3)
![Master Execution Latency](./results/plots/master_latency_comparison.png)

### Pareto Analysis: Execution Latency vs. Image Quality
![Pareto Frontier: Latency vs SSIM](./results/plots/pareto_latency_vs_ssim.png)

### CUDA Output vs. Official Reference Comparison (Hypoechoic Cyst)
![CUDA Template O3 Hypoechoic Comparison](./results/plots/cuda_template_O3_FP32_hypoechoic_comparison.png)

---

## 4. Algorithmic Equivalence: Single vs Double Precision (FP64 Reference)

To verify that single precision (FP32), compiler flags (`-O1`, `-O2`, `-O3`), and template metaprogramming introduce no numerical degradation, all implementations were compared against the exact **Double-Precision (FP64) Reference**:

| Evaluated Implementation | Max Absolute Error | RMSE | Normalized RMSE (NRMSE) | PSNR (dB) | Pearson Correlation ($r$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CPU Naive -O3** | 286.31 | 1.0480 | $1.896 \times 10^{-5}$ (0.0019%) | **94.44 dB** | **0.99999997** |
| **CPU Template -O3** | 286.31 | 1.0480 | $1.896 \times 10^{-5}$ (0.0019%) | **94.44 dB** | **0.99999997** |
| **CUDA Naive -O3 (FP32)** | 286.21 | 1.0451 | $1.891 \times 10^{-5}$ (0.0019%) | **94.47 dB** | **0.99999997** |
| **CUDA Template -O3 (FP32)** | 286.21 | 1.0451 | $1.891 \times 10^{-5}$ (0.0019%) | **94.47 dB** | **0.99999997** |

> [!TIP]
> The single precision implementations achieve **PSNR = 94.47 dB** and **Correlation = 0.99999997** against double precision. Floating-point round-off is well below the noise floor of 12-bit/14-bit medical ultrasound ADCs.

---

## 5. Architectural Findings: Hardware FP64 Throttle on GeForce GTX 1650

A key empirical finding of this benchmark is the massive performance gap between FP32 and FP64 on consumer Turing GPUs:
- **FP32 Latency**: **42.75 ms / frame (~23.4 FPS)**.
- **FP64 Latency**: **1,319.47 ms / frame (~0.76 FPS)**.
- **Hardware FP64 Penalty**: **~30.9× slowdown**.

Consumer GeForce GPUs (GTX 1650, Turing TU117) have an FP64-to-FP32 ALU ratio of **1:32** (one FP64 unit per 32 FP32 cores). In ultrasound beamforming, double precision incurs a 31× latency penalty with virtually zero difference in perceptible image quality (correlation difference $< 10^{-7}$). FP32 is overwhelmingly optimal for real-time deployment.

---

## 6. Clinical Ground Truth Validation (Zea Pipeline Analysis)

When compared against the external clinical reference generated by `zea`:
- **SSIM**: `0.7631`
- **PSNR**: `21.17 dB`
- **Correlation**: `0.7579`
- **RMSE**: `5.24 dB`

### Why Does Time-Domain Real-RF DAS Differ from Zea?
Zea's pipeline is not real-RF DAS. It performs a multi-stage **Complex Baseband IQ Demodulation**:
1. **Ring-down Windowing (`apply_window`)**: Zea zeroes out the initial 16 samples and tapers samples 16–48 to suppress transducer ring-down.
2. **Demodulation (`demodulate`)**: Multiplies raw RF by $e^{-j 2\pi f_c t}$ ($f_c = 5.2$ MHz) followed by a low-pass FIR filter to generate baseband IQ data.
3. **Carrier Phase Rotation**: In baseband IQ, fractional delays $\Delta t$ require explicit complex carrier phase rotation:
   $$I_\Delta = I' \cos(2\pi f_c \Delta t) - Q' \sin(2\pi f_c \Delta t)$$
   $$Q_\Delta = Q' \cos(2\pi f_c \Delta t) + I' \sin(2\pi f_c \Delta t)$$
4. **Envelope Detection**: Zea takes the direct complex magnitude $\sqrt{I_{\text{sum}}^2 + Q_{\text{sum}}^2}$, whereas Real-RF DAS takes a 1D axial Hilbert transform along the depth column.

Our Time-Domain RF DAS correctly resolves the clinical target (hypoechoic cyst at $x = -8.4$ mm, $z = 40.0$ mm, $-60$ dB dynamic range) while processing raw RF at **42.7 ms per frame**.

---

## 7. Phase 2 Roadmap: DMAS & Deep Kernel Optimization

### 7.1 Delay Multiply and Sum (DMAS) Formulation
In Delay Multiply and Sum (*Matrone et al., IEEE TUFFC 2015*), received signals are coupled pairwise to enhance coherence:
$$s_{\text{DMAS}}(p) = \sum_{i=1}^{M-1} \sum_{j=i+1}^M \text{sign}(s_i(p) s_j(p)) \sqrt{|s_i(p) s_j(p)|}$$

- **Naive Pairwise DMAS**: Requires $O(M^2)$ operations per pixel. With $M = 21 \times 128 = 2,688$, $M^2 \approx 7.2 \times 10^6$ ops/pixel ($2.2 \times 10^{12}$ total operations).
- **Fast Algebraic Identity (Lim et al.)**:
  $$s_{\text{DMAS}}(p) = \frac{1}{2} \left[ \left( \sum_{i=1}^M \text{sign}(s_i(p)) \sqrt{|s_i(p)|} \right)^2 - \sum_{i=1}^M |s_i(p)| \right]$$
  Reduces complexity to **$O(M)$**—a **2,688× reduction** in floating-point operations.

### 7.2 CUDA Kernel Optimization Targets
1. **2D Thread-Block Tiling**: Switch from 1D thread blocks (`blockDim.x = 256`) to 2D tiles (`16 × 16` threads) matching 2D spatial locality in ultrasound RF data.
2. **Constant Memory Optimization (`__constant__`)**: Place transducer geometry (`128 × 3`), transmit delays (`21 × 128`), and initial times (`21`) into the 64 KB GPU constant memory cache for zero-latency broadcast to all warps.
3. **Texture / Surface Memory**: Leverage hardware texture units with native linear interpolation (`tex1Dfetch`) for RF sample lookups.
