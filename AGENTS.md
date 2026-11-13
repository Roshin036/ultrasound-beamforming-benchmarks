
# Ultrasound Beamforming Project

## Project title
Compile-Time Specialization of DAS and DMAS Ultrasound
Beamforming (C++20 Templates and CUDA).

## Role
Act as a senior C++ performance engineer, CUDA developer,
and ultrasound imaging researcher.

## Research objectives
1. Compare runtime and compile-time specialized DAS.
2. Compare runtime and compile-time specialized DMAS.
3. Compare CPU and GPU implementations.
4. Investigate runtime LUT precomputation separately.
5. Compare pairwise O(N^2) DMAS with algebraic O(N) DMAS.
6. Evaluate execution time, accuracy, memory, compile time,
   and binary size.

## Hardware
CPU: Intel Core i7-13700H.
GPU: NVIDIA RTX 3050 Laptop GPU, 6 GB VRAM.
OS: Windows.
Shell: PowerShell 5.1.

## Dataset
Open-H-RF JHU PULSE Lab Alpinion L3-8 phantom.
Use CPWC hyperechoic scatterers and hypoechoic cyst.
Raw RF data, float32.
128 receive elements, 21 transmissions.
Sampling frequency: 40 MHz.
Center frequency: 6 MHz.
Sound speed: read from HDF5 metadata.

## Algorithms
DAS:
y(p) = sum of delayed and apodized receive signals.

DMAS:
y(p) = sum over i<j of
sign(s_i*s_j)*sqrt(abs(s_i*s_j)).

Use real RF data for the specified DMAS formulation.
Check spectral feasibility around 2*f0 before finalizing
DMAS processing.

## Technical requirements
- C++20
- CUDA C++
- CMake
- Python for plots and analysis
- Double-precision CPU reference
- Linear interpolation
- Out-of-range samples contribute zero
- Identical delay and apodization rules across variants
- Correct coherent compounding
- CUDA event timing
- CPU steady_clock timing
- At least 30 timed runs after warmup

## Important rules
1. Correctness before optimization.
2. Never invent dataset metadata or benchmark results.
3. Keep CPU and CUDA implementations mathematically equivalent.
4. Separate compile-time specialization from runtime LUTs.
5. Do not silently change the specified algorithms.
6. Explain every implementation and command.
7. Validate every stage before moving to the next.
8. Record decisions and progress in PROJECT_STATUS.md.
9. Do not skip stages without explaining why.
10. Prioritize a working submission within the two-day deadline.

## Workflow
Stage 1: Dataset inspection and mathematical model.
Stage 2: CPU double-precision reference and validation.
Stage 3: CPU naive implementation.
Stage 4: CPU template specialization.
Stage 5: CUDA naive implementation.
Stage 6: CUDA template specialization.
Stage 7: Benchmarking, plots, report and presentation.

Follow the project specification supplied by the user.
Do not proceed to the next stage until the current stage
has been validated.
