# Project status

## Current stage

Stage 7, DAS benchmarking: initial performance, build-time, binary-size, and CUDA-memory measurements are recorded for all four DAS variants on both CPWC datasets. Stages 1 through 6 are implemented and validated; final reporting, physical acquisition cross-checks, and DMAS research objectives remain.

## DAS model used

The run uses the file `datasets/JHU_Alpinion_L3-8_CPWC_hyperechoic_scatterers.hdf5` (one frame, 21 transmissions, 4352 axial samples, 128 receive channels, real float32 RF). For pixel `p` and transmit `n`, the transmit arrival time is

```text
tx_time(n,p) = min over active transmit elements e of
               (distance(p, element[e]) / sound_speed + t0_delays[n,e])
               - initial_times[n]
```

This is the first-arrival convention for plane-wave entries (`focus_distances == 0`) described by the zea beamforming documentation. Receive time is `distance(p, receive_element[e]) / sound_speed`. The sample index is `(tx_time + receive_time) * sampling_frequency`. Samples are linearly interpolated in double precision; indices outside `[0, n_samples - 1]` contribute zero. The real RF contributions are summed over all transmit/receive pairs without normalization, giving coherent compounding.

The CPWC file has unit transmit apodization for all entries and no receive-apodization field, so this reference uses unit receive weights. The HDF5 file contains no waveform peak-time field; the reference explicitly uses `t_peak = 0`. The image grid is supplied at runtime rather than treated as dataset metadata. The demonstration invocation uses `x = -19.05..19.05 mm` in `0.3 mm` steps and `z = 3..60 mm` in `0.3 mm` steps, with `y = 0`.

The delay-field descriptions and reference first-arrival behavior are documented in the [zea 0.1.5 beamformer API](https://zea.readthedocs.io/en/stable/_autosummary/zea.beamform.beamformer.html). The dataset itself reports `zea_version = 0.1.6`; this code uses the documented `initial_times`, `t0_delays`, and zero-focus plane-wave semantics, and does not call the zea implementation.

## Implementation files

- `das_reference.cpp`: C++20 CPU reference; keeps calculations and accumulation in `double`, with stored RF samples promoted from `float32`.
- `export_das_input.py`: read-only HDF5 export to the local `DASREF01` binary input format. Requires `h5py` and `hdf5plugin` for Blosc-compressed arrays.
- `plot_das.py`: analytic-envelope conversion and log-compressed display of the RF sum.
- `CMakeLists.txt`: C++20 build target.
- `das_cuda_naive.cu`: runtime-sized CUDA DAS, one GPU thread per image pixel, reading the same `DASREF01` bundle format as the CPU variants.
- `das_cuda_template.cu`: CUDA DAS specialized at compile time for 21 transmit events and 128 receive channels; it rejects other channel/transmit dimensions and uses the same `DASREF01` input.
- `plot_das_comparison.py`: CPU/CUDA envelope and RF-difference comparison plot.
- `plot_das_benchmark.py`: plot benchmark medians and run-to-run ranges from the benchmark summary CSV.
- `measure_build_times.py`: repeatable clean per-target CMake/NMake Release build-time measurement; restores all four DAS executables after measurement.

Generated demonstration artifacts are in `build/`: exported bundles, CPU/CUDA RF CSVs, envelope/comparison plots, benchmark result CSVs, and clean-build timing CSV. CPU and CUDA DAS benchmark modes accept `--benchmark` and run two warmups followed by 30 timed repetitions. The HDF5 source files were not changed.

## Validation performed

- Compiled with g++ 13.1.0 using `-std=c++20 -O2 -Wall -Wextra -Wpedantic`.
- `das_reference --self-check` passed integer/fractional interpolation, exact last-sample handling, and zero output outside the sample range.
- Processed the CPWC hyperechoic file on the demonstration grid (128 x 191 pixels); the RF output is finite and nonzero, and the envelope PNG was generated.
- Independently recomputed four selected pixels in Python from the HDF5 data and obtained agreement with the C++ output to floating-point roundoff (maximum absolute error observed: `4.46e-11`).
- `pythonfiles/validate_delays.py` checked seven of the 21 CPWC steering events against `(x * sin(angle) - min(x * sin(angle))) / c`; maximum reported error was `4.547e-13 s`.
- The runtime CPU DAS baseline (`das_cpu_naive.cpp`, float32 arithmetic) was compared pixel-by-pixel with the double reference on both CPWC datasets (24,448 pixels each). Hyperechoic: normalized RMSE `3.61e-5`, correlation `0.99999999935`; hypoechoic: normalized RMSE `4.44e-5`, correlation `0.99999999901`.

## Remaining questions before treating the model as final

- The dataset has no waveform samples or waveform peak-time metadata. `t_peak = 0` is a stated reference convention, not a measured dataset value; a later comparison with acquisition documentation or a trusted reconstruction could establish whether an offset is needed.
- The dataset has no prescribed image grid. The bounds and spacing above are demonstration settings, not recovered acquisition settings.
- No receive apodization is stored. The reference therefore uses uniform receive weights; any future receive window would be an explicit experimental variant.
- The MATLAB/zea version compatibility of the first-arrival model has not yet been checked against a separately generated trusted image.

## Implemented variants

- Stage 2: `das_reference.cpp`, double-precision CPU reference.
- Stage 3: `das_cpu_naive.cpp`, runtime-sized scalar CPU baseline using float32 arithmetic.
- Stage 4: `das_cpu_template.cpp`, compile-time specialization for 21 transmits and 128 receive channels, using the same float32 arithmetic as Stage 3. It has no runtime LUT.

The Stage 4 output matched Stage 3 exactly for every pixel on both CPWC datasets (24,448 pixels per dataset).

- Stage 5: compiled `das_cuda_naive.cu` with CUDA 13.5.37 for `sm_86` and ran the 128 x 191 demonstration grid on both CPWC datasets. All 24,448 output samples per dataset were finite and nonzero. Comparing output values in corresponding row order:
  - Hyperechoic: vs. CPU naive, NRMSE `2.23e-5`, correlation `0.999999999752`; vs. double reference, NRMSE `3.63e-5`, correlation `0.999999999347`.
  - Hypoechoic: vs. CPU template, NRMSE `2.88e-5`, correlation `0.999999999586`; vs. double reference, NRMSE `4.25e-5`, correlation `0.999999999099`.
  Small CPU/CUDA differences are expected from floating-point operation order and CUDA math implementation; these comparisons establish numerical agreement, not bitwise identity. The CSV coordinate fields use float coordinates in the runtime implementations and double coordinates in the reference; RF values were compared by corresponding grid row order.

- Stage 6: built `das_cuda_template.cu` as the CMake target `das_cuda_template` using NMake and processed both 128 x 191 CPWC demonstration grids. All 24,448 outputs per dataset were finite. Template CUDA output was bitwise identical to CUDA naive at every pixel on both datasets (max absolute difference 0; NRMSE 0; correlation 1). It therefore has the same comparisons to CPU and double reference as Stage 5. This validates correctness for these datasets, not speedup.

- Stage 7 benchmarks: Release executables processed the 128 x 191 (24,448 pixel) grid on each dataset, with 2 warmups and 30 timed runs per variant. CPU `steady_clock` covers only the compute loop; CUDA events cover only the kernel, excluding allocation and transfers. Hyperechoic median/mean/p95 (ms): CPU naive 2572.62/2053.47/3459.83; CPU template 723.136/723.608/758.058; CUDA naive 18.7658/18.7569/19.3812; CUDA template 19.0638/19.1156/19.7734. Hypoechoic: CPU naive 2893.47/2984.56/3757.42; CPU template 745.472/1183.25/2908.93; CUDA naive 18.8641/18.9546/19.5471; CUDA template 19.0566/19.1217/19.7448. CPU outliers are large; treat median ratios as preliminary. Median CPU-naive/CUDA-naive ratios were 137.1x and 153.4x; CPU-template/CUDA-template ratios were 37.9x and 39.1x. CUDA template specialization did not show a speedup over CUDA naive in these measurements. Summary CSVs and PNG plots are in `build/`.

- Benchmark outputs were finite and agreed with established CPU results. Hyperechoic CUDA vs CPU NRMSE `2.23e-5`, correlation `0.999999999752`; hypoechoic NRMSE `2.88e-5`, correlation `0.999999999586`. CUDA naive and template outputs matched each other for both datasets.

- Release binary sizes: CPU naive 50,176 bytes; CPU template 50,176 bytes; CUDA naive 90,112 bytes; CUDA template 83,456 bytes. CUDA buffer payload is 46,913,620 bytes for this grid. `cudaMemGetInfo` showed a 50,331,648-byte free-memory decrease after allocation (total device memory reported 6,441,926,656 bytes); the difference from payload reflects allocation/runtime overhead. CUDA context memory is not included in the free-memory delta measurement.

## Current environment and next stages

- Runtime LUT precomputation remains a separate variant and is not included in the CPU template specialization.
- The RTX 3050 is visible to `nvidia-smi` (driver 610.91; CUDA UMD 13.3). CUDA Toolkit 13.5 is now present at `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.5`; `nvcc --version` reports 13.5.37.
- A small CUDA device sanity program (`cuda_device_check.cu`) compiled for `sm_86` with the available MSVC 19.44 toolchain and ran successfully on the RTX 3050. It allocated device memory, incremented a value on the GPU, copied it back, and verified the result was 42. This verifies basic CUDA compile/link/device execution only; it does not validate ultrasound DAS.
- Visual Studio Build Tools is still under the nonstandard `C:\Program` path and `VsDevCmd.bat` prints a missing `vswhere.exe` warning, but it exposes `cl.exe` and the CUDA sanity build completed. The desired MSVC 14.38 component was not found. Reassess whether a repair or supported toolset is needed if later CUDA compilation/builds expose a compatibility issue.
- The initial Toolkit installer attempt logged `0xE4000008` while another installer instance held the global install event. Since then, the Toolkit directory has appeared and passed the compile/run check; the precise cause of the installation completing after that failed attempt is unknown.
- Stage 5 uses one thread per pixel and runtime loops over transmit events and receive channels. Stage 6 uses the same computation with `N_TX=21` and `N_RX=128` as compile-time template parameters. Stage 7 records compute-only timings and CUDA device allocation measurements; timings do not include end-to-end I/O or transfers.
- Clean Release build wall times (include VsDevCmd, CMake/NMake, compilation, and link): CPU naive 9.431 s, CPU template 7.304 s, CUDA naive 14.491 s, CUDA template 14.402 s. These are single-target clean-build wall times, not compiler-only timings; CSV is `build/das_compile_times.csv`.

- The CUDA target is added conditionally in CMake. CMake/Ninja stalled during toolchain checks/build in this environment. CMake/NMake configured with explicit CUDA and host compiler paths, built and linked `das_cuda_naive`, and the CMake-built executable ran successfully. Its 24,448 hyperechoic RF values were bitwise identical to the direct-`nvcc` executable. The successful CMake invocation skipped CMake's automatic CXX/CUDA compiler ABI and “works” probes because those stalled; the actual CUDA target compile, link, and execution succeeded. Use NMake for this toolchain unless the Visual Studio installation is repaired.
- Remaining: repeat CPU timing under controlled load because outliers remain high; complete final report and presentation; address the DMAS variants and 2*f0 spectral feasibility check. Physical transmit timing offset, receive apodization, and image-grid questions remain separate validation items.
