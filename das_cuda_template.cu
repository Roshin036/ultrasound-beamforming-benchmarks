#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr std::array<char, 8> kMagic1{'D', 'A', 'S', 'R', 'E', 'F', '0', '1'};
constexpr std::array<char, 8> kMagic2{'D', 'A', 'S', 'R', 'E', 'F', '0', '2'};

struct Data {
  std::uint32_t n_tx{}, n_samples{}, n_rx{};
  double fs{}, c{}, fc{6.0e6}, t_peak{1.6666667e-7}, f_num{1.75};
  std::vector<double> geometry, initial_times, t0_delays, tx_apod;
  std::vector<float> raw;
};

template <class T>
void read_exact(std::ifstream& in, T* p, std::size_t n, const char* label) {
  in.read(reinterpret_cast<char*>(p), static_cast<std::streamsize>(n * sizeof(T)));
  if (!in) throw std::runtime_error(std::string("Truncated input reading ") + label);
}

Data load(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("Cannot open input bundle: " + path);
  std::array<char, 8> magic{};
  read_exact(in, magic.data(), magic.size(), "magic");
  bool is_v1 = (magic == kMagic1);
  bool is_v2 = (magic == kMagic2);
  if (!is_v1 && !is_v2) throw std::runtime_error("Not a valid DASREF bundle");

  Data d;
  read_exact(in, &d.n_tx, 1, "header");
  read_exact(in, &d.n_samples, 1, "header");
  read_exact(in, &d.n_rx, 1, "header");
  read_exact(in, &d.fs, 1, "header");
  read_exact(in, &d.c, 1, "header");

  if (is_v2) {
    read_exact(in, &d.fc, 1, "fc");
    read_exact(in, &d.t_peak, 1, "t_peak");
    read_exact(in, &d.f_num, 1, "f_num");
  } else {
    d.fc = 6.0e6;
    d.t_peak = 1.0 / d.fc;
    d.f_num = 1.75;
  }

  if (!d.n_tx || !d.n_samples || !d.n_rx || !(d.fs > 0.0) || !(d.c > 0.0))
    throw std::runtime_error("Invalid dimensions or acquisition constants");

  d.geometry.resize(static_cast<std::size_t>(d.n_rx) * 3);
  d.initial_times.resize(d.n_tx);
  d.t0_delays.resize(static_cast<std::size_t>(d.n_tx) * d.n_rx);
  d.tx_apod.resize(static_cast<std::size_t>(d.n_tx) * d.n_rx);
  read_exact(in, d.geometry.data(), d.geometry.size(), "geometry");
  read_exact(in, d.initial_times.data(), d.initial_times.size(), "initial times");
  read_exact(in, d.t0_delays.data(), d.t0_delays.size(), "transmit delays");
  read_exact(in, d.tx_apod.data(), d.tx_apod.size(), "transmit apodizations");

  d.raw.resize(static_cast<std::size_t>(d.n_tx) * d.n_samples * d.n_rx);
  read_exact(in, d.raw.data(), d.raw.size(), "raw RF samples");
  if (in.peek() != std::char_traits<char>::eof())
    throw std::runtime_error("Unexpected trailing input bytes");
  return d;
}

std::uint32_t resolve_count(double lo, double hi, double param) {
  if (param >= 1.0 && std::abs(param - std::round(param)) < 1e-6) {
    return static_cast<std::uint32_t>(std::round(param));
  }
  if (param <= 0.0 || hi < lo) throw std::runtime_error("Invalid grid step or span");
  const double n = (hi - lo) / param;
  return static_cast<std::uint32_t>(std::llround(n) + 1);
}

void cuda_check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

template <class T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t count) {
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&ptr_), count * sizeof(T)), "cudaMalloc");
  }
  ~DeviceBuffer() { if (ptr_) cudaFree(ptr_); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  T* get() const { return ptr_; }
  void upload(const std::vector<T>& src) {
    cuda_check(cudaMemcpy(ptr_, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy HtoD");
  }
  void download(std::vector<T>& dst) const {
    cuda_check(cudaMemcpy(dst.data(), ptr_, dst.size() * sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy DtoH");
  }

 private:
  T* ptr_{};
};

// Constant memory for ultra-fast broadcast to all warps
__constant__ float c_geom_x32[128];
__constant__ float c_initial_times32[21];
__constant__ float c_t0_delays32[21 * 128];

__constant__ double c_geom_x64[128];
__constant__ double c_initial_times64[21];
__constant__ double c_t0_delays64[21 * 128];

template <typename Real>
struct ConstMemoryTraits;

template <>
struct ConstMemoryTraits<float> {
  __device__ static inline float geom_x(std::size_t i) { return c_geom_x32[i]; }
  __device__ static inline float initial_time(std::size_t i) { return c_initial_times32[i]; }
  __device__ static inline float t0_delay(std::size_t i) { return c_t0_delays32[i]; }
};

template <>
struct ConstMemoryTraits<double> {
  __device__ static inline double geom_x(std::size_t i) { return c_geom_x64[i]; }
  __device__ static inline double initial_time(std::size_t i) { return c_initial_times64[i]; }
  __device__ static inline double t0_delay(std::size_t i) { return c_t0_delays64[i]; }
};

template <typename Real>
__device__ inline Real interp_zero_fast(const float* __restrict__ raw, std::uint32_t n_samples,
                                        std::size_t tx_stride, std::size_t rx, Real index) {
  if (!isfinite(index) || index < Real(0.0) || index > static_cast<Real>(n_samples - 1)) return Real(0.0);
  const auto i0 = static_cast<std::uint32_t>(floor(index));
  const auto pos = ((tx_stride + i0) << 7) + rx;
  const Real a = static_cast<Real>(raw[pos]);
  if (i0 + 1 >= n_samples) return a;
  const Real f = index - static_cast<Real>(i0);
  return a + f * (static_cast<Real>(raw[pos + 128]) - a);
}

template <typename Real, std::size_t N_TX, std::size_t N_RX>
__global__ void das_fast_template_kernel(const float* __restrict__ raw, Real* __restrict__ output,
                                         std::uint32_t n_samples, Real fs, Real inv_c,
                                         Real t_peak, Real kappa, Real geom_x0, Real inv_pitch,
                                         std::uint32_t nx, std::uint32_t nz,
                                         double xmin_mm, double xmax_mm,
                                         double zmin_mm, double zmax_mm) {
  const std::uint32_t pixel = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t pixel_count = nx * nz;
  if (pixel >= pixel_count) return;

  const std::uint32_t iz = pixel / nx;
  const std::uint32_t ix = pixel - iz * nx;
  const Real x = (nx > 1) ? static_cast<Real>((xmin_mm + (xmax_mm - xmin_mm) * ix / double(nx - 1)) * 1e-3)
                          : static_cast<Real>(xmin_mm * 1e-3);
  const Real z = (nz > 1) ? static_cast<Real>((zmin_mm + (zmax_mm - zmin_mm) * iz / double(nz - 1)) * 1e-3)
                          : static_cast<Real>(zmin_mm * 1e-3);
  const Real z2 = z * z;

  // 1. Transmit arrival times: 21 values computed once per pixel
  Real tx_times[N_TX];
  #pragma unroll
  for (std::size_t tx = 0; tx < N_TX; ++tx) {
    Real min_arr = Real(1e30);
    const std::size_t tx_offset = tx * N_RX;
    #pragma unroll 16
    for (std::size_t el = 0; el < N_RX; ++el) {
      const Real ex = x - ConstMemoryTraits<Real>::geom_x(el);
      const Real dist = sqrt(ex * ex + z2);
      const Real cand = dist * inv_c + ConstMemoryTraits<Real>::t0_delay(tx_offset + el);
      min_arr = (cand < min_arr) ? cand : min_arr;
    }
    tx_times[tx] = min_arr - ConstMemoryTraits<Real>::initial_time(tx) + t_peak;
  }

  // 2. Analytical Dynamic Aperture bounds
  const Real half_aperture = z * kappa;
  const int rx_min = max(0, min(static_cast<int>(N_RX - 1), static_cast<int>(ceil((x - half_aperture - geom_x0) * inv_pitch))));
  const int rx_max = max(0, min(static_cast<int>(N_RX - 1), static_cast<int>(floor((x + half_aperture - geom_x0) * inv_pitch))));

  // 3. Receive Loop: compute element distance ONCE per receiver, then accumulate across TX
  Real sum = Real(0.0);
  for (int rx = rx_min; rx <= rx_max; ++rx) {
    const Real ex = x - ConstMemoryTraits<Real>::geom_x(rx);
    const Real dist = sqrt(ex * ex + z2);
    const Real rx_time = dist * inv_c;

    #pragma unroll
    for (std::size_t tx = 0; tx < N_TX; ++tx) {
      const auto tx_stride = tx * n_samples;
      sum += interp_zero_fast<Real>(raw, n_samples, tx_stride, rx, (tx_times[tx] + rx_time) * fs);
    }
  }

  output[pixel] = sum;
}

template <typename Real, std::size_t N_TX, std::size_t N_RX>
void run_typed(const Data& d, const std::string& output_path,
               double xmin, double xmax, std::uint32_t nx,
               double zmin, double zmax, std::uint32_t nz,
               bool benchmark_mode) {
  if (d.n_tx != N_TX || d.n_rx != N_RX)
    throw std::runtime_error("Input dimensions do not match template specialization");

  const std::size_t pixel_count = static_cast<std::size_t>(nx) * nz;

  std::size_t free_before{}, total_device_bytes{};
  cuda_check(cudaMemGetInfo(&free_before, &total_device_bytes), "cudaMemGetInfo before");

  // Constant memory upload
  if constexpr (std::is_same_v<Real, float>) {
    std::vector<float> h_geom(N_RX), h_init(N_TX), h_delays(N_TX * N_RX);
    for (std::size_t i = 0; i < N_RX; ++i) h_geom[i] = static_cast<float>(d.geometry[i * 3]);
    for (std::size_t i = 0; i < N_TX; ++i) h_init[i] = static_cast<float>(d.initial_times[i]);
    for (std::size_t i = 0; i < N_TX * N_RX; ++i) h_delays[i] = static_cast<float>(d.t0_delays[i]);
    cuda_check(cudaMemcpyToSymbol(c_geom_x32, h_geom.data(), N_RX * sizeof(float)), "c_geom_x32");
    cuda_check(cudaMemcpyToSymbol(c_initial_times32, h_init.data(), N_TX * sizeof(float)), "c_initial_times32");
    cuda_check(cudaMemcpyToSymbol(c_t0_delays32, h_delays.data(), N_TX * N_RX * sizeof(float)), "c_t0_delays32");
  } else {
    std::vector<double> h_geom(N_RX), h_init(N_TX), h_delays(N_TX * N_RX);
    for (std::size_t i = 0; i < N_RX; ++i) h_geom[i] = d.geometry[i * 3];
    for (std::size_t i = 0; i < N_TX; ++i) h_init[i] = d.initial_times[i];
    for (std::size_t i = 0; i < N_TX * N_RX; ++i) h_delays[i] = d.t0_delays[i];
    cuda_check(cudaMemcpyToSymbol(c_geom_x64, h_geom.data(), N_RX * sizeof(double)), "c_geom_x64");
    cuda_check(cudaMemcpyToSymbol(c_initial_times64, h_init.data(), N_TX * sizeof(double)), "c_initial_times64");
    cuda_check(cudaMemcpyToSymbol(c_t0_delays64, h_delays.data(), N_TX * N_RX * sizeof(double)), "c_t0_delays64");
  }

  DeviceBuffer<float> d_raw(d.raw.size());
  d_raw.upload(d.raw);
  DeviceBuffer<Real> d_output(pixel_count);

  std::size_t free_after{};
  cuda_check(cudaMemGetInfo(&free_after, &total_device_bytes), "cudaMemGetInfo after");
  const std::size_t device_buffer_bytes = free_before - free_after;

  const Real inv_c = Real(1.0) / static_cast<Real>(d.c);
  const Real t_peak = static_cast<Real>(d.t_peak);
  const Real kappa = Real(1.0) / (Real(2.0) * static_cast<Real>(d.f_num) + Real(1e-7));
  const Real geom_x0 = static_cast<Real>(d.geometry[0]);
  const Real pitch = static_cast<Real>(d.geometry[3] - d.geometry[0]);
  const Real inv_pitch = Real(1.0) / pitch;
  const Real fs = static_cast<Real>(d.fs);

  constexpr int threads = 256;
  const auto blocks = static_cast<unsigned>((pixel_count + threads - 1) / threads);

  const auto launch = [&] {
    das_fast_template_kernel<Real, N_TX, N_RX><<<blocks, threads>>>(
        d_raw.get(), d_output.get(),
        d.n_samples, fs, inv_c, t_peak, kappa, geom_x0, inv_pitch,
        nx, nz, xmin, xmax, zmin, zmax);
    cuda_check(cudaGetLastError(), "das_fast_template_kernel launch");
  };

  if (benchmark_mode) {
    constexpr int warmups = 2, runs = 30;
    for (int i = 0; i < warmups; ++i) launch();
    cuda_check(cudaDeviceSynchronize(), "CUDA warmup sync");

    cudaEvent_t start{}, stop{};
    cuda_check(cudaEventCreate(&start), "cudaEventCreate start");
    cuda_check(cudaEventCreate(&stop), "cudaEventCreate stop");
    std::vector<float> ms;
    ms.reserve(runs);

    for (int i = 0; i < runs; ++i) {
      cuda_check(cudaEventRecord(start), "cudaEventRecord start");
      launch();
      cuda_check(cudaEventRecord(stop), "cudaEventRecord stop");
      cuda_check(cudaEventSynchronize(stop), "cudaEventSynchronize");
      float elapsed{};
      cuda_check(cudaEventElapsedTime(&elapsed, start, stop), "cudaEventElapsedTime");
      ms.push_back(elapsed);
    }
    cuda_check(cudaEventDestroy(start), "cudaEventDestroy start");
    cuda_check(cudaEventDestroy(stop), "cudaEventDestroy stop");
    std::sort(ms.begin(), ms.end());

    double total = 0.0;
    for (const float value : ms) total += value;
    std::string prec_name = (sizeof(Real) == 8) ? "cuda_template_f64" : "cuda_template_f32";
    std::cout << "BENCHMARK variant=" << prec_name << " timer=cuda_event warmups=" << warmups
              << " runs=" << runs << " pixels=" << pixel_count << " mean_ms=" << total / runs
              << " median_ms=" << ms[runs / 2] << " p95_ms=" << ms[static_cast<std::size_t>(0.95 * (runs - 1))]
              << " min_ms=" << ms.front() << " max_ms=" << ms.back()
              << " device_buffer_bytes=" << device_buffer_bytes
              << " device_total_bytes=" << total_device_bytes << '\n';
  } else {
    launch();
  }

  cuda_check(cudaDeviceSynchronize(), "DAS kernel execution");
  std::vector<Real> pixels(pixel_count);
  d_output.download(pixels);

  std::ofstream out(output_path);
  if (!out) throw std::runtime_error("Cannot open output CSV: " + output_path);
  out << "x_mm,z_mm,rf_sum\n" << std::setprecision(9);
  for (std::uint32_t iz = 0; iz < nz; ++iz) {
    const double z = (nz > 1) ? (zmin + (zmax - zmin) * iz / double(nz - 1)) : zmin;
    for (std::uint32_t ix = 0; ix < nx; ++ix) {
      const double x = (nx > 1) ? (xmin + (xmax - xmin) * ix / double(nx - 1)) : xmin;
      const std::size_t index = static_cast<std::size_t>(iz) * nx + ix;
      out << x << ',' << z << ',' << pixels[index] << '\n';
    }
  }
  std::cout << "Wrote template CUDA DAS (" << (sizeof(Real) == 8 ? "FP64" : "FP32")
            << "): " << nx << " x " << nz << " pixels to " << output_path << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 9) {
      std::cerr << "Usage: das_cuda_template <input.bundle> <output.csv> <xmin_mm> <xmax_mm> <nx|dx> <zmin_mm> <zmax_mm> <nz|dz> [--double] [--benchmark]\n";
      return 2;
    }
    const auto v = [](const char* s) { return std::stod(s); };
    const Data d = load(argv[1]);
    const double xmin = v(argv[3]), xmax = v(argv[4]);
    const double zmin = v(argv[6]), zmax = v(argv[7]);
    const std::uint32_t nx = resolve_count(xmin, xmax, v(argv[5]));
    const std::uint32_t nz = resolve_count(zmin, zmax, v(argv[8]));

    bool benchmark_mode = false;
    bool double_precision = false;

    for (int i = 9; i < argc; ++i) {
      std::string arg = argv[i];
      if (arg == "--benchmark") benchmark_mode = true;
      else if (arg == "--double" || arg == "--f64") double_precision = true;
      else if (arg == "--float" || arg == "--f32") double_precision = false;
      else throw std::runtime_error("Unknown option: " + arg);
    }

    if (double_precision) {
      run_typed<double, 21, 128>(d, argv[2], xmin, xmax, nx, zmin, zmax, nz, benchmark_mode);
    } else {
      run_typed<float, 21, 128>(d, argv[2], xmin, xmax, nx, zmin, zmax, nz, benchmark_mode);
    }
  } catch (const std::exception& error) {
    std::cerr << "CUDA template DAS error: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
