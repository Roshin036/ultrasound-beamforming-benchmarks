#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr std::size_t K_TX = 21;
constexpr std::size_t K_RX = 128;

// Constant memory for zero-latency broadcast across warps
__constant__ float c_geometry[K_RX * 3];
__constant__ float c_initial_times[K_TX];
__constant__ float c_t0_delays[K_TX * K_RX];
__constant__ float c_tx_apod[K_TX * K_RX];

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

  if (d.n_tx != K_TX || d.n_rx != K_RX)
    throw std::runtime_error("Dimensions do not match optimized specialization (K_TX=21, K_RX=128)");

  d.geometry.resize(d.n_rx * 3);
  d.initial_times.resize(d.n_tx);
  d.t0_delays.resize(d.n_tx * d.n_rx);
  d.tx_apod.resize(d.n_tx * d.n_rx);
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

__device__ inline float interp_zero(const float* __restrict__ raw, std::uint32_t n_samples,
                                    std::uint32_t tx, std::uint32_t rx, float index) {
  if (!isfinite(index) || index < 0.0f || index > static_cast<float>(n_samples - 1)) return 0.0f;
  const auto i0 = static_cast<std::uint32_t>(floorf(index));
  const auto pos = (static_cast<std::size_t>(tx) * n_samples + i0) * K_RX + rx;
  const float a = raw[pos];
  if (i0 + 1 >= n_samples) return a;
  const float f = index - static_cast<float>(i0);
  return a + f * (raw[pos + K_RX] - a);
}

// 2D Block-Tiled DMAS Kernel with Constant Memory Broadcast
__global__ void __launch_bounds__(256, 4)
dmas_2d_tiled_kernel(const float* __restrict__ raw, float* __restrict__ output,
                     std::uint32_t n_samples, float fs, float inv_c,
                     float t_peak, float max_alpha,
                     std::uint32_t nx, std::uint32_t nz,
                     float xmin_m, float dx_m,
                     float zmin_m, float dz_m) {
  const std::uint32_t ix = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t iz = blockIdx.y * blockDim.y + threadIdx.y;
  if (ix >= nx || iz >= nz) return;

  const float x = xmin_m + static_cast<float>(ix) * dx_m;
  const float z = zmin_m + static_cast<float>(iz) * dz_m;
  float total_dmas = 0.0f;

  #pragma unroll 1
  for (std::uint32_t tx = 0; tx < K_TX; ++tx) {
    float tx_arrival = 1e30f;
    const std::uint32_t tx_offset = tx * K_RX;

    #pragma unroll 8
    for (std::uint32_t el = 0; el < K_RX; ++el) {
      if (c_tx_apod[tx_offset + el] == 0.0f) continue;
      const std::uint32_t g = el * 3;
      const float ex = x - c_geometry[g];
      const float ey = -c_geometry[g + 1];
      const float ez = z - c_geometry[g + 2];
      const float candidate = sqrtf(ex * ex + ey * ey + ez * ez) * inv_c + c_t0_delays[tx_offset + el];
      tx_arrival = fminf(tx_arrival, candidate);
    }
    if (tx_arrival >= 1e29f) continue;
    const float tx_time = tx_arrival - c_initial_times[tx] + t_peak;

    float s1 = 0.0f;
    float s2 = 0.0f;

    #pragma unroll 8
    for (std::uint32_t rx = 0; rx < K_RX; ++rx) {
      const std::uint32_t g = rx * 3;
      const float ex = x - c_geometry[g];
      const float ey = -c_geometry[g + 1];
      const float ez = z - c_geometry[g + 2];
      const float dist = sqrtf(ex * ex + ey * ey + ez * ez);
      const float norm = dist + 1e-6f;
      const float alpha = acosf(fminf(fmaxf(ez / norm, -1.0f), 1.0f));
      if (alpha > max_alpha) continue;

      const float rx_time = dist * inv_c;
      const float val = interp_zero(raw, n_samples, tx, rx, (tx_time + rx_time) * fs);
      const float abs_val = fabsf(val);
      const float s_hat = copysignf(sqrtf(abs_val), val);
      s1 += s_hat;
      s2 += abs_val;
    }
    total_dmas += 0.5f * (s1 * s1 - s2);
  }

  output[iz * nx + ix] = total_dmas;
}

void execute_optimized(const Data& d, const std::string& output, double xmin, double xmax,
                       std::uint32_t nx, double zmin, double zmax, std::uint32_t nz,
                       bool benchmark_mode) {
  const std::size_t pixel_count = static_cast<std::size_t>(nx) * nz;

  // Upload to Constant Memory
  std::vector<float> geom_f(d.geometry.size());
  for (std::size_t i = 0; i < d.geometry.size(); ++i) geom_f[i] = static_cast<float>(d.geometry[i]);
  std::vector<float> init_f(d.initial_times.size());
  for (std::size_t i = 0; i < d.initial_times.size(); ++i) init_f[i] = static_cast<float>(d.initial_times[i]);
  std::vector<float> t0_f(d.t0_delays.size());
  for (std::size_t i = 0; i < d.t0_delays.size(); ++i) t0_f[i] = static_cast<float>(d.t0_delays[i]);
  std::vector<float> apod_f(d.tx_apod.size());
  for (std::size_t i = 0; i < d.tx_apod.size(); ++i) apod_f[i] = static_cast<float>(d.tx_apod[i]);

  cuda_check(cudaMemcpyToSymbol(c_geometry, geom_f.data(), geom_f.size() * sizeof(float)), "copy c_geometry");
  cuda_check(cudaMemcpyToSymbol(c_initial_times, init_f.data(), init_f.size() * sizeof(float)), "copy c_initial_times");
  cuda_check(cudaMemcpyToSymbol(c_t0_delays, t0_f.data(), t0_f.size() * sizeof(float)), "copy c_t0_delays");
  cuda_check(cudaMemcpyToSymbol(c_tx_apod, apod_f.data(), apod_f.size() * sizeof(float)), "copy c_tx_apod");

  DeviceBuffer<float> raw_d(d.raw.size());
  DeviceBuffer<float> out_d(pixel_count);
  raw_d.upload(d.raw);

  const float inv_c = 1.0f / static_cast<float>(d.c);
  const float max_alpha = atanf(1.0f / (2.0f * static_cast<float>(d.f_num) + 1e-7f));
  const float xmin_m = static_cast<float>(xmin * 1e-3);
  const float dx_m = (nx > 1) ? static_cast<float>((xmax - xmin) / double(nx - 1) * 1e-3) : 0.0f;
  const float zmin_m = static_cast<float>(zmin * 1e-3);
  const float dz_m = (nz > 1) ? static_cast<float>((zmax - zmin) / double(nz - 1) * 1e-3) : 0.0f;

  // 2D Thread Block Configuration (16 x 16 = 256 threads)
  const dim3 block_dim(16, 16);
  const dim3 grid_dim((nx + block_dim.x - 1) / block_dim.x, (nz + block_dim.y - 1) / block_dim.y);

  const auto launch_kernel = [&]() {
    dmas_2d_tiled_kernel<<<grid_dim, block_dim>>>(
        raw_d.get(), out_d.get(),
        d.n_samples, static_cast<float>(d.fs), inv_c,
        static_cast<float>(d.t_peak), max_alpha,
        nx, nz, xmin_m, dx_m, zmin_m, dz_m);
    cuda_check(cudaGetLastError(), "dmas_2d_tiled_kernel execution");
  };

  if (!benchmark_mode) {
    launch_kernel();
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize run");
    std::vector<float> pixels(pixel_count);
    out_d.download(pixels);

    std::ofstream out(output);
    if (!out) throw std::runtime_error("Cannot open output CSV: " + output);
    out << "x_mm,z_mm,rf_sum\n" << std::setprecision(9);
    for (std::uint32_t iz = 0; iz < nz; ++iz) {
      const double z = (nz > 1) ? (zmin + (zmax - zmin) * iz / double(nz - 1)) : zmin;
      for (std::uint32_t ix = 0; ix < nx; ++ix) {
        const double x = (nx > 1) ? (xmin + (xmax - xmin) * ix / double(nx - 1)) : xmin;
        out << x << ',' << z << ',' << pixels[static_cast<std::size_t>(iz) * nx + ix] << '\n';
      }
    }
    std::cout << "Wrote 2D-tiled Constant-Memory CUDA DMAS: " << nx << " x " << nz
              << " pixels to " << output << '\n';
    return;
  }

  constexpr int warmups = 2, runs = 30;
  for (int i = 0; i < warmups; ++i) {
    launch_kernel();
    cuda_check(cudaDeviceSynchronize(), "warmup sync");
  }

  cudaEvent_t start{}, stop{};
  cuda_check(cudaEventCreate(&start), "cudaEventCreate start");
  cuda_check(cudaEventCreate(&stop), "cudaEventCreate stop");

  std::vector<double> ms;
  ms.reserve(runs);
  for (int i = 0; i < runs; ++i) {
    cuda_check(cudaEventRecord(start), "cudaEventRecord start");
    launch_kernel();
    cuda_check(cudaEventRecord(stop), "cudaEventRecord stop");
    cuda_check(cudaEventSynchronize(stop), "cudaEventSynchronize stop");
    float elapsed_ms{};
    cuda_check(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime");
    ms.push_back(elapsed_ms);
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  std::sort(ms.begin(), ms.end());
  const double mean = std::accumulate(ms.begin(), ms.end(), 0.0) / runs;
  cudaDeviceProp prop{};
  cuda_check(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");

  std::cout << "BENCHMARK variant=dmas_cuda_optimized_2d_tiled"
            << " timer=cuda_event warmups=" << warmups << " runs=" << runs
            << " pixels=" << pixel_count << " mean_ms=" << mean
            << " median_ms=" << ms[runs / 2] << " p95_ms=" << ms[static_cast<std::size_t>(0.95 * (runs - 1))]
            << " min_ms=" << ms.front() << " max_ms=" << ms.back()
            << " device_buffer_bytes=" << (d.raw.size() + pixel_count) * sizeof(float)
            << " device_total_bytes=" << prop.totalGlobalMem << '\n';
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 9) {
      std::cerr << "Usage: dmas_cuda_optimized <input.bundle> <output.csv> <xmin_mm> <xmax_mm> <nx|dx> <zmin_mm> <zmax_mm> <nz|dz> [--benchmark]\n";
      return 2;
    }
    const auto v = [](const char* s) { return std::stod(s); };
    const Data d = load(argv[1]);
    const double xmin = v(argv[3]), xmax = v(argv[4]);
    const double zmin = v(argv[6]), zmax = v(argv[7]);
    const std::uint32_t nx = resolve_count(xmin, xmax, v(argv[5]));
    const std::uint32_t nz = resolve_count(zmin, zmax, v(argv[8]));

    bool benchmark_mode = (argc > 9 && std::string(argv[9]) == "--benchmark");
    execute_optimized(d, argv[2], xmin, xmax, nx, zmin, zmax, nz, benchmark_mode);
  } catch (const std::exception& e) {
    std::cerr << "CUDA DMAS optimized error: " << e.what() << '\n';
    return 1;
  }
  return 0;
}
