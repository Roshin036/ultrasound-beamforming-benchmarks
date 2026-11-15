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

template <typename Real>
__device__ inline Real interp_zero(const float* raw, std::uint32_t n_samples,
                                   std::uint32_t n_rx, std::uint32_t tx,
                                   std::uint32_t rx, Real index) {
  if (!isfinite(index) || index < Real(0.0) || index > static_cast<Real>(n_samples - 1)) return Real(0.0);
  const auto i0 = static_cast<std::uint32_t>(floor(index));
  const auto pos = (static_cast<std::size_t>(tx) * n_samples + i0) * n_rx + rx;
  const Real a = static_cast<Real>(raw[pos]);
  if (i0 + 1 >= n_samples) return a;
  const Real f = index - static_cast<Real>(i0);
  return a + f * (static_cast<Real>(raw[pos + n_rx]) - a);
}

template <typename Real>
__global__ void dmas_kernel(const Real* geometry, const Real* initial_times,
                            const Real* t0_delays, const Real* tx_apod,
                            const float* raw, Real* output,
                            std::uint32_t n_tx, std::uint32_t n_samples,
                            std::uint32_t n_rx, Real fs, Real c,
                            Real t_peak, Real f_num,
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
  const Real inv_c = Real(1.0) / c;
  const Real max_alpha = atan(Real(1.0) / (Real(2.0) * f_num + Real(1e-7)));
  Real total_dmas = Real(0.0);

  for (std::uint32_t tx = 0; tx < n_tx; ++tx) {
    Real tx_arrival = Real(1e30);
    const std::size_t tx_offset = static_cast<std::size_t>(tx) * n_rx;
    for (std::uint32_t el = 0; el < n_rx; ++el) {
      if (tx_apod[tx_offset + el] == Real(0.0)) continue;
      const std::size_t g = static_cast<std::size_t>(el) * 3;
      const Real ex = x - geometry[g];
      const Real ey = -geometry[g + 1];
      const Real ez = z - geometry[g + 2];
      const Real candidate = sqrt(ex * ex + ey * ey + ez * ez) * inv_c + t0_delays[tx_offset + el];
      tx_arrival = (candidate < tx_arrival) ? candidate : tx_arrival;
    }
    if (tx_arrival >= Real(1e29)) continue;
    const Real tx_time = tx_arrival - initial_times[tx] + t_peak;

    Real s1 = Real(0.0);
    Real s2 = Real(0.0);
    for (std::uint32_t rx = 0; rx < n_rx; ++rx) {
      const std::size_t g = static_cast<std::size_t>(rx) * 3;
      const Real ex = x - geometry[g];
      const Real ey = -geometry[g + 1];
      const Real ez = z - geometry[g + 2];
      const Real dist = sqrt(ex * ex + ey * ey + ez * ez);
      const Real norm = dist + Real(1e-6);
      const Real alpha = acos(fmin(fmax(ez / norm, Real(-1.0)), Real(1.0)));
      if (alpha > max_alpha) continue;
      const Real rx_time = dist * inv_c;
      const Real val = interp_zero(raw, n_samples, n_rx, tx, rx, (tx_time + rx_time) * fs);
      const Real s_hat = copysign(sqrt(fabs(val)), val);
      s1 += s_hat;
      s2 += fabs(val);
    }
    total_dmas += Real(0.5) * (s1 * s1 - s2);
  }
  output[pixel] = total_dmas;
}

template <typename Real>
void execute_dmas(const Data& d, const std::string& output, double xmin, double xmax,
                  std::uint32_t nx, double zmin, double zmax, std::uint32_t nz,
                  bool benchmark_mode) {
  const std::size_t pixel_count = static_cast<std::size_t>(nx) * nz;
  const auto cast_vec = [](const std::vector<double>& src) {
    std::vector<Real> dst(src.size());
    for (std::size_t i = 0; i < src.size(); ++i) dst[i] = static_cast<Real>(src[i]);
    return dst;
  };

  const std::vector<Real> geom_h = cast_vec(d.geometry);
  const std::vector<Real> init_h = cast_vec(d.initial_times);
  const std::vector<Real> t0_h = cast_vec(d.t0_delays);
  const std::vector<Real> apod_h = cast_vec(d.tx_apod);

  DeviceBuffer<Real> geom_d(geom_h.size());
  DeviceBuffer<Real> init_d(init_h.size());
  DeviceBuffer<Real> t0_d(t0_h.size());
  DeviceBuffer<Real> apod_d(apod_h.size());
  DeviceBuffer<float> raw_d(d.raw.size());
  DeviceBuffer<Real> out_d(pixel_count);

  geom_d.upload(geom_h);
  init_d.upload(init_h);
  t0_d.upload(t0_h);
  apod_d.upload(apod_h);
  raw_d.upload(d.raw);

  const std::size_t allocated_bytes =
      (geom_h.size() + init_h.size() + t0_h.size() + apod_h.size() + pixel_count) * sizeof(Real) +
      d.raw.size() * sizeof(float);

  constexpr std::uint32_t kBlockSize = 256;
  const std::uint32_t grid_size = (pixel_count + kBlockSize - 1) / kBlockSize;

  const auto launch_kernel = [&]() {
    dmas_kernel<Real><<<grid_size, kBlockSize>>>(
        geom_d.get(), init_d.get(), t0_d.get(), apod_d.get(), raw_d.get(), out_d.get(),
        d.n_tx, d.n_samples, d.n_rx, static_cast<Real>(d.fs), static_cast<Real>(d.c),
        static_cast<Real>(d.t_peak), static_cast<Real>(d.f_num),
        nx, nz, xmin, xmax, zmin, zmax);
    cuda_check(cudaGetLastError(), "dmas_kernel execution");
  };

  if (!benchmark_mode) {
    launch_kernel();
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize run");
    std::vector<Real> pixels(pixel_count);
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
    std::cout << "Wrote naive CUDA DMAS (" << (sizeof(Real) == 8 ? "FP64" : "FP32")
              << "): " << nx << " x " << nz << " pixels to " << output << '\n';
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

  std::cout << "BENCHMARK variant=" << (sizeof(Real) == 8 ? "dmas_cuda_naive_f64" : "dmas_cuda_naive_f32")
            << " timer=cuda_event warmups=" << warmups << " runs=" << runs
            << " pixels=" << pixel_count << " mean_ms=" << mean
            << " median_ms=" << ms[runs / 2] << " p95_ms=" << ms[static_cast<std::size_t>(0.95 * (runs - 1))]
            << " min_ms=" << ms.front() << " max_ms=" << ms.back()
            << " device_buffer_bytes=" << allocated_bytes
            << " device_total_bytes=" << prop.totalGlobalMem << '\n';
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 9) {
      std::cerr << "Usage: dmas_cuda_naive <input.bundle> <output.csv> <xmin_mm> <xmax_mm> <nx|dx> <zmin_mm> <zmax_mm> <nz|dz> [--double] [--benchmark]\n";
      return 2;
    }
    const auto v = [](const char* s) { return std::stod(s); };
    const Data d = load(argv[1]);
    const double xmin = v(argv[3]), xmax = v(argv[4]);
    const double zmin = v(argv[6]), zmax = v(argv[7]);
    const std::uint32_t nx = resolve_count(xmin, xmax, v(argv[5]));
    const std::uint32_t nz = resolve_count(zmin, zmax, v(argv[8]));

    bool use_double = false;
    bool benchmark_mode = false;
    for (int i = 9; i < argc; ++i) {
      std::string arg = argv[i];
      if (arg == "--double") use_double = true;
      if (arg == "--benchmark") benchmark_mode = true;
    }

    if (use_double)
      execute_dmas<double>(d, argv[2], xmin, xmax, nx, zmin, zmax, nz, benchmark_mode);
    else
      execute_dmas<float>(d, argv[2], xmin, xmax, nx, zmin, zmax, nz, benchmark_mode);
  } catch (const std::exception& e) {
    std::cerr << "CUDA DMAS naive error: " << e.what() << '\n';
    return 1;
  }
  return 0;
}
