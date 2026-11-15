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

template <typename Real>
__device__ inline Real interp_zero(const float* raw, std::uint32_t n_samples,
                                   std::uint32_t n_rx, std::size_t tx,
                                   std::size_t rx, Real index) {
  if (!isfinite(index) || index < Real(0.0) || index > static_cast<Real>(n_samples - 1)) return Real(0.0);
  const auto i0 = static_cast<std::uint32_t>(floor(index));
  const auto pos = (tx * n_samples + i0) * n_rx + rx;
  const Real a = static_cast<Real>(raw[pos]);
  if (i0 + 1 >= n_samples) return a;
  const Real f = index - static_cast<Real>(i0);
  return a + f * (static_cast<Real>(raw[pos + n_rx]) - a);
}

template <typename Real, std::size_t N_TX, std::size_t N_RX>
__global__ void das_template_kernel(const Real* geometry, const Real* initial_times,
                                    const Real* t0_delays, const Real* tx_apod,
                                    const float* raw, Real* output,
                                    std::uint32_t n_samples, Real fs, Real c,
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
  Real sum = Real(0.0);

  for (std::size_t tx = 0; tx < N_TX; ++tx) {
    Real tx_arrival = Real(1e30);
    const std::size_t tx_offset = tx * N_RX;
    for (std::size_t el = 0; el < N_RX; ++el) {
      if (tx_apod[tx_offset + el] == Real(0.0)) continue;
      const std::size_t g = el * 3;
      const Real ex = x - geometry[g];
      const Real ey = -geometry[g + 1];
      const Real ez = z - geometry[g + 2];
      const Real candidate = sqrt(ex * ex + ey * ey + ez * ez) * inv_c + t0_delays[tx_offset + el];
      tx_arrival = (candidate < tx_arrival) ? candidate : tx_arrival;
    }
    if (tx_arrival >= Real(1e29)) continue;
    const Real tx_time = tx_arrival - initial_times[tx] + t_peak;
    for (std::size_t rx = 0; rx < N_RX; ++rx) {
      const std::size_t g = rx * 3;
      const Real ex = x - geometry[g];
      const Real ey = -geometry[g + 1];
      const Real ez = z - geometry[g + 2];
      const Real dist = sqrt(ex * ex + ey * ey + ez * ez);
      const Real norm = dist + Real(1e-6);
      const Real alpha = acos(fmin(fmax(ez / norm, Real(-1.0)), Real(1.0)));
      if (alpha > max_alpha) continue;
      const Real rx_time = dist * inv_c;
      sum += interp_zero<Real>(raw, n_samples, N_RX, tx, rx, (tx_time + rx_time) * fs);
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
  if (pixel_count > std::numeric_limits<std::uint32_t>::max())
    throw std::runtime_error("Image grid exceeds CUDA kernel indexing range");

  std::vector<Real> h_geometry(d.geometry.begin(), d.geometry.end());
  std::vector<Real> h_initial(d.initial_times.begin(), d.initial_times.end());
  std::vector<Real> h_delays(d.t0_delays.begin(), d.t0_delays.end());
  std::vector<Real> h_apod(d.tx_apod.begin(), d.tx_apod.end());

  std::size_t free_before{}, total_device_bytes{};
  cuda_check(cudaMemGetInfo(&free_before, &total_device_bytes), "cudaMemGetInfo before");
  DeviceBuffer<Real> d_geometry(h_geometry.size()), d_initial(h_initial.size());
  DeviceBuffer<Real> d_delays(h_delays.size()), d_apod(h_apod.size());
  DeviceBuffer<float> d_raw(d.raw.size());
  DeviceBuffer<Real> d_output(pixel_count);

  d_geometry.upload(h_geometry);
  d_initial.upload(h_initial);
  d_delays.upload(h_delays);
  d_apod.upload(h_apod);
  d_raw.upload(d.raw);

  std::size_t free_after{}, total_after{};
  cuda_check(cudaMemGetInfo(&free_after, &total_after), "cudaMemGetInfo after");
  const std::size_t device_buffer_bytes = free_before - free_after;

  constexpr std::uint32_t threads = 128;
  const auto blocks = static_cast<unsigned>((pixel_count + threads - 1) / threads);

  const auto launch = [&] {
    das_template_kernel<Real, N_TX, N_RX><<<blocks, threads>>>(
        d_geometry.get(), d_initial.get(), d_delays.get(), d_apod.get(),
        d_raw.get(), d_output.get(),
        d.n_samples, static_cast<Real>(d.fs), static_cast<Real>(d.c),
        static_cast<Real>(d.t_peak), static_cast<Real>(d.f_num),
        nx, nz, xmin, xmax, zmin, zmax);
    cuda_check(cudaGetLastError(), "das_template_kernel launch");
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
