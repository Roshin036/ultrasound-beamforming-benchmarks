#include <cuda_runtime.h>

#include <algorithm>
#include <array>
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
constexpr std::array<char, 8> kMagic{'D', 'A', 'S', 'R', 'E', 'F', '0', '1'};

struct Data {
  std::uint32_t n_tx{}, n_samples{}, n_rx{};
  float fs{}, c{};
  std::vector<float> geometry, initial_times, t0_delays, tx_apod, raw;
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
  if (magic != kMagic) throw std::runtime_error("Not a DASREF01 bundle");

  Data d;
  double fs64{}, c64{};
  read_exact(in, &d.n_tx, 1, "header");
  read_exact(in, &d.n_samples, 1, "header");
  read_exact(in, &d.n_rx, 1, "header");
  read_exact(in, &fs64, 1, "header");
  read_exact(in, &c64, 1, "header");
  d.fs = static_cast<float>(fs64);
  d.c = static_cast<float>(c64);
  if (!d.n_tx || !d.n_samples || !d.n_rx || !(d.fs > 0.0F) || !(d.c > 0.0F))
    throw std::runtime_error("Invalid dimensions or acquisition constants");

  std::vector<double> temp;
  const auto read_doubles = [&](std::size_t count, std::vector<float>& dst, const char* label) {
    temp.resize(count);
    read_exact(in, temp.data(), temp.size(), label);
    dst.resize(count);
    for (std::size_t i = 0; i < count; ++i) dst[i] = static_cast<float>(temp[i]);
  };
  read_doubles(static_cast<std::size_t>(d.n_rx) * 3, d.geometry, "geometry");
  read_doubles(d.n_tx, d.initial_times, "initial times");
  read_doubles(static_cast<std::size_t>(d.n_tx) * d.n_rx, d.t0_delays, "transmit delays");
  read_doubles(static_cast<std::size_t>(d.n_tx) * d.n_rx, d.tx_apod, "transmit apodizations");
  d.raw.resize(static_cast<std::size_t>(d.n_tx) * d.n_samples * d.n_rx);
  read_exact(in, d.raw.data(), d.raw.size(), "raw RF samples");
  if (in.peek() != std::char_traits<char>::eof())
    throw std::runtime_error("Unexpected trailing input bytes");
  return d;
}

std::uint32_t count(double lo, double hi, double step, const char* axis) {
  if (!std::isfinite(lo) || !std::isfinite(hi) || !std::isfinite(step) || step <= 0.0 || hi < lo)
    throw std::runtime_error(std::string("Invalid ") + axis + " grid");
  const double n = (hi - lo) / step;
  const auto nr = std::llround(n);
  if (std::abs(n - static_cast<double>(nr)) > 1e-8 ||
      nr >= std::numeric_limits<std::uint32_t>::max())
    throw std::runtime_error(std::string(axis) + " range must be an integer multiple of step");
  return static_cast<std::uint32_t>(nr + 1);
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
    cuda_check(cudaMemcpy(ptr_, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice),
               "cudaMemcpy host to device");
  }
  void download(std::vector<T>& dst) const {
    cuda_check(cudaMemcpy(dst.data(), ptr_, dst.size() * sizeof(T), cudaMemcpyDeviceToHost),
               "cudaMemcpy device to host");
  }

 private:
  T* ptr_{};
};

__device__ float interp_zero(const float* raw, std::uint32_t n_samples,
                             std::uint32_t n_rx, std::uint32_t tx,
                             std::uint32_t rx, float index) {
  if (!isfinite(index) || index < 0.0F || index > static_cast<float>(n_samples - 1)) return 0.0F;
  const auto i0 = static_cast<std::uint32_t>(floorf(index));
  const auto pos = (static_cast<std::size_t>(tx) * n_samples + i0) * n_rx + rx;
  const float a = raw[pos];
  if (i0 + 1 >= n_samples) return a;
  const float f = index - static_cast<float>(i0);
  return a + f * (raw[pos + n_rx] - a);
}

template <std::uint32_t N_TX, std::uint32_t N_RX>
__global__ void das_kernel(const float* geometry, const float* initial_times,
                           const float* t0_delays, const float* tx_apod,
                           const float* raw, float* output,
                           std::uint32_t n_samples, float fs, float c,
                           std::uint32_t nx, std::uint32_t nz,
                           double xmin_mm, double dx_mm,
                           double zmin_mm, double dz_mm) {
  const std::uint32_t pixel = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t pixel_count = nx * nz;
  if (pixel >= pixel_count) return;
  const std::uint32_t iz = pixel / nx;
  const std::uint32_t ix = pixel - iz * nx;
  const float x = static_cast<float>((xmin_mm + static_cast<double>(ix) * dx_mm) * 1e-3);
  const float z = static_cast<float>((zmin_mm + static_cast<double>(iz) * dz_mm) * 1e-3);
  const float inv_c = 1.0F / c;
  float sum = 0.0F;

  for (std::uint32_t tx = 0; tx < N_TX; ++tx) {
    float tx_arrival = __int_as_float(0x7f800000);
    const std::size_t tx_offset = static_cast<std::size_t>(tx) * N_RX;
    for (std::uint32_t el = 0; el < N_RX; ++el) {
      if (tx_apod[tx_offset + el] == 0.0F) continue;
      const std::size_t g = static_cast<std::size_t>(el) * 3;
      const float ex = x - geometry[g];
      const float ey = -geometry[g + 1];
      const float ez = z - geometry[g + 2];
      const float candidate = sqrtf(ex * ex + ey * ey + ez * ez) * inv_c +
                              t0_delays[tx_offset + el];
      tx_arrival = fminf(tx_arrival, candidate);
    }
    if (!isfinite(tx_arrival)) continue;
    const float tx_time = tx_arrival - initial_times[tx];  // t_peak = 0, matching CPU reference.
    for (std::uint32_t rx = 0; rx < N_RX; ++rx) {
      const std::size_t g = static_cast<std::size_t>(rx) * 3;
      const float ex = x - geometry[g];
      const float ey = -geometry[g + 1];
      const float ez = z - geometry[g + 2];
      const float rx_time = sqrtf(ex * ex + ey * ey + ez * ez) * inv_c;
      sum += interp_zero(raw, n_samples, N_RX, tx, rx, (tx_time + rx_time) * fs);
    }
  }
  output[pixel] = sum;
}

void run(const Data& d, const std::string& output_path,
         double xmin, double xmax, double dx, double zmin, double zmax, double dz,
         bool benchmark_mode) {
  constexpr std::uint32_t n_tx = 21;
  constexpr std::uint32_t n_rx = 128;
  if (d.n_tx != n_tx || d.n_rx != n_rx)
    throw std::runtime_error("Input dimensions do not match CUDA specialization (21 tx, 128 rx)");
  const std::uint32_t nx = count(xmin, xmax, dx, "x");
  const std::uint32_t nz = count(zmin, zmax, dz, "z");
  const std::size_t pixel_count = static_cast<std::size_t>(nx) * nz;
  if (pixel_count > std::numeric_limits<std::uint32_t>::max())
    throw std::runtime_error("Image grid exceeds CUDA kernel indexing range");

  std::size_t free_before{}, total_device_bytes{};
  cuda_check(cudaMemGetInfo(&free_before, &total_device_bytes), "cudaMemGetInfo before buffers");
  DeviceBuffer<float> d_geometry(d.geometry.size()), d_initial(d.initial_times.size());
  DeviceBuffer<float> d_delays(d.t0_delays.size()), d_apod(d.tx_apod.size());
  DeviceBuffer<float> d_raw(d.raw.size()), d_output(pixel_count);
  d_geometry.upload(d.geometry);
  d_initial.upload(d.initial_times);
  d_delays.upload(d.t0_delays);
  d_apod.upload(d.tx_apod);
  d_raw.upload(d.raw);
  std::size_t free_after{}, total_after{};
  cuda_check(cudaMemGetInfo(&free_after, &total_after), "cudaMemGetInfo after buffers");
  if (total_after != total_device_bytes) throw std::runtime_error("CUDA total memory changed during setup");
  const std::size_t device_buffer_bytes = free_before - free_after;

  constexpr std::uint32_t threads = 128;
  const auto blocks = static_cast<unsigned>((pixel_count + threads - 1) / threads);
  const auto launch = [&] {
    das_kernel<n_tx, n_rx><<<blocks, threads>>>(d_geometry.get(), d_initial.get(), d_delays.get(),
                                                 d_apod.get(), d_raw.get(), d_output.get(),
                                                 d.n_samples, d.fs, d.c, nx, nz,
                                                 xmin, dx, zmin, dz);
    cuda_check(cudaGetLastError(), "DAS kernel launch");
  };
  if (benchmark_mode) {
    constexpr int warmups = 2, runs = 30;
    for (int i = 0; i < warmups; ++i) launch();
    cuda_check(cudaDeviceSynchronize(), "CUDA warmup synchronization");
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
    std::cout << "BENCHMARK variant=cuda_template timer=cuda_event warmups=" << warmups
              << " runs=" << runs << " pixels=" << pixel_count << " mean_ms=" << total / runs
              << " median_ms=" << ms[runs / 2] << " p95_ms=" << ms[static_cast<std::size_t>(0.95 * (runs - 1))]
              << " min_ms=" << ms.front() << " max_ms=" << ms.back()
              << " device_buffer_bytes=" << device_buffer_bytes
              << " device_total_bytes=" << total_device_bytes << '\n';
  } else {
    launch();
  }
  cuda_check(cudaDeviceSynchronize(), "DAS kernel execution");
  std::vector<float> pixels(pixel_count);
  d_output.download(pixels);

  std::ofstream out(output_path);
  if (!out) throw std::runtime_error("Cannot open output CSV: " + output_path);
  out << "x_mm,z_mm,rf_sum\n" << std::setprecision(9);
  for (std::uint32_t iz = 0; iz < nz; ++iz) {
    for (std::uint32_t ix = 0; ix < nx; ++ix) {
      const std::size_t index = static_cast<std::size_t>(iz) * nx + ix;
      const float x = static_cast<float>((xmin + ix * dx) * 1e-3);
      const float z = static_cast<float>((zmin + iz * dz) * 1e-3);
      out << static_cast<double>(x) * 1e3 << ',' << static_cast<double>(z) * 1e3
          << ',' << pixels[index] << '\n';
    }
  }
  if (!out) throw std::runtime_error("Failed while writing CSV");
  std::cout << "Wrote template-specialized CUDA DAS (N_TX=21, N_RX=128): "
            << nx << " x " << nz << " pixels to "
            << output_path << '\n';
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 9 && argc != 10) {
      std::cerr << "Usage: das_cuda_template <input.bundle> <output.csv> <xmin_mm> <xmax_mm> "
                   "<dx_mm> <zmin_mm> <zmax_mm> <dz_mm> [--benchmark]\n";
      return 2;
    }
    const auto value = [](const char* s) { return std::stod(s); };
    const bool benchmark_mode = argc == 10 && std::string(argv[9]) == "--benchmark";
    if (argc == 10 && !benchmark_mode) throw std::runtime_error("Unknown option: " + std::string(argv[9]));
    const Data data = load(argv[1]);
    run(data, argv[2], value(argv[3]), value(argv[4]), value(argv[5]),
        value(argv[6]), value(argv[7]), value(argv[8]), benchmark_mode);
  } catch (const std::exception& error) {
    std::cerr << "CUDA template DAS error: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
