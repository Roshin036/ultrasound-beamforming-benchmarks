#include <array>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
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

template<class T>
void read_exact(std::ifstream& in, T* p, std::size_t n, const char* label) {
  in.read(reinterpret_cast<char*>(p), static_cast<std::streamsize>(n * sizeof(T)));
  if (!in) throw std::runtime_error(std::string("Truncated input reading ") + label);
}

Data load(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("Cannot open input bundle: " + path);
  std::array<char, 8> magic{}; read_exact(in, magic.data(), magic.size(), "magic");
  if (magic != kMagic) throw std::runtime_error("Not a DASREF01 bundle");
  Data d; double fs64{}, c64{};
  read_exact(in, &d.n_tx, 1, "header"); read_exact(in, &d.n_samples, 1, "header");
  read_exact(in, &d.n_rx, 1, "header"); read_exact(in, &fs64, 1, "header"); read_exact(in, &c64, 1, "header");
  d.fs = static_cast<float>(fs64); d.c = static_cast<float>(c64);
  if (!d.n_tx || !d.n_samples || !d.n_rx || !(d.fs > 0) || !(d.c > 0))
    throw std::runtime_error("Invalid dimensions or acquisition constants");

  std::vector<double> temp;
  const auto read_doubles = [&](std::size_t n, std::vector<float>& dst, const char* label) {
    temp.resize(n); read_exact(in, temp.data(), temp.size(), label); dst.resize(n);
    for (std::size_t i = 0; i < n; ++i) dst[i] = static_cast<float>(temp[i]);
  };
  read_doubles(static_cast<std::size_t>(d.n_rx) * 3, d.geometry, "geometry");
  read_doubles(d.n_tx, d.initial_times, "initial times");
  read_doubles(static_cast<std::size_t>(d.n_tx) * d.n_rx, d.t0_delays, "transmit delays");
  read_doubles(static_cast<std::size_t>(d.n_tx) * d.n_rx, d.tx_apod, "transmit apodizations");
  d.raw.resize(static_cast<std::size_t>(d.n_tx) * d.n_samples * d.n_rx);
  read_exact(in, d.raw.data(), d.raw.size(), "raw RF samples");
  if (in.peek() != std::char_traits<char>::eof()) throw std::runtime_error("Unexpected trailing input bytes");
  return d;
}

float interp_zero(const Data& d, std::size_t tx, std::size_t rx, float index) {
  if (!std::isfinite(index) || index < 0.0F || index > static_cast<float>(d.n_samples - 1)) return 0.0F;
  const auto i0 = static_cast<std::uint32_t>(std::floor(index));
  const auto pos = (tx * d.n_samples + i0) * d.n_rx + rx;
  const float a = d.raw[pos];
  if (i0 + 1 >= d.n_samples) return a;
  const float fraction = index - static_cast<float>(i0);
  return a + fraction * (d.raw[pos + d.n_rx] - a);
}

std::uint32_t count(double lo, double hi, double step, const char* axis) {
  if (!std::isfinite(lo) || !std::isfinite(hi) || !std::isfinite(step) || step <= 0 || hi < lo)
    throw std::runtime_error(std::string("Invalid ") + axis + " grid");
  const double intervals = (hi - lo) / step;
  const auto rounded = std::llround(intervals);
  if (std::abs(intervals - static_cast<double>(rounded)) > 1e-8 || rounded >= std::numeric_limits<std::uint32_t>::max())
    throw std::runtime_error(std::string(axis) + " range must be an integer multiple of step");
  return static_cast<std::uint32_t>(rounded + 1);
}

template<std::size_t N_TX, std::size_t N_RX>
void run(const Data& d, const std::string& output, double xmin, double xmax, double dx,
         double zmin, double zmax, double dz) {
  if (d.n_tx != N_TX || d.n_rx != N_RX)
    throw std::runtime_error("Input dimensions do not match this compile-time specialization");
  const auto nx = count(xmin, xmax, dx, "x"), nz = count(zmin, zmax, dz, "z");
  std::ofstream out(output);
  if (!out) throw std::runtime_error("Cannot open output CSV: " + output);
  out << "x_mm,z_mm,rf_sum\n" << std::setprecision(9);
  const float inv_c = 1.0F / d.c;
  for (std::uint32_t iz = 0; iz < nz; ++iz) {
    const float z = static_cast<float>((zmin + iz * dz) * 1e-3);
    for (std::uint32_t ix = 0; ix < nx; ++ix) {
      const float x = static_cast<float>((xmin + ix * dx) * 1e-3);
      float sum = 0.0F;
      for (std::size_t tx = 0; tx < N_TX; ++tx) {
        float tx_arrival = std::numeric_limits<float>::infinity();
        const std::size_t tx_offset = tx * N_RX;
        for (std::size_t el = 0; el < N_RX; ++el) {
          if (d.tx_apod[tx_offset + el] == 0.0F) continue;
          const std::size_t g = el * 3;
          const float ex = x - d.geometry[g], ey = -d.geometry[g + 1], ez = z - d.geometry[g + 2];
          tx_arrival = std::min(tx_arrival, std::sqrt(ex * ex + ey * ey + ez * ez) * inv_c + d.t0_delays[tx_offset + el]);
        }
        if (!std::isfinite(tx_arrival)) continue;
        const float tx_time = tx_arrival - d.initial_times[tx];  // t_peak = 0, as in reference
        for (std::size_t rx = 0; rx < N_RX; ++rx) {
          const std::size_t g = rx * 3;
          const float ex = x - d.geometry[g], ey = -d.geometry[g + 1], ez = z - d.geometry[g + 2];
          const float rx_time = std::sqrt(ex * ex + ey * ey + ez * ez) * inv_c;
          sum += interp_zero(d, tx, rx, (tx_time + rx_time) * d.fs);
        }
      }
      out << static_cast<double>(x) * 1e3 << ',' << static_cast<double>(z) * 1e3 << ',' << sum << '\n';
    }
  }
  if (!out) throw std::runtime_error("Failed while writing CSV");
  std::cout << "Wrote template-specialized DAS (N_TX=" << N_TX << ", N_RX=" << N_RX
            << "): " << nx << " x " << nz << " pixels to " << output << '\n';
}
template<std::size_t N_TX, std::size_t N_RX>
void benchmark(const Data& d, double xmin, double xmax, double dx,
               double zmin, double zmax, double dz) {
  if (d.n_tx != N_TX || d.n_rx != N_RX)
    throw std::runtime_error("Input dimensions do not match this compile-time specialization");
  const auto nx = count(xmin, xmax, dx, "x"), nz = count(zmin, zmax, dz, "z");
  std::vector<float> pixels(static_cast<std::size_t>(nx) * nz);
  const float inv_c = 1.0F / d.c;
  const auto compute = [&] {
    for (std::uint32_t iz = 0; iz < nz; ++iz) {
      const float z = static_cast<float>((zmin + iz * dz) * 1e-3);
      for (std::uint32_t ix = 0; ix < nx; ++ix) {
        const float x = static_cast<float>((xmin + ix * dx) * 1e-3);
        float sum = 0.0F;
        for (std::size_t tx = 0; tx < N_TX; ++tx) {
          float tx_arrival = std::numeric_limits<float>::infinity();
          const std::size_t tx_offset = tx * N_RX;
          for (std::size_t el = 0; el < N_RX; ++el) {
            if (d.tx_apod[tx_offset + el] == 0.0F) continue;
            const std::size_t g = el * 3;
            const float ex = x - d.geometry[g], ey = -d.geometry[g + 1], ez = z - d.geometry[g + 2];
            tx_arrival = std::min(tx_arrival, std::sqrt(ex * ex + ey * ey + ez * ez) * inv_c + d.t0_delays[tx_offset + el]);
          }
          if (!std::isfinite(tx_arrival)) continue;
          const float tx_time = tx_arrival - d.initial_times[tx];
          for (std::size_t rx = 0; rx < N_RX; ++rx) {
            const std::size_t g = rx * 3;
            const float ex = x - d.geometry[g], ey = -d.geometry[g + 1], ez = z - d.geometry[g + 2];
            const float rx_time = std::sqrt(ex * ex + ey * ey + ez * ez) * inv_c;
            sum += interp_zero(d, tx, rx, (tx_time + rx_time) * d.fs);
          }
        }
        pixels[static_cast<std::size_t>(iz) * nx + ix] = sum;
      }
    }
  };
  constexpr int warmups = 2, runs = 30;
  for (int i = 0; i < warmups; ++i) compute();
  std::vector<double> ms;
  ms.reserve(runs);
  for (int i = 0; i < runs; ++i) {
    const auto start = std::chrono::steady_clock::now();
    compute();
    const auto stop = std::chrono::steady_clock::now();
    ms.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
  }
  std::sort(ms.begin(), ms.end());
  double checksum = 0.0;
  for (const float value : pixels) checksum += value;
  const double mean = std::accumulate(ms.begin(), ms.end(), 0.0) / runs;
  std::cout << "BENCHMARK variant=cpu_template timer=steady_clock warmups=" << warmups
            << " runs=" << runs << " pixels=" << pixels.size() << " mean_ms=" << mean
            << " median_ms=" << ms[runs / 2] << " p95_ms=" << ms[static_cast<std::size_t>(0.95 * (runs - 1))]
            << " min_ms=" << ms.front() << " max_ms=" << ms.back() << " checksum=" << checksum << '\n';
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 9 && argc != 10) {
      std::cerr << "Usage: das_cpu_template <input.bundle> <output.csv> <xmin_mm> <xmax_mm> <dx_mm> <zmin_mm> <zmax_mm> <dz_mm>\n";
      return 2;
    }
    const auto v = [](const char* s) { return std::stod(s); };
    const Data d = load(argv[1]);
    if (argc == 10 && std::string(argv[9]) == "--benchmark")
      benchmark<21, 128>(d, v(argv[3]), v(argv[4]), v(argv[5]), v(argv[6]), v(argv[7]), v(argv[8]));
    else
      run<21, 128>(d, argv[2], v(argv[3]), v(argv[4]), v(argv[5]), v(argv[6]), v(argv[7]), v(argv[8]));
  } catch (const std::exception& e) {
    std::cerr << "Template DAS error: " << e.what() << '\n';
    return 1;
  }
  return 0;
}
