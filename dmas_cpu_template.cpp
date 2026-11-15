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
constexpr std::array<char, 8> kMagic1{'D', 'A', 'S', 'R', 'E', 'F', '0', '1'};
constexpr std::array<char, 8> kMagic2{'D', 'A', 'S', 'R', 'E', 'F', '0', '2'};

struct Data {
  std::uint32_t n_tx{}, n_samples{}, n_rx{};
  float fs{}, c{}, fc{6.0e6f}, t_peak{1.6666667e-7f}, f_num{1.75f};
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
  std::array<char, 8> magic{};
  read_exact(in, magic.data(), magic.size(), "magic");
  bool is_v1 = (magic == kMagic1);
  bool is_v2 = (magic == kMagic2);
  if (!is_v1 && !is_v2) throw std::runtime_error("Not a valid DASREF bundle");

  Data d;
  double fs64{}, c64{};
  read_exact(in, &d.n_tx, 1, "header");
  read_exact(in, &d.n_samples, 1, "header");
  read_exact(in, &d.n_rx, 1, "header");
  read_exact(in, &fs64, 1, "header");
  read_exact(in, &c64, 1, "header");
  d.fs = static_cast<float>(fs64);
  d.c = static_cast<float>(c64);

  if (is_v2) {
    double fc64{}, tpeak64{}, fnum64{};
    read_exact(in, &fc64, 1, "fc");
    read_exact(in, &tpeak64, 1, "t_peak");
    read_exact(in, &fnum64, 1, "f_num");
    d.fc = static_cast<float>(fc64);
    d.t_peak = static_cast<float>(tpeak64);
    d.f_num = static_cast<float>(fnum64);
  } else {
    d.fc = 6.0e6f;
    d.t_peak = 1.0f / d.fc;
    d.f_num = 1.75f;
  }

  if (!d.n_tx || !d.n_samples || !d.n_rx || !(d.fs > 0) || !(d.c > 0))
    throw std::runtime_error("Invalid dimensions or acquisition constants");

  std::vector<double> temp;
  const auto read_doubles = [&](std::size_t count, std::vector<float>& dst, const char* label) {
    temp.resize(count); read_exact(in, temp.data(), temp.size(), label); dst.resize(count);
    for (std::size_t i = 0; i < count; ++i) dst[i] = static_cast<float>(temp[i]);
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

inline float interp_zero(const Data& d, std::size_t tx, std::size_t rx, float index) {
  if (!std::isfinite(index) || index < 0.0F || index > static_cast<float>(d.n_samples - 1)) return 0.0F;
  const auto i0 = static_cast<std::uint32_t>(std::floor(index));
  const auto pos = (tx * d.n_samples + i0) * d.n_rx + rx;
  const float a = d.raw[pos];
  if (i0 + 1 >= d.n_samples) return a;
  const float f = index - static_cast<float>(i0);
  return a + f * (d.raw[pos + d.n_rx] - a);
}

std::uint32_t resolve_count(double lo, double hi, double param) {
  if (param >= 1.0 && std::abs(param - std::round(param)) < 1e-6) {
    return static_cast<std::uint32_t>(std::round(param));
  }
  if (param <= 0.0 || hi < lo) throw std::runtime_error("Invalid grid step or span");
  const double n = (hi - lo) / param;
  return static_cast<std::uint32_t>(std::llround(n) + 1);
}

template<std::size_t N_TX, std::size_t N_RX>
void run(const Data& d, const std::string& output, double xmin, double xmax, std::uint32_t nx,
         double zmin, double zmax, std::uint32_t nz) {
  if (d.n_tx != N_TX || d.n_rx != N_RX)
    throw std::runtime_error("Input dimensions do not match compile-time specialization");
  std::ofstream out(output);
  if (!out) throw std::runtime_error("Cannot open output CSV: " + output);
  out << "x_mm,z_mm,rf_sum\n" << std::setprecision(9);
  const float inv_c = 1.0F / d.c;
  const float max_alpha = std::atan(1.0F / (2.0F * d.f_num + 1e-7F));

  for (std::uint32_t iz = 0; iz < nz; ++iz) {
    const float z = (nz > 1) ? static_cast<float>((zmin + (zmax - zmin) * iz / double(nz - 1)) * 1e-3)
                             : static_cast<float>(zmin * 1e-3);
    for (std::uint32_t ix = 0; ix < nx; ++ix) {
      const float x = (nx > 1) ? static_cast<float>((xmin + (xmax - xmin) * ix / double(nx - 1)) * 1e-3)
                               : static_cast<float>(xmin * 1e-3);
      float dmas_pixel_sum = 0.0F;
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
        const float tx_time = tx_arrival - d.initial_times[tx] + d.t_peak;

        float s1 = 0.0F;
        float s2 = 0.0F;
        for (std::size_t rx = 0; rx < N_RX; ++rx) {
          const std::size_t g = rx * 3;
          const float ex = x - d.geometry[g], ey = -d.geometry[g + 1], ez = z - d.geometry[g + 2];
          const float dist = std::sqrt(ex * ex + ey * ey + ez * ez);
          const float norm = dist + 1e-6F;
          const float alpha = std::acos(std::clamp(ez / norm, -1.0F, 1.0F));
          if (alpha > max_alpha) continue;
          const float rx_time = dist * inv_c;
          const float val = interp_zero(d, tx, rx, (tx_time + rx_time) * d.fs);
          const float s_hat = std::copysign(std::sqrt(std::abs(val)), val);
          s1 += s_hat;
          s2 += std::abs(val);
        }
        dmas_pixel_sum += 0.5F * (s1 * s1 - s2);
      }
      out << static_cast<double>(x) * 1e3 << ',' << static_cast<double>(z) * 1e3 << ',' << dmas_pixel_sum << '\n';
    }
  }
  std::cout << "Wrote template CPU DMAS (N_TX=" << N_TX << ", N_RX=" << N_RX
            << "): " << nx << " x " << nz << " pixels to " << output << '\n';
}

template<std::size_t N_TX, std::size_t N_RX>
void benchmark(const Data& d, double xmin, double xmax, std::uint32_t nx,
               double zmin, double zmax, std::uint32_t nz) {
  if (d.n_tx != N_TX || d.n_rx != N_RX)
    throw std::runtime_error("Input dimensions do not match compile-time specialization");
  std::vector<float> pixels(static_cast<std::size_t>(nx) * nz);
  const float inv_c = 1.0F / d.c;
  const float max_alpha = std::atan(1.0F / (2.0F * d.f_num + 1e-7F));

  const auto compute = [&] {
    for (std::uint32_t iz = 0; iz < nz; ++iz) {
      const float z = (nz > 1) ? static_cast<float>((zmin + (zmax - zmin) * iz / double(nz - 1)) * 1e-3)
                               : static_cast<float>(zmin * 1e-3);
      for (std::uint32_t ix = 0; ix < nx; ++ix) {
        const float x = (nx > 1) ? static_cast<float>((xmin + (xmax - xmin) * ix / double(nx - 1)) * 1e-3)
                                 : static_cast<float>(xmin * 1e-3);
        float dmas_pixel_sum = 0.0F;
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
          const float tx_time = tx_arrival - d.initial_times[tx] + d.t_peak;

          float s1 = 0.0F;
          float s2 = 0.0F;
          for (std::size_t rx = 0; rx < N_RX; ++rx) {
            const std::size_t g = rx * 3;
            const float ex = x - d.geometry[g], ey = -d.geometry[g + 1], ez = z - d.geometry[g + 2];
            const float dist = std::sqrt(ex * ex + ey * ey + ez * ez);
            const float norm = dist + 1e-6F;
            const float alpha = std::acos(std::clamp(ez / norm, -1.0F, 1.0F));
            if (alpha > max_alpha) continue;
            const float rx_time = dist * inv_c;
            const float val = interp_zero(d, tx, rx, (tx_time + rx_time) * d.fs);
            const float s_hat = std::copysign(std::sqrt(std::abs(val)), val);
            s1 += s_hat;
            s2 += std::abs(val);
          }
          dmas_pixel_sum += 0.5F * (s1 * s1 - s2);
        }
        pixels[static_cast<std::size_t>(iz) * nx + ix] = dmas_pixel_sum;
      }
    }
  };

  constexpr int warmups = 2, runs = 10;
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
  std::cout << "BENCHMARK variant=dmas_cpu_template timer=steady_clock warmups=" << warmups
            << " runs=" << runs << " pixels=" << pixels.size() << " mean_ms=" << mean
            << " median_ms=" << ms[runs / 2] << " p95_ms=" << ms[static_cast<std::size_t>(0.95 * (runs - 1))]
            << " min_ms=" << ms.front() << " max_ms=" << ms.back() << " checksum=" << checksum << '\n';
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 9 && argc != 10) {
      std::cerr << "Usage: dmas_cpu_template <input.bundle> <output.csv> <xmin_mm> <xmax_mm> <nx|dx> <zmin_mm> <zmax_mm> <nz|dz> [--benchmark]\n";
      return 2;
    }
    const auto v = [](const char* s) { return std::stod(s); };
    const Data d = load(argv[1]);
    const double xmin = v(argv[3]), xmax = v(argv[4]);
    const double zmin = v(argv[6]), zmax = v(argv[7]);
    const std::uint32_t nx = resolve_count(xmin, xmax, v(argv[5]));
    const std::uint32_t nz = resolve_count(zmin, zmax, v(argv[8]));

    if (argc == 10 && std::string(argv[9]) == "--benchmark")
      benchmark<21, 128>(d, xmin, xmax, nx, zmin, zmax, nz);
    else
      run<21, 128>(d, argv[2], xmin, xmax, nx, zmin, zmax, nz);
  } catch (const std::exception& e) {
    std::cerr << "CPU DMAS template error: " << e.what() << '\n';
    return 1;
  }
  return 0;
}
