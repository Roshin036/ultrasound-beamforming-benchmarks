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
  std::array<char, 8> magic{}; read_exact(in, magic.data(), magic.size(), "magic");
  bool is_v1 = (magic == kMagic1);
  bool is_v2 = (magic == kMagic2);
  if (!is_v1 && !is_v2) throw std::runtime_error("Not a valid DASREF bundle");

  Data d; double fs64{}, c64{};
  read_exact(in, &d.n_tx, 1, "header"); read_exact(in, &d.n_samples, 1, "header");
  read_exact(in, &d.n_rx, 1, "header"); read_exact(in, &fs64, 1, "header"); read_exact(in, &c64, 1, "header");
  d.fs = static_cast<float>(fs64); d.c = static_cast<float>(c64);

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

template<std::size_t N_TX, std::size_t N_RX>
class FastDMASBeamformer {
 public:
  static constexpr std::size_t RX_SHIFT = 7; // 128 = 2^7

  explicit FastDMASBeamformer(const Data& d, double xmin, double xmax, std::uint32_t nx,
                              double zmin, double zmax, std::uint32_t nz)
      : d_(d), nx_(nx), nz_(nz), inv_c_(1.0F / d.c), kappa_(1.0F / (2.0F * d.f_num + 1e-7F)),
        x_coords_(nx), z_coords_(nz), z2_lut_(nz), dx2_lut_(static_cast<std::size_t>(nx) * N_RX) {
    for (std::uint32_t ix = 0; ix < nx; ++ix) {
      x_coords_[ix] = (nx > 1) ? static_cast<float>((xmin + (xmax - xmin) * ix / double(nx - 1)) * 1e-3)
                               : static_cast<float>(xmin * 1e-3);
    }
    for (std::uint32_t iz = 0; iz < nz; ++iz) {
      const float z = (nz > 1) ? static_cast<float>((zmin + (zmax - zmin) * iz / double(nz - 1)) * 1e-3)
                               : static_cast<float>(zmin * 1e-3);
      z_coords_[iz] = z;
      z2_lut_[iz] = z * z;
    }
    // Precompute separable dx2 table: [nx x N_RX]
    for (std::uint32_t ix = 0; ix < nx; ++ix) {
      const float x = x_coords_[ix];
      for (std::size_t rx = 0; rx < N_RX; ++rx) {
        const float ex = x - d_.geometry[rx * 3];
        dx2_lut_[ix * N_RX + rx] = ex * ex;
      }
    }
    geom_x0_ = d_.geometry[0];
    pitch_ = d_.geometry[3] - d_.geometry[0];
    inv_pitch_ = 1.0F / pitch_;
  }

  void compute(std::vector<float>& pixels) const {
    pixels.resize(static_cast<std::size_t>(nx_) * nz_);
    alignas(32) std::array<float, N_RX> dist;
    alignas(32) std::array<float, N_TX> tx_times;

    for (std::uint32_t iz = 0; iz < nz_; ++iz) {
      const float z = z_coords_[iz];
      const float z2 = z2_lut_[iz];
      const float half_aperture = z * kappa_;
      const auto row_offset = static_cast<std::size_t>(iz) * nx_;

      for (std::uint32_t ix = 0; ix < nx_; ++ix) {
        const float x = x_coords_[ix];
        const float* __restrict__ dx2_ptr = &dx2_lut_[ix * N_RX];

        // 1. Hoisted distance calculation: 128 elements once per pixel!
        #pragma GCC ivdep
        for (std::size_t el = 0; el < N_RX; ++el) {
          dist[el] = std::sqrt(dx2_ptr[el] + z2);
        }

        // 2. Transmit arrival times across all 21 transmits
        for (std::size_t tx = 0; tx < N_TX; ++tx) {
          const float* __restrict__ t0_ptr = &d_.t0_delays[tx * N_RX];
          float min_arrival = std::numeric_limits<float>::infinity();
          #pragma GCC ivdep
          for (std::size_t el = 0; el < N_RX; ++el) {
            min_arrival = std::min(min_arrival, dist[el] * inv_c_ + t0_ptr[el]);
          }
          tx_times[tx] = min_arrival - d_.initial_times[tx] + d_.t_peak;
        }

        // 3. Dynamic active aperture bounds
        const float x_left = x - half_aperture;
        const float x_right = x + half_aperture;
        const int rx_min = std::clamp(static_cast<int>(std::ceil((x_left - geom_x0_) * inv_pitch_)), 0, static_cast<int>(N_RX - 1));
        const int rx_max = std::clamp(static_cast<int>(std::floor((x_right - geom_x0_) * inv_pitch_)), 0, static_cast<int>(N_RX - 1));

        // 4. Filtered DMAS accumulation
        float dmas_pixel_sum = 0.0F;
        const auto n_samples = d_.n_samples;
        const float* __restrict__ raw_ptr = d_.raw.data();

        for (std::size_t tx = 0; tx < N_TX; ++tx) {
          const float tx_time = tx_times[tx];
          const auto tx_stride = tx * n_samples;

          float s1 = 0.0F;
          float s2 = 0.0F;

          for (int rx = rx_min; rx <= rx_max; ++rx) {
            const float rx_time = dist[rx] * inv_c_;
            const float index = (tx_time + rx_time) * d_.fs;

            if (index >= 0.0F && index <= static_cast<float>(n_samples - 1)) {
              const auto i0 = static_cast<std::uint32_t>(index);
              const auto pos = ((tx_stride + i0) << RX_SHIFT) + rx;
              const float a = raw_ptr[pos];
              float val = a;
              if (i0 + 1 < n_samples) {
                const float frac = index - static_cast<float>(i0);
                val += frac * (raw_ptr[pos + N_RX] - a);
              }
              const float s_hat = std::copysign(std::sqrt(std::abs(val)), val);
              s1 += s_hat;
              s2 += std::abs(val);
            }
          }
          dmas_pixel_sum += 0.5F * (s1 * s1 - s2);
        }

        pixels[row_offset + ix] = dmas_pixel_sum;
      }
    }
  }

 private:
  const Data& d_;
  std::uint32_t nx_{}, nz_{};
  float inv_c_{}, kappa_{}, geom_x0_{}, pitch_{}, inv_pitch_{};
  std::vector<float> x_coords_, z_coords_, z2_lut_, dx2_lut_;
};
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 9) {
      std::cerr << "Usage: dmas_cpu_template <input.bundle> <output.csv> <xmin> <xmax> <nx> <zmin> <zmax> <nz> [--benchmark]\n";
      return 2;
    }
    const auto v = [](const char* s) { return std::stod(s); };
    const Data d = load(argv[1]);
    const double xmin = v(argv[3]), xmax = v(argv[4]);
    const double zmin = v(argv[6]), zmax = v(argv[7]);
    const std::uint32_t nx = resolve_count(xmin, xmax, v(argv[5]));
    const std::uint32_t nz = resolve_count(zmin, zmax, v(argv[8]));

    FastDMASBeamformer<21, 128> beamformer(d, xmin, xmax, nx, zmin, zmax, nz);
    std::vector<float> pixels;

    bool benchmark_mode = (argc >= 10 && std::string(argv[9]) == "--benchmark");
    if (benchmark_mode) {
      constexpr int warmups = 2, runs = 10;
      for (int i = 0; i < warmups; ++i) beamformer.compute(pixels);
      std::vector<double> ms;
      ms.reserve(runs);
      for (int i = 0; i < runs; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        beamformer.compute(pixels);
        const auto t1 = std::chrono::steady_clock::now();
        ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
      }
      std::sort(ms.begin(), ms.end());
      const double mean = std::accumulate(ms.begin(), ms.end(), 0.0) / runs;
      std::cout << "BENCHMARK variant=dmas_cpu_template timer=steady_clock warmups=" << warmups
                << " runs=" << runs << " pixels=" << pixels.size() << " mean_ms=" << mean
                << " median_ms=" << ms[runs / 2] << " p95_ms=" << ms[static_cast<std::size_t>(0.95 * (runs - 1))]
                << " min_ms=" << ms.front() << " max_ms=" << ms.back() << '\n';
    } else {
      beamformer.compute(pixels);
      std::ofstream out(argv[2]);
      out << "x_mm,z_mm,rf_sum\n" << std::setprecision(9);
      for (std::uint32_t iz = 0; iz < nz; ++iz) {
        const double z = (nz > 1) ? (zmin + (zmax - zmin) * iz / double(nz - 1)) : zmin;
        for (std::uint32_t ix = 0; ix < nx; ++ix) {
          const double x = (nx > 1) ? (xmin + (xmax - xmin) * ix / double(nx - 1)) : xmin;
          out << x << ',' << z << ',' << pixels[static_cast<std::size_t>(iz) * nx + ix] << '\n';
        }
      }
      std::cout << "Wrote template-specialized DMAS: " << nx << " x " << nz << " pixels to " << argv[2] << '\n';
    }
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << '\n';
    return 1;
  }
  return 0;
}
