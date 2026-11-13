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

struct Input {
  std::uint32_t n_tx{};
  std::uint32_t n_samples{};
  std::uint32_t n_rx{};
  double sampling_frequency{};
  double sound_speed{};
  std::vector<double> geometry;       // [rx][xyz]
  std::vector<double> initial_times;  // [tx]
  std::vector<double> t0_delays;      // [tx][tx element]
  std::vector<double> tx_apod;        // [tx][tx element]
  std::vector<float> raw;             // [tx][sample][rx]
};

template <class T>
void read_exact(std::ifstream& in, T* dst, std::size_t count, const char* what) {
  in.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(count * sizeof(T)));
  if (!in) throw std::runtime_error(std::string("Truncated input while reading ") + what);
}

Input read_input(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("Cannot open input bundle: " + path);
  std::array<char, 8> magic{};
  read_exact(in, magic.data(), magic.size(), "magic");
  if (magic != kMagic) throw std::runtime_error("Not a DASREF01 input bundle");

  Input d;
  read_exact(in, &d.n_tx, 1, "header");
  read_exact(in, &d.n_samples, 1, "header");
  read_exact(in, &d.n_rx, 1, "header");
  read_exact(in, &d.sampling_frequency, 1, "header");
  read_exact(in, &d.sound_speed, 1, "header");
  if (!d.n_tx || !d.n_samples || !d.n_rx || !(d.sampling_frequency > 0.0) || !(d.sound_speed > 0.0))
    throw std::runtime_error("Invalid dimensions or acquisition constants in input header");

  d.geometry.resize(static_cast<std::size_t>(d.n_rx) * 3);
  d.initial_times.resize(d.n_tx);
  d.t0_delays.resize(static_cast<std::size_t>(d.n_tx) * d.n_rx);
  d.tx_apod.resize(static_cast<std::size_t>(d.n_tx) * d.n_rx);
  const auto raw_count = static_cast<std::size_t>(d.n_tx) * d.n_samples * d.n_rx;
  d.raw.resize(raw_count);
  read_exact(in, d.geometry.data(), d.geometry.size(), "probe geometry");
  read_exact(in, d.initial_times.data(), d.initial_times.size(), "initial times");
  read_exact(in, d.t0_delays.data(), d.t0_delays.size(), "transmit delays");
  read_exact(in, d.tx_apod.data(), d.tx_apod.size(), "transmit apodizations");
  read_exact(in, d.raw.data(), d.raw.size(), "raw RF samples");
  if (in.peek() != std::char_traits<char>::eof()) throw std::runtime_error("Unexpected trailing bytes in input bundle");
  return d;
}

double sample_linear_zero(const Input& d, std::uint32_t tx, std::uint32_t rx, double index) {
  if (!std::isfinite(index) || index < 0.0 || index > static_cast<double>(d.n_samples - 1)) return 0.0;
  const auto i0 = static_cast<std::uint32_t>(std::floor(index));
  const auto base = (static_cast<std::size_t>(tx) * d.n_samples + i0) * d.n_rx + rx;
  const double a = static_cast<double>(d.raw[base]);
  if (i0 + 1 >= d.n_samples) return a;
  const double fraction = index - static_cast<double>(i0);
  const double b = static_cast<double>(d.raw[base + d.n_rx]);
  return a + fraction * (b - a);
}

std::uint32_t grid_count(double lo, double hi, double step, const char* axis) {
  if (!std::isfinite(lo) || !std::isfinite(hi) || !std::isfinite(step) || !(step > 0.0) || hi < lo)
    throw std::runtime_error(std::string("Invalid ") + axis + " grid bounds or step");
  const double intervals = (hi - lo) / step;
  const auto rounded = std::llround(intervals);
  if (std::abs(intervals - static_cast<double>(rounded)) > 1e-8)
    throw std::runtime_error(std::string(axis) + " range must be an integer multiple of step");
  if (rounded >= std::numeric_limits<std::uint32_t>::max()) throw std::runtime_error("Grid is too large");
  return static_cast<std::uint32_t>(rounded + 1);
}

void self_check() {
  Input d;
  d.n_tx = 1; d.n_samples = 3; d.n_rx = 1;
  d.raw = {0.0F, 10.0F, 20.0F};
  const auto near = [](double a, double b) { return std::abs(a - b) < 1e-12; };
  if (!near(sample_linear_zero(d, 0, 0, 1.0), 10.0) ||
      !near(sample_linear_zero(d, 0, 0, 1.25), 12.5) ||
      !near(sample_linear_zero(d, 0, 0, 2.0), 20.0) ||
      !near(sample_linear_zero(d, 0, 0, -0.01), 0.0) ||
      !near(sample_linear_zero(d, 0, 0, 2.01), 0.0))
    throw std::runtime_error("Linear interpolation/out-of-range self-check failed");
  std::cout << "DAS reference self-check passed (linear interpolation and zero outside sample range).\n";
}

void run(const Input& d, const std::string& output,
         double xmin_mm, double xmax_mm, double dx_mm,
         double zmin_mm, double zmax_mm, double dz_mm) {
  const auto nx = grid_count(xmin_mm, xmax_mm, dx_mm, "x");
  const auto nz = grid_count(zmin_mm, zmax_mm, dz_mm, "z");
  std::ofstream out(output);
  if (!out) throw std::runtime_error("Cannot open output CSV: " + output);
  out << "x_mm,z_mm,rf_sum\n" << std::setprecision(17);

  const double c = d.sound_speed;
  const double fs = d.sampling_frequency;
  const double inv_c = 1.0 / c;
  for (std::uint32_t iz = 0; iz < nz; ++iz) {
    const double z = (zmin_mm + static_cast<double>(iz) * dz_mm) * 1e-3;
    for (std::uint32_t ix = 0; ix < nx; ++ix) {
      const double x = (xmin_mm + static_cast<double>(ix) * dx_mm) * 1e-3;
      const std::array<double, 3> pixel{x, 0.0, z};
      double sum = 0.0;
      for (std::uint32_t tx = 0; tx < d.n_tx; ++tx) {
        double first_arrival = std::numeric_limits<double>::infinity();
        const auto tx_offset = static_cast<std::size_t>(tx) * d.n_rx;
        for (std::uint32_t el = 0; el < d.n_rx; ++el) {
          const double apod = d.tx_apod[tx_offset + el];
          if (apod == 0.0) continue;
          const auto g = static_cast<std::size_t>(el) * 3;
          const double ex = pixel[0] - d.geometry[g];
          const double ey = pixel[1] - d.geometry[g + 1];
          const double ez = pixel[2] - d.geometry[g + 2];
          const double distance = std::sqrt(ex * ex + ey * ey + ez * ez);
          first_arrival = std::min(first_arrival, distance * inv_c + d.t0_delays[tx_offset + el]);
        }
        if (!std::isfinite(first_arrival)) continue;
        // The data format has no waveform peak-time field. Reference convention: t_peak = 0.
        const double tx_time = first_arrival - d.initial_times[tx];
        for (std::uint32_t rx = 0; rx < d.n_rx; ++rx) {
          const auto g = static_cast<std::size_t>(rx) * 3;
          const double ex = pixel[0] - d.geometry[g];
          const double ey = pixel[1] - d.geometry[g + 1];
          const double ez = pixel[2] - d.geometry[g + 2];
          const double rx_time = std::sqrt(ex * ex + ey * ey + ez * ez) * inv_c;
          const double sample_index = (tx_time + rx_time) * fs;
          // No receive-apodization array is stored, so the reference uses unit receive weights.
          sum += sample_linear_zero(d, tx, rx, sample_index);
        }
      }
      out << x * 1e3 << ',' << z * 1e3 << ',' << sum << '\n';
    }
  }
  if (!out) throw std::runtime_error("Failed while writing output CSV: " + output);
  std::cout << "Wrote " << nx << " x " << nz << " pixels to " << output << " (coherent RF sum).\n";
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--self-check") { self_check(); return 0; }
    if (argc != 9) {
      std::cerr << "Usage: das_reference <input.bundle> <output.csv> <xmin_mm> <xmax_mm> <dx_mm> <zmin_mm> <zmax_mm> <dz_mm>\n"
                   "       das_reference --self-check\n";
      return 2;
    }
    const auto number = [](const char* s) { return std::stod(s); };
    const Input input = read_input(argv[1]);
    run(input, argv[2], number(argv[3]), number(argv[4]), number(argv[5]),
        number(argv[6]), number(argv[7]), number(argv[8]));
  } catch (const std::exception& e) {
    std::cerr << "DAS reference error: " << e.what() << '\n';
    return 1;
  }
  return 0;
}
