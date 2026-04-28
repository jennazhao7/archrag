// File-backed plaintext centroid scoring kernel for gem5 comparison.
// Computes one query vector against a centroid matrix loaded from raw float32 files.

#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace archkernels {

enum Metric : int {
  METRIC_DOT = 0,
  METRIC_L2 = 1,  // returns negative squared L2 so larger is better
};

extern "C" void pt_clustered_scores(
    const float* query,
    const float* centroids,
    int K,
    int d,
    int metric,
    float* out) {
  for (int k = 0; k < K; ++k) {
    const float* c = centroids + static_cast<std::size_t>(k) * static_cast<std::size_t>(d);
    float acc = 0.0f;
    if (metric == METRIC_DOT) {
      for (int j = 0; j < d; ++j) {
        acc += query[j] * c[j];
      }
      out[k] = acc;
    } else {
      for (int j = 0; j < d; ++j) {
        const float diff = query[j] - c[j];
        acc += diff * diff;
      }
      out[k] = -acc;
    }
  }
}

}  // namespace archkernels

namespace {

int parse_positive_int(const char* value, const char* name) {
  const int parsed = std::atoi(value);
  if (parsed <= 0) {
    throw std::runtime_error(std::string(name) + " must be > 0");
  }
  return parsed;
}

std::vector<float> read_float_file(const std::string& path, std::size_t expected_count) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("failed to open input file: " + path);
  }

  std::vector<float> values(expected_count);
  input.read(
      reinterpret_cast<char*>(values.data()),
      static_cast<std::streamsize>(expected_count * sizeof(float)));
  if (input.gcount() != static_cast<std::streamsize>(expected_count * sizeof(float))) {
    throw std::runtime_error("input file is smaller than expected: " + path);
  }
  return values;
}

void print_usage(const char* prog) {
  std::cout << "Usage: " << prog
            << " <query.bin> <centroids.bin> <K> <d> [metric:0|1] [iters]\n"
            << "  query.bin: raw float32 vector with d values\n"
            << "  centroids.bin: raw float32 matrix with K*d values, row-major\n"
            << "  metric: 0=dot, 1=negative squared L2 (default: 1)\n"
            << "  iters: repeat count for the compute loop (default: 1)\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 1 && std::string(argv[1]) == "--help") {
    print_usage(argv[0]);
    return 0;
  }
  if (argc < 5) {
    print_usage(argv[0]);
    return 1;
  }

  try {
    const std::string query_path = argv[1];
    const std::string centroids_path = argv[2];
    const int K = parse_positive_int(argv[3], "K");
    const int d = parse_positive_int(argv[4], "d");
    const int metric = (argc > 5) ? std::atoi(argv[5]) : archkernels::METRIC_L2;
    const int iters = (argc > 6) ? parse_positive_int(argv[6], "iters") : 1;

    std::vector<float> query = read_float_file(query_path, static_cast<std::size_t>(d));
    std::vector<float> centroids = read_float_file(
        centroids_path, static_cast<std::size_t>(K) * static_cast<std::size_t>(d));
    std::vector<float> scores(static_cast<std::size_t>(K), 0.0f);

    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    for (int it = 0; it < iters; ++it) {
      archkernels::pt_clustered_scores(
          query.data(), centroids.data(), K, d, metric, scores.data());
    }
    const auto t1 = clock::now();

    double checksum = 0.0;
    int best_idx = 0;
    float best_score = scores.empty() ? 0.0f : scores[0];
    for (int k = 0; k < K; ++k) {
      const float score = scores[static_cast<std::size_t>(k)];
      checksum += static_cast<double>(score);
      if (score > best_score) {
        best_score = score;
        best_idx = k;
      }
    }

    const double elapsed_ms =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t1 - t0).count();
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "pt_centroids_file done\n";
    std::cout << "K=" << K << " d=" << d
              << " metric=" << ((metric == archkernels::METRIC_DOT) ? "dot" : "l2")
              << " iters=" << iters << "\n";
    std::cout << "elapsed_ms=" << elapsed_ms << "\n";
    std::cout << "checksum=" << checksum << "\n";
    std::cout << "best_idx=" << best_idx << " best_score=" << best_score << "\n";
  } catch (const std::exception& exc) {
    std::cerr << "pt_centroids_file error: " << exc.what() << "\n";
    return 1;
  }
  return 0;
}
