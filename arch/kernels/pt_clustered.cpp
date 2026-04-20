// Standalone plaintext clustered kernel for architecture simulation.
// Computes query-to-centroid scores for one query and K centroids.

#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace archkernels {

enum Metric : int {
  METRIC_DOT = 0,
  METRIC_L2 = 1,  // returns negative squared L2 so larger is better
};

// query:     [d]
// centroids: [K, d] row-major
// out:       [K]
extern "C" void pt_clustered_scores(
    const float* query,
    const float* centroids,
    int K,
    int d,
    int metric,
    float* out) {
  // Main compute loops kept explicit for easy cycle/memory instrumentation.
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

int parse_int_arg(const char* s, int fallback) {
  if (s == nullptr) {
    return fallback;
  }
  const int v = std::atoi(s);
  return v > 0 ? v : fallback;
}

float fill_value(int idx) {
  const unsigned int x = static_cast<unsigned int>(idx * 1664525u + 1013904223u);
  return static_cast<float>(x % 10000u) / 10000.0f;
}

void print_usage(const char* prog) {
  std::cout << "Usage: " << prog << " [K] [d] [metric:0|1] [iters]\n"
            << "Defaults: K=64 d=384 metric=1 (L2) iters=10\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 1 && std::string(argv[1]) == "--help") {
    print_usage(argv[0]);
    return 0;
  }

  const int K = (argc > 1) ? parse_int_arg(argv[1], 64) : 64;
  const int d = (argc > 2) ? parse_int_arg(argv[2], 384) : 384;
  const int metric = (argc > 3) ? std::atoi(argv[3]) : archkernels::METRIC_L2;
  const int iters = (argc > 4) ? parse_int_arg(argv[4], 10) : 10;

  std::vector<float> query(static_cast<std::size_t>(d));
  std::vector<float> centroids(static_cast<std::size_t>(K) * static_cast<std::size_t>(d));
  std::vector<float> scores(static_cast<std::size_t>(K), 0.0f);

  for (int j = 0; j < d; ++j) {
    query[static_cast<std::size_t>(j)] = fill_value(j + 13);
  }
  for (int k = 0; k < K; ++k) {
    for (int j = 0; j < d; ++j) {
      const std::size_t idx = static_cast<std::size_t>(k) * static_cast<std::size_t>(d) +
                              static_cast<std::size_t>(j);
      centroids[idx] = fill_value(static_cast<int>(idx) + 31);
    }
  }

  using clock = std::chrono::steady_clock;
  const auto t0 = clock::now();
  for (int it = 0; it < iters; ++it) {
    archkernels::pt_clustered_scores(
        query.data(), centroids.data(), K, d, metric, scores.data());
  }
  const auto t1 = clock::now();
  const double elapsed_ms =
      std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t1 - t0).count();

  double checksum = 0.0;
  for (int k = 0; k < K; ++k) {
    checksum += static_cast<double>(scores[static_cast<std::size_t>(k)]);
  }

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "pt_clustered done\n";
  std::cout << "K=" << K << " d=" << d
            << " metric=" << ((metric == archkernels::METRIC_DOT) ? "dot" : "l2")
            << " iters=" << iters << "\n";
  std::cout << "elapsed_ms=" << elapsed_ms << "\n";
  std::cout << "checksum=" << checksum << "\n";
  return 0;
}
