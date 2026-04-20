// Standalone plaintext non-clustered kernel for architecture simulation.
// Computes query-to-database scores for one query and N database vectors.

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

// query: [d]
// db:    [N, d] row-major
// out:   [N]
extern "C" void pt_nonclustered_scores(
    const float* query,
    const float* db,
    int N,
    int d,
    int metric,
    float* out) {
  // Main compute loops kept explicit for easy cycle/memory instrumentation.
  for (int i = 0; i < N; ++i) {
    const float* vec = db + static_cast<std::size_t>(i) * static_cast<std::size_t>(d);
    float acc = 0.0f;
    if (metric == METRIC_DOT) {
      for (int j = 0; j < d; ++j) {
        acc += query[j] * vec[j];
      }
      out[i] = acc;
    } else {
      for (int j = 0; j < d; ++j) {
        const float diff = query[j] - vec[j];
        acc += diff * diff;
      }
      out[i] = -acc;
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
  const unsigned int x = static_cast<unsigned int>(idx * 1103515245u + 12345u);
  return static_cast<float>(x % 10000u) / 10000.0f;
}

void print_usage(const char* prog) {
  std::cout << "Usage: " << prog << " [N] [d] [metric:0|1] [iters]\n"
            << "Defaults: N=4096 d=384 metric=1 (L2) iters=3\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 1 && std::string(argv[1]) == "--help") {
    print_usage(argv[0]);
    return 0;
  }

  const int N = (argc > 1) ? parse_int_arg(argv[1], 4096) : 4096;
  const int d = (argc > 2) ? parse_int_arg(argv[2], 384) : 384;
  const int metric = (argc > 3) ? std::atoi(argv[3]) : archkernels::METRIC_L2;
  const int iters = (argc > 4) ? parse_int_arg(argv[4], 3) : 3;

  std::vector<float> query(static_cast<std::size_t>(d));
  std::vector<float> db(static_cast<std::size_t>(N) * static_cast<std::size_t>(d));
  std::vector<float> scores(static_cast<std::size_t>(N), 0.0f);

  for (int j = 0; j < d; ++j) {
    query[static_cast<std::size_t>(j)] = fill_value(j + 7);
  }
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < d; ++j) {
      const std::size_t idx = static_cast<std::size_t>(i) * static_cast<std::size_t>(d) +
                              static_cast<std::size_t>(j);
      db[idx] = fill_value(static_cast<int>(idx) + 97);
    }
  }

  using clock = std::chrono::steady_clock;
  const auto t0 = clock::now();
  for (int it = 0; it < iters; ++it) {
    archkernels::pt_nonclustered_scores(
        query.data(), db.data(), N, d, metric, scores.data());
  }
  const auto t1 = clock::now();
  const double elapsed_ms =
      std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t1 - t0).count();

  double checksum = 0.0;
  for (int i = 0; i < N; ++i) {
    checksum += static_cast<double>(scores[static_cast<std::size_t>(i)]);
  }

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "pt_nonclustered done\n";
  std::cout << "N=" << N << " d=" << d
            << " metric=" << ((metric == archkernels::METRIC_DOT) ? "dot" : "l2")
            << " iters=" << iters << "\n";
  std::cout << "elapsed_ms=" << elapsed_ms << "\n";
  std::cout << "checksum=" << checksum << "\n";
  return 0;
}
