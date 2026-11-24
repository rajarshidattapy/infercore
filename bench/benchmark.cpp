// InferCore benchmark + evaluation harness.
//
// Measures, on the machine that runs it:
//   - fp32 vs int8 vs ONNX Runtime latency (mean / p50 / p95, batch 1)
//   - per-op timing breakdown (native backend)
//   - thread scaling of the native executor (1 vs hardware threads)
//   - MNIST test accuracy fp32 vs int8 (real PTQ accuracy drop)
//   - max abs output diff: native-fp32 vs ONNX Runtime, fp32 vs int8
//   - on-disk model sizes
// Every result row is written to a SQLite database and printed to stdout.
//
// Usage: infercore_benchmark [--models-dir DIR] [--iterations N] [--eval-n N]
//                            [--db PATH]

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <numeric>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include <sqlite3.h>

#include "infercore/model.hpp"
#include "infercore/weights.hpp"

namespace {

struct TestSet {
  std::vector<uint8_t> labels;
  std::vector<std::vector<float>> images;  // normalized [0,1]
};

uint32_t read_be32(std::ifstream& f) {
  uint8_t b[4];
  f.read(reinterpret_cast<char*>(b), 4);
  return (static_cast<uint32_t>(b[0]) << 24) | (b[1] << 16) | (b[2] << 8) | b[3];
}

TestSet load_mnist(const std::string& dir, size_t max_n) {
  std::ifstream img(dir + "/mnist/t10k-images-idx3-ubyte", std::ios::binary);
  std::ifstream lbl(dir + "/mnist/t10k-labels-idx1-ubyte", std::ios::binary);
  if (!img || !lbl) throw std::runtime_error("MNIST test set not found under " + dir);
  if (read_be32(img) != 0x803 || read_be32(lbl) != 0x801)
    throw std::runtime_error("bad MNIST magic");
  const uint32_t n_img = read_be32(img);
  const uint32_t rows = read_be32(img);
  const uint32_t cols = read_be32(img);
  const uint32_t n_lbl = read_be32(lbl);
  if (n_img != n_lbl) throw std::runtime_error("MNIST count mismatch");
  TestSet ts;
  const size_t n = std::min<size_t>(n_img, max_n);
  std::vector<uint8_t> buf(static_cast<size_t>(rows) * cols);
  for (size_t i = 0; i < n; ++i) {
    img.read(reinterpret_cast<char*>(buf.data()), buf.size());
    std::vector<float> px(buf.size());
    for (size_t j = 0; j < buf.size(); ++j) px[j] = buf[j] / 255.0f;
    ts.images.push_back(std::move(px));
    uint8_t l;
    lbl.read(reinterpret_cast<char*>(&l), 1);
    ts.labels.push_back(l);
  }
  return ts;
}

struct Latency {
  double mean_us, p50_us, p95_us;
  int iterations;
};

Latency measure_latency(infercore::IModel& model, const TestSet& ts, int iterations) {
  infercore::Tensor in({1, 1, 28, 28});
  std::vector<double> us;
  us.reserve(iterations);
  for (int i = 0; i < 20; ++i) {  // warmup
    std::copy(ts.images[i % ts.images.size()].begin(),
              ts.images[i % ts.images.size()].end(), in.data());
    model.run(in);
  }
  for (int i = 0; i < iterations; ++i) {
    std::copy(ts.images[i % ts.images.size()].begin(),
              ts.images[i % ts.images.size()].end(), in.data());
    const auto t0 = std::chrono::steady_clock::now();
    model.run(in);
    const auto t1 = std::chrono::steady_clock::now();
    us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
  }
  std::sort(us.begin(), us.end());
  Latency l;
  l.iterations = iterations;
  l.mean_us = std::accumulate(us.begin(), us.end(), 0.0) / us.size();
  l.p50_us = us[us.size() / 2];
  l.p95_us = us[static_cast<size_t>(us.size() * 0.95)];
  return l;
}

int top1(const infercore::Tensor& t) {
  int best = 0;
  for (size_t i = 1; i < t.numel(); ++i)
    if (t[i] > t[best]) best = static_cast<int>(i);
  return best;
}

void sql_exec(sqlite3* db, const char* sql) {
  char* err = nullptr;
  if (sqlite3_exec(db, sql, nullptr, nullptr, &err) != SQLITE_OK) {
    std::string msg = err ? err : "?";
    sqlite3_free(err);
    throw std::runtime_error("sqlite: " + msg);
  }
}

void sql_insert(sqlite3* db, const std::string& sql) { sql_exec(db, sql.c_str()); }

std::string esc(const std::string& s) {
  std::string out;
  for (char c : s) out += (c == '\'') ? "''" : std::string(1, c);
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  std::string dir = "models";
  std::string db_path = "bench_results.db";
  int iterations = 500;
  int eval_n = 1000;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string k = argv[i];
    const std::string v = argv[i + 1];
    if (k == "--models-dir") dir = v;
    else if (k == "--db") db_path = v;
    else if (k == "--iterations") iterations = std::stoi(v);
    else if (k == "--eval-n") eval_n = std::stoi(v);
    else { std::fprintf(stderr, "unknown arg %s\n", k.c_str()); return 1; }
  }

  const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
  std::printf("InferCore benchmark\n");
  std::printf("  hardware_concurrency: %u\n", hw);

  TestSet ts;
  try {
    ts = load_mnist(dir, static_cast<size_t>(std::max(eval_n, iterations)));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n(download the MNIST test set first)\n", e.what());
    return 1;
  }
  std::printf("  MNIST test images loaded: %zu\n", ts.images.size());

  const std::string graph = dir + "/mnist_cnn.json";
  const std::string w32 = dir + "/mnist_fp32.weights";
  const std::string w8 = dir + "/mnist_int8.weights";
  const std::string onnx = dir + "/mnist_cnn.onnx";

  auto fp32_1t = infercore::load_native_model(graph, w32, infercore::Precision::kFp32, 1);
  auto fp32_mt = infercore::load_native_model(graph, w32, infercore::Precision::kFp32, hw);
  auto int8_mt = infercore::load_native_model(graph, w8, infercore::Precision::kInt8, hw);

  // ---- latency ----
  const Latency l_fp32_1t = measure_latency(*fp32_1t, ts, iterations);
  const Latency l_fp32 = measure_latency(*fp32_mt, ts, iterations);
  const Latency l_int8 = measure_latency(*int8_mt, ts, iterations);
  std::printf("\nLatency (batch 1, %d iterations, after 20 warmup)\n", iterations);
  std::printf("  native fp32 (1 thread) : mean %8.1f us | p50 %8.1f | p95 %8.1f\n",
              l_fp32_1t.mean_us, l_fp32_1t.p50_us, l_fp32_1t.p95_us);
  std::printf("  native fp32 (%2u threads): mean %8.1f us | p50 %8.1f | p95 %8.1f\n",
              hw, l_fp32.mean_us, l_fp32.p50_us, l_fp32.p95_us);
  std::printf("  native int8 (%2u threads): mean %8.1f us | p50 %8.1f | p95 %8.1f\n",
              hw, l_int8.mean_us, l_int8.p50_us, l_int8.p95_us);

  bool have_onnx = false;
  Latency l_onnx{};
  std::unique_ptr<infercore::IModel> onnx_model;
  try {
    onnx_model = infercore::load_onnx_model(onnx);
    have_onnx = true;
    l_onnx = measure_latency(*onnx_model, ts, iterations);
    std::printf("  onnxruntime fp32       : mean %8.1f us | p50 %8.1f | p95 %8.1f\n",
                l_onnx.mean_us, l_onnx.p50_us, l_onnx.p95_us);
  } catch (const std::exception& e) {
    std::printf("  onnxruntime fp32       : unavailable (%s)\n", e.what());
  }

  // ---- per-op breakdown (native) ----
  auto op_report = [&](infercore::IModel& m, const char* tag) {
    std::vector<std::pair<std::string, double>> agg;
    std::vector<size_t> counts;
    for (int i = 0; i < 50; ++i) {
      infercore::Tensor in({1, 1, 28, 28}, ts.images[i]);
      m.run(in);
      const auto& timings = m.last_timings();
      if (agg.empty()) {
        agg.resize(timings.size());
        counts.assign(timings.size(), 0);
        for (size_t j = 0; j < timings.size(); ++j)
          agg[j].first = timings[j].op_name + " (" + timings[j].op_type + ")";
      }
      for (size_t j = 0; j < timings.size(); ++j) {
        agg[j].second += timings[j].micros;
        counts[j]++;
      }
    }
    std::printf("\nPer-op mean latency [%s] over 50 runs\n", tag);
    for (size_t j = 0; j < agg.size(); ++j)
      std::printf("  %-24s %8.1f us\n", agg[j].first.c_str(), agg[j].second / counts[j]);
    return agg;
  };
  const auto ops_fp32 = op_report(*fp32_mt, "fp32");
  const auto ops_int8 = op_report(*int8_mt, "int8");

  // ---- accuracy + output diffs ----
  int correct32 = 0, correct8 = 0;
  double max_diff_q = 0.0, max_diff_onnx = 0.0;
  infercore::Tensor in({1, 1, 28, 28});
  for (int i = 0; i < eval_n; ++i) {
    std::copy(ts.images[i].begin(), ts.images[i].end(), in.data());
    infercore::Tensor a = fp32_mt->run(in);
    infercore::Tensor b = int8_mt->run(in);
    if (top1(a) == ts.labels[i]) ++correct32;
    if (top1(b) == ts.labels[i]) ++correct8;
    for (size_t j = 0; j < a.numel(); ++j)
      max_diff_q = std::max(max_diff_q, static_cast<double>(std::fabs(a[j] - b[j])));
    if (have_onnx && i < 50) {
      infercore::Tensor c = onnx_model->run(in);
      for (size_t j = 0; j < a.numel(); ++j)
        max_diff_onnx = std::max(max_diff_onnx, static_cast<double>(std::fabs(a[j] - c[j])));
    }
  }
  const double acc32 = 100.0 * correct32 / eval_n;
  const double acc8 = 100.0 * correct8 / eval_n;
  std::printf("\nAccuracy on %d MNIST test images\n", eval_n);
  std::printf("  fp32: %.2f%% (%d/%d)\n", acc32, correct32, eval_n);
  std::printf("  int8: %.2f%% (%d/%d)  | accuracy drop: %.2f pp\n", acc8, correct8, eval_n,
              acc32 - acc8);
  std::printf("  max abs output diff fp32 vs int8: %.4f\n", max_diff_q);
  if (have_onnx)
    std::printf("  max abs output diff native vs ONNX Runtime (50 inputs): %.3e\n",
                max_diff_onnx);

  // ---- sizes ----
  const size_t sz32 = infercore::file_size_bytes(w32);
  const size_t sz8 = infercore::file_size_bytes(w8);
  std::printf("\nModel size on disk\n");
  std::printf("  fp32 weights: %zu bytes\n", sz32);
  std::printf("  int8 weights: %zu bytes (%.2fx smaller)\n", sz8,
              static_cast<double>(sz32) / sz8);
  if (have_onnx) std::printf("  onnx model  : %zu bytes\n", infercore::file_size_bytes(onnx));

  // ---- persist to SQLite ----
  sqlite3* db = nullptr;
  if (sqlite3_open(db_path.c_str(), &db) != SQLITE_OK) {
    std::fprintf(stderr, "cannot open db %s\n", db_path.c_str());
    return 1;
  }
  sql_exec(db,
           "CREATE TABLE IF NOT EXISTS runs("
           "id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT DEFAULT CURRENT_TIMESTAMP,"
           "hw_threads INT, iterations INT, eval_n INT);"
           "CREATE TABLE IF NOT EXISTS latency("
           "run_id INT, backend TEXT, precision TEXT, threads INT,"
           "mean_us REAL, p50_us REAL, p95_us REAL);"
           "CREATE TABLE IF NOT EXISTS accuracy("
           "run_id INT, precision TEXT, n INT, correct INT, accuracy_pct REAL);"
           "CREATE TABLE IF NOT EXISTS op_times("
           "run_id INT, precision TEXT, op TEXT, mean_us REAL);"
           "CREATE TABLE IF NOT EXISTS artifacts(run_id INT, name TEXT, bytes INT);");
  char buf[512];
  std::snprintf(buf, sizeof(buf),
                "INSERT INTO runs(hw_threads, iterations, eval_n) VALUES(%u, %d, %d);",
                hw, iterations, eval_n);
  sql_insert(db, buf);
  const int64_t run_id = sqlite3_last_insert_rowid(db);
  auto ins_lat = [&](const char* backend, const char* prec, unsigned threads,
                     const Latency& l) {
    std::snprintf(buf, sizeof(buf),
                  "INSERT INTO latency VALUES(%lld, '%s', '%s', %u, %.3f, %.3f, %.3f);",
                  (long long)run_id, backend, prec, threads, l.mean_us, l.p50_us,
                  l.p95_us);
    sql_insert(db, buf);
  };
  ins_lat("native", "fp32", 1, l_fp32_1t);
  ins_lat("native", "fp32", hw, l_fp32);
  ins_lat("native", "int8", hw, l_int8);
  if (have_onnx) ins_lat("onnxruntime", "fp32", 0, l_onnx);
  std::snprintf(buf, sizeof(buf), "INSERT INTO accuracy VALUES(%lld, 'fp32', %d, %d, %.4f);",
                (long long)run_id, eval_n, correct32, acc32);
  sql_insert(db, buf);
  std::snprintf(buf, sizeof(buf), "INSERT INTO accuracy VALUES(%lld, 'int8', %d, %d, %.4f);",
                (long long)run_id, eval_n, correct8, acc8);
  sql_insert(db, buf);
  for (const auto& [op, us] : ops_fp32) {
    std::snprintf(buf, sizeof(buf), "INSERT INTO op_times VALUES(%lld, 'fp32', '%s', %.3f);",
                  (long long)run_id, esc(op).c_str(), us / 50.0);
    sql_insert(db, buf);
  }
  for (const auto& [op, us] : ops_int8) {
    std::snprintf(buf, sizeof(buf), "INSERT INTO op_times VALUES(%lld, 'int8', '%s', %.3f);",
                  (long long)run_id, esc(op).c_str(), us / 50.0);
    sql_insert(db, buf);
  }
  std::snprintf(buf, sizeof(buf), "INSERT INTO artifacts VALUES(%lld, 'mnist_fp32.weights', %zu);",
                (long long)run_id, sz32);
  sql_insert(db, buf);
  std::snprintf(buf, sizeof(buf), "INSERT INTO artifacts VALUES(%lld, 'mnist_int8.weights', %zu);",
                (long long)run_id, sz8);
  sql_insert(db, buf);
  sqlite3_close(db);
  std::printf("\nresults written to %s (run id %lld)\n", db_path.c_str(), (long long)run_id);
  return 0;
}
