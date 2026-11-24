# InferCore

**Small, fast C++ runtime for ML inference. Native kernels, ONNX Runtime backend, INT8.**

InferCore is a dependency-light C++17 inference and orchestration runtime. It loads a
model described by a small JSON graph, executes it with its own native kernels
(fp32 or INT8) through a thread-pool scheduler, and can run the same model through
ONNX Runtime for validation. Everything is packaged as a shared library with a C ABI.

The repo ships a real CNN trained on MNIST (`tools/train_model.py`, 97.28% test
accuracy) so every benchmark and test below runs against an actual model.

## Architecture

```
            model JSON (graph)          ICW1 / ICQ1 weights          .onnx
                    |                          |                        |
                    v                          v                        v
            +-------------------------------------------------------------+
            |                        InferCore                             |
            |  Graph loader (zero-dep JSON) -> topological scheduler       |
            |  +----------------------+   +---------------------------+    |
            |  | Native backend       |   | ONNX Runtime backend      |    |
            |  |  fp32: im2col+GEMM   |   |  (Ort::Session, C++ API)  |    |
            |  |  int8: PTQ, i32 acc  |   +---------------------------+    |
            |  +----------------------+                                    |
            |  ThreadPool: intra-op GEMM rows + independent graph nodes    |
            +-------------------------------------------------------------+
                    | libinfercore.so / infercore.dll (C ABI)
        ic_load / ic_run / ic_free      (see include/infercore/infercore.h)
```

- **Operator graph + scheduler** — `Tensor`, an `Op` interface (`Conv2D`, `ReLU`,
  `MaxPool`, `Gemm`, `Softmax`), a `Graph` with topological execution, and an
  executor with a thread pool that runs independent nodes and GEMM row blocks in
  parallel. Per-op timings are recorded on every run.
- **Two backends, golden-tested** — the native backend reads `ICW1`/`ICQ1` weight
  files; the ONNX Runtime backend runs the exported `mnist_cnn.onnx`. A GoogleTest
  golden test asserts the outputs match within `1e-4` (measured: `4.2e-07`).
- **INT8 post-training quantization** — per-output-channel symmetric weight
  quantization, dynamic activation quantization, int8 x int8 -> int32 GEMM.
- **Shared library + C ABI** — `ic_load` / `ic_load_onnx` / `ic_run` / `ic_free`,
  plus `demo/demo.cpp`, an app that `dlopen`/`LoadLibrary`s the library at runtime
  and classifies a digit with the fp32 and int8 models concurrently.
- **Benchmarks that persist** — `bench/benchmark.cpp` measures latency
  (mean/p50/p95), per-op breakdown, accuracy fp32 vs int8, output diffs, and model
  sizes, and writes every row into a SQLite database (`bench_results.db`).

## Build

```bash
# ONNX Runtime backend (optional but recommended):
#   download a release from https://github.com/microsoft/onnxruntime/releases
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DONNXRUNTIME_ROOT=/path/to/onnxruntime
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Without `-DONNXRUNTIME_ROOT`, everything except the ONNX backend still builds.

## Run

```bash
# Benchmark (writes results to SQLite):
./build/infercore_benchmark --models-dir models --iterations 1000 --eval-n 2000

# Runtime-loaded demo (dlopen):
cd build && INFERCORE_LIB=./libinfercore.so ./infercore_demo \
    ../models/mnist_cnn.json ../models/mnist_fp32.weights \
    ../models/mnist_int8.weights ../models/sample_demo.bin
```

## Measured results

All numbers are produced by `bench/benchmark.cpp`; nothing is hand-written. The
benchmark is part of CI and its raw output is uploaded as a workflow artifact on
every push (see the Actions tab).

Initial run — sandbox, 1 vCPU Intel Xeon Platinum 8375C, GCC 11.4, `-O3`
(`--iterations 2000 --eval-n 10000`, i.e. the full MNIST test set):

| Metric | fp32 (native) | int8 (native) | ONNX Runtime fp32 |
|---|---|---|---|
| Mean latency, batch 1 | 108.4 us | 162.8 us | 21.1 us |
| p50 / p95 latency | 105.6 / 115.5 us | 161.5 / 168.9 us | 20.7 / 21.1 us |
| MNIST test accuracy | 97.28% | 97.28% | - |
| Weights on disk | 21,254 B | 5,730 B (**3.71x smaller**) | - |
| Max abs diff vs native fp32 | - | 0.118 (probs) | **4.2e-07** |

Per-op breakdown (native, mean over 50 runs): conv1 34.9 us, conv2 50.1 us,
everything else < 15 us — convolutions dominate, as expected.

Honest notes:

- **INT8 here trades speed for size, not speed for accuracy.** The int8 kernels are
  portable scalar C++ (int32 accumulate); without AVX-VNNI/NEON dot-product
  intrinsics they run slower than the auto-vectorized fp32 GEMM on this CPU while
  keeping accuracy identical (0.00 pp drop on 10,000 images). Vectorized int8
  kernels are the roadmap item that turns the 3.71x size win into a latency win too.
- ONNX Runtime is faster than the native fp32 kernels (21 us vs 108 us) — it is a
  heavily SIMD-optimized production engine, and matching it is not the goal. The
  point of the ORT backend here is *validation*: the golden test proves the native
  kernels compute the same math.

## Repository layout

```
include/infercore/   public headers (tensor, kernels, ops, model, C ABI)
src/                 runtime implementation (native + ONNX backends, scheduler)
demo/                dlopen/LoadLibrary demo using only the C ABI
bench/               latency/accuracy benchmark with SQLite logging
tests/               GoogleTest: kernels, model accuracy, golden ONNX, C ABI
tools/               train_model.py (MNIST CNN training, PTQ, ONNX export)
models/              trained model: JSON graph, ICW1/ICQ1 weights, ONNX, samples
third_party/         vendored GoogleTest 1.15.2 + SQLite amalgamation
.github/workflows/   CI: Linux + Windows builds, ASan/UBSan job, benchmark job
```

## CI

Every push builds and tests on **Ubuntu and Windows**, runs the test suite under
**ASan+UBSan**, and runs the benchmark, uploading `bench_results.txt` and
`bench_results.db` as artifacts.

## Regenerating the model

```bash
pip install numpy onnx onnxruntime
python3 tools/train_model.py   # trains, quantizes, exports, self-checks vs ORT
```

## License

MIT — see [LICENSE](LICENSE). Vendored components keep their own licenses
(GoogleTest: BSD-3-Clause in `third_party/googletest/LICENSE`; SQLite: public domain).
