
#include <arrayfire.h>
#include <arrayfire_benchmark.h>
#include <benchmark/benchmark.h>

#include <cmath>
#include <cstddef>
#include <vector>

using af::array;
using af::deviceGC;
using af::deviceMemInfo;
using af::dim4;
using af::matmul;
using af::randu;
using std::vector;
using std::size_t;

enum class Tile { none, lhs, rhs };

static void addBase(benchmark::State &state, dim4 aDims) {
  deviceGC();
  array A = randu(aDims);
  array B = randu(aDims);
  array C = randu(aDims);
  af::sync();

  size_t alloc_bytes, alloc_buffers, lock_bytes, lock_buffers;
  deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);

  for (auto _ : state) {
    C = A + B;
    C.eval();
    af::sync();
  }
  size_t alloc_bytes2, alloc_buffers2, lock_bytes2, lock_buffers2;
  deviceMemInfo(&alloc_bytes2, &alloc_buffers2, &lock_bytes2, &lock_buffers2);
  state.counters["alloc_megabytes"] = (alloc_bytes2 - alloc_bytes) / 1e6;
  deviceGC();
}

static void addTiled(benchmark::State &state, dim4 aDims) {
  deviceGC();
  array A = randu(aDims);
  array B = randu(aDims[0]); // 1d
  array C = randu(aDims);
  af::sync();

  size_t alloc_bytes, alloc_buffers, lock_bytes, lock_buffers;
  deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);

  for (auto _ : state) {
    C = A + af::tile(B, 1, aDims[1], aDims[2]);
    C.eval();
    af::sync();
  }
  size_t alloc_bytes2, alloc_buffers2, lock_bytes2, lock_buffers2;
  deviceMemInfo(&alloc_bytes2, &alloc_buffers2, &lock_bytes2, &lock_buffers2);
  state.counters["alloc_megabytes"] = (alloc_bytes2 - alloc_bytes) / 1e6;
  deviceGC();
}

int main(int argc, char **argv) {
  using af::benchmark::RegisterBenchmark;

  vector<af_dtype> types = {f32};

  RegisterBenchmark("A[N,N,batch]+B[N,N,batch]", types,
                    [](benchmark::State &state, af_dtype type) {
                      unsigned dim = state.range(0);
                      unsigned bat = state.range(1);
                      af::dim4 dims(dim, dim, bat);
                      addBase(state, dims);
                    })
      ->RangeMultiplier(2)
      ->Ranges({{512, 2048}, {2, 64}})
      ->ArgNames({"N", "batch"})
      ->Unit(benchmark::kMicrosecond);

  RegisterBenchmark("A[N,N,batch]+tile(B)", types,
                    [](benchmark::State &state, af_dtype type) {
                      unsigned dim = state.range(0);
                      unsigned bat = state.range(1);
                      af::dim4 dims(dim, dim, bat);
                      addTiled(state, dims);
                    })
      ->RangeMultiplier(2)
      ->Ranges({{512, 2048}, {2, 64}})
      ->ArgNames({"N", "batch"})
      ->Unit(benchmark::kMicrosecond);

  benchmark::Initialize(&argc, argv);

  af::benchmark::AFReporter r;
  benchmark::RunSpecifiedBenchmarks(&r);
}
