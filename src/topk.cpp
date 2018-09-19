
#include <arrayfire_benchmark.h>
#include <benchmark/benchmark.h>
#include <arrayfire.h>

#include <cmath>

using af::array;
using af::deviceGC;
using af::deviceInfo;
using af::deviceMemInfo;
using af::dim4;
using af::randu;
using af::topk;
using std::cbrt;
using std::to_string;
using std::vector;

static void topkBenchBase(benchmark::State& state, dim4 dims, int k, af_dtype type) {
  //af_dtype type = type;
  array a = randu(dims, type);
  {
    // allocate output memory
    dim4 out_dims = dims;
    out_dims[0] = k;
    array vals = randu(out_dims, type);
    array idx = randu(out_dims, s32);
  }

  //printMemInfo();
  size_t alloc_bytes, alloc_buffers, lock_bytes, lock_buffers;
  deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);
  for(auto _ : state) {
    array idx;
    array vals;
    topk(vals, idx, a, k);
    af::sync();
  }
  size_t alloc_bytes2, alloc_buffers2, lock_bytes2, lock_buffers2;
  deviceMemInfo(&alloc_bytes2, &alloc_buffers2, &lock_bytes2, &lock_buffers2);

  state.counters["alloc_bytes"] = alloc_bytes2 - alloc_bytes;
  state.counters["alloc_buffers"] = alloc_buffers2 - alloc_buffers;
  deviceGC();
}

// Benchmarks the internal memory allocation as K increases
static void topkBench(benchmark::State& state, af_dtype type) {
  int k = state.range(0);
  dim4 dims = dim4(state.range(1), state.range(2));
  topkBenchBase(state, dims, k, type);
}

int main(int argc, char** argv) {
  vector<af_dtype> types = {f32, f64, u32, s32};

  af::benchmark::RegisterBenchmark("topkMemK", types, topkBench)
        ->RangeMultiplier(2)
        ->Ranges({{2, 128}, {30000, 30000}, {10, 10}})
        ->ArgNames({"[k]", "dim0", "dim1"})
        ->Iterations(1);

  af::benchmark::RegisterBenchmark("topkMemDims0", types, topkBench)
        ->Ranges({{5, 5}, {8,1<<19}, {10, 10}})
        ->ArgNames({"k", "[dim0]", "dim1"})
        ->Iterations(1);

  af::benchmark::RegisterBenchmark("topkMemDims1", types, topkBench)
        ->Ranges({{5, 5}, {100, 100}, {1, 1<<15}})
        ->ArgNames({"k", "dim0", "[dim1]"})
        ->Iterations(1);

  af::benchmark::RegisterBenchmark("topk", types, topkBench)
      ->RangeMultiplier(2)
      ->Ranges({{2, 256}, {30000, 30000}, {10, 10}})
      ->ArgNames({"[k]", "dim0", "dim1"});

  af::benchmark::RegisterBenchmark("topk", types, topkBench)
      ->RangeMultiplier(2)
      ->Ranges({{5, 5}, {8, 1<<24}, {10, 10}})
      ->ArgNames({"k", "[dim0]", "dim1"});

  af::benchmark::RegisterBenchmark("topk", types, topkBench)
      ->RangeMultiplier(2)
      ->Ranges({{5, 5}, {100, 100}, {1, 1<<15}})
      ->ArgNames({"k", "dim0", "[dim1]"});

  benchmark::Initialize(&argc, argv);

  af::benchmark::AFReporter r;
  benchmark::RunSpecifiedBenchmarks(&r);
}
