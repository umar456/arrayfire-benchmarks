
#include <arrayfire_benchmark.h>
#include <benchmark/benchmark.h>
#include <arrayfire.h>

using af::array;
using af::randu;
using af::dim4;
using af::sync;
using af::deviceMemInfo;
using af::deviceGC;
using af::deviceInfo;
using af::span;

using std::vector;

static void jitBench(benchmark::State& state, af_dtype type) {

  dim4 dims(state.range(0), state.range(1), state.range(2), state.range(3));
  int nops = state.range(4);
  {
      array a = randu(dims, type);
      sync();
      for(int i = 0; i < nops; i++) {
        a += 1.0f;
      }
      a.eval();
      sync();
  }

  array a = randu(dims, type);

  sync();
  for(auto _ : state) {
    for(int i = 0; i < nops; i++) {
      a += 1.0f;
    }
    a.eval();
    sync();
  }
  state.counters["elements"] = dims.elements();
  state.counters["flops"] = benchmark::Counter(dims.elements() * nops, benchmark::Counter::kIsRate);

  state.counters["bandwidth"] = benchmark::Counter(2*a.bytes(), benchmark::Counter::kIsRate);
}


static void jitBench2(benchmark::State& state, af_dtype type) {

  dim4 dims(state.range(0), state.range(1), state.range(2), state.range(3));
  int nops = state.range(4);

  {
    array a = randu(dims, type);
    array b = randu(dims, type);
    array cond = constant(1, dims, b8);
    sync();
    for(int i = 0; i < nops; i++) {
      a += select(cond, a, b);
    }
    a.eval();
    sync();
  }

  //array blah = randu(dims*4, type);
  array a = randu(dims, type);
  array b = randu(dims, type);
  array cond = constant(1, dims, b8);
  //array b = randu(dims, type);
  //array c = randu(dims, type);
  //array d = randu(dims, type);
  sync();
  for(auto _ : state) {
    for(int i = 0; i < nops; i++) {
      a = select(cond, a, b);
    }
    a.eval();
    sync();
  }
  state.counters["elements"] = dims.elements();
}


long long calc_elements(long long d0, long long d1, long long d2, long long d3) {
  return (1<<d0) * (1<<d1) * (1<<d2) * (1<<d3);
}

void
SameSize(benchmark::internal::Benchmark* b) {

  int max_shift = 27;
  int dim2_max_shift = 8;
  int dim3_max_shift = 8;
  int min_shift = 0;
  long long elements = 1<<max_shift;

  for(long long d3 = min_shift; d3 <= dim3_max_shift; d3++ ) {
    for(long long d2 = min_shift; d2 <= dim2_max_shift; d2++ ) {
      for(long long d1 = min_shift; d1 <= max_shift; d1++ ) {
        for(long long d0 = min_shift; d0 <= max_shift; d0++ ) {
          if(elements == calc_elements(d0, d1, d2, d3)) {
            printf("%d %d %d %d\n", d0, d1, d2, d3);
            b->Args({1<<d0, 1<<d1, 1<<d2, 1<<d3, 16});
          }
        }
      }
    }
  }

}


int main(int argc, char** argv) {
  //af::setBackend(AF_BACKEND_CUDA);
  //af::setBackend(AF_BACKEND_OPENCL);

  vector<af_dtype> types = {f32};

  af::benchmark::RegisterBenchmark("jitDim0", types, jitBench)
    //->Iterations(1)
    ->Ranges({{1, 1<<28}, {1, 1}, {1, 1}, {1, 1}, {1, 1<<8}})
    ->ArgNames({"dim0", "dim1", "dim2", "dim3", "nops"})
    ->Unit(benchmark::kMicrosecond);

  af::benchmark::RegisterBenchmark("jitDim0Select", types, jitBench2)
    //->Iterations(1)
    ->Ranges({{1, 1<<28}, {1, 1}, {1, 1}, {1, 1}, {1, 1<<8}})
    ->ArgNames({"dim0", "dim1", "dim2", "dim3", "nops"})
    ->Unit(benchmark::kMicrosecond);

  af::benchmark::RegisterBenchmark("jit2", types, jitBench)
    //->Iterations(2)
    ->Ranges({{1<<13, 1<<13}, {1, 1<<16}, {1, 1}, {1, 1}, {1, 1<<8}})
    ->ArgNames({"dim0", "dim1", "dim2", "dim3", "nops"})
    ->Unit(benchmark::kMicrosecond);

  benchmark::Initialize(&argc, argv);

  af::benchmark::AFReporter r;
  benchmark::RunSpecifiedBenchmarks(&r);
}
