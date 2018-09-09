
#include <arrayfire_benchmark.h>
#include <benchmark/benchmark.h>
#include <arrayfire.h>

#include <cmath>

using std::vector;
using std::cbrt;
using std::to_string;
using af::array;
using af::randu;
using af::dim4;
using af::deviceGC;

struct bench_params {
  int ndims;
};

static void randuBench(benchmark::State& state, af_dtype type, bench_params params) {
  dim_t dim_size = 0;
  dim4 dims;
  switch(params.ndims) {
  case 1:
    dim_size = state.range(0);
    dims = dim4(dim_size);
    break;
  case 2:
    dim_size = sqrt(state.range(0));
    dims = dim4(dim_size, dim_size);
    break;
  case 3:
    dim_size = cbrt(state.range(0));
    dims = dim4(dim_size, dim_size, dim_size);
    break;
  }

  // Perform 2 randu operations to avoid allocation during test
  array a = randu(dims, type);
  a = randu(dims, type);
  af::sync();

  for (auto _ : state) {
    array a = randu(dims, type);
    af::sync();
  }
  state.counters["bytes"] =
    benchmark::Counter(a.bytes(),
                       benchmark::Counter::kIsIterationInvariantRate,
                       benchmark::Counter::OneK::kIs1024);
  deviceGC();
}


static void randuDimsBench(benchmark::State& state) {
  dim_t dim_size = 0;
  dim4 dims(state.range(0), state.range(1));

  // run randu twice to avoid allocating in the test
  array a = randu(dims, f32);
  a = randu(dims, f32);
  af::sync();
  for (auto _ : state) {
    array a = randu(dims, f32);
    af::sync();
  }
  state.counters["bytes"] =
    benchmark::Counter(a.bytes(),
                       benchmark::Counter::kIsIterationInvariantRate,
                       benchmark::Counter::OneK::kIs1024);
  deviceGC();
}

int main(int argc, char** argv) {

  // Types to benchmark
  vector<af_dtype> types = { f32,
                             c32,
                             f64,
                             c64,
                             b8 ,
                             s32,
                             u32,
                             u8 ,
                             s64,
                             u64,
                             s16,
                             u16};

  // test randu for all types and dimensions
  //for (auto& type : types) {
    for (int ndims = 1; ndims < 3; ndims++) {
      bench_params params = {ndims};
      //benchmark::RegisterBenchmark(("randu" + to_string(ndims) + "D/" + to_string(type)).c_str(), randuBench, params)->Range(2,(1<<26))->ArgNames({"elements"});
      af::benchmark::RegisterBenchmark(("randu" + to_string(ndims) + "D").c_str(), types, randuBench, params)
        ->Range(2,(1<<26))
        ->ArgNames({"elements"});
    }
    //}

  benchmark::RegisterBenchmark("randuDimsBench", randuDimsBench)->ArgNames({"d0", "d1"})
    ->Args({1,  1<<27})
    ->Args({2,  1<<26})
    ->Args({4,  1<<25})
    ->Args({8,  1<<24})
    ->Args({16, 1<<23})
    ->Args({32, 1<<22})
    ->Args({64, 1<<21})
    ->Args({128, 1<<20})
    ->Args({256, 1<<19});

  benchmark::Initialize(&argc, argv);

  af::benchmark::AFReporter r;
  benchmark::RunSpecifiedBenchmarks(&r);
}
