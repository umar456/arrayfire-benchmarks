
#include <arrayfire.h>
#include <arrayfire_benchmark.h>

using af::array;
using af::randu;

auto meanBench(benchmark::State& state, af_dtype type, dim_t dim) {
  array input = randu(state.range(0), state.range(1), type);
  mean(input, dim);
  af::sync();

  for(auto _ : state) {
    mean(input, dim);
    af::sync();
  }
}

int main(int argc, char **argv) {

  af::benchmark::Initialize(&argc, argv);

  af::benchmark::RegisterBenchmark("meand0", {(af_dtype)f32}, meanBench, 0)
      ->Ranges({{512, 1<<16}, {8, 1<<8}})
      ->Unit(benchmark::kMicrosecond)
      ->Iterations(10);

  af::benchmark::RegisterBenchmark("meand1", {(af_dtype)f32}, meanBench, 1)
      ->Ranges({{512, 1 << 16}, {8, 1 << 8}})
      ->Unit(benchmark::kMicrosecond)
      ->Iterations(10);

  af::benchmark::AFReporter r;
  benchmark::RunSpecifiedBenchmarks(&r);
}
