
#include <arrayfire_benchmark.h>
#include <vector>
#include <arrayfire.h>

using std::vector;
using af::array;
using af::randu;

static void unifiedRandu(benchmark::State& state, af_dtype type) {
  array a = randu(state.range(0), type);
  a = randu(state.range(0), type);

  af::sync();
  for(auto _ : state) {
    a = randu(state.range(0), type);
    af::sync();
  }
}

int main(int argc, char** argv) {
  vector<af_dtype> types = {f32};
  af::benchmark::RegisterBenchmark("unified", types, unifiedRandu)
    ->Range(1, 1<<27)
    ->Unit(benchmark::kMicrosecond);

  benchmark::Initialize(&argc, argv);
  af::benchmark::AFReporter r;
  benchmark::RunSpecifiedBenchmarks(&r);
}
