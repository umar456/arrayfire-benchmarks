
#include <arrayfire_benchmark.h>
#include <arrayfire.h>
#include <vector>

using af::array;
using af::randu;
using af::sum;

using std::vector;

void piBench(benchmark::State& state, af_dtype type) {
  int elements = state.range(0);
  array x = randu(elements, type); array y = randu(elements, type);
  float pi = 4 * sum<float>(x * x + y * y < 1.0f)/elements;
  for(auto _ : state) {
    array x = randu(elements, type); array y = randu(elements, type);
    pi = 4 * sum<float>(x * x + y * y < 1.0f)/elements;
  }
  state.counters["pi"] = pi;
}

int main(int argc, char** argv) {
  vector<af_dtype> types = {f32, f64};

  benchmark::Initialize(&argc, argv);
  af::benchmark::RegisterBenchmark("pi", types, piBench)
    ->RangeMultiplier(4)
    ->Range(8, 1<<26)
    ->ArgName("elements");

  af::benchmark::AFReporter r;
  benchmark::RunSpecifiedBenchmarks(&r);
}
