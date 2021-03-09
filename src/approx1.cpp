
#include <arrayfire_benchmark.h>
#include <arrayfire.h>

#include <vector>

using namespace std;
using namespace af::benchmark;
using af::dim4;

static void approx1Bench(benchmark::State &state, af_dtype type) {

  dim4 dims = dim4(state.range(0));
  dim4 pdims = dim4(state.range(1));

  auto input = iota(dims, dim4(1), type);
  auto pos = iota(pdims, dim4(1), type);

  for (auto _ : state) {
    auto outBatch = approx1(input, pos, AF_INTERP_LINEAR);
    af::sync();
  }

}

int main(int argc, char **argv) {

  vector<af_dtype> types = {f32, f64};

  RegisterBenchmark("approx1", types, approx1Bench)
      ->RangeMultiplier(2)
      ->Ranges({{1<<12, 1 << 21}, {1<<12, 1 << 21}})
      ->ArgNames({"dim0", "pos_dims"})
      ->Unit(::benchmark::kMicrosecond);

  af::benchmark::Initialize(&argc, argv);

  af::benchmark::AFReporter r;
  benchmark::RunSpecifiedBenchmarks(&r);
}
