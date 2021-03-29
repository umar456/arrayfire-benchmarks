
#include <arrayfire_benchmark.h>
#include <benchmark/benchmark.h>
#include <arrayfire.h>

#include <vector>

using af::array;
using af::dim4;
using af::nearestNeighbour;
using af::randu;
using af::sync;
using std::vector;

static void nnBench(benchmark::State& state, af_dtype type) {
  dim4 qdim(state.range(0), state.range(2));
  dim4 tdim(state.range(1), state.range(2));
  int dist_dim = state.range(3);
  int n_dist   = state.range(4);

  array query = af::randu(qdim, type);
  array train = af::randu(tdim, type);
  sync();
  for(auto _ : state) {
    array idx;
    array dist;
    nearestNeighbour(idx, dist, query, train, dist_dim, n_dist);
  }
}

int main(int argc, char** argv) {
  vector<af_dtype> types = {f32, f64, u32, s32};

  af::benchmark::RegisterBenchmark("NearestNeighbour", types, nnBench)
        ->Ranges({{4, 1<<14}, {2, 1<<8}, {2, 256}, {1, 1}, {1, 1}})
        ->ArgNames({"qdim0", "tdim0", "fdim", "dist_dim", "n_dist"});

  benchmark::Initialize(&argc, argv);

  af::benchmark::AFReporter r;
  benchmark::RunSpecifiedBenchmarks(&r);
}
