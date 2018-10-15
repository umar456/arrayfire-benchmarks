#include <arrayfire_benchmark.h>
#include <arrayfire.h>
#include <vector>
#include <iostream>

using af::array;
using af::randu;
using af::constant;
using af::sum;

using std::cout;
using std::endl;
using std::vector;

void convBench(benchmark::State& state, af_dtype type) {
  int w = state.range(0);
  int h = state.range(1);
  int c = 1;
  int n = state.range(2);

  //array im = randu(w, h, 1, n, type);
  array im = constant(1.f, w, h, 1, n, type);
  cout << im.dims() << endl;

  int win_sz = state.range(3);
  //array filt = randu(win_sz, win_sz, type);
  array filt = constant(1.f / (win_sz * win_sz) , win_sz, win_sz);

  af::array res;
  convolve2(im, filt);
  af::sync();
  for(auto _ : state) {
    convolve2(im, filt);
    af::sync();
  }
  //af_print(res);
}

int main(int argc, char** argv) {
  vector<af_dtype> types = {f32};

  benchmark::Initialize(&argc, argv);
  af::benchmark::RegisterBenchmark("conv_relu_pool", types, convBench)
    ->RangeMultiplier(4)
    ->Ranges({{32, 1<<10}, {32, 1<<10}, {1, 512}, {5, 5}})
    //->Ranges({{8,8}, {8, 8}, {1, 1}, {5, 5}})
    ->ArgNames({"dim0", "dim1", "batchsize", "dim_filt"});

  af::benchmark::AFReporter r;
  af::benchmark::AFJSONReporter jsr;
  benchmark::RunSpecifiedBenchmarks(&r, &jsr);

}


