#include <arrayfire_benchmark.h>
#include <benchmark/benchmark.h>
#include <arrayfire.h>

#include <cmath>
#include <vector>

using std::vector;
using af::array;
using af::randu;
using af::fft;
using af::fft2;
using af::fft3;
using af::dim4;
using af::deviceMemInfo;
using af::deviceGC;

static
void fftBase(benchmark::State& state,
             dim4 dims,
             af_dtype type,
             unsigned fftDim)
{
    array in = randu(dims, type);

   //allocate output once to bypass alloc calls
   //when smoothing function is actually called
   if (type==f64)
       array outTemp = randu(dims, c64);
   else
       array outTemp = randu(dims, c32);
   af::sync();

  if (fftDim==3) {
      for (auto _ : state) {
          array out = fft3(in);
          out.eval();
      }
  } else if (fftDim==2) {
      for (auto _ : state) {
          array out = fft2(in);
          out.eval();
      }
  } else {
      for (auto _ : state) {
          array out = fft(in);
          out.eval();
      }
  }
  af::sync();

  deviceGC();
}

static
void fftBench(benchmark::State& state, af_dtype type)
{
    unsigned dim0 = state.range(0);
    unsigned dim1 = state.range(1);
    unsigned dim2 = state.range(2);
    unsigned fftDim = state.range(3);
    af::dim4 inDims(dim0, dim1, dim2);
    fftBase(state, inDims, type, fftDim);
}

int main(int argc, char** argv)
{
    vector<af_dtype> types = {f64, f32};

    af::benchmark::RegisterBenchmark("fft1", types, fftBench)
        ->RangeMultiplier(2)
        ->Ranges({{64, 1<<19}, {1, 1}, {1, 1}, {1, 1}})
        ->ArgNames({"dim0", "dim1", "dim2", "fft_dim"})
        ->Unit(benchmark::kMicrosecond);

    af::benchmark::RegisterBenchmark("fft2", types, fftBench)
        ->RangeMultiplier(2)
        ->Ranges({{1<<4, 1<<12}, {1<<4, 1<<12}, {1, 1}, {2, 2}})
        ->ArgNames({"dim0", "dim1", "dim2", "fft_dim"})
        ->Unit(benchmark::kMicrosecond);

    af::benchmark::RegisterBenchmark("fft3", types, fftBench)
        ->RangeMultiplier(2)
        ->Ranges({{1<<4, 1<<7}, {1<<4, 1<<7}, {1<<4, 1<<7}, {3, 3}})
        ->ArgNames({"dim0", "dim1", "dim2", "fft_dim"})
        ->Unit(benchmark::kMicrosecond);

    benchmark::Initialize(&argc, argv);

    af::benchmark::AFReporter r;
    benchmark::RunSpecifiedBenchmarks(&r);
}
