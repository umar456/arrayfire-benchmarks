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

  size_t alloc_bytes, alloc_buffers, lock_bytes, lock_buffers;
  deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);

  for (auto _ : state) {
      if (fftDim==3)
          array out = fft3(in);
      else if (fftDim==2)
          array out = fft2(in);
      else
          array out = fft(in);
      af::sync();
  }

  size_t alloc_bytes2, alloc_buffers2, lock_bytes2, lock_buffers2;
  deviceMemInfo(&alloc_bytes2, &alloc_buffers2, &lock_bytes2, &lock_buffers2);

  state.counters["alloc_bytes"] = alloc_bytes2 - alloc_bytes;
  state.counters["alloc_buffers"] = alloc_buffers2 - alloc_buffers;
  deviceGC();
}

static
void fftBench(benchmark::State& state, af_dtype type)
{
    unsigned dim = state.range(0);
    unsigned fftDim = state.range(1);
    af::dim4 inDims(1);
    for (int i=0; i<fftDim; ++i)
        inDims[i] = dim;
    fftBase(state, inDims, type, fftDim);
}

int main(int argc, char** argv)
{
    vector<af_dtype> types = {f64, f32};

    //warm up: causes to cache kernels
    //helps in offsetting the skewed first run time
    fft(randu(100), f32);
    fft2(randu(10, 10), f32);
    fft3(randu(5, 5, 2), f32);

    af::benchmark::RegisterBenchmark("fft1", types, fftBench)
        ->RangeMultiplier(2)
        ->Ranges({{64, 1<<19}, {1, 1}})
        ->ArgNames({"dim", "fft_dim"})
        ->Iterations(20)
        ->Unit(benchmark::kMicrosecond);

    af::benchmark::RegisterBenchmark("fft2", types, fftBench)
        ->RangeMultiplier(2)
        ->Ranges({{64, 4096}, {2, 2}})
        ->ArgNames({"dim", "fft_dim"})
        ->Iterations(100)
        ->Unit(benchmark::kMicrosecond);

    //TODO(pradeep) may be add more dimensions explicitly
    //              using Args member function
    af::benchmark::RegisterBenchmark("fft3", types, fftBench)
        ->RangeMultiplier(2)
        ->Ranges({{16, 64}, {3, 3}})
        ->ArgNames({"dim", "fft_dim"})
        ->Iterations(100)
        ->Unit(benchmark::kMicrosecond);

    benchmark::Initialize(&argc, argv);

    af::benchmark::AFReporter r;
    benchmark::RunSpecifiedBenchmarks(&r);
}

