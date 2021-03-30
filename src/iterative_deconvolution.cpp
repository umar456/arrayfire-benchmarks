#include <arrayfire_benchmark.h>
#include <benchmark/benchmark.h>
#include <arrayfire.h>

#include <cmath>
#include <vector>

using std::vector;
using af::array;
using af::randu;
using af::gaussianKernel;
using af::anisotropicDiffusion;
using af::dim4;
using af::deviceMemInfo;
using af::deviceGC;

static
void deconvBase(benchmark::State& state,
                dim4 dims, af_dtype type, int klen,
                unsigned iterations, float relaxation_factor, int algo)
{
    array inImg = randu(dims, type);
    array psf   = gaussianKernel(klen, klen, f32);

   //allocate output once to bypass alloc calls
   //when smoothing function is actually called
   if (type==f64) {
       array outTemp = randu(dims, f64);
   } else {
       array outTemp = randu(dims, f32);
   }
   af::sync();

  size_t alloc_bytes, alloc_buffers, lock_bytes, lock_buffers;
  deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);

  for (auto _ : state) {
      array out = iterativeDeconv(inImg, psf,
              iterations, relaxation_factor, (af::iterativeDeconvAlgo)algo);
      af::sync();
  }

  size_t alloc_bytes2, alloc_buffers2, lock_bytes2, lock_buffers2;
  deviceMemInfo(&alloc_bytes2, &alloc_buffers2, &lock_bytes2, &lock_buffers2);

  state.counters["alloc_bytes"] = alloc_bytes2 - alloc_bytes;
  state.counters["alloc_buffers"] = alloc_buffers2 - alloc_buffers;
  deviceGC();
}

static
void asmBench(benchmark::State& state, af_dtype type)
{
    unsigned dim0  = state.range(0);
    unsigned dim1  = state.range(1);
    unsigned klen  = state.range(2);
    unsigned iters = state.range(3);
    unsigned algo  = state.range(4);
    deconvBase(state, af::dim4(dim0, dim1), type, klen, iters, 0.05f, algo);
}

int main(int argc, char** argv)
{
    vector<af_dtype> types = {f32, u16, u8};

    af::benchmark::RegisterBenchmark("varying_iterations", types, asmBench)
        ->Iterations(20)
        ->RangeMultiplier(2)
        ->Ranges({{3648, 3648}, {2432, 2432}, {13, 13}, {2, 1<<7}, {1, 1}})
        ->Ranges({{3648, 3648}, {2432, 2432}, {13, 13}, {2, 1<<7}, {2, 2}})
        ->ArgNames({"dim0", "dim1", "psf_radius", "algo_iterations", "AlgoEnum"})
        ->Unit(benchmark::kMillisecond);

    benchmark::Initialize(&argc, argv);

    af::benchmark::AFReporter r;
    benchmark::RunSpecifiedBenchmarks(&r);
}
