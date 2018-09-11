#include <arrayfire_benchmark.h>
#include <benchmark/benchmark.h>
#include <arrayfire.h>

#include <cmath>
#include <vector>

using std::vector;
using af::array;
using af::randu;
using af::anisotropicDiffusion;
using af::dim4;
using af::deviceMemInfo;
using af::deviceGC;

static
void asmBase(benchmark::State& state,
             dim4 dims,
             af_dtype type,
             float timestep,
             float conductance,
             unsigned iterations)
{
    array inImg = randu(dims, type);

   //allocate output once to bypass alloc calls
   //when smoothing function is actually called
   if (type==f64)
       array outTemp = randu(dims, f64);
   else
       array outTemp = randu(dims, f32);

  size_t alloc_bytes, alloc_buffers, lock_bytes, lock_buffers;
  deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);

  for (auto _ : state) {
      array out = anisotropicDiffusion(inImg, timestep, conductance, iterations);
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
    unsigned dim   = state.range(0);
    unsigned iters = state.range(1);
    asmBase(state, af::dim4(dim, dim), type, 0.0105833, 0.35, iters);
}

int main(int argc, char** argv)
{
    vector<af_dtype> types = {f32, u32, u16, u8};

    af::benchmark::RegisterBenchmark("asm_ImageSizes", types, asmBench)
        ->RangeMultiplier(2)
        ->Ranges({{16, 8192}, {32, 32}})
        ->ArgNames({"dim", "asm_iterations"})
        ->Iterations(100)
        ->Unit(benchmark::kMillisecond);

    af::benchmark::RegisterBenchmark("asm_iterations", types, asmBench)
        ->RangeMultiplier(2)
        ->Ranges({{2048, 2048}, {2, 128}})
        ->ArgNames({"dim", "asm_iterations"})
        ->Iterations(100)
        ->Unit(benchmark::kMillisecond);

    benchmark::Initialize(&argc, argv);

    af::benchmark::AFReporter r;
    benchmark::RunSpecifiedBenchmarks(&r);
}
