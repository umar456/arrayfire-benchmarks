#include <arrayfire_benchmark.h>
#include <benchmark/benchmark.h>
#include <arrayfire.h>

#include <cmath>
#include <vector>

using std::vector;
using af::array;
using af::randu;
using af::matmul;
using af::dim4;
using af::deviceMemInfo;
using af::deviceGC;

enum class Tile { none, lhs, rhs };

static
void gemmBase(benchmark::State& state,
              dim4 aDims, dim4 bDims, af_dtype type,
              Tile opt = Tile::none)
{
    array A = randu(aDims, type);
    array B = randu(bDims, type);
    af::sync();

    size_t alloc_bytes, alloc_buffers, lock_bytes, lock_buffers;
    deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);

    if (opt == Tile::lhs) {
        for (auto _ : state) {
            array tiledA = tile(A, dim4(1, 1, bDims[2], bDims[3]));
            array C = matmul(tiledA, B);
            af::sync();
        }
    } else if (opt == Tile::rhs) {
        for (auto _ : state) {
            array tiledB = tile(B, dim4(1, 1, aDims[2], aDims[3]));
            array C = matmul(A, tiledB);
            af::sync();
        }
    } else {
        for (auto _ : state) {
            array C = matmul(A, B);
            C.eval();
            af::sync();
        }
    }
    size_t alloc_bytes2, alloc_buffers2, lock_bytes2, lock_buffers2;
    deviceMemInfo(&alloc_bytes2, &alloc_buffers2, &lock_bytes2, &lock_buffers2);

    state.counters["alloc_megabytes"] = (alloc_bytes2 - alloc_bytes)/1e6;

    deviceGC();
}

int main(int argc, char** argv)
{
    using af::benchmark::RegisterBenchmark;

    vector<af_dtype> types = {f32};

    RegisterBenchmark("A[N,N,batch]xB[N,N,batch]",
                      types, [](benchmark::State& state, af_dtype type) {
                                    unsigned dim = state.range(0);
                                    unsigned bat = state.range(1);
                                    af::dim4 dims(dim, dim, bat);
                                    gemmBase(state, dims, dims, type);
                               })
        ->RangeMultiplier(2)
        ->Ranges({{64, 2048}, {2, 8}})
        ->ArgNames({"N", "batch"})
        ->Unit(benchmark::kMicrosecond);

    RegisterBenchmark("A[M,K,batch]xB[K,N,batch]",
                      types, [](benchmark::State& state, af_dtype type) {
                                    unsigned M = state.range(0);
                                    unsigned K = state.range(1);
                                    unsigned N = state.range(2);
                                    unsigned b = state.range(3);
                                    af::dim4 aDims(M, K, b);
                                    af::dim4 bDims(K, N, b);
                                    gemmBase(state, aDims, bDims, type);
                               })
        ->RangeMultiplier(2)
        ->Ranges({{64, 1024}, {64, 1024}, {64, 1024}, {2, 8}})
        ->ArgNames({"M", "K", "N", "batch"})
        ->Unit(benchmark::kMicrosecond);

    RegisterBenchmark("TILE(A[N,N])xB[N,N,batch]",
                      types, [](benchmark::State& state, af_dtype type) {
                                    unsigned N = state.range(0);
                                    unsigned b = state.range(1);
                                    af::dim4 aDims(N, N);
                                    af::dim4 bDims(N, N, b);
                                    gemmBase(state, aDims, bDims, type, Tile::lhs);
                               })
        ->RangeMultiplier(2)
        ->Ranges({{128, 128}, {2, 4096}})
        ->ArgNames({"N", "batch"})
        ->Unit(benchmark::kMicrosecond);

#if 0 //Broadcast batched batch is disabled in current master
    RegisterBenchmark("A[N,N]xB[N,N,batch]",
                      types, [](benchmark::State& state, af_dtype type) {
                                    unsigned N = state.range(0);
                                    unsigned b = state.range(1);
                                    af::dim4 aDims(N, N);
                                    af::dim4 bDims(N, N, b);
                                    gemmBase(state, aDims, bDims, type);
                               })
        ->RangeMultiplier(2)
        ->Ranges({{128, 128}, {2, 4096}})
        ->ArgNames({"N", "batch"})
        ->Unit(benchmark::kMicrosecond);
#endif

    benchmark::Initialize(&argc, argv);

    af::benchmark::AFReporter r;
    benchmark::RunSpecifiedBenchmarks(&r);
}

