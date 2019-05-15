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
    array A = array(aDims, type);
    array B = array(bDims, type);
    A.eval();
    B.eval();
    af::sync();
    {
        if (opt == Tile::lhs) {
          array tiledA = tile(A, dim4(1, 1, aDims[2], aDims[3]));
          tiledA.eval();
        } else if(opt == Tile::rhs) {
          array tiledB = tile(B, dim4(1, 1, bDims[2], bDims[3]));
          tiledB.eval();
        }
        array C = matmul(A, B);
        af::sync();
    }

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
            af::sync();
        }
    }
    size_t alloc_bytes2, alloc_buffers2, lock_bytes2, lock_buffers2;
    deviceMemInfo(&alloc_bytes2, &alloc_buffers2, &lock_bytes2, &lock_buffers2);

    state.counters["bytes"] = (alloc_bytes2 - alloc_bytes);

    size_t m = aDims[0];
    size_t n = aDims[1];
    size_t k = bDims[1];
    size_t batches = aDims[2];
    size_t elements = m * n * k * batches;
    state.counters["flops"] = benchmark::Counter(
        2.0 * elements, benchmark::Counter::kIsIterationInvariantRate,
        benchmark::Counter::OneK::kIs1024);

    deviceGC();
}

int main(int argc, char** argv)
{
    using af::benchmark::RegisterBenchmark;

    vector<af_dtype> types = {f32};

    RegisterBenchmark("Matmul", types,
                      [](benchmark::State &state, af_dtype type) {
                        unsigned M = state.range(0);
                        unsigned N = state.range(1);
                        unsigned K = state.range(2);
                        unsigned b = state.range(3);
                        af::dim4 aDims(M, N, b);
                        af::dim4 bDims(N, K);
                        gemmBase(state, aDims, bDims, type);
                      })
        ->Ranges({{64, 2048}, {64, 1024}, {64, 1024}, {2, 128}})
        ->ArgNames({"M", "N", "K", "batch"})
        ->Unit(benchmark::kMicrosecond);

    RegisterBenchmark("ExplicitTileA",
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

    benchmark::Initialize(&argc, argv);

    af::benchmark::AFReporter r;
    benchmark::RunSpecifiedBenchmarks(&r);
}

