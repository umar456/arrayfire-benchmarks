#include <arrayfire_benchmark.h>
#include <arrayfire.h>
#include <vector>

using af::array;
using af::randu;
using af::deviceGC;
using af::deviceMemInfo;
using std::vector;
using af::sum;

void stdevBench(benchmark::State& state, af_dtype type) {
    af::dim4 aDim(state.range(0), state.range(1));
    array a = randu(aDim, type);
    unsigned stdevDim = state.range(2) - 1;
    array output;
    size_t alloc_bytes, alloc_buffers, lock_bytes, lock_buffers;
    deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);
    af::sync();

    output = stdev(a, stdevDim);
    af::sync();

    for(auto _ : state) {
        output = stdev(a, stdevDim);
        af::sync();
    }

    size_t alloc_bytes2, alloc_buffers2, lock_bytes2, lock_buffers2;
    deviceMemInfo(&alloc_bytes2, &alloc_buffers2, &lock_bytes2, &lock_buffers2);

    state.counters["Bytes"] = (alloc_bytes2 - alloc_bytes);
    deviceGC(); 
}

int main(int argc, char** argv) {
    vector<af_dtype> types = {f32, f64};

    benchmark::Initialize(&argc, argv);
    af::benchmark::RegisterBenchmark("stdev", types, stdevBench)
        ->Ranges({{8, 1<<12}, {8, 1<<12}, {0, 1}})
        ->ArgNames({"dim0", "dim1", "stdevDim"})
        ->Unit(benchmark::kMicrosecond);

    af::benchmark::AFReporter r;
    benchmark::RunSpecifiedBenchmarks(&r);
}
