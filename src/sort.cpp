
#include <arrayfire_benchmark.h>
#include <arrayfire.h>
#include <vector>

using af::array;
using af::randu;
using af::deviceGC;
using af::deviceMemInfo;
using std::vector;
using af::sum;

void sortBench(benchmark::State& state, af_dtype type) {
    af::dim4 dataDims(state.range(0), state.range(1));
    unsigned sortDim = state.range(2);
    array input = randu(dataDims, type);
    array output;
    size_t alloc_bytes, alloc_buffers, lock_bytes, lock_buffers;
    deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);
    af::sync();

    output = sort(input, sortDim);
    af::sync();

    for(auto _ : state) {
        output = sort(input);
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
    af::benchmark::RegisterBenchmark("sort", types, sortBench)
        ->Ranges({{8, 1<<12}, {8, 1<<12}, {0, 1}})
        ->ArgNames({"dim0", "dim1", "sortDim"})
        ->Unit(benchmark::kMicrosecond);

    af::benchmark::AFReporter r;
    benchmark::RunSpecifiedBenchmarks(&r);
}
