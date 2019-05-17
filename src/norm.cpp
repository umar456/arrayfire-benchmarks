#include <arrayfire_benchmark.h>
#include <arrayfire.h>
#include <vector>

using af::array;
using af::randu;
using af::deviceGC;
using af::deviceMemInfo;
using std::vector;
using af::sum;

void normBench(benchmark::State& state, af_dtype type) {
    af::dim4 dataDims(state.range(0));
    af::normType nType = static_cast<af::normType>(state.range(1));
    array input = randu(dataDims, type);
    array output;
    size_t alloc_bytes, alloc_buffers, lock_bytes, lock_buffers;
    deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);
    af::sync();

    output = norm(input, nType);
    af::sync();

    for(auto _ : state) {
        output = norm(input, nType);
        af::sync();
    }

    size_t alloc_bytes2, alloc_buffers2, lock_bytes2, lock_buffers2;
    deviceMemInfo(&alloc_bytes2, &alloc_buffers2, &lock_bytes2, &lock_buffers2);

    state.counters["Bytes"] = (alloc_bytes2 - alloc_bytes);
    deviceGC(); 
}

static void CustomArguments(benchmark::internal::Benchmark* b) {
    for (int i = 0; i < 8; ++i) {
        for (int j = 2; j < 16; j+=2) {
            
            if (i != 6) b->Args({1 << j, i});
        }
    }
}

int main(int argc, char** argv) {
    vector<af_dtype> types = {f32, f64};

    benchmark::Initialize(&argc, argv);
    af::benchmark::RegisterBenchmark("norm", types, normBench)
        // ->Ranges({{8, 1<<26}})
        ->Apply(CustomArguments)
        ->ArgNames({"dim0", "normtype"})
        ->Unit(benchmark::kMicrosecond);

    af::benchmark::AFReporter r;
    benchmark::RunSpecifiedBenchmarks(&r);
}
