#include <arrayfire_benchmark.h>
#include <arrayfire.h>
#include <vector>
#include <string>
#include <utility>

using af::array;
using af::randu;
using af::deviceGC;
using af::deviceMemInfo;
using std::vector;
using std::string;
using af::sum;
using std::pair;

static auto genNorm(af::normType nType) {
    return [nType](benchmark::State& state, af_dtype type) {
        af::dim4 dataDims(state.range(0));
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
    }; 
}

void registerBenchmark(std::string name, af::normType type) {
    static vector<af_dtype> types = {f32, f64};
    af::benchmark::RegisterBenchmark(("norm:" + name).c_str() , types, genNorm(type))
        ->Ranges({{8, 1<<24}})
        ->ArgNames({"dim0"})
        ->Unit(benchmark::kMicrosecond);
}

int main(int argc, char** argv) {
    vector<pair<string, af::normType>> normTypes = {
        { "AF_NORM_VECTOR_1", AF_NORM_VECTOR_1 },
        { "AF_NORM_VECTOR_INF", AF_NORM_VECTOR_INF },
        { "AF_NORM_VECTOR_2", AF_NORM_VECTOR_2 },
        { "AF_NORM_VECTOR_P", AF_NORM_VECTOR_P },
        { "AF_NORM_MATRIX_1", AF_NORM_MATRIX_1 },
        { "AF_NORM_MATRIX_INF", AF_NORM_MATRIX_INF },
        { "AF_NORM_MATRIX_L_PG", AF_NORM_MATRIX_L_PQ },
        { "AF_NORM_EUCLID", AF_NORM_EUCLID }
    };

    benchmark::Initialize(&argc, argv);
    af::benchmark::AFReporter r;

    for (auto type : normTypes) {
        registerBenchmark(type.first, type.second); 
    }
    benchmark::RunSpecifiedBenchmarks(&r);
}
