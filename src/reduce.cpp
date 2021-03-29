#include <arrayfire_benchmark.h>
#include <arrayfire.h>
#include <vector>


using af::array;
using af::randu;

using std::vector;
using af::sum;

void sumBench(benchmark::State& state, af_dtype type) {
    af::dim4 dataDimensions(state.range(0));
    array x = randu(dataDimensions, type);
    af::sync();
    array output;
    output = sum(x);
    output.eval();
    af::sync();

    for (auto _ : state) {
        output = sum(x);
        output.eval();
        af::sync();
    }
}

int main(int argc, char** argv) {
    vector<af_dtype> types = {f32, f64};

    benchmark::Initialize(&argc, argv);
    af::benchmark::RegisterBenchmark("sum", types, sumBench)
        // ->RangeMultiplier(2)
        ->Ranges({{8, 1<<26}, {8, 1<<12}})
        ->ArgNames({"dim0", "dim1"})
        //->Iterations(2)
        ->Unit(benchmark::kMicrosecond);

    //af::benchmark::AFJSONReporter r;
    af::benchmark::AFReporter r;
    benchmark::RunSpecifiedBenchmarks(&r);
}
