#include <arrayfire_benchmark.h>
#include <arrayfire.h>
#include <vector>
#include <string>
#include <utility>

using af::array;
using af::randu;
using std::vector;
using std::string;
using af::sum;
using std::pair;
using std::to_string;

static auto genRotateBench(af_interp_type rType, float angle) {
  angle = angle * M_PI / 180;
  return [rType, angle](benchmark::State &state, af_dtype type) {
    af::dim4 dataDims(state.range(0), state.range(1));
    bool crop = state.range(2);
    array input = randu(dataDims, type);
    { rotate(input, angle, crop, rType); }
    af::sync();

    for (auto _ : state) {
      rotate(input, angle, crop, rType);
      af::sync();
    }
  }; 
}

void registerBenchmark(string name, af_interp_type type, float angle) {
    static vector<af_dtype> types = {f32, u8};
    string benchName = "rotate:" + name + "/angle:" + to_string((int)angle);
    af::benchmark::RegisterBenchmark(benchName.c_str(), types, genRotateBench(type, angle))
        ->Ranges({{8, 1 << 12}, {8, 1 << 11}, {0, 1}})
        ->ArgNames({"dim0", "dim1", "crop"})
        ->Unit(benchmark::kMicrosecond);
}

int main(int argc, char** argv) {
    vector<pair<string, af_interp_type>> interpTypes = {
        { "AF_INTERP_NEAREST", AF_INTERP_NEAREST },
        { "AF_INTERP_BILINEAR", AF_INTERP_BILINEAR },
        { "AF_INTERP_LOWER", AF_INTERP_LOWER },
        { "AF_INTERP_BILINEAR_COSINE", AF_INTERP_BILINEAR_COSINE },
        { "AF_INTERP_BICUBIC", AF_INTERP_BICUBIC },
        { "AF_INTERP_BICUBIC_SPLINE", AF_INTERP_BICUBIC_SPLINE }
        // TODO: Add the rest of the interpolation types when they become supported
    };

    benchmark::Initialize(&argc, argv);
    af::benchmark::AFReporter r;

    for (auto type : interpTypes) {
        for (int x = 0; x < 360; x+=30) { 
            registerBenchmark(type.first, type.second, x);
        }
    }

    benchmark::RunSpecifiedBenchmarks(&r);
}
