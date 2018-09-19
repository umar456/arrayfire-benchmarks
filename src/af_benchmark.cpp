
#include "arrayfire_benchmark.h"

namespace af {
  namespace benchmark {
    std::string to_string(af_dtype type) {
      std::string out;
      switch(type) {
      case f32: out = "f32"; break;
      case c32: out = "c32"; break;
      case f64: out = "f64"; break;
      case c64: out = "c64"; break;
      case b8: out = "b8"; break;
      case s32: out = "s32"; break;
      case u32: out = "u32"; break;
      case u8: out = "u8"; break;
      case s64: out = "s64"; break;
      case u64: out = "u64"; break;
      case s16: out = "s16"; break;
      case u16: out = "u16"; break;
      }
      return out;
    }

    BenchmarkCollection::BenchmarkCollection() : benchmarks_() {}

    void BenchmarkCollection::Add(::benchmark::internal::Benchmark* bench) {
      benchmarks_.push_back(bench);
    }

    // TODO(umar): Some functions are not implemented yet
    BenchmarkCollection* BenchmarkCollection::Arg(int64_t x)                                                  { for(auto& b : benchmarks_) b->Arg(x); return this; }
    BenchmarkCollection* BenchmarkCollection::Unit(::benchmark::TimeUnit unit)                                { for(auto& b : benchmarks_) b->Unit(unit); return this; }
    BenchmarkCollection* BenchmarkCollection::Range(int64_t start, int64_t limit)                             { for(auto& b : benchmarks_) b->Range(start, limit); return this; }
    BenchmarkCollection* BenchmarkCollection::DenseRange(int64_t start, int64_t limit, int step)              { for(auto& b : benchmarks_) b->DenseRange(start, limit, step); return this; }
    BenchmarkCollection* BenchmarkCollection::Args(const std::vector<int64_t>& args)                          { for(auto& b : benchmarks_) b->Args(args); return this; }
    BenchmarkCollection* BenchmarkCollection::Ranges(const std::vector<std::pair<int64_t, int64_t> >& ranges) { for(auto& b : benchmarks_) b->Ranges(ranges); return this; }
    BenchmarkCollection* BenchmarkCollection::ArgName(const std::string& name)                                { for(auto& b : benchmarks_) b->ArgName(name); return this; }
    BenchmarkCollection* BenchmarkCollection::ArgNames(const std::vector<std::string>& names)                 { for(auto& b : benchmarks_) b->ArgNames(names); return this; }
    BenchmarkCollection* BenchmarkCollection::RangeMultiplier(int multiplier)                                 { for(auto& b : benchmarks_) b->RangeMultiplier(multiplier); return this; }
    BenchmarkCollection* BenchmarkCollection::MinTime(double t)                                               { for(auto& b : benchmarks_) b->MinTime(t); return this; }
    BenchmarkCollection* BenchmarkCollection::Apply(void (*func)(::benchmark::internal::Benchmark* benchmark)){ for(auto& b : benchmarks_) b->Apply(func); return this; }
    BenchmarkCollection* BenchmarkCollection::Iterations(size_t n)                                            { for(auto& b : benchmarks_) b->Iterations(n); return this; }
  }

}
