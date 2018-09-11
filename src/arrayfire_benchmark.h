
#include <string>
#include <vector>

#include <af/defines.h>
#include <benchmark/benchmark.h>
#include <af/device.h>
#include <af/util.h>

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

  // Same as the google Benchmark class but performs the same operation over a set of benchmarks
  class BenchmarkCollection {
    std::vector<::benchmark::internal::Benchmark*> benchmarks_;

  public:
    BenchmarkCollection(): benchmarks_() {}

    void Add(::benchmark::internal::Benchmark* bench) {
      benchmarks_.push_back(bench);
    }

    // TODO(umar): Some functions are not implemented yet
    BenchmarkCollection* Arg(int64_t x)                                                  { for(auto& b : benchmarks_) b->Arg(x); return this; }
    BenchmarkCollection* Unit(::benchmark::TimeUnit unit)                                { for(auto& b : benchmarks_) b->Unit(unit); return this; }
    BenchmarkCollection* Range(int64_t start, int64_t limit)                             { for(auto& b : benchmarks_) b->Range(start, limit); return this; }
    BenchmarkCollection* DenseRange(int64_t start, int64_t limit, int step = 1)          { for(auto& b : benchmarks_) b->DenseRange(start, limit, step); return this; }
    BenchmarkCollection* Args(const std::vector<int64_t>& args)                          { for(auto& b : benchmarks_) b->Args(args); return this; }
    BenchmarkCollection* Ranges(const std::vector<std::pair<int64_t, int64_t> >& ranges) { for(auto& b : benchmarks_) b->Ranges(ranges); return this; }
    BenchmarkCollection* ArgName(const std::string& name)                                { for(auto& b : benchmarks_) b->ArgName(name); return this; }
    BenchmarkCollection* ArgNames(const std::vector<std::string>& names)                 { for(auto& b : benchmarks_) b->ArgNames(names); return this; }
    BenchmarkCollection* RangeMultiplier(int multiplier)                                 { for(auto& b : benchmarks_) b->RangeMultiplier(multiplier); return this; }
    BenchmarkCollection* MinTime(double t)                                               { for(auto& b : benchmarks_) b->MinTime(t); return this; }
    BenchmarkCollection* Iterations(size_t n)                                            { for(auto& b : benchmarks_) b->Iterations(n); return this; }
  };

  class AFReporter : public ::benchmark::ConsoleReporter {
    virtual bool ReportContext(const Context& context) {
      ::benchmark::ConsoleReporter::ReportContext(context);
      char device_name[256];
      char platform[256];
      char toolkit[256];
      char compute[256];
      deviceInfo(device_name, platform, toolkit, compute);

      int major, minor, patch;
      af_get_version(&major, &minor, &patch);
      const char* revision = af_get_revision();

      printf("Context:\n"
             "ArrayFire v%d.%d.%d (%s) (Backend: %s Platform: %s)\n"
             "Device: %s (%s %s)\n",
             major, minor, patch, revision, platform, platform,
             device_name, compute, toolkit);
      return true;
    }

  public:
    AFReporter() :
      ::benchmark::ConsoleReporter() {
    }
  };

  // class AFMemoryManager : public ::benchmark::MemoryManager {
  //   size_t alloc_bytes, alloc_buffers, lock_bytes, lock_buffers;
  //   size_t alloc_bytes2, alloc_buffers2, lock_bytes2, lock_buffers2;
  // public:
  //   AFMemoryManager() : ::benchmark::MemoryManager() {}
  //
  //   virtual void Start() {
  //     deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);
  //     printf("start\n");
  //   }
  //   virtual void Stop(Result* result) {
  //     deviceMemInfo(&alloc_bytes2, &alloc_buffers2, &lock_bytes2, &lock_buffers2);
  //     result->num_allocs = alloc_buffers2 - alloc_buffers;
  //     result->max_bytes_used = alloc_bytes2 - alloc_bytes;
  //     printf("stop\n");
  //     deviceGC();
  //   }
  // };

template <class Lambda, class... Args>
BenchmarkCollection* RegisterBenchmark(const char* name, std::vector<af_dtype> types, Lambda&& fn,
                                        Args&&... args) {
  using namespace af;

  BenchmarkCollection *collection = new BenchmarkCollection();
  for (auto& type : types) {
    char test_name[2048];
    snprintf(test_name, 2048, "%s/%s", name, to_string(type).c_str());
    collection->Add(::benchmark::RegisterBenchmark(test_name, [=](::benchmark::State& st) { fn(st, type, args...); }));
  }
  return collection;
}

}
}

