
#pragma once
#include <string>
#include <vector>

#include <af/defines.h>
#include <benchmark/benchmark.h>
#include <af/device.h>
#include <af/util.h>

namespace af {
namespace benchmark {
  class AFJSONReporter : public ::benchmark::BenchmarkReporter {
  public:
    AFJSONReporter() : first_report_(true) {}
    virtual bool ReportContext(const Context& context);
    virtual void ReportRuns(const std::vector<Run>& reports);
    virtual void Finalize();

  private:
    void PrintRunData(const Run& report);

    bool first_report_;
  };

  std::string to_string(af_dtype type);

  // Same as the google Benchmark class but performs the same operation over a set of benchmarks
  class BenchmarkCollection {
    std::vector<::benchmark::internal::Benchmark*> benchmarks_;

  public:
    BenchmarkCollection();

    void Add(::benchmark::internal::Benchmark* bench);

    // TODO(umar): Some functions are not implemented yet
    BenchmarkCollection* Arg(int64_t x);
    BenchmarkCollection* Unit(::benchmark::TimeUnit unit);
    BenchmarkCollection* Range(int64_t start, int64_t limit);
    BenchmarkCollection* DenseRange(int64_t start, int64_t limit, int step = 1);
    BenchmarkCollection* Args(const std::vector<int64_t>& args);
    BenchmarkCollection* Ranges(const std::vector<std::pair<int64_t, int64_t> >& ranges);
    BenchmarkCollection* ArgName(const std::string& name);
    BenchmarkCollection* ArgNames(const std::vector<std::string>& names);
    BenchmarkCollection* RangeMultiplier(int multiplier);
    BenchmarkCollection* MinTime(double t);
    BenchmarkCollection* Apply(void (*func)(::benchmark::internal::Benchmark* benchmark));
    BenchmarkCollection* Iterations(size_t n);
  };

  class AFReporter : public ::benchmark::ConsoleReporter {
    virtual bool ReportContext(const Context& context);

  public:
    AFReporter();
  };

  // class AFMemoryManager : public ::benchmark::MemoryManager {
  //   size_t alloc_bytes, alloc_buffers, lock_bytes, lock_buffers;
  //   size_t alloc_bytes2, alloc_buffers2, lock_bytes2, lock_buffers2;
  // public:
  //   AFMemoryManager() : ::benchmark::MemoryManager() {}
  //
  //   virtual void Start() {
  //     deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);
  //
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
BenchmarkCollection* RegisterBenchmark(const char* name, std::vector<af_dtype> types, Lambda fn,
                                        Args... args) {
  using namespace af;

  BenchmarkCollection *collection = new BenchmarkCollection();
  for (auto& type : types) {
    char test_name[2048];
    snprintf(test_name, 2048, "%s/%s", name, to_string(type).c_str());
    collection->Add(
      ::benchmark::RegisterBenchmark(test_name, [=](::benchmark::State& st) {
          fn(st, type, args...); }));
  }
  return collection;
}

static inline void Initialize(int *argc, char **argv) {
  ::benchmark::Initialize(argc, argv);
}

}
}

