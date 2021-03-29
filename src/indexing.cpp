
#include <arrayfire_benchmark.h>
#include <arrayfire.h>
#include <vector>
#include <functional>

#include <iostream>

using af::array;
using af::randu;
using af::sum;
using af::dim4;
using af::seq;
using af::span;
using af::deviceGC;
using af::deviceMemInfo;

using std::vector;
using std::pair;

void indexingBench(benchmark::State& state, af_dtype type) {

  int dim = state.range(0);
  dim4 dims(state.range(2),  state.range(3), state.range(4), state.range(5));
  dim4 odims(state.range(2), state.range(3), state.range(4), state.range(5));

  dims[dim] = state.range(1);
  //std::cout << dims << std::endl;
  //std::cout << odims << std::endl;

  {
    array a = randu(dims, type);
    array out = randu(odims, type);

      switch(dim) {
        case 0: out(seq(state.range(1)), span, span, span) = a; break;
        case 1: out(span, seq(state.range(1)), span, span) = a; break;
        case 2: out(span, span, seq(state.range(1)), span) = a; break;
        case 3: out(span, span, span, seq(state.range(1))) = a; break;
      }

    size_t alloc_bytes, alloc_buffers, lock_bytes, lock_buffers;
    deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);
    for(auto _ : state) {
      switch(dim) {
        case 0: out(seq(state.range(1)), span, span, span) = a; break;
        case 1: out(span, seq(state.range(1)), span, span) = a; break;
        case 2: out(span, span, seq(state.range(1)), span) = a; break;
        case 3: out(span, span, span, seq(state.range(1))) = a; break;
      }
      out.eval();
      af::sync();
    }

    size_t alloc_bytes2, alloc_buffers2, lock_bytes2, lock_buffers2;
    deviceMemInfo(&alloc_bytes2, &alloc_buffers2, &lock_bytes2, &lock_buffers2);

    state.counters["alloc_bytes"] = alloc_bytes2 - alloc_bytes;
    state.counters["alloc_buffers"] = alloc_buffers2 - alloc_buffers;
    state.counters["bytes"] = a.bytes();
  }
  deviceGC();
  //af::printMemInfo();
}

void indexingBench2(benchmark::State& state, af_dtype type) {

  int dim = state.range(0);
  dim4 dims(state.range(2),  state.range(3), state.range(4), state.range(5));
  dim4 odims(state.range(2), state.range(3), state.range(4), state.range(5));

  dims[dim] = state.range(1);
  //std::cout << dims << std::endl;
  //std::cout << odims << std::endl;

  {
    array a = randu(dims, type);
    array out = randu(odims, type);

      switch(dim) {
        case 0: a = out(seq(state.range(1)), span, span, span); break;
        case 1: a = out(span, seq(state.range(1)), span, span); break;
        case 2: a = out(span, span, seq(state.range(1)), span); break;
        case 3: a = out(span, span, span, seq(state.range(1))); break;
      }

    size_t alloc_bytes, alloc_buffers, lock_bytes, lock_buffers;
    deviceMemInfo(&alloc_bytes, &alloc_buffers, &lock_bytes, &lock_buffers);
    for(auto _ : state) {
      switch(dim) {
        case 0: a = out(seq(state.range(1)), span, span, span); break;
        case 1: a = out(span, seq(state.range(1)), span, span); break;
        case 2: a = out(span, span, seq(state.range(1)), span); break;
        case 3: a = out(span, span, span, seq(state.range(1))); break;
      }
      out.eval();
      af::sync();
    }

    size_t alloc_bytes2, alloc_buffers2, lock_bytes2, lock_buffers2;
    deviceMemInfo(&alloc_bytes2, &alloc_buffers2, &lock_bytes2, &lock_buffers2);

    state.counters["alloc_bytes"] = alloc_bytes2 - alloc_bytes;
    state.counters["alloc_buffers"] = alloc_buffers2 - alloc_buffers;
    state.counters["bytes"] = a.bytes();
  }
  deviceGC();
  //af::printMemInfo();
}


void
CustomArgs(benchmark::internal::Benchmark* b) {
    vector<pair<int, int>> sizes = {
        {1, 1<<26}, {1, 1<<26}, {1, 1}, {1, 1}
    };
    pair<int, int> elements = {1, 1<<8};
    int max = 1<<28;

    //printf("el: %d %d\n", elements.first, elements.second);

    vector<dim_t> d(4, 1);
    for(int dim = 0; dim < 4; dim++) {
      for(d[3] = sizes[3].first; d[3] <= sizes[3].second; d[3]*=8) {
        for(d[2] = sizes[2].first; d[2] <= sizes[2].second; d[2]*=8) {
          for(d[1] = sizes[1].first; d[1] <= sizes[1].second; d[1]*=8) {
            for(d[0] = sizes[0].first; d[0] <= sizes[0].second; d[0]*=8) {
              for(int el = elements.first; el <= d[dim]; el*=8) {
                if((d[0]*d[1]*d[2]*d[3]) < max) {
                  b->Args({dim, el, d[0], d[1], d[2], d[3]});
                  //printf("elements: %d %d %d %d = %d\n", d[0],d[1],d[2],d[3], d[0]*d[1]*d[2]*d[3]);
                }
              }
            }
          }
        }
      }
    }
}

int main(int argc, char** argv) {
  vector<af_dtype> types = {f32, f64};

  benchmark::Initialize(&argc, argv);

  af::benchmark::RegisterBenchmark("indexing", types, indexingBench)
    ->Apply(CustomArgs)
    ->ArgNames({"dim", "elements", "dim0", "dim1", "dim2", "dim3"});

  //af::benchmark::AFJSONReporter jsonr;
  // benchmark::RunSpecifiedBenchmarks(&r, &jsonr);
  af::benchmark::AFReporter r;
  benchmark::RunSpecifiedBenchmarks(&r);
}
