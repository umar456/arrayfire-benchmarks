# ArrayFire Benchmarking Library

# Requirements

* ArrayFire v3.6
* Google Benchmark

# Installation

Separate build directory:
```
git clone --recursive http://github.com/umar456/arrayfire-benchmarks
cd arrayfire-benchmarks
git clone --branch v1.5.2 http://github.com/google/benchmark
mkdir build && cd build
cmake -DArrayFire_DIR=/path/to/build/dir ..
make -j8
```

ArrayFire Inline:
```
git clone --recursive http://github.com/umar456/arrayfire-benchmarks
cd arrayfire-benchmarks
git clone http://github.com/arrayfire/arrayfire
mkdir build && cd build
cmake ..
make -j8
```
