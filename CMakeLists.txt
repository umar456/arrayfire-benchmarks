cmake_minimum_required(VERSION 3.5)

project(ArrayFire-Benchmarks VERSION 3.6)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)

if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/arrayfire)
  set(CMAKE_BUILD_RPATH
    arrayfire/src/backend/cpu
    arrayfire/src/backend/cuda
    arrayfire/src/backend/opencl)
  add_subdirectory(arrayfire EXCLUDE_FROM_ALL)
else ()
  find_package(ArrayFire REQUIRED)
  set(CMAKE_BUILD_RPATH
    ${ArrayFire_DIR}/src/backend/cpu
    ${ArrayFire_DIR}/src/backend/cuda
    ${ArrayFire_DIR}/src/backend/opencl)
endif ()

option(ENABLE_ITK_BENCHMARKS "Enable ITK Benchmarks" OFF)
if (ENABLE_ITK_BENCHMARKS)
  find_package(ITK 5.0.1 REQUIRED)
endif ()

set(BENCHMARK_ENABLE_TESTING OFF CACHE INTERNAL "")
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE INTERNAL "")
set(BENCHMARK_ENABLE_ASSEMPLY_TEST OFF CACHE INTERNAL "")
add_subdirectory(benchmark EXCLUDE_FROM_ALL)

include_directories(src)
if (ENABLE_ITK_BENCHMARKS)
  include(${ITK_USE_FILE})
endif ()

add_library(af_benchmark
  src/af_benchmark.cpp
  src/af_console_reporter.cpp
  src/af_json_reporter.cpp)

target_link_libraries(af_benchmark
  PUBLIC
    benchmark
    ArrayFire::af
    ${ITK_LIBRARIES}
    ${CMAKE_DL_LIBS}
    )

# add_executable(cudnn_conv_relu_pool src/cudnn_conv_relu_pool.cpp)
# target_include_directories(cudnn_conv_relu_pool PRIVATE /opt/cuda/include)
# target_link_libraries(cudnn_conv_relu_pool -L/opt/cuda/lib64 -lcudnn -lcudart af_benchmark)

add_executable(unified_af src/unified.cpp)
add_executable(unified_cuda src/unified.cpp)
target_link_libraries(unified_af benchmark ArrayFire::af af_benchmark)
target_link_libraries(unified_cuda benchmark ArrayFire::afcuda af_benchmark)

enable_testing()

function(addBenchmark name)
  add_executable(${name} src/${name}.cpp)
  add_test(${name} ${name})
  target_link_libraries(${name} af_benchmark)
  target_compile_definitions(${name}
    PRIVATE ASSETS_DIR="${ArrayFire-Benchmarks_SOURCE_DIR}/assets"
  )
  if (ENABLE_ITK_BENCHMARKS)
    target_compile_definitions(${name} PRIVATE ENABLE_ITK)
  endif ()
endfunction()

addBenchmark(approx1)
addBenchmark(anisotropic_smoothing)
addBenchmark(randu)
addBenchmark(topk)
addBenchmark(nearest_neighbor)
addBenchmark(pi)
addBenchmark(af_conv_relu_pool)
addBenchmark(fft)
addBenchmark(indexing)
addBenchmark(jit)
addBenchmark(tile)
addBenchmark(matmul)
addBenchmark(mean)
addBenchmark(reduce)
addBenchmark(norm)
addBenchmark(regions)
addBenchmark(sort)
addBenchmark(stdev)
addBenchmark(rotate)
addBenchmark(iterative_deconvolution)

mark_as_advanced(
  BENCHMARK_BUILD_32_BITS
  BENCHMARK_DOWNLOAD_DEPENDENCIES
  BENCHMARK_ENABLE_ASSEMBLY_TESTS
  BENCHMARK_ENABLE_EXCEPTIONS
  BENCHMARK_ENABLE_INSTALL
  BENCHMARK_ENABLE_LTO
  BENCHMARK_USE_LIBCXX
  LIBRT
  LLVM_FILECHECK_EXE
)
