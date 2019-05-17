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

set(BENCHMARK_ENABLE_TESTING OFF CACHE INTERNAL "")
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE INTERNAL "")
set(BENCHMARK_ENABLE_ASSEMPLY_TEST OFF CACHE INTERNAL "")
add_subdirectory(benchmark EXCLUDE_FROM_ALL)

include_directories(src)

add_library(af_benchmark
  src/af_benchmark.cpp
  src/af_console_reporter.cpp
  src/af_json_reporter.cpp)

target_link_libraries(af_benchmark
  PUBLIC
    benchmark ArrayFire::af ${CMAKE_DL_LIBS})

# add_executable(cudnn_conv_relu_pool src/cudnn_conv_relu_pool.cpp)
# target_include_directories(cudnn_conv_relu_pool PRIVATE /opt/cuda/include)
# target_link_libraries(cudnn_conv_relu_pool -L/opt/cuda/lib64 -lcudnn -lcudart af_benchmark)

add_executable(anisotropic_smoothing src/anisotropic_smoothing.cpp)
target_link_libraries(anisotropic_smoothing af_benchmark)

#add_executable(fft src/fft.cpp)
#target_link_libraries(fft af_benchmark)

#add_executable(indexing src/indexing.cpp)
#target_link_libraries(indexing af_benchmark)
#add_dependencies(indexing test_index_unified)

#add_executable(jit src/jit.cpp)
#target_link_libraries(jit af_benchmark)
#add_dependencies(jit test_jit_unified)


#add_executable(tile src/tile.cpp)
#target_link_libraries(tile af_benchmark)
#add_dependencies(tile test_tile_unified)

add_executable(unified_af src/unified.cpp)
add_executable(unified_cuda src/unified.cpp)
target_link_libraries(unified_af benchmark ArrayFire::af af_benchmark)
target_link_libraries(unified_cuda benchmark ArrayFire::afcuda af_benchmark)

SET(benchmarks randu topk nearest_neighbor pi af_conv_relu_pool fft indexing jit tile matmul reduce norm sort)

function(addBenchmark name)
  add_executable(${name} src/${name}.cpp)
  target_link_libraries(${name} af_benchmark)
endfunction()

foreach(benchmark ${benchmarks})
  addBenchmark(${benchmark})
endforeach() 
