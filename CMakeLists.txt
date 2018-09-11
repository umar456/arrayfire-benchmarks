cmake_minimum_required(VERSION 3.5)

project(ArrayFire-Benchmarks
  VERSION 3.6)

add_subdirectory(arrayfire EXCLUDE_FROM_ALL)

set(BENCHMARK_ENABLE_TESTING OFF CACHE INTERNAL "")
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE INTERNAL "")
set(BENCHMARK_ENABLE_ASSEMPLY_TEST OFF CACHE INTERNAL "")
add_subdirectory(benchmark EXCLUDE_FROM_ALL)

include_directories(src)

add_executable(randu src/randu.cpp)
target_link_libraries(randu benchmark ArrayFire::af)

add_executable(topk src/topk.cpp)
target_link_libraries(topk benchmark ArrayFire::af)

add_executable(pi src/pi.cpp)
target_link_libraries(pi benchmark ArrayFire::af)

add_executable(unified_af src/unified.cpp)
add_executable(unified_cuda src/unified.cpp)
target_link_libraries(unified_af benchmark ArrayFire::af)
target_link_libraries(unified_cuda benchmark ArrayFire::afcuda)
