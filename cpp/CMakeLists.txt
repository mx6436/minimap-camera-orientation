cmake_minimum_required(VERSION 3.21)
project(camori_preprocess CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenCV (core only). Point OpenCV_DIR at any OpenCV 4 install, e.g. MaaDeps:
#   <MaaDeps>/vcpkg/installed/maa-x64-windows/share/opencv4
#   <MaaDeps>/vcpkg/installed/maa-arm64-android/share/opencv4
if(OpenCV_DIR)
    # MaaDeps' OpenCV config looks up its own deps (Eigen3, ...) next to it
    get_filename_component(_deps_prefix "${OpenCV_DIR}/../.." ABSOLUTE)
    list(APPEND CMAKE_PREFIX_PATH "${_deps_prefix}")
endif()
find_package(OpenCV REQUIRED COMPONENTS core)

add_library(camori_preprocess STATIC src/preprocess.cpp)
target_include_directories(camori_preprocess PUBLIC include)
target_link_libraries(camori_preprocess PUBLIC opencv_core)
# keep IEEE float32 semantics: no fast-math, no FMA contraction, so the op order matches ORT / torch
if(MSVC)
    target_compile_options(camori_preprocess PRIVATE /fp:precise /utf-8)
else()
    target_compile_options(camori_preprocess PRIVATE -ffp-contract=off -fno-fast-math)
endif()

option(CAMORI_BUILD_TESTS "build the fixture test" ON)
if(CAMORI_BUILD_TESTS)
    add_executable(camori_fixture_test tests/fixture_test.cpp)
    target_link_libraries(camori_fixture_test PRIVATE camori_preprocess)
    if(MSVC)
        target_compile_options(camori_fixture_test PRIVATE /utf-8)
    endif()
endif()

# optional: time a preprocess.onnx with onnxruntime on the same fixture inputs
option(CAMORI_BUILD_ORT_BENCH "build camori_ort_bench (needs onnxruntime)" OFF)
if(CAMORI_BUILD_ORT_BENCH)
    find_package(onnxruntime CONFIG REQUIRED PATHS "${_deps_prefix}/share/maa-onnxruntime" "${_deps_prefix}/share/onnxruntime")
    add_executable(camori_ort_bench tests/ort_bench.cpp)
    target_link_libraries(camori_ort_bench PRIVATE onnxruntime::onnxruntime)
endif()
