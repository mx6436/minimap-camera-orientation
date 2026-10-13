cmake_minimum_required(VERSION 3.28)
project(maplocator-data-prep CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# MaaEnd submodule 提供定位与条带预处理语义；MaaUtils.cmake 负责 MaaDeps 接线与依赖查找。
set(MAAEND_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/MaaEnd")
set(CPP_ALGO_SOURCE "${MAAEND_ROOT}/agent/cpp-algo/source")
set(MAAUTILS "${MAAEND_ROOT}/agent/cpp-algo/MaaUtils")

include("${MAAUTILS}/MaaUtils.cmake")

set(maplocator_sources
    "${CPP_ALGO_SOURCE}/MapLocator/MapLocator.cpp"
    "${CPP_ALGO_SOURCE}/MapLocator/MapAlgorithm.cpp"
    "${CPP_ALGO_SOURCE}/MapLocator/MatchStrategy.cpp"
    "${CPP_ALGO_SOURCE}/MapLocator/MotionTracker.cpp"
    "${CPP_ALGO_SOURCE}/MapLocator/YoloPredictor.cpp"
    "${CPP_ALGO_SOURCE}/MapLocator/CameraOrientationPredictor.cpp"
    "${CPP_ALGO_SOURCE}/MapLocator/CameraOrientationPreprocess.cpp"
)

add_executable(dataset-prep
    src/main.cpp
    ${maplocator_sources}
)

target_include_directories(dataset-prep PRIVATE "${CPP_ALGO_SOURCE}")

# OpenCV_INCLUDE_DIRS / OpenCV_LIBS / ONNXRuntime::ONNXRuntime / Boost::* / ZLIB::ZLIB
# 由 MaaUtils.cmake 的 find_package 提供。
target_link_libraries(dataset-prep PRIVATE
    MaaUtils
    Boost::regex Boost::system
    ONNXRuntime::ONNXRuntime
    ZLIB::ZLIB
    ${OpenCV_LIBS}
)
if(UNIX)
    target_link_libraries(dataset-prep PRIVATE pthread dl)
endif()
add_dependencies(dataset-prep MaaUtils)
