#!/bin/bash
# usage: cpp/scripts/build-android.sh <NDK> <OpenCV_DIR (arm64-android)> [extra cmake args]
#   e.g. cpp/scripts/build-android.sh /f/Android/SDK/ndk/29.0.13599879 \
#        /f/Project/Golang/MaaEnd/agent/cpp-algo/MaaUtils/MaaDeps/vcpkg/installed/maa-arm64-android/share/opencv4
set -e
HERE="$(cd "$(dirname "$0")/.." && pwd)"
NDK="$1"; OPENCV="$2"; shift 2
cmake -S "$HERE" -B "$HERE/build/android" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-26 -DANDROID_STL=c++_shared \
  -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=BOTH -DOpenCV_DIR="$OPENCV" "$@"
cmake --build "$HERE/build/android"
