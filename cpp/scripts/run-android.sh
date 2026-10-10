#!/bin/bash
# Push build/android + fixtures to a device and run the fixture test there.
# usage: cpp/scripts/run-android.sh <fixtures_dir> <dir with libopencv_world4.so / libc++_shared.so> [--bench N]
set -e
export MSYS_NO_PATHCONV=1
winpath() { if command -v cygpath >/dev/null; then cygpath -w "$1"; else echo "$1"; fi; }
HERE="$(cd "$(dirname "$0")/.." && pwd)"
FIXTURES="$1"; LIBS="$2"; shift 2
D=/data/local/tmp/camori
adb shell "rm -rf $D && mkdir -p $D"
adb push "$(winpath "$HERE/build/android/camori_fixture_test")" $D/ >/dev/null
for so in "$LIBS"/libopencv_world4.so "$LIBS"/libc++_shared.so; do adb push "$(winpath "$so")" $D/ >/dev/null; done
tar -C "$(dirname "$FIXTURES")" -cf "$HERE/build/fixtures.tar" "$(basename "$FIXTURES")"
adb push "$(winpath "$HERE/build/fixtures.tar")" $D/ >/dev/null
adb shell "cd $D && tar -xf fixtures.tar && rm fixtures.tar && chmod +x camori_fixture_test && LD_LIBRARY_PATH=. ./camori_fixture_test $(basename "$FIXTURES") $*"
