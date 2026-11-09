#!/usr/bin/env bash
# Host (Linux x86_64, g++) unit tests for the two pure-logic headers the patches add:
#   neuron/npustt_filters.h   (0003: --npustt-exclude / --npustt-max-softmax parsing)
#   neuron/npustt_dla_cache.h (0004: cache key, paths, mkdir -p, tmp-then-rename write, mmap helper)
#
# usage: tests/run_host_tests.sh [WORK_DIR]
#   WORK_DIR = the --work folder of build/build.sh (default ~/mtk-npu-bridge-work), i.e. a patched clone at
#   WORK_DIR/tflite-neuron-delegate that has been built once, so Bazel's external/ folder holds the
#   TF 2.8 and FlatBuffers headers the filters test needs. The DLA cache test needs no TF headers.
# Environment: BAZEL (default `bazel`), BAZEL_STARTUP_OPTS (same as for the build, e.g. "--nohome_rc").
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${1:-$HOME/mtk-npu-bridge-work}"
SRC="$WORK/tflite-neuron-delegate"
BAZEL="${BAZEL:-bazel}"
STARTUP="${BAZEL_STARTUP_OPTS:-}"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT
[ -f "$SRC/neuron/npustt_dla_cache.h" ] || { echo "no patched clone at $SRC (run build/build.sh first)" >&2; exit 1; }

echo "== dla cache test"
g++ -std=c++17 -Wall -Wextra -Werror -O1 -I"$SRC" "$HERE/dla_cache_test.cc" -o "$OUT/dla_cache_test"
"$OUT/dla_cache_test"

echo "== filters test"
# shellcheck disable=SC2086
EXT="$(cd "$SRC" && "$BAZEL" $STARTUP info output_base 2>/dev/null)/external"
TF="$EXT/org_tensorflow"
[ -d "$TF/tensorflow/lite" ] || { echo "no TF sources under $EXT (build once first)" >&2; exit 1; }
g++ -std=c++17 -Wall -Wextra -O1 -I"$SRC" -I"$TF" -I"$EXT/flatbuffers/include" \
  "$HERE/filters_test.cc" "$TF/tensorflow/lite/minimal_logging.cc" \
  "$TF/tensorflow/lite/minimal_logging_default.cc" -o "$OUT/filters_test"
"$OUT/filters_test"
