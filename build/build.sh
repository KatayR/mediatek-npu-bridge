#!/usr/bin/env bash
# Builds the patched TFLite Neuron delegate AAR:
#   1. clones MediaTek's tflite-neuron-delegate at the pinned commit (or reuses a clean clone),
#   2. applies patches/0001..0004 in order (git apply --check first),
#   3. runs Bazel for //neuron/java:tensorflow-lite-neuron (and optionally the benchmark binary),
#   4. copies the AAR to the output folder and prints its sha256 and members.
#
# Linux x86_64 only (WSL2 Ubuntu 24.04 was used; keep the work dir on a Linux file system, not on a mounted Windows drive).
# Toolchain and versions: build/README.md. Nothing here needs root.
#
# usage: build/build.sh [--work DIR] [--out DIR] [--bench] [--check-only]
#   --work DIR     where the upstream clone lives (default ~/mtk-npu-bridge-work)
#   --out DIR      where the AAR is copied (default <work>/out)
#   --bench        also build benchmark_model_plus_neuron_delegate (command-line tool for adb)
#   --check-only   clone + git apply --check, no build
#
# Environment (required unless noted):
#   ANDROID_HOME       Android SDK with platforms;android-28 and build-tools;30.0.3
#   ANDROID_NDK_HOME   NDK r21e (21.4.7075529); Bazel 4's android_ndk_repository supports NDK <= r21
#   JAVA_HOME          JDK 11
#   PYTHON_BIN_PATH    a Python 3 with numpy (and setuptools, for distutils on Python >= 3.12)
#   BAZEL              optional, the bazel/bazelisk binary (default: `bazel` on PATH)
#   USE_BAZEL_VERSION  optional, default 4.2.1 (read by bazelisk)
#   JOBS               optional, default 8
#   BAZEL_DISTDIR      optional, a --distdir with pre-fetched dependency archives
#   BAZEL_STARTUP_OPTS optional, e.g. "--nohome_rc"
#   BAZEL_BUILD_OPTS   optional, extra build flags, e.g. "--local_ram_resources=11000"
set -euo pipefail

UPSTREAM_URL=https://github.com/MediaTek-NeuroPilot/tflite-neuron-delegate
UPSTREAM_COMMIT=abb45cca4d22a7b02cdd87191d7c1fef44235c1c
VERSION=0.1.0
PATCHES=(0001-expose-options.patch 0002-batch-matmul-gelu.patch 0003-op-exclusion-filters.patch 0004-dla-cache.patch)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
WORK="$HOME/mtk-npu-bridge-work"
OUT=""
BENCH=0
CHECK_ONLY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --work) WORK="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --bench) BENCH=1; shift ;;
    --check-only) CHECK_ONLY=1; shift ;;
    -h|--help) sed -n '2,27p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
OUT="${OUT:-$WORK/out}"
SRC="$WORK/tflite-neuron-delegate"
export USE_BAZEL_VERSION="${USE_BAZEL_VERSION:-4.2.1}"
BAZEL="${BAZEL:-bazel}"
JOBS="${JOBS:-8}"
STARTUP="${BAZEL_STARTUP_OPTS:-}"

step() { echo; echo "== [$(date +%T)] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

step "clone $UPSTREAM_URL @ $UPSTREAM_COMMIT -> $SRC"
mkdir -p "$WORK"
if [ ! -d "$SRC/.git" ]; then
  git clone --quiet "$UPSTREAM_URL" "$SRC"
fi
git -C "$SRC" fetch --quiet origin "$UPSTREAM_COMMIT" 2>/dev/null || true
if [ -n "$(git -C "$SRC" status --porcelain --untracked-files=no)" ]; then
  die "$SRC has local changes; use a fresh --work folder (patches are applied to a pristine checkout)"
fi
git -C "$SRC" -c advice.detachedHead=false checkout --quiet "$UPSTREAM_COMMIT"
echo "HEAD $(git -C "$SRC" rev-parse HEAD)"

step "apply patches"
for p in "${PATCHES[@]}"; do
  f="$REPO/patches/$p"
  [ -f "$f" ] || die "missing $f"
  if grep -q $'\r' "$f"; then
    die "$p has CRLF line endings (a Windows checkout with core.autocrlf?); re-checkout with LF"
  fi
  git -C "$SRC" apply --check "$f" || die "git apply --check failed for $p"
  git -C "$SRC" apply "$f"
  echo "applied $p ($(sha256sum "$f" | cut -c1-16)...)"
done
git -C "$SRC" status --short

[ "$CHECK_ONLY" = 1 ] && { step "check only: done"; exit 0; }

step "preflight"
bad=0
for v in ANDROID_HOME ANDROID_NDK_HOME JAVA_HOME PYTHON_BIN_PATH; do
  if [ -z "${!v:-}" ]; then echo "MISSING env $v"; bad=1; else echo "ok  $v=${!v}"; fi
done
[ "$bad" = 0 ] || die "set the variables above (build/README.md)"
[ -d "$ANDROID_HOME/platforms/android-28" ] || { echo "MISSING $ANDROID_HOME/platforms/android-28"; bad=1; }
[ -d "$ANDROID_HOME/build-tools/30.0.3" ] || { echo "MISSING $ANDROID_HOME/build-tools/30.0.3"; bad=1; }
grep -q "Pkg.Revision = 21\." "$ANDROID_NDK_HOME/source.properties" 2>/dev/null \
  || { echo "ANDROID_NDK_HOME is not an NDK r21 (Bazel 4 cannot use newer NDKs)"; bad=1; }
[ -x "$JAVA_HOME/bin/java" ] || { echo "MISSING $JAVA_HOME/bin/java"; bad=1; }
command -v "$BAZEL" > /dev/null || { echo "MISSING bazel (set BAZEL=/path/to/bazelisk)"; bad=1; }
command -v zip > /dev/null || { echo "MISSING zip"; bad=1; }
PYTHON_LIB_PATH="${PYTHON_LIB_PATH:-$("$PYTHON_BIN_PATH" -c 'import site; print(site.getsitepackages()[0])')}"
"$PYTHON_BIN_PATH" -c 'import numpy; from distutils import sysconfig' \
  || { echo "PYTHON_BIN_PATH needs numpy and distutils (pip install numpy setuptools)"; bad=1; }
[ "$bad" = 0 ] || die "preflight failed"
echo "ok  PYTHON_LIB_PATH=$PYTHON_LIB_PATH"
"$BAZEL" $STARTUP version 2>/dev/null | grep "Build label" || true

# TF 2.8's genrules call "/usr/bin/env python"; Ubuntu 24.04 only has python3. A private bin dir with a
# `python` link (first on PATH) avoids installing python-is-python3.
mkdir -p "$WORK/bin"
ln -sf "$PYTHON_BIN_PATH" "$WORK/bin/python"
export PATH="$WORK/bin:$PATH"
export PYTHON_BIN_PATH PYTHON_LIB_PATH

# Bazel 4.2.1 crashes (NullPointerException in the server) on a value-less --repo_env=NAME, so the values
# are given explicitly.
FLAGS=(--config android_arm64 -c opt --jobs="$JOBS"
       --repo_env=PYTHON_BIN_PATH="$PYTHON_BIN_PATH" --repo_env=PYTHON_LIB_PATH="$PYTHON_LIB_PATH")
[ -n "${BAZEL_DISTDIR:-}" ] && FLAGS+=(--distdir="$BAZEL_DISTDIR")
# shellcheck disable=SC2206
[ -n "${BAZEL_BUILD_OPTS:-}" ] && FLAGS+=($BAZEL_BUILD_OPTS)

cd "$SRC"
step "bazel build //neuron/java:tensorflow-lite-neuron"
t0=$SECONDS
# shellcheck disable=SC2086
"$BAZEL" $STARTUP build "${FLAGS[@]}" //neuron/java:tensorflow-lite-neuron
echo "AAR build: $((SECONDS - t0)) s"
if [ "$BENCH" = 1 ]; then
  step "bazel build //neuron:benchmark_model_plus_neuron_delegate"
  t0=$SECONDS
  # shellcheck disable=SC2086
  "$BAZEL" $STARTUP build "${FLAGS[@]}" //neuron:benchmark_model_plus_neuron_delegate
  echo "benchmark build: $((SECONDS - t0)) s"
fi

step "copy to $OUT"
mkdir -p "$OUT"
AAR="$OUT/tensorflow-lite-neuron-mtk-bridge-$VERSION.aar"
cp -f "$SRC/bazel-bin/neuron/java/tensorflow-lite-neuron.aar" "$AAR"
chmod u+w "$AAR"
if [ "$BENCH" = 1 ]; then
  cp -f "$SRC/bazel-bin/neuron/benchmark_model_plus_neuron_delegate" "$OUT/"
  chmod u+w "$OUT/benchmark_model_plus_neuron_delegate"
fi

step "verify"
sha256sum "$AAR"
unzip -l "$AAR"
T="$(mktemp -d)"
unzip -q -o "$AAR" -d "$T"
echo "-- member sha256 (the zip itself differs between identical rebuilds: entry timestamps)"
(cd "$T" && sha256sum classes.jar jni/arm64-v8a/*.so)
NDKBIN="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin"
echo "-- exported symbols"
"$NDKBIN/llvm-readelf" --dyn-syms --wide "$T"/jni/arm64-v8a/*.so | awk '$5=="GLOBAL" && $7!="UND" {print $8}' | sort -u
echo "-- NEEDED"
"$NDKBIN/llvm-readelf" -d "$T"/jni/arm64-v8a/*.so | grep NEEDED
echo "-- Java API"
"$JAVA_HOME/bin/javap" -cp "$T/classes.jar" 'org.tensorflow.lite.NeuronDelegate$Options'
rm -rf "$T"
step "done: $AAR"
