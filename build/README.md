# Building the AAR from source

`build/build.sh` is the recipe the release AAR was built with: pinned upstream commit, the four patches,
TensorFlow 2.8.0 (fixed by the upstream `WORKSPACE`), Bazel 4.2.1, NDK r21e, Android platform 28 and
build-tools 30.0.3. A rebuild gives an AAR whose members (`classes.jar`, the `.so`, manifest, header,
LICENSE) are byte-identical to the release; the `.aar` file's own sha256 differs because of zip entry
timestamps, so compare members, not the zip.

Tested on WSL2 Ubuntu 24.04 (x86_64), 8 Bazel jobs: the first build of the AAR and the benchmark tool took
about 5.5 minutes (dependency archives pre-fetched); a fresh clone with Bazel's download cache already filled
took 2.5 minutes for both (AAR 79 s). Keep the work folder on a Linux file system (not on a mounted Windows drive).

## Toolchain (user space, no root except the apt line)

```bash
# compilers, JDKs, zip (the only step that needs sudo)
sudo apt update && sudo apt install -y git curl unzip zip patch file build-essential python3-venv python3-dev \
  openjdk-17-jdk-headless openjdk-11-jdk-headless

T=~/mtk-npu-toolchain; mkdir -p $T/android-sdk ~/bin && cd $T
# bazelisk; build.sh sets USE_BAZEL_VERSION=4.2.1 (starts on Ubuntu 24.04 with its embedded JDK)
curl -L -o ~/bin/bazel https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64 && chmod +x ~/bin/bazel
# Android command-line tools -> SDK packages (sdkmanager wants JDK 17)
curl -L -o cmdline-tools.zip https://dl.google.com/android/repository/commandlinetools-linux-15859902_latest.zip
unzip -q cmdline-tools.zip -d android-sdk/cmdline-tools && mv android-sdk/cmdline-tools/cmdline-tools android-sdk/cmdline-tools/latest
J17=/usr/lib/jvm/java-17-openjdk-amd64
yes | JAVA_HOME=$J17 android-sdk/cmdline-tools/latest/bin/sdkmanager --sdk_root=$T/android-sdk --licenses
JAVA_HOME=$J17 android-sdk/cmdline-tools/latest/bin/sdkmanager --sdk_root=$T/android-sdk \
  "platforms;android-28" "build-tools;30.0.3" "platform-tools"
# NDK r21e: Bazel 4's android_ndk_repository does not support newer NDKs
curl -L -o ndk-r21e.zip https://dl.google.com/android/repository/android-ndk-r21e-linux-x86_64.zip && unzip -q ndk-r21e.zip
# Python with numpy for TF 2.8's python_configure (setuptools brings back distutils on Python 3.12)
python3 -m venv $T/venv && $T/venv/bin/pip install numpy setuptools
```

| Item | Version | sha256 of the download |
|---|---|---|
| Bazel | 4.2.1 via bazelisk | |
| Android cmdline-tools | 22.0 (`commandlinetools-linux-15859902_latest.zip`) | `4e4c464f145a7512b57d088ac6c278c03c9eea610886b35a5e0804e74eedf583` |
| NDK | r21e (21.4.7075529, clang 9.0.9) | `ad7ce5467e18d40050dc51b8e7affc3e635c85bd8c59be62de32352328ed467e` |
| SDK packages | `platforms;android-28` rev 6, `build-tools;30.0.3`, `platform-tools` | |
| JDK | OpenJDK 11 (`JAVA_HOME` for the build), 17 (sdkmanager only) | |
| TensorFlow | 2.8.0, fetched by Bazel (`v2.8.0.zip`) | `d2948c066a0bc3f45cb8072def03c85f50af8a75606bbdff91715ef8c5f2a28c` |
| Host compiler | GCC 13 (Ubuntu 24.04) builds TF 2.8's host tools without changes | |

## Build

```bash
export ANDROID_HOME=~/mtk-npu-toolchain/android-sdk
export ANDROID_NDK_HOME=~/mtk-npu-toolchain/android-ndk-r21e
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PYTHON_BIN_PATH=~/mtk-npu-toolchain/venv/bin/python
export BAZEL=~/bin/bazel
build/build.sh                 # -> ~/mtk-npu-bridge-work/out/tensorflow-lite-neuron-mtk-bridge-0.1.0.aar
build/build.sh --bench         # also benchmark_model_plus_neuron_delegate (adb command-line tool)
build/build.sh --check-only    # clone + git apply --check only
tests/run_host_tests.sh        # host unit tests of the two headers the patches add (after one build)
```

Inside, the script runs
`bazel build --config android_arm64 -c opt //neuron/java:tensorflow-lite-neuron` (and
`//neuron:benchmark_model_plus_neuron_delegate` with `--bench`) with
`--repo_env=PYTHON_BIN_PATH=... --repo_env=PYTHON_LIB_PATH=...`, and verifies the result (sha256, members,
exported symbols, NEEDED libraries, `javap` of `NeuronDelegate$Options`).

Two problems it works around, both met on Ubuntu 24.04:
1. Bazel 4.2.1 crashes with `NullPointerException: null value in entry: PYTHON_BIN_PATH=null` on a
   value-less `--repo_env=NAME`; the script passes `NAME=value`.
2. TF 2.8's `gen_git_source` runs `/usr/bin/env python`, and Ubuntu 24.04 has no `python` command
   (`Exit 127`); the script puts a `python` link to `PYTHON_BIN_PATH` first on `PATH` (in `<work>/bin`).

Other failures seen in TF 2.8 builds and their usual fix: `cc1plus killed` → lower `JOBS`; "based on
version 21" is only a warning, a hard NDK error means the wrong `ANDROID_NDK_HOME` or an NDK newer than r21;
a link error on `libneuron_performance.a` for three ABIs means `--config android_arm64` is missing; a dead
archive URL → download it into a folder and pass it as `BAZEL_DISTDIR`.

## Expected result (release 0.1.0)

| File | Size | sha256 |
|---|---|---|
| `tensorflow-lite-neuron-mtk-bridge-0.1.0.aar` (release file) | 190,644 | `d0e581d52cf5fb2745b41961033d601e2c825b6190a41b9b6cfc76ae91fc3e88` |
| member `jni/arm64-v8a/libtensorflowlite_neuron_jni.so` | 448,648 | `ea375f2737af1c7dcb8a3358914703714dac27cb880e2bec38053917c621e819` |
| member `classes.jar` | 3,513 | `9a9afd4015f68b4c0e59aeda597d96e5da91b192894c9c8b459f207474195e9a` |
| `benchmark_model_plus_neuron_delegate` (`--bench`) | 6,311,856 | `b9ceaef05a1428393c68c303be67fdacfa7f0578a237fbde19c2ac38b0c39dcd` |

Exported symbols of the `.so`: `Java_org_tensorflow_lite_NeuronDelegate_createDelegate`, `..._deleteDelegate`,
`TfLiteNeuronDelegateCreate`, `TfLiteNeuronDelegateDelete`, `TfLiteNeuronDelegateOptionsDefault` (version
`VERS_1.0`). NEEDED: `libm.so libandroid.so libdl.so liblog.so libc.so`. No TFLite symbol is imported or
exported: TFLite is reached through the `TfLiteContext` function pointers, which is why the runtime must be
exactly TFLite 2.8 (main README).
