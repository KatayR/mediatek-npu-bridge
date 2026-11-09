# mediatek-npu-bridge

Run TensorFlow Lite models on the NPU (MDLA) of MediaTek phones from an ordinary Android app, using
MediaTek's open-source TFLite Neuron delegate with four patches, a build recipe, and what we learned making
large models work on a Dimensity 1080.

**What this is:** a patched copy of MediaTek's MIT
[tflite-neuron-delegate](https://github.com/MediaTek-NeuroPilot/tflite-neuron-delegate) (as patches against
the pinned upstream commit), a script that builds it into an Android AAR, a prebuilt AAR on the Releases
page, small Kotlin/C++ examples, and measured notes: which ops and shapes the NPU accepts, compile caching,
memory, and splitting big models.

**What this is not:** not a driver and not a new runtime. The NPU driver, the on-device Neuron compiler and
the `libneuronusdk_adapter.mtk.so` library are MediaTek's and come preinstalled on the phone; the delegate
loads them at run time. Nothing here is model-specific: any `.tflite` model (vision, audio, text, generative)
goes through the same path, as far as its ops are accepted by the NPU.

## Status and tested hardware

| | |
|---|---|
| Tested chip | MediaTek MT6877 (Dimensity 1080), NPU = MDLA 2.0 (plus 2 VPU cores, `mtk-dsp`) |
| Tested phone | one model: Samsung Galaxy A34 5G (SM-A346E), 8 GB RAM, Android 16 (API 36), 4 KB pages |
| Neuron runtime on that phone | `libneuronusdk_adapter.mtk.so` 8.2.26 in `/system_ext/lib64` |
| Other MediaTek NPUs | probably work (the delegate and the adapter API are the same family), **untested**. Which ops and shapes are accepted differs per MDLA generation |
| Non-MediaTek chips | no. Without a Neuron adapter the delegate silently hands the graph to NNAPI (see pitfalls) |
| ABI | `arm64-v8a` only |
| TFLite runtime | exactly `org.tensorflow:tensorflow-lite:2.8.0` |

MediaTek's upstream README now points to Google's LiteRT MediaTek support for newer SoCs. We have not checked
which chips that covers; this repository is what worked on the MT6877.

Measured on that phone (medians, after warm-up):

| Workload | NPU (MDLA) | CPU, 4 threads |
|---|---|---|
| MobileNetV2 int8 | 1.92 ms | 12.55 ms |
| One `FULLY_CONNECTED` 1024→4096 on [1,512,1024], fp16-relaxed | 14.3 ms | 91.7 ms |
| One transformer layer (hidden 1024, 16 heads, FFN 4096, 512 tokens) in NPU-friendly forms | 53–57 ms | 315–320 ms |
| 24-layer, 1.3 GB float32 transformer encoder as 4 sections | 1.28–1.35 s, 1.0 GB process memory, cached load 0.8–2.0 s | 8.8 s |

## Quick start

1. Download `tensorflow-lite-neuron-mtk-bridge-0.1.0.aar` from the [Releases page](../../releases/tag/v0.1.0)
   (sha256 `d0e581d52cf5fb2745b41961033d601e2c825b6190a41b9b6cfc76ae91fc3e88`) into `app/libs/`.
2. Gradle (`app/build.gradle.kts`):

   ```kotlin
   android {
       defaultConfig {
           minSdk = 29                      // the AAR's manifest says 29
           ndk { abiFilters += "arm64-v8a" }
       }
   }
   dependencies {
       implementation("org.tensorflow:tensorflow-lite:2.8.0")     // exactly 2.8.0, see pitfalls
       implementation(files("libs/tensorflow-lite-neuron-mtk-bridge-0.1.0.aar"))
   }
   ```
3. Manifest (inside `<application>`; needed for apps targeting API 31+, otherwise the vendor library cannot be
   opened):

   ```xml
   <uses-native-library android:name="libneuronusdk_adapter.mtk.so" android:required="false" />
   ```
4. Kotlin:

   ```kotlin
   import org.tensorflow.lite.Interpreter
   import org.tensorflow.lite.NeuronDelegate

   val options = NeuronDelegate.Options()
       .setExecutionPreference(NeuronDelegate.Options.EXECUTION_PREFERENCE_FAST_SINGLE_ANSWER)
       .setAllowFp16(true)                 // float models: MDLA 2.0 runs fp16 only
       .setAcceleratorName("mtk-mdla")     // the NPU; exact name
       .setCompileOptions("--relax-fp32")
       .setCacheDir(File(context.codeCacheDir, "neuron/${modelToken}").apply { mkdirs() }.path)
   val delegate = NeuronDelegate(options)
   val interpreter = Interpreter(modelFile, Interpreter.Options().addDelegate(delegate))  // compiles here
   interpreter.run(input, output)
   interpreter.close()                     // interpreter first,
   delegate.close()                        // then the delegate
   ```

   `modelToken` must change whenever the model file changes (see "Compile cache"). A complete helper with
   both cache modes is in [`examples/kotlin`](examples/kotlin): `NeuronModel.open(context, file).use { it.run(in, out) }`.
5. Check logcat (tag `tflite`) on the first run:
   - `NeuronApi version: 8.2.26` (or similar) and **no** `neuron error: Neuron adapter API exists: 0` line, or
     the graph went to NNAPI instead of the NPU;
   - `Replacing N node(s) with delegate (TFLiteNeuronDelegate) node, yielding K partitions`: K = 1 is the goal;
   - `OP <NAME> (v<n>) is not supported (<reason>)` for each op that stays on the CPU.
   - Compare the outputs with a CPU run of the same model. A graph can compile, run and still be wrong.

To list the Neuron devices of a phone without the delegate, see
[`examples/native/neuron_devices.cpp`](examples/native/neuron_devices.cpp).

## Options reference

All setters return the `Options` object. Patch 0001 adds everything after `setAllowFp16`.

| Setter | Default | What it does, and what we learned |
|---|---|---|
| `setExecutionPreference(int)` | `EXECUTION_PREFERENCE_FAST_SINGLE_ANSWER` (1) | also `_LOW_POWER` (0), `_SUSTAINED_SPEED` (2), `_UNDEFINED` (-1). FAST_SINGLE_ANSWER is turned into the compiler's TurboBoost preference and a 2000 ms performance boost. We only used the default. |
| `setAllowFp16(boolean)` | false | `NeuronModel_relaxComputationFloat32toFloat16`. Required for float models on MDLA 2.0 ("MDLA: cannot support Float32 input/output" otherwise). We always set it together with `--relax-fp32`. |
| `setExecutionPriority(int)` | `EXECUTION_PRIORITY_HIGH` (110) | `_LOW` 90, `_MEDIUM` 100. Not measured. |
| `setOptimizationHint(int)` | `OPTIMIZATION_LOW_LATENCY` (1) | bit mask: `_NONE` 0, `_LOW_LATENCY` 1, `_DEEP_FUSION` 2, `_BATCH_PROCESSOR` 4. Not measured. |
| `setBoostDuration(int ms)` | 0 | 0 = 2000 ms for FAST_SINGLE_ANSWER, 10 ms otherwise. Negative throws. Not measured. |
| `setAcceleratorName(String)` | null (Neuron picks) | `"mtk-mdla"` (NPU), `"mtk-dsp"` (VPU), `"mtk-gpu"`, or a comma list. Exact Neuron names, not NNAPI's `*_shim` names. An unknown name only logs `Could not find the specified Neuron accelerator` and compiles for *all* devices. Max 49 bytes (throws above). The VPU on MT6877 takes no float at all. |
| `setCompileOptions(String)` | null | Passed to the Neuron compiler after the built-in `--opt-accuracy --opt 3` (a second `setOptimizationString` call; whether it appends or replaces is not documented). We used `--relax-fp32`. Also carries patch 0003's `--npustt-exclude=OP1,OP2` and `--npustt-max-softmax=N` tokens, which keep op types on the CPU and are stripped before the compiler sees the string. Max 199 bytes including tokens (throws above). |
| `setCacheDir(String)` | null | Absolute folder for the compile cache; null/empty = no cache. Alone it selects MediaTek's own cache. |
| `setModelToken(String)` | null | With `setCacheDir`, selects patch 0004's DLA cache in `<cacheDir>/npustt-dla/<token>/`. (The Javadoc in patch 0001 still says the token is unused; that predates 0004.) |

`NeuronDelegate(options)` throws `IllegalArgumentException` for a too-long string. The compile itself runs
inside the `Interpreter` constructor.

## Compile cache

The first time a model is loaded, the Neuron compiler compiles every delegated partition on the phone. That
is fast for small models and slow for big ones:

| Model | First load (compile) | Cached load |
|---|---|---|
| MobileNetV2 int8 | 0.3–0.4 s | 37 ms (MediaTek cache) / 57 ms (DLA cache) |
| 6 transformer layers, 300 MB float32, one partition | 207 s | 0.19 s (MediaTek) / 0.24 s (DLA) |
| 24 layers, 1.3 GB float32, as 4 one-partition files | 854–870 s | 0.8–2.0 s |
| 24 layers as one file cut into 49 partitions by CPU ops | 745 s | 2.9–3.2 s (DLA) / 3.1 s (MediaTek) |

Two cache routes, both work on the tested phone:

**MediaTek's own cache** (`setCacheDir(dir)` without a model token; upstream behaviour): the delegate calls
`NeuronCompilation_setCaching(dir, token)` and the adapter writes two files per compilation. Logcat (tag
`neuron`): `Cache is not available` = ordinary miss (it is printed before every first compile; it does not
mean caching is broken), `Loaded compiled network` = hit. It keeps no heap copy of the network, so it is the
lighter route. **Its key is a fingerprint of the graph structure (tensor count, node and I/O indices), not of
the weights.** Two models with the same structure in one cache folder load each other's compiled networks:
we got silently wrong outputs (no error) from three same-shape sections of one model, and the same would
happen with a retrained model of the same architecture. **Give every model file its own cache folder**,
named by a token over the file's identity, e.g. MD5 of `name|size|mtime|accelerator|fp16|compileOptions`
([`examples/kotlin/NeuronCache.kt`](examples/kotlin/NeuronCache.kt) does this).

**DLA cache** (patch 0004; `setCacheDir(root)` + `setModelToken(token)`): the delegate stores each compiled
network with `NeuronCompilation_storeCompiledNetwork` as `<root>/npustt-dla/<token>/<key>.dla` and restores it
with `NeuronModel_restoreFromCompiledNetwork`. The per-partition `<key>` is FNV-1a 64 over the graph
fingerprint, accelerator name, compile options, fp16 flag, Neuron runtime version and a format tag; the weights
are again not in it, so the token must identify the model file. It does not depend on the vendor cache, but
the adapter copies every restored network into its heap: about the size of the `.dla` file per network
(0.6 GB for a 1.3 GB float32 model), on top of what the MediaTek route costs. Logs (tag `tflite`):
`npustt: dla restore key=... result=hit|miss|fail`, `npustt: dla store key=... result=ok|fail`.

Our recommendation: MediaTek's cache with one folder per model token; DLA mode when you cannot rely on the
vendor cache. Either way:

- Not in any key: execution preference, priority, optimization hint, boost. Clear the cache after changing them.
- A firmware update can change the NPU driver. `Context.codeCacheDir` is emptied by Android on every app
  update and every platform update, which makes it a safe place; for development, a folder that survives
  `adb install -r` (e.g. under `getExternalFilesDir(null)`) saves recompiles.
- Disk: about the compiled network size, roughly the weights at fp16 (150 MB per 300 MB float32 section).
- The folders the delegate creates may not be deletable from the adb shell; clear them from the app.
- One model per process at a time per cache folder (the DLA store uses one fixed `.tmp` name per key).

## What the NPU accepts (MDLA 2.0)

Summary of probe runs (full table and method: [`probes/README.md`](probes/README.md)):

- Float models need fp16 relaxation; int8 CNNs run as they are.
- Weights must be float32 constants in the file (the compiler converts them); fp16-weight files with
  `DEQUANTIZE` are refused.
- Refused: `BATCH_MATMUL`, the `GELU` op, `nn.LayerNorm` as litert-torch lowers it, big `SOFTMAX` inputs
  (e.g. [16,256,256], while [4,512,512] and [16,128,128] pass), reshapes with a dimension > 65535, wide 1-D
  convolutions (filter width 128) and a filter-width-10, stride-5 depthwise conv.
- Plain 1-D convolutions with kernel 2 or 3 and stride 2 (`nn.Conv1d` as exported: `CONV_2D` with
  `TRANSPOSE`s) are accepted. Faster still: a kernel-3 stride-2 conv as two `FULLY_CONNECTED`s on the frames
  reshaped into pairs plus an `ADD` (2× faster, compiles in 0.2 s instead of 90 s), kernel 2 as reshape + one FC.
- Accepted replacements: LayerNorm written out (mean, subtract, mean of squares, rsqrt), GELU as
  `x·sigmoid(1.5957691·(x + 0.044715·x³))`, matmuls as `FULLY_CONNECTED` / 1×1 `CONV_2D` (with a runtime weight
  tensor the sizes must be 16-aligned, e.g. 512 instead of 499), softmax per attention head.
- Silent miscompiles (compiled, ran, wrong numbers): a LayerNorm whose means were 1×1 convolutions (cosine
  0.41 at one shape), and a `CONCATENATION` along channels feeding one FC (im2col of a conv: cosine 0.95 alone,
  0.66 in a stack; the same maths as a sum of FCs is right). Always compare NPU outputs with the CPU, and probe
  composed graphs whole: parts that were right alone drifted or were refused together.
- One refused op fails the whole compile (`NEURON_UNMAPPABLE`, `Compile error: NoExecPlan`); the compiler logs
  the real reason per op (`<OP>: MDLA: <reason>`). Fix the export, or keep that op type on the CPU with
  `--npustt-exclude=<OP>` (patch 0003), at the price of more partitions.

## Memory

Measured with `dumpsys meminfo` / `Debug.MemoryInfo` on the 8 GB phone:

- **The driver holds memory per loaded network.** It shows up as "EGL mtrack" (Graphics) in `dumpsys meminfo`,
  not in the process's smaps: about 0.38–0.40 GB for four networks of a 1.3 GB float32 model, from the load on.
  It is returned at once when the interpreter and delegate are closed (379–401 MB → 24–46 MB, the same
  337.9 MB returned in each of 10 load/close rounds).
- **Partitions cost memory, not only compile time.** Every compiled network gets heap blocks that are first
  touched at the first execution. A 6-layer section as one partition added about 80 MB at the first
  inference; the same section cut into 13 partitions by CPU softmaxes added 432 MB. A 24-layer model in 49
  partitions added 2.6 GB at the first inference, and one with 145 partitions failed its first run with
  `APUSys failed to allocate data buffer` (APU memory exhausted).
- **Compile peaks are high.** Compiling one 6-layer partition peaked at 2.44 GB RSS; a 24-layer single
  partition was killed by the low-memory killer at 4.4 GB after 13.5 minutes. The cache pays this once, but
  the first start must survive it: compile with little else in memory.
- **Model files:** use float32 weights and let TFLite memory-map the file. With a cache hit the compiler never
  reads the weights, and the file pages are not resident. (An fp16-weight file is not only refused by the NPU;
  TFLite's CPU path also unpacks it to float32 in RAM.)
- **After close**, freed native heap can stay resident (246 MB in our case) until the allocator returns it:
  `mallopt(M_PURGE_ALL, 0)` (API 34+, else `M_PURGE`) from your own JNI code took the process from about
  310 MB to about 100 MB, every round, without creep. `Debug.getPss()` read about 346 MB too high right after a
  close; `Debug.MemoryInfo` was right.
- Result for the 1.3 GB float32 transformer: 3.9 GB process memory (one file, 49 partitions, DLA restore)
  → 1.0 GB (4 one-partition sections, MediaTek cache in per-model folders, and the CPU-side stage run in
  tiles to cap its work buffers) → about 0.1 GB between uses (closed and purged; reload 0.85–0.94 s).

## Splitting large models into sections

Big graphs hit three limits at once: the compile peak of one big partition (RAM), the per-partition memory
and APU buffers of many small partitions, and ops the NPU refuses. What worked:

1. **Cut the model into several `.tflite` files ("sections")** at block boundaries in the source framework:
   a module that runs blocks i..j, takes the hidden state in and returns it, exported with fixed shapes.
   Pick the section size by the compile peak: the largest section whose whole compile fits in RAM with margin
   (6 of 24 layers, 2.4 GB peak, on the 8 GB phone).
2. **Make every NPU section one whole partition**: only accepted forms inside (see above), no CPU islands.
   Parts the NPU cannot take go into their own CPU-only `.tflite`. A frame-local front end on a long input
   (e.g. a conv feature encoder on 10 s of audio) can be cut into tiles so that every dimension stays ≤ 65535;
   then the tile runs as a short chain, a CPU file for the layer the NPU handles badly and an NPU file for the
   rest (a 6-block conv encoder: 263 ms on the CPU → 65 ms on the NPU per tile; details in the probe notes).
3. **Run the sections in order** in one process, one `Interpreter` + `NeuronDelegate` per section, the output
   buffer of one feeding the next.
4. **Check the cut on the PC**: the chained sections must reproduce the unsplit model bit for bit on the same
   CPU kernels; then check each section on the device against its CPU output.
5. **Cache per file**: sections of one model usually share one structure, which is exactly the case where
   MediaTek's structure-keyed cache mixes them up. One cache folder (or DLA token) per section file.

For the 24-layer model this took the NPU time from 2.24 s (one file, 49 partitions, softmax on the CPU) to
1.28–1.35 s (4 sections, softmax on the NPU), and the memory from 3.25–3.9 GB to 1.0 GB.

## Known limits and pitfalls

- **Exactly TFLite 2.8.** The delegate is built against TF 2.8 and the runtime reads its `TfLiteDelegate` /
  `TfLiteRegistration` structs. With LiteRT 1.4.2 it fails in `prepare` with a misleading error
  (`conv.cc:335 has_bias || node->inputs->size == 2 was not true ... (TFLiteNeuronDelegate) failed to prepare`),
  not a crash. Use `org.tensorflow:tensorflow-lite:2.8.0` and do not mix in another TFLite/LiteRT runtime
  (untested). The `.so` exports and imports no TFLite symbols.
- **No Neuron device means NNAPI, not CPU.** If the adapter cannot be opened or lists no `mtk-dsp*` /
  `mtk-mdla*` device, the delegate logs `neuron error: Neuron adapter API exists: 0` and passes the graph to an
  internal NNAPI kernel, while TFLite still reports `TFLiteNeuronDelegate`. Same when the system property
  `debug.tflite.neuron.redirect_nnapi` is `1`. Check for `NeuronApi version: x.y.z`.
- **4 KB page alignment.** The `.so` has 4 KB-aligned segments: it loads on 4 KB-page devices (like the A34)
  but not on 16 KB-page devices, and app stores that check 16 KB alignment may flag it.
- **Wrong results without errors** happen: a miscompiled construct (see probes) and a cache shared between
  same-structure models both gave silently wrong outputs. Compare against the CPU for every new model or form.
- **Transient APU buffer failures** (`APUSys failed to allocate data buffer`, `NEURON_BAD_DATA`) appeared
  with a 49-partition model and went away on a retry; a 4-section model never showed them. Retry a failed first
  run once; run a small model as a health check before blaming a change.
- **Timing:** the first inference after a load is slower than later ones; take timings after warm-ups
  (10 for small models).
- The delegate prints its node summary once per process.
- The benchmark tool `benchmark_model_plus_neuron_delegate` (built with `build/build.sh --bench`) has no flags
  for the cache directory or model token.
- The NNAPI path (`mtk-*_shim` accelerators through TFLite's NNAPI delegate) runs small int8 CNNs well but took
  at most a quarter of a transformer graph and was slower than the CPU there.

## Building from source

```bash
build/build.sh            # clone upstream at abb45cca, apply patches 0001-0004, bazel build the AAR
```

Toolchain (Bazel 4.2.1, NDK r21e, JDK 11, SDK platform 28, Python with numpy), versions, hashes and the
expected output: [`build/README.md`](build/README.md). A rebuild reproduces the release AAR's members byte for
byte. Host unit tests for the patch helpers: `tests/run_host_tests.sh`.

## Repository layout

```
patches/            the four patches (unchanged) + per-patch description
build/build.sh      clone, patch, build, verify; build/README.md: toolchain
tests/              host unit tests of the filter parser and the DLA cache helpers
examples/kotlin/    NeuronModel.kt (load, run, close), NeuronCache.kt (cache folders and tokens)
examples/native/    neuron_devices.cpp: list the Neuron runtime version and devices
probes/             what MDLA 2.0 accepts, and how to probe your own constructs
LICENSES/, NOTICE   third-party licences (MediaTek MIT, TensorFlow Apache-2.0, FarmHash MIT)
```

## Licence and credits

MIT, see [LICENSE](LICENSE). The delegate is MediaTek's work
([tflite-neuron-delegate](https://github.com/MediaTek-NeuroPilot/tflite-neuron-delegate), Copyright (c) 2021
MediaTek Inc., MIT); the patches only add to it. The prebuilt AAR also contains parts of TensorFlow Lite 2.8.0
(Apache-2.0) and FarmHash (MIT), linked statically; details and the licence texts are in [NOTICE](NOTICE) and
[LICENSES/](LICENSES). Operation codes and Neuron API signatures used by patches 0002 and 0004 were checked
against Google's LiteRT MediaTek plugin (Apache-2.0), ONNX Runtime's NeuroPilot execution provider (MIT) and
ExecuTorch's MediaTek backend (BSD). MediaTek, NeuroPilot and Dimensity are trademarks of MediaTek Inc.; this
project is not affiliated with MediaTek or Samsung.
