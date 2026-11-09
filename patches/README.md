# patches

Four patches against MediaTek's MIT TFLite Neuron delegate
(`https://github.com/MediaTek-NeuroPilot/tflite-neuron-delegate`, commit
`abb45cca4d22a7b02cdd87191d7c1fef44235c1c`, the last upstream commit; its last code change is from 2022).
Apply them in order, one `git apply` each, on a pristine checkout (`build/build.sh` does this):

```bash
git clone https://github.com/MediaTek-NeuroPilot/tflite-neuron-delegate
cd tflite-neuron-delegate
git checkout abb45cca4d22a7b02cdd87191d7c1fef44235c1c
for p in 0001-expose-options 0002-batch-matmul-gelu 0003-op-exclusion-filters 0004-dla-cache; do
  git apply --check ../mediatek-npu-bridge/patches/$p.patch && git apply ../mediatek-npu-bridge/patches/$p.patch
done
```

| Patch | sha256 | Files | Lines |
|---|---|---|---|
| `0001-expose-options.patch` | `43bff266ba8037eba8b7e4dc03077b0439df779043e32a4c557d499ff97d0f18` | 2 | +220 −6 |
| `0002-batch-matmul-gelu.patch` | `239c5f18c14cceea8b9d8e5a2578630ca7fce7cb564aff74383820a13e12dacb` | 4 | +74 |
| `0003-op-exclusion-filters.patch` | `7193cfcbd95af37966aa5227bfba4a84dcad8efbb6026037ab287f46c320f970` | 3 | +136 −1 |
| `0004-dla-cache.patch` | `06a3e9cd59849afd6a3358425abc0f746805763237dc29e16ed84112d76188cb` | 8 | +413 −1 |

The patches must keep LF line endings (`.gitattributes` marks `*.patch` as binary-safe `-text`). If a
Windows checkout turned them into CRLF, `git apply` fails.

Names: the patches were written inside an earlier project, so the option tokens, log lines, two new header
files and one compile macro carry the prefix `npustt` (`--npustt-exclude`, `npustt: dla ...`,
`neuron/npustt_filters.h`, `NPUSTT_GELU_APPROX_INT32`, the cache folder `npustt-dla`), and some code comments
say "npu-stt 0002" etc. One comment in 0004 points to `notes/spec-dla-cache.md`, a document of that project;
its content (the cache-key inputs) is the 0004 section below. They are kept as they are, so the patches stay
byte-identical to the ones the release AAR was built from.

## 0001-expose-options.patch

The upstream Java API only exposes `setExecutionPreference(int)` and `setAllowFp16(boolean)`. This patch
exposes the rest of `NeuronDelegateOptions` (`neuron/neuron_delegate.h`) to Java. No C++ delegate code changes.

`NeuronDelegate.java`, class `Options`:

| Setter | Native field | Default | Notes |
|---|---|---|---|
| `setExecutionPriority(int)` | `execution_priority` | `EXECUTION_PRIORITY_HIGH` (110) | constants `EXECUTION_PRIORITY_LOW/MEDIUM/HIGH` = 90/100/110 |
| `setOptimizationHint(int)` | `optimization_hint` | `OPTIMIZATION_LOW_LATENCY` (1) | bit mask: `OPTIMIZATION_NONE/LOW_LATENCY/DEEP_FUSION/BATCH_PROCESSOR` = 0/1/2/4 |
| `setBoostDuration(int ms)` | `boost_duration` | 0 | 0 = the delegate picks 2000 ms (fast single answer) or 10 ms; negative throws `IllegalArgumentException` |
| `setAcceleratorName(String)` | `accelerator_name[50]` | null (Neuron chooses) | exact device names, comma list, e.g. `"mtk-mdla"`; > 49 bytes throws |
| `setCompileOptions(String)` | `compile_options[200]` | null | e.g. `"--relax-fp32"`; applied after the built-in `"--opt-accuracy --opt 3"`; > 199 bytes throws |
| `setCacheDir(String)` | `cache_dir` | null (no caching) | null or empty disables compile caching |
| `setModelToken(String)` | `model_token` | null | unused by the upstream delegate; with 0004, cache dir + token switch on the DLA cache |

The Javadoc of `setModelToken` in this patch still says the delegate never reads the token: true for
0001–0003, out of date once 0004 is applied.

The native method becomes `createDelegate(int preference, boolean allowFp16, int executionPriority,
int optimizationHint, int boostDuration, String acceleratorName, String compileOptions, String cacheDir,
String modelToken)`.

`neuron_delegate_jni.cc`:
- `CopyJString` copies a Java string into the fixed `char[]` fields with `snprintf` (always NUL-terminated);
  a string that does not fit throws `IllegalArgumentException` instead of being cut.
- `DupJString` `strdup`s `cache_dir` / `model_token`, because the delegate keeps only the pointers. The copies
  are kept in a `std::map<TfLiteDelegate*, pair<char*, char*>>` under a mutex and freed in `deleteDelegate`
  (the entry is taken out of the map before the delegate is deleted).

## 0002-batch-matmul-gelu.patch

Lets the delegate hand TFLite `BATCH_MATMUL` and `GELU` to the Neuron compiler instead of rejecting them.
No Java/JNI change. These are plain Neuron operation codes (not the `MTKEXT_*` extension strings), used the
same way by Google's LiteRT MediaTek compiler plugin and ONNX Runtime's NeuroPilot EP:

| TFLite op | Neuron op (code) | Operands, in order | Output |
|---|---|---|---|
| `BATCH_MATMUL` (v1-4) | `NEURON_BATCH_MATMUL` (102, Neuron ≥ 5.1.2) | `A`, `B`, `BOOL adj_x`, `BOOL adj_y` | 1 tensor |
| `GELU` (v1) | `NEURON_GELU_V2` (109) | `x`, `UINT32 approximate` (1 if TFLite `approximate`, else 0) | 1 tensor |

Delegated only when: `BATCH_MATMUL` float32 A, B and output, same rank 2..4, equal batch dimensions (no
broadcasting), **B not a constant** (ONNX Runtime's EP reports garbage results for a constant B on the MDLA;
export weight matmuls as `FULLY_CONNECTED`); `GELU` float32, op version 1. Everything else stays on the CPU
with an `OP … is not supported (…)` log line.

**On MT6877 / MDLA 2.0 both ops are refused by the compiler** ("MDLA: unsupported operation"), so on that
chip the patch only changes the error, and you exclude them with 0003. It is kept because newer MDLA versions
may accept them (untested).

The scalar type of `approximate` is uncertain: LiteRT uses `UINT32` (the default build), ONNX Runtime's EP
`INT32` (build with `--copt=-DNPUSTT_GELU_APPROX_INT32`). `approximate = 0` (exact GELU) has no public precedent.

Sources (open licences, read 2026-11-04): `google-ai-edge/LiteRT` (Apache-2.0)
`litert/vendors/mediatek/compiler/legalizations/batch_matmul_op_legalization.cc`, `gelu_op_legalization.cc`
(commit `6aec6fa8e2949f326c64e1e0714ffcf83356597e`); `onnxruntime/onnxruntime-ep-neuropilot` (MIT)
`neuron_lib/neuron_types.h`, `builders/op_impl/gelu_op_builder.cc`, `builders/op_builder_helpers.cc`
(commit `d940a3450a992d2066625dbf75c1b460ba732410`).

## 0003-op-exclusion-filters.patch

Keeps chosen op types on the CPU on request, so one op the MDLA compiler refuses no longer fails the whole
compile. New `neuron/npustt_filters.h`, changes in `neuron/neuron_delegate.cc` and `neuron/BUILD`. No Java/JNI
change: the filters travel inside the compile-options string.

| Token (inside `setCompileOptions(...)`) | Effect |
|---|---|
| `--npustt-exclude=OP1,OP2,...` | ops with these TFLite builtin names stay on the CPU (case-insensitive; names as printed by `EnumNameBuiltinOperator`, e.g. `SOFTMAX`, `BATCH_MATMUL`, `GELU`, `MEAN`, `FULLY_CONNECTED`, `CONV_2D`). May be repeated; unknown names are logged and ignored. |
| `--npustt-max-softmax=N` | a `SOFTMAX` node whose input has more than N elements stays on the CPU. `0` or absent = no limit. |

- The tokens are removed before the rest reaches `NeuronCompilation_setOptimizationString`; the remaining
  options keep their order (joined by single spaces). Without any `--npustt-` token the string is passed on
  unchanged.
- Log (tag `tflite`), once per delegate when a filter is active:
  `npustt: excluded ops: [SOFTMAX,BATCH_MATMUL,GELU], max softmax elements: 0, compile options passed on: "--relax-fp32"`;
  per filtered node: `OP SOFTMAX (v1) is not supported (excluded by --npustt-exclude)`.
- The whole string, tokens included, must stay within 199 bytes. The benchmark tool copies the flag with an
  unchecked `strcpy`, so keep it short there too.

Examples:
```
setCompileOptions("--relax-fp32 --npustt-exclude=BATCH_MATMUL,GELU,SOFTMAX")
setCompileOptions("--relax-fp32 --npustt-exclude=BATCH_MATMUL,GELU --npustt-max-softmax=1000000")
benchmark_model_plus_neuron_delegate --use_neuron=true --neuron_accelerator_name=mtk-mdla --neuron_allow_fp16=true \
  --neuron_compile_options="--relax-fp32 --npustt-exclude=SOFTMAX" --graph=model.tflite
```

Every CPU island splits the graph into more partitions, and each partition costs compile time and memory
(main README, "Memory").

## 0004-dla-cache.patch

Compiled-network ("DLA") cache: each Neuron compilation is stored once with
`NeuronCompilation_storeCompiledNetwork` and restored at the next start with
`NeuronModel_restoreFromCompiledNetwork`, so the second start skips the compile and never reads the weight
bytes. New `neuron/npustt_dla_cache.h` (pure helpers, host-tested), restore/store in
`neuron_delegate_kernel.{h,cc}` (`Prepare`), one log line in `neuron_delegate.cc`, two optional function
pointers in `neuron_implementation.{h,cc}` + `neuron_types.h`, `BUILD`. No Java/JNI change: `setCacheDir` /
`setModelToken` from 0001 are reused.

| `cache_dir` | `model_token` | Behaviour |
|---|---|---|
| null/empty | any | no caching (unchanged) |
| set | null/empty | MediaTek's own `NeuronCompilation_setCaching` path (unchanged) |
| set | set | **DLA mode**: files `<cache_dir>/npustt-dla/<model_token>/<key>.dla`, no `setCaching` call |
| set | set, but an adapter symbol is missing | log `npustt: dla cache unavailable (missing <symbol>)`, `setCaching` path |

- Key per partition: FNV-1a 64, 16 lowercase hex chars, over the kernel's `nn_compilation_cache_token_`
  (32 bytes, a fingerprint of the graph structure, not of the weights), the accelerator name, the built-in
  `--opt-accuracy --opt 3`, `compile_options` (after 0003 removed its tokens), the fp16 flag, the Neuron
  version string (e.g. `8.2.26`) and the literal `0004`; each field is followed by its length. The weights
  are not in the key: the model token keeps different model files apart.
- Token characters outside `[A-Za-z0-9._-]` become `_` in the path (an MD5 hex token is unchanged); `.` and
  `..` become `_`.
- Restore (start of `Prepare`): `open` + `fstat` + `mmap(PROT_READ, MAP_PRIVATE)` → restore → the restored model
  and compilation replace the ones from `Init`; `createForDevices`, `setPreference`, `setPriority`,
  `setOptimizationString`, `setCaching`, `setL1MemorySizeKb` and `finish` are skipped. The mapping is kept until
  the kernel is destroyed; its pages are dropped with `madvise(MADV_DONTNEED)` right after the restore (the
  adapter copies the buffer into its own heap). Any error: log, `unlink`, normal compile.
- Store (after a successful `NeuronCompilation_finish`, before any execution): `getCompiledNetworkSize` →
  `malloc` → `storeCompiledNetwork` → write `<key>.tmp` + `fsync` → `rename` to `<key>.dla` → `free`.
  Directories are created as needed. Never fatal.
- Log lines (tag `tflite`, INFO, one line each):
  ```
  npustt: dla cache dir=<cache_dir> token=<model_token> neuron=<x.y.z>      (once per delegate)
  npustt: dla restore key=<key> result=hit bytes=<n> ms=<n>
  npustt: dla restore key=<key> result=miss
  npustt: dla restore key=<key> result=fail code=<n>
  npustt: dla store key=<key> result=ok bytes=<n> ms=<n>
  npustt: dla store key=<key> result=fail code=<n> stage=size|store|write
  npustt: dla cache unavailable (missing <symbol>)                           (instead of the first)
  ```
  `code` > 0 is a Neuron error code; `code` < 0 is `-errno` (`-22` = empty file or size 0, `-12` = malloc
  failed, `-14` = restore said OK but returned a null pointer).
- Caveats: execution preference, priority, optimization hint, L1 size and boost are not in the key (clear the
  cache after changing them); one fixed `<key>.tmp` name, so two processes writing the same cache dir + token
  at once could collide; on a Neuron restore error nothing the adapter may have allocated is freed; the
  benchmark tool has no flags for `cache_dir` / `model_token`, so DLA mode is reachable from Java only.
- Sources (open licences, read 2026-11-04): ExecuTorch (BSD, MediaTek copyright)
  `backends/mediatek/runtime/include/api/NeuronAdapterShim.h` for the exact signatures; LiteRT (Apache-2.0)
  `litert/vendors/mediatek/compiler/compiler_plugin.cc` (size → store),
  `dispatch/litert_dispatch_invocation_context.cc` (restore) and `dispatch/file_backed_pages.cc` (`MADV_DONTNEED`).

Host tests for the two new headers: `tests/run_host_tests.sh` (58 checks for the cache helpers, plus the
filter parser).
