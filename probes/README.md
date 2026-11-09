# What the MDLA 2.0 accepts (MT6877 / Dimensity 1080)

Measured with the patched delegate on one phone (Galaxy A34 5G, Android 16, Neuron adapter 8.2.26),
`setAcceleratorName("mtk-mdla")`, `setAllowFp16(true)`, `setCompileOptions("--relax-fp32")`, float32 TFLite
models exported from PyTorch with litert-torch. Each probe is a tiny fixed-shape graph that holds one
construct; "ok" means it compiled, ran, and matched the CPU reference (cosine ≥ 0.9999 unless noted).
Compiler messages are quoted as the adapter logs them (log tag `neuron`/`tflite`).

## Method (generic)

1. Build one small `.tflite` per construct at the real shapes of your model (one op, or a short op chain),
   fixed shapes, float32 weights. Save a seeded random input and the framework's float32 output next to it.
2. Run each on the device twice: CPU (TFLite built-in kernels) and the delegate on `mtk-mdla`. Record:
   delegated nodes / total, number of partitions, compile time, median run time, cosine and max-abs
   difference against the reference, and the compiler's per-op reasons when the compile fails.
3. Read the reason lines: the compiler prints one line per refused op (e.g. `SOFTMAX: MDLA: cannot fit on
   internal buffers`), even though the error returned to the app is only `NEURON_UNMAPPABLE` /
   `Compile error: NoExecPlan`.

A probe compiles in seconds; a whole large model can take many minutes per attempt. Probe first, then change
the export.

## Results

| Construct | Shapes tried | Result on MDLA 2.0 |
|---|---|---|
| Float model without fp16 relaxation | any | refused: "MDLA: cannot support Float32 input / Float32 output". Use `setAllowFp16(true)` + `--relax-fp32` |
| Quantized int8 CNN (MobileNetV2) | 224×224 | ok, 65/65 nodes, 1 partition, 1.92 ms (4-thread CPU 12.55 ms) |
| `FULLY_CONNECTED`, constant fp32 weights | 1024→4096 on [1,512,1024] | ok, 14.3 ms (CPU 91.7 ms) |
| Same with fp16 weights + `DEQUANTIZE` (fp16-weight file) | same | refused (no compiler report): keep float32 weights in the file, the compiler relaxes them itself |
| `nn.LayerNorm` exported as is (litert-torch's MEAN lowering) | [1,499,1024], [1,1500,512] | refused: "MEAN: MDLA: Cannot support the Mean which needs to create op with unsupported batch size" |
| LayerNorm written out: `d = x - mean(x)`, `var = mean(d·d)`, `d·rsqrt(var+eps)·w + b` (MEAN/SUB/MUL/ADD/RSQRT over the last axis) | [1,499,1024], [1,1500,512], also on a [1,T,1,C] view | ok, 1 partition |
| `MEAN` (LayerNorm) on a very long axis | [1,31999,512] | refused: "cannot fit on internal buffers" |
| LayerNorm mean/variance as `FULLY_CONNECTED` with a constant 1/C weight | [1,499,1024] | ok |
| LayerNorm mean/variance as 1×1 `CONV_2D` with constant 1/C weights | [1,499,1024] | **compiles and runs but wrong (cosine 0.41)**; correct at [1,1500,512]. Silent miscompile: always check outputs |
| `GELU` op (tanh or exact) | [1,499,4096] | not delegated by the upstream delegate; with patch 0002 mapped to `NEURON_GELU_V2` and refused: "MDLA: unsupported operation" |
| GELU written as x·sigmoid(1.5957691·(x + 0.044715·x³)) (MUL/ADD/LOGISTIC) | [1,499,4096] | ok, 1 partition (same function as tanh-GELU) |
| `BATCH_MATMUL` | [16,499,64] attention | refused: "MDLA: unsupported operation" (with 0002); no batch matmul on MDLA 2.0 |
| `FULLY_CONNECTED` / 1×1 `CONV_2D` with a **runtime** (non-constant) weight tensor | T = 499 | refused: "dynamic filter batch should be 16-aligned", "dynamic filter channel should be 32-bytes aligned" |
| Same | T = 512 (padded) | ok: attention as q·kᵀ via FC with runtime weight [512,64] and p·v via 1×1 CONV with runtime filter, 248/248 nodes, 1 partition, 27.9 ms per 16-head layer core |
| `SOFTMAX` (last axis) | [1,512,512], [1,499,499], [4,512,512], [16,128,128] | ok (3.8 / 4.0 / 13.6 / 9.6 ms) |
| `SOFTMAX` | [16,256,256], [16,499,499], [1,16,499,499], [16,512,512] | refused: "cannot fit on internal buffers" ([16,256,256] has the same element count as the accepted [4,512,512], so the limit is not a plain element count) |
| `RESHAPE` with a dimension > 65535 | a [1,163920] input | refused: "Cannot support the Reshape of which dimension may exceed 65535" |
| `DEPTHWISE_CONV_2D` with filter width 10, stride 5, multiplier 512 | 1-D strided conv on a long input | refused: "filter width (10)", "stride width (5)", "invalid multiplier (512)" |
| `CONV_2D` with filter width 128, padding 64 | wide 1-D conv | refused: "filter width (128)", "invalid padding.left: 64" |
| VPU (`mtk-dsp`) with float16 / float32 | any float model | refused: "VPU: unsupported data type: Float16" (and Float32); int8 MobileNetV2 runs at 19.97 ms (slower than the CPU) |

A full transformer layer (hidden 1024, 16 heads, FFN 4096, T = 512) built only from the accepted forms above
(decomposed LayerNorm, sigmoid-form GELU, FC/CONV attention with runtime weights, per-head softmax) compiles
as one partition and runs at 53–57 ms per layer on the MDLA against 315–320 ms on the 4-thread CPU, with
cosine ≥ 0.99997 against float32.

One refused op fails the compile of the whole graph (no per-op fallback inside the Neuron compiler), so
either change the export form or keep that op type on the CPU with patch 0003's `--npustt-exclude=...`.
