package com.example.npu

import android.content.Context
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.NeuronDelegate
import java.io.Closeable
import java.io.File

/**
 * One TFLite model on the MediaTek NPU through the patched Neuron delegate.
 *
 * Dependencies: `org.tensorflow:tensorflow-lite:2.8.0` (exactly this runtime, see the README) and
 * `tensorflow-lite-neuron-mtk-bridge-0.1.0.aar`.
 *
 * ```
 * NeuronModel.open(context, File(context.filesDir, "model.tflite")).use { m ->
 *     m.run(input, output)          // same buffers/arrays as Interpreter.run
 * }
 * ```
 *
 * The first open of a model compiles it on the device (seconds to minutes for big models); with a cache
 * (see [NeuronCache]) later opens load the compiled network instead. Open and run on a background thread.
 */
class NeuronModel private constructor(
    val interpreter: Interpreter,
    private val delegate: NeuronDelegate,
) : Closeable {

    data class Config(
        /** Exact Neuron device name: "mtk-mdla" (NPU), "mtk-dsp" (VPU) or "mtk-gpu"; a comma list is allowed. */
        val accelerator: String = "mtk-mdla",
        /** fp16 relaxation of float models; MDLA 2.0 does not run float32 at all. */
        val fp16: Boolean = true,
        /** Extra compiler options; null = "--relax-fp32" when [fp16], else none. Max 199 bytes. */
        val compileOptions: String? = null,
        val cache: NeuronCache.Mode = NeuronCache.Mode.ADAPTER,
        /** Threads for the TFLite CPU kernels that run the ops the delegate does not take. */
        val cpuThreads: Int = 4,
        /** XNNPACK for the CPU leftovers. Off in all our NPU measurements. */
        val useXnnpack: Boolean = false,
    ) {
        val effectiveCompileOptions: String?
            get() = compileOptions ?: if (fp16) "--relax-fp32" else null
    }

    fun run(input: Any, output: Any) = interpreter.run(input, output)

    fun runForMultipleInputsOutputs(inputs: Array<Any>, outputs: Map<Int, Any>) =
        interpreter.runForMultipleInputsOutputs(inputs, outputs)

    /** Interpreter first, then the delegate. Closing returns the driver's memory for the compiled network. */
    override fun close() {
        interpreter.close()
        delegate.close()
    }

    companion object {
        fun open(context: Context, model: File, config: Config = Config()): NeuronModel {
            val compile = config.effectiveCompileOptions
            val options = NeuronDelegate.Options()
                .setExecutionPreference(NeuronDelegate.Options.EXECUTION_PREFERENCE_FAST_SINGLE_ANSWER)
                .setAllowFp16(config.fp16)
                .setAcceleratorName(config.accelerator)
                .setCompileOptions(compile)
            NeuronCache.configure(context, options, model, config.cache, config.accelerator, config.fp16, compile)
            // Throws IllegalArgumentException when acceleratorName > 49 or compileOptions > 199 bytes.
            val delegate = NeuronDelegate(options)
            val interpreterOptions = Interpreter.Options()
                .addDelegate(delegate)
                .setUseXNNPACK(config.useXnnpack)
                .setNumThreads(config.cpuThreads)
            val interpreter = try {
                Interpreter(model, interpreterOptions)   // the on-device compile happens here
            } catch (t: Throwable) {
                delegate.close()
                throw t
            }
            return NeuronModel(interpreter, delegate)
        }
    }
}
