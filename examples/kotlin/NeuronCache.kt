package com.example.npu

import android.content.Context
import org.tensorflow.lite.NeuronDelegate
import java.io.File
import java.security.MessageDigest

/**
 * Compile cache for the Neuron delegate: folder, model token, clear.
 *
 * Two cache routes exist (README, "Compile cache"):
 * - [Mode.ADAPTER]: MediaTek's own cache (`NeuronCompilation_setCaching`), selected by `setCacheDir` without a
 *   model token. Lighter (no heap copy of the compiled network). Its key covers only the graph structure, not
 *   the weights, so two models with the same structure must never share a folder: this helper gives every
 *   model file its own folder, named by [token].
 * - [Mode.DLA]: patch 0004's store/restore, selected by `setCacheDir` + `setModelToken`. Files land in
 *   `<root>/npustt-dla/<token>/<key>.dla` (the folder name `npustt-dla` is fixed in the patch). Does not depend
 *   on the vendor cache, but keeps a heap copy of each restored network (about the size of its .dla file).
 *
 * Folder choice: [Context.getCodeCacheDir] is emptied by Android on every app update and every platform
 * (OTA) update, which is a safe point to recompile (a new firmware can bring a new NPU driver). During
 * development a folder that survives `adb install -r` saves recompiles, e.g.
 * `File(context.getExternalFilesDir(null), "neuron-cache")`.
 */
object NeuronCache {
    enum class Mode { OFF, ADAPTER, DLA }

    fun root(context: Context): File = File(context.codeCacheDir, "neuron-cache")

    /**
     * Hex MD5 of `"<file name>|<size>|<mtime>|<accelerator>|<fp16>|<compile options>"`. Any new model file
     * (new weights, re-export) gets a new token, and so do changed compile settings. Execution preference,
     * priority, optimization hint and boost are not part of it (nor of the delegate's own key): call [clear]
     * after changing those.
     */
    fun token(model: File, accelerator: String, fp16: Boolean, compileOptions: String?): String =
        md5Hex("${model.name}|${model.length()}|${model.lastModified()}|$accelerator|$fp16|${compileOptions ?: ""}")

    /** Sets cacheDir (and modelToken for DLA) on [options] for [model]; returns the folder used, or null. */
    fun configure(
        context: Context,
        options: NeuronDelegate.Options,
        model: File,
        mode: Mode,
        accelerator: String,
        fp16: Boolean,
        compileOptions: String?,
    ): File? {
        if (mode == Mode.OFF) return null
        val token = token(model, accelerator, fp16, compileOptions)
        return when (mode) {
            Mode.ADAPTER -> File(root(context), "adapter/$token").also {
                it.mkdirs()
                options.setCacheDir(it.path)   // absolute path required by the adapter
            }
            Mode.DLA -> File(root(context), "dla").also {
                it.mkdirs()
                options.setCacheDir(it.path)
                options.setModelToken(token)
            }
            Mode.OFF -> null
        }
    }

    /**
     * Deletes the whole cache. Do it from the app: the folders the delegate creates may not be deletable from
     * the adb shell (no group write).
     */
    fun clear(context: Context): Boolean {
        val dir = root(context)
        return !dir.exists() || dir.deleteRecursively()
    }

    private fun md5Hex(s: String): String =
        MessageDigest.getInstance("MD5").digest(s.toByteArray(Charsets.UTF_8)).joinToString("") { "%02x".format(it) }
}
