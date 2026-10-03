package com.tealeafnet.benchmark

import android.app.ActivityManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Build
import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.FileWriter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.sqrt

/**
 * LightTeaNet on-device edge benchmark.
 *
 * Measures, separately, for each test image:
 *   1. Preprocessing latency  (decode + resize + normalize -> ByteBuffer)
 *   2. Model inference latency (interpreter.run only)
 *   3. Total pipeline latency  (1 + 2)
 *
 * Does WARMUP_ITERATIONS un-timed runs first, then MEASURED_ITERATIONS
 * timed runs (looping over the bundled test images if there are fewer
 * images than MEASURED_ITERATIONS).
 *
 * Writes a per-iteration CSV and a summary CSV to the app's external
 * files directory (no storage permission needed on modern Android):
 *   /Android/data/com.tealeafnet.benchmark/files/benchmark_raw.csv
 *   /Android/data/com.tealeafnet.benchmark/files/benchmark_summary.csv
 *
 * Pull them with:
 *   adb pull /sdcard/Android/data/com.tealeafnet.benchmark/files/benchmark_raw.csv
 *   adb pull /sdcard/Android/data/com.tealeafnet.benchmark/files/benchmark_summary.csv
 */
class MainActivity : AppCompatActivity() {

    companion object {
        const val MODEL_FILE = "lightteanet.tflite"   // placed in assets/
        const val TEST_IMAGES_DIR = "test_images"     // assets/test_images/*.jpg
        const val IMG_SIZE = 224
        const val NUM_CLASSES = 4
        const val WARMUP_ITERATIONS = 20
        const val MEASURED_ITERATIONS = 150
        const val NUM_THREADS = 1 // fixed thread count for reproducible, reportable numbers
    }

    private lateinit var logView: TextView
    private lateinit var interpreter: Interpreter
    private lateinit var activityManager: ActivityManager

    private val preprocessTimesMs = mutableListOf<Double>()
    private val inferenceTimesMs = mutableListOf<Double>()
    private val totalTimesMs = mutableListOf<Double>()
    private val pssSamplesKb = mutableListOf<Int>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        logView = findViewById(R.id.logView)
        val runButton: Button = findViewById(R.id.runButton)
        activityManager = getSystemService(ACTIVITY_SERVICE) as ActivityManager

        runButton.setOnClickListener {
            runButton.isEnabled = false
            log("Starting benchmark...\n")
            Thread {
                try {
                    runBenchmark()
                } catch (e: Exception) {
                    log("ERROR: ${e.message}\n${e.stackTraceToString()}")
                }
                runOnUiThread { runButton.isEnabled = true }
            }.start()
        }
    }

    private fun log(msg: String) {
        android.util.Log.d("TeaLeafBenchmark", msg)
        runOnUiThread { logView.append(msg + "\n") }
    }

    // ------------------------------------------------------------
    // Load the .tflite model from assets as a MappedByteBuffer
    // ------------------------------------------------------------
    private fun loadModelFile(): MappedByteBuffer {
        val fd = assets.openFd(MODEL_FILE)
        val inputStream = fd.createInputStream()
        val fileChannel = inputStream.channel
        return fileChannel.map(
            FileChannel.MapMode.READ_ONLY,
            fd.startOffset,
            fd.declaredLength
        )
    }

    // ------------------------------------------------------------
    // Load bundled test images from assets/test_images/
    // ------------------------------------------------------------
    private fun loadTestImages(): List<Bitmap> {
        val fileNames = assets.list(TEST_IMAGES_DIR) ?: emptyArray()
        return fileNames.sorted().map { name ->
            val input = assets.open("$TEST_IMAGES_DIR/$name")
            BitmapFactory.decodeStream(input)
        }
    }

    // ------------------------------------------------------------
    // Preprocessing: resize + normalize to [0,1] + pack into ByteBuffer
    // (mirrors the Python training pipeline: rescale=1./255)
    // ------------------------------------------------------------
    private fun preprocess(bitmap: Bitmap): ByteBuffer {
        val resized = Bitmap.createScaledBitmap(bitmap, IMG_SIZE, IMG_SIZE, true)

        val buffer = ByteBuffer.allocateDirect(4 * IMG_SIZE * IMG_SIZE * 3)
        buffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(IMG_SIZE * IMG_SIZE)
        resized.getPixels(pixels, 0, IMG_SIZE, 0, 0, IMG_SIZE, IMG_SIZE)

        for (pixel in pixels) {
            buffer.putFloat(((pixel shr 16) and 0xFF) / 255.0f) // R
            buffer.putFloat(((pixel shr 8) and 0xFF) / 255.0f)  // G
            buffer.putFloat((pixel and 0xFF) / 255.0f)          // B
        }

        buffer.rewind()
        return buffer
    }

    // ------------------------------------------------------------
    // Current process memory (PSS, in KB) — same metric `adb shell
    // dumpsys meminfo` reports; take the max across the run for
    // "peak RAM"
    // ------------------------------------------------------------
    private fun currentPssKb(): Int {
        val pid = android.os.Process.myPid()
        val memInfo = activityManager.getProcessMemoryInfo(intArrayOf(pid))
        return memInfo[0].totalPss
    }

    private fun runBenchmark() {

        log("Loading model...")
        val options = Interpreter.Options()
        options.setNumThreads(NUM_THREADS)
        interpreter = Interpreter(loadModelFile(), options)

        log("Loading test images...")
        val images = loadTestImages()
        if (images.isEmpty()) {
            log("No images found in assets/$TEST_IMAGES_DIR/ — add some test images first.")
            return
        }
        log("Loaded ${images.size} test image(s).")

        val outputBuffer = Array(1) { FloatArray(NUM_CLASSES) }

        // ---------------- WARM-UP (not timed) ----------------
        log("Warm-up: $WARMUP_ITERATIONS iterations...")
        for (i in 0 until WARMUP_ITERATIONS) {
            val bitmap = images[i % images.size]
            val input = preprocess(bitmap)
            interpreter.run(input, outputBuffer)
        }

        // ---------------- MEASURED ITERATIONS ----------------
        log("Measuring: $MEASURED_ITERATIONS iterations...")
        pssSamplesKb.add(currentPssKb())

        for (i in 0 until MEASURED_ITERATIONS) {
            val bitmap = images[i % images.size]

            val tPreStart = System.nanoTime()
            val input = preprocess(bitmap)
            val tPreEnd = System.nanoTime()

            val tInfStart = System.nanoTime()
            interpreter.run(input, outputBuffer)
            val tInfEnd = System.nanoTime()

            val preMs = (tPreEnd - tPreStart) / 1_000_000.0
            val infMs = (tInfEnd - tInfStart) / 1_000_000.0
            val totalMs = preMs + infMs

            preprocessTimesMs.add(preMs)
            inferenceTimesMs.add(infMs)
            totalTimesMs.add(totalMs)

            if (i % 10 == 0) {
                pssSamplesKb.add(currentPssKb())
            }
        }
        pssSamplesKb.add(currentPssKb())

        writeResults()
        printSummary()
    }

    private fun mean(values: List<Double>) = values.sum() / values.size

    private fun std(values: List<Double>): Double {
        val m = mean(values)
        val variance = values.sumOf { (it - m) * (it - m) } / values.size
        return sqrt(variance)
    }

    private fun printSummary() {
        val meanPre = mean(preprocessTimesMs)
        val meanInf = mean(inferenceTimesMs)
        val meanTotal = mean(totalTimesMs)
        val stdTotal = std(totalTimesMs)
        val fps = 1000.0 / meanTotal
        val peakPssMb = (pssSamplesKb.maxOrNull() ?: 0) / 1024.0

        val summary = StringBuilder()
        summary.append("\n================ RESULTS ================\n")
        summary.append("Device          : ${Build.MANUFACTURER} ${Build.MODEL}\n")
        summary.append("Android version : ${Build.VERSION.RELEASE} (SDK ${Build.VERSION.SDK_INT})\n")
        summary.append("Hardware/Board  : ${Build.HARDWARE} / ${Build.BOARD}\n")
        summary.append("CPU cores avail : ${Runtime.getRuntime().availableProcessors()}\n")
        summary.append("Threads used    : $NUM_THREADS\n")
        summary.append("Model format    : TFLite (float32), input ${IMG_SIZE}x${IMG_SIZE}x3\n")
        summary.append("Warm-up iters   : $WARMUP_ITERATIONS\n")
        summary.append("Measured iters  : $MEASURED_ITERATIONS\n")
        summary.append(String.format("Preprocessing   : mean=%.3f ms, std=%.3f ms\n", meanPre, std(preprocessTimesMs)))
        summary.append(String.format("Inference       : mean=%.3f ms, std=%.3f ms\n", meanInf, std(inferenceTimesMs)))
        summary.append(String.format("Total pipeline  : mean=%.3f ms, std=%.3f ms\n", meanTotal, stdTotal))
        summary.append(String.format("FPS             : %.2f\n", fps))
        summary.append(String.format("Peak PSS (RAM)  : %.2f MB\n", peakPssMb))
        summary.append("===========================================\n")

        log(summary.toString())
    }

    private fun writeResults() {
        val dir = getExternalFilesDir(null)

        // ---- per-iteration raw CSV ----
        val rawFile = File(dir, "benchmark_raw.csv")
        FileWriter(rawFile).use { w ->
            w.append("iteration,preprocess_ms,inference_ms,total_ms\n")
            for (i in totalTimesMs.indices) {
                w.append("$i,${preprocessTimesMs[i]},${inferenceTimesMs[i]},${totalTimesMs[i]}\n")
            }
        }

        // ---- summary CSV (one row — append more rows if you run
        //      this on multiple devices and merge later) ----
        val meanPre = mean(preprocessTimesMs)
        val meanInf = mean(inferenceTimesMs)
        val meanTotal = mean(totalTimesMs)
        val stdTotal = std(totalTimesMs)
        val fps = 1000.0 / meanTotal
        val peakPssMb = (pssSamplesKb.maxOrNull() ?: 0) / 1024.0

        val summaryFile = File(dir, "benchmark_summary.csv")
        FileWriter(summaryFile).use { w ->
            w.append(
                "device,android_version,hardware,cpu_cores,threads,model_format,input_size," +
                        "warmup_iters,measured_iters,mean_preprocess_ms,mean_inference_ms," +
                        "mean_total_ms,std_total_ms,fps,peak_ram_mb\n"
            )
            w.append(
                "${Build.MANUFACTURER} ${Build.MODEL},${Build.VERSION.RELEASE}," +
                        "${Build.HARDWARE},${Runtime.getRuntime().availableProcessors()},$NUM_THREADS," +
                        "TFLite-float32,${IMG_SIZE}x${IMG_SIZE}x3,$WARMUP_ITERATIONS,$MEASURED_ITERATIONS," +
                        "$meanPre,$meanInf,$meanTotal,$stdTotal,$fps,$peakPssMb\n"
            )
        }

        log("Results written to:\n${rawFile.absolutePath}\n${summaryFile.absolutePath}")
    }
}
