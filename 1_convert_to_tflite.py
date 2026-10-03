# ============================================================
# CONVERT LightTeaNet TO TFLite FOR ANDROID BENCHMARKING
# Run this in Colab, AFTER training (Part 1 of the previous script),
# using the saved model at /content/lightteanet.keras
# ============================================================

import tensorflow as tf

MODEL_PATH = "/content/lightteanet.keras"
TFLITE_PATH = "/content/lightteanet.tflite"

# ------------------------------------------------------------
# Load the trained Keras model
# ------------------------------------------------------------

model = tf.keras.models.load_model(MODEL_PATH)

# ------------------------------------------------------------
# Convert to TFLite — float32, NO quantization.
# Keeping full precision means the latency/accuracy you report
# for the phone corresponds directly to the same model you
# benchmarked for params/FLOPs in the manuscript. (You can
# additionally report an int8-quantized variant as a secondary,
# clearly-labeled result if you want a stronger edge-efficiency
# claim, but don't substitute it for the float32 number.)
# ------------------------------------------------------------

converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

print(f"Saved TFLite model to {TFLITE_PATH}")
print(f"Size: {len(tflite_model) / 1024:.2f} KB")

# ------------------------------------------------------------
# OPTIONAL — int8 dynamic-range quantized variant, if you
# decide to report it as a secondary deployment-optimized result
# ------------------------------------------------------------

TFLITE_QUANT_PATH = "/content/lightteanet_int8.tflite"

converter_quant = tf.lite.TFLiteConverter.from_keras_model(model)
converter_quant.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_quant_model = converter_quant.convert()

with open(TFLITE_QUANT_PATH, "wb") as f:
    f.write(tflite_quant_model)

print(f"Saved quantized TFLite model to {TFLITE_QUANT_PATH}")
print(f"Size: {len(tflite_quant_model) / 1024:.2f} KB")

# ------------------------------------------------------------
# Download both files from the Colab file browser (left sidebar),
# or:
# ------------------------------------------------------------
from google.colab import files
files.download(TFLITE_PATH)
# files.download(TFLITE_QUANT_PATH)  # uncomment if you want this too
