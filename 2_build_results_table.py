# ============================================================
# BUILD THE FINAL "EDGE HARDWARE" TABLE FOR THE MANUSCRIPT
# Run this locally (not in Colab) after pulling benchmark_summary.csv
# from the phone via:
#   adb pull /sdcard/Android/data/com.tealeafnet.benchmark/files/benchmark_summary.csv
#
# If you benchmark more than one device (e.g. phone + a Colab T4
# GPU timing, or a second phone), append each device's
# benchmark_summary.csv row into summary_rows below, or just
# concatenate multiple CSVs before running this.
# ============================================================

import pandas as pd

# ------------------------------------------------------------
# 1. Load the phone's summary CSV
# ------------------------------------------------------------

phone_df = pd.read_csv("benchmark_summary.csv")

# ------------------------------------------------------------
# 2. (Optional) Add a GPU/server-side row for comparison, if you
#    separately timed inference on your Colab T4 using the same
#    style of measurement (see measure_latency() from the earlier
#    fine-tuning script, or time interpreter.run() equivalently).
#    Fill these in with real measured numbers, or delete this block.
# ------------------------------------------------------------

gpu_row = {
    "device": "NVIDIA T4 (Colab)",
    "android_version": "-",
    "hardware": "-",
    "cpu_cores": "-",
    "threads": "-",
    "model_format": "Keras (float32)",
    "input_size": "224x224x3",
    "warmup_iters": 20,
    "measured_iters": 150,
    "mean_preprocess_ms": None,   # fill in
    "mean_inference_ms": None,    # fill in
    "mean_total_ms": None,        # fill in
    "std_total_ms": None,         # fill in
    "fps": None,                  # fill in
    "peak_ram_mb": None           # fill in
}

all_rows = pd.concat([pd.DataFrame([gpu_row]), phone_df], ignore_index=True)
# If you don't want the GPU row, just use: all_rows = phone_df

# ------------------------------------------------------------
# 3. Print the manuscript-ready table
# ------------------------------------------------------------

print("\n==================== FULL DETAIL TABLE ====================")
print(all_rows.to_string(index=False))

all_rows.to_csv("edge_benchmark_full_detail.csv", index=False)

# ------------------------------------------------------------
# 4. Compact version matching the table format in the reviewer
#    discussion (Device | Mean latency (ms) | FPS | Peak RAM (MB))
# ------------------------------------------------------------

compact = all_rows[["device", "mean_total_ms", "std_total_ms", "fps", "peak_ram_mb"]].copy()
compact.columns = ["Device", "Mean latency (ms)", "Std (ms)", "FPS", "Peak RAM (MB)"]

print("\n==================== MANUSCRIPT TABLE ====================")
print(compact.to_string(index=False))

print("\n\nMarkdown table for the manuscript:\n")
print("| Device | Mean latency (ms) | Std (ms) | FPS | Peak RAM (MB) |")
print("|---|---|---|---|---|")
for _, row in compact.iterrows():
    def fmt(v):
        return "—" if pd.isna(v) else f"{v:.2f}"
    print(f"| {row['Device']} | {fmt(row['Mean latency (ms)'])} | {fmt(row['Std (ms)'])} "
          f"| {fmt(row['FPS'])} | {fmt(row['Peak RAM (MB)'])} |")

# ------------------------------------------------------------
# 5. Preprocessing vs. inference breakdown (the methodological
#    detail the reviewer specifically asked for)
# ------------------------------------------------------------

breakdown = all_rows[["device", "mean_preprocess_ms", "mean_inference_ms", "mean_total_ms"]].copy()
breakdown.columns = ["Device", "Preprocessing (ms)", "Inference (ms)", "Total (ms)"]

print("\n==================== LATENCY BREAKDOWN ====================")
print(breakdown.to_string(index=False))

breakdown.to_csv("edge_latency_breakdown.csv", index=False)
