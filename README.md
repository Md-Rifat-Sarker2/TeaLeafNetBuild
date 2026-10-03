# LightTeaNet — Android Edge Benchmark (builds with no Android Studio)

Answers Reviewer #1 Comment 4 + Reviewer #2 Comment 5: actual on-device
inference latency, with preprocessing and model inference timed
**separately**, plus peak RAM and device specs — not just
params/FLOPs. This version builds entirely in the cloud via GitHub
Actions, so you don't need Android Studio or any local SDK install.

## 1. Convert the model (Colab)
Run `1_convert_to_tflite.py` (included separately, or recreate it from
our earlier conversation) against your saved `lightteanet.keras`. It
produces `lightteanet.tflite` — download it from Colab.

## 2. Add your files to this project
Before pushing, put these two things into
`app/src/main/assets/`:
- `lightteanet.tflite` (from step 1)
- `test_images/` — ~20–30 representative images (mix in-domain test
  images and some from the external field set you're collecting for
  the other reviewer comment)

Then delete the two placeholder files in
`app/src/main/assets/` and `app/src/main/assets/test_images/`
(`README_PLACE_FILES_HERE.txt` and `.gitkeep`).

## 3. Push this project to GitHub (no git install needed)
Easiest path, entirely in the browser:
1. Go to github.com → **New repository** (any name, e.g.
   `tealeafnet-benchmark`) → Create.
2. On the repo page, click **"uploading an existing file"** and drag
   the whole project folder in (GitHub preserves the folder structure
   when you drop a folder). Commit to the `main` branch.

(If you do have `git` available anywhere — e.g. Colab's terminal —
`git init`, `git add .`, `git commit -m "init"`, then push to the repo
URL GitHub gives you works too, and is less fiddly for a 15+ file
project than the web upload. Either way works.)

## 4. Let GitHub Actions build it
The push itself triggers `.github/workflows/build.yml` automatically.
In your repo, go to the **Actions** tab — you'll see a "Build APK" run
in progress (takes ~3–5 minutes). If it doesn't start automatically,
click **"Run workflow"** on the left.

## 5. Download the APK
Once the run finishes (green check), open it, scroll to
**Artifacts**, and download `tealeafnet-benchmark-debug-apk` — it's a
zip containing `app-debug.apk`.

## 6. Install it on your phone
- Transfer the APK to your phone (email it to yourself, Google Drive,
  USB cable — anything).
- On the phone, open the file. You'll need to allow "install from
  unknown sources" the first time Android asks.
- Open the app, tap **"Run Edge Benchmark"**.

## 7. Get the results off the phone
Simplest without `adb`: in the app, after the benchmark finishes, the
results are at (use any file manager app on the phone, or Google
Files):
```
Android/data/com.tealeafnet.benchmark/files/benchmark_raw.csv
Android/data/com.tealeafnet.benchmark/files/benchmark_summary.csv
```
Share/email these two files to yourself.

(If you do have `adb` available somewhere — e.g. installed via `pip
install adb-shell` or on another machine — `adb pull
/sdcard/Android/data/com.tealeafnet.benchmark/files/benchmark_summary.csv`
works too.)

## 8. Build the manuscript table
Run `2_build_results_table.py` in the same folder as the two CSVs —
it prints and saves the `Device | Mean latency | FPS | Peak RAM` table
and the preprocessing/inference breakdown table, both plain and as
Markdown ready to paste into the paper.

## What to report in the methods text
All automatically captured in `benchmark_summary.csv`: phone model +
manufacturer, Android version, CPU core count, thread count (fixed at
1 for reproducibility — note this explicitly), TFLite float32 format,
224×224×3 input, warm-up/measured iteration counts, mean ± std
latency, FPS, peak RAM (PSS).

## If GitHub Actions feels like too much too
A lighter fallback that still gets a real device number without any
app at all: the official **TFLite `benchmark_model`** CLI tool, run
over `adb shell` directly against `lightteanet.tflite`. Gives
model-only inference latency (no app build, no Android project), but
can't separate out preprocessing — you'd note that as a limitation.
Ask and I'll write out the exact commands.
