# Clash-Royale-Agent

Preview and record live Clash Royale gameplay from a monitor or a rectangular screen region. Open the game in an emulator or mirror your phone to the desktop, then keep the game visible while capturing.

## Setup

Use Python 3.10 or newer. On Linux, run from an X11 desktop session with `DISPLAY` set. On macOS, grant Screen Recording permission to the app you use to launch Python.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, use `py -3 -m venv .venv` and activate with `.venv\Scripts\Activate.ps1` in PowerShell. On Debian or Ubuntu, install the matching `python3-venv` package if creating the environment reports that `ensurepip` is unavailable.

## Run

List all commands with `python -m clash_royale --help`.

```sh
# List available monitor IDs, sizes and desktop offsets.
python -m clash_royale capture --list-monitors

# Preview the entire first monitor.
python -m clash_royale capture --monitor 1

# Preview a region and record it at a target of 30 FPS.
python -m clash_royale capture --monitor 1 --region 100 80 540 960 --fps 30 --record recordings/match01.mp4
```

`--region X Y WIDTH HEIGHT` uses pixels relative to the selected monitor's top-left corner. Replace the example numbers with the bounds of your game window. Keep the OpenCV preview outside the capture region to avoid capturing it, or add `--no-preview` to record without opening a window.

If the game window moves, draw a new capture box on a still screenshot of the monitor:

```sh
python -m clash_royale capture --monitor 1 --select-region hud_data/game_region.json
python -m clash_royale capture --monitor 1 --region-file hud_data/game_region.json
```

Drag around the game and press **Enter** to save the box; press **C** to cancel without changing the file. Check the preview before recording with `--record`. The saved coordinates apply to future captures; they cannot change footage already recorded.

Press **Q** or **Esc** in the preview window, close the window, or press **Ctrl+C** in the terminal to stop and finalize the recording. A temporary capture error triggers a brief retry. Five consecutive failures end the run with an error.

## Recordings

`recordings/match01.mp4` contains video, and `recordings/match01.timestamps.csv` contains one row per frame submitted to the writer. `frame_index` starts at 0, and `capture_monotonic_ns` is sampled immediately after each successful screen grab. Subtract the first timestamp from a frame's timestamp and divide by 1 billion to get elapsed seconds. The MP4 uses a fixed playback rate, so use the CSV for actual capture intervals, especially when capture falls behind. If a recording has an odd width or height, the encoder input gets one black column on the right or row at the bottom as needed.

## Crop an existing recording

```sh
python -m clash_royale crop recordings/full-match.mp4 recordings/game-only.mp4 \
  --frame 120 --save-region hud_data/game-region.json
```

Choose a clear frame with `--frame` (zero-based, default 0). Drag a rectangle around the game and press **Enter** to accept or **C** to cancel. Keep the game window in the same place throughout the recording. The tool writes a new video at the source's nominal FPS and preserves frame order. It copies the source capture timestamps by frame index into a matching `.timestamps.csv` when one exists. The export is video-only. An odd crop width or height gets one pixel of padding on the right or bottom.

For a full-monitor source recording, reuse the saved rectangle on that same monitor and desktop layout.

```sh
python -m clash_royale capture --monitor 1 --region-file hud_data/game-region.json
python -m clash_royale crop recordings/another-full-match.mp4 recordings/another-game.mp4 \
  --region-file hud_data/game-region.json
```

Pick the source monitor ID with `python -m clash_royale capture --list-monitors`. Regions are relative to the source frame, so an already cropped source needs its original monitor offset added before using its coordinates for live capture.

## Measure performance

The terminal reports three metrics about once per second.

- **Capture FPS** is the number of successful frames per elapsed second, including time spent previewing, recording, pacing and retrying.
- **Grab latency** is the average time spent in the MSS screen grab.
- **Processing latency** is the average and maximum time from grab completion through BGR conversion, recording and preview event handling when enabled. Pacing sleep happens after this measurement.

Run the same region for several seconds with preview only, then add `--record` to measure the recording overhead. Use `--fps 0` without recording to measure uncapped throughput, and add `--no-preview` to see how much the preview costs. For recorded footage, compare successive CSV timestamps against the target frame interval (about 33.3 ms at 30 FPS) to check spacing and spot gaps.

`clash_royale/capture/screen.py` provides timestamped BGR NumPy frames. `clash_royale/capture/recording.py` writes video and CSV output. Run the capture CLI with `python -m clash_royale capture`.

## Offline replay and annotation

The annotator uses the same Python environment and dependencies as capture. Open any MP4 from a desktop session. Keep a capture's `.timestamps.csv` next to the video to preserve its monotonic frame timestamps.

### Build a review queue

The current detector was trained from only 48 labelled frames across two matches. Before relying on its predictions, add labels from other matches. Generate a queue that samples the recording every two seconds, skips frames already labelled for that source video, and spreads suggestions across time and visually different scenes:

```sh
python -m clash_royale suggest-frames recordings/match03.mp4 --dataset dataset --limit 30
python -m clash_royale annotate recordings/match03.mp4 --dataset dataset --queue dataset/queues/match03.json
python -m clash_royale suggest-frames recordings/match04.mp4 --dataset dataset --limit 30
python -m clash_royale annotate recordings/match04.mp4 --dataset dataset --queue dataset/queues/match04.json
```

The queue contains frame indices and the source video's SHA-256 digest. The annotator checks the digest before opening it, starts at the first unlabelled suggestion, and uses **,** and **.** for the previous and next unlabelled suggestion. Save the boxes with **S**; the next queue jump then skips that frame. Use `--start-seconds` and `--end-seconds` to restrict a clip, and `--stride-seconds` to adjust how densely candidate frames are scanned. Review every suggested frame and label every visible instance of each class you train. Run `prepare-dataset` again after adding labels, keeping whole matches in the validation split, then retrain and evaluate on a separate match. The queue proposes frames for human review; it does not create labels or assess their correctness.

```sh
python -m clash_royale annotate recordings/match01.mp4 --dataset dataset
python -m clash_royale annotate recordings/match01.mp4 --dataset dataset --review
```

| Control | Action |
| --- | --- |
| **Space** | Play or pause at the video's nominal frame rate |
| **A / D** | Step backward or forward one frame |
| **J / L** or **Frame** slider | Seek about a second or jump to any frame |
| **[ / ]** | Choose the class for new boxes |
| **Left drag** | Draw a box |
| **Right click** | Select a box |
| **E**, then **left drag** | Redraw the selected box to move or resize it |
| **C / X** | Assign the active class to the selected box or delete it |
| **S / R** | Save the current frame and boxes, or revert to the last saved boxes |
| **P / N** | Jump to the previous or next saved frame, or to a flagged frame when using `--review-report` |
| **, / .** | Jump to the previous or next unlabelled queue suggestion when using `--queue` |
| **Q / Esc** | Quit |

Save or revert pending edits before navigating. Use **S** on a frame with no boxes to save it as a background image. Reopen the same video with `--review` to go straight to its first saved frame. Saved boxes appear automatically when you visit an exported frame.

The default classes, in ID order, are Friendly troop, Enemy troop, Friendly tower, Enemy tower, Card, and Other / unknown. For a custom list, pass `--classes path/to/classes.txt` when starting a dataset, with one name per line. The dataset stores that order in its own `classes.txt`, which it reuses on later runs.

### Dataset layout

Each MP4 gets its own match group by default. If a match spans multiple clips, pass the same `--match match01` when first opening each clip. The dataset remembers each source's group. Record one match per MP4 when possible.

- `images/<match>/` contains full-resolution PNGs from the decoded video.
- `labels/<match>/` contains matching YOLO label files with zero-based class IDs and normalized center, width, and height coordinates computed from the original image pixels. [YOLO detection format](https://docs.ultralytics.com/datasets/detect/)
- `metadata/<match>/` contains matching JSON files with the original video filename, zero-based frame index, playback timestamp in nanoseconds, and the capture monotonic timestamp when available. The playback timestamp falls back to the frame index divided by nominal FPS when the decoder supplies no timestamp.

Saving the same video and frame index again updates its labels in place. The tool identifies source videos by their content, so reopening a renamed copy in the same dataset keeps the same exports.

### Review dataset quality

```sh
python -m clash_royale check-dataset --dataset dataset --troop-only
python -m clash_royale annotate recordings/match01.mp4 --dataset dataset --review-report dataset/quality.json
```

The checker writes `dataset/quality.json` with a list of flagged images, their source videos and frame numbers, and the reason for each flag. It reports invalid boxes by line and class counts across the dataset and for each match. Classes with zero boxes or fewer than one quarter of the most common class's boxes are flagged as underrepresented. Omit `--troop-only` to assess coverage and balance for all annotation classes.

Review empty annotation files and frames with no target boxes for missed troops. A frame is flagged for few boxes when it has at most half the median positive box count for its match. Exact and potential near-duplicates include the matching frame for comparison. The near-duplicate check compares thumbnails from same-size images. Confirm the flags visually, including intentional background frames and similar frames with distinct gameplay. If preparation rejects an exact duplicate across matches, keep one of the exports in the active dataset and archive the other PNG, label, and metadata together.

Open a source MP4 named in the report. The annotator starts at its first flagged frame and prints the reasons in the terminal. Use **P / N** to step through that video's flags and **A / D** or the slider to see surrounding frames. Use **[ / ]** to select the appropriate troop class and drag a box around each visible unlabeled troop. Use **E**, **C**, or **X** on a selected box to resize, relabel, or remove it. The editor displays valid boxes when some rows are malformed and lists the skipped rows in the terminal. Inspect the frame and redraw any affected boxes before pressing **S** to save. Re-run the checker after edits to refresh the report.

## YOLO training and evaluation

### Prepare the annotated dataset

Use the same environment as capture and annotation. Preparation only needs the existing dependencies. It expects the annotator's `classes.txt` and matching files under `images/`, `labels/`, and `metadata/`.

```sh
python -m clash_royale prepare-dataset --dataset dataset --val-fraction 0.2 --seed 0
```

For a troop-only experiment, prepare the two-class dataset and optional box previews with this command.

```sh
python -m clash_royale prepare-dataset --dataset dataset --troop-only --val-fraction 0.2 --seed 0 --preview
```

The troop-only bundle lives in `dataset/splits/troops/`. Its image lists point to copied PNGs with separate YOLO labels for `0 friendly_troop` and `1 enemy_troop`, mapped from the names in the original `classes.txt`. The original annotations and class IDs are preserved. Draw troop boxes only around troops. Buildings and spell effects can remain visible without a label in this bundle. Frames with zero troop boxes become background examples, so confirm they contain no visible troops before training. Open the JPEGs in the printed preview directory to inspect the boxes actually exported. Omit `--preview` during preparation to skip rendering previews.

The fraction applies to **matches**, with at least one match in each split. Use `--val-match MATCH_ID` (repeat for multiple matches) to keep specific matches in validation as you collect more footage. Use the match IDs printed by the annotator or listed in the metadata.

For the same validation match as the first run, use `--val-match match-f381ad78d798fcc8` if that match is present. Review the match lists, background counts, and class counts in the selected bundle's `split.json` and check that both troop classes have examples in both splits. The bundle also contains `data.yaml`, `train.txt`, and `val.txt`. Preparation rejects missing or unreadable exports, invalid labels, duplicate source video/frame IDs and identical pixels across matches. It reports repeated pixels within a match and keeps those frames together. [Ultralytics dataset layout](https://docs.ultralytics.com/datasets/detect/)

### Train after collecting real labeled gameplay

```sh
python -m pip install -r requirements-training.txt
python -m clash_royale train --dataset dataset --model yolo11n.pt --epochs 50 --imgsz 640 --batch 8 --device cpu --output runs/troops
```

When ready to launch the troop-only experiment, use the prepared troop split and a new run directory.

```sh
python -m clash_royale train --dataset dataset --troop-only --model yolo11n.pt --epochs 50 --imgsz 640 --batch 8 --device cpu --output runs/troops-only
```

Use `--device 0` for the first CUDA GPU with a compatible PyTorch installation. The default pretrained YOLO11 nano checkpoint can download on first use. You can also pass the path to a local pretrained detection checkpoint. Training requires annotated objects in both splits and checks that the exports still match preparation. Rerun preparation after editing or adding samples. [Ultralytics training options](https://docs.ultralytics.com/modes/train/)

On Windows, training defaults to zero data loader workers. Use `--workers` to override the default.

The new run directory contains `results.csv`, `weights/best.pt`, `weights/last.pt`, `args.yaml`, `run_config.json`, and the split files under `dataset/`. An all-class run references the original exports. A troop-only run stores its own copies of the PNGs and filtered labels. Keep the run's image and label files unchanged to reproduce evaluation.

### Evaluate the validation matches

```sh
python -m clash_royale evaluate --weights runs/troops/weights/best.pt --data runs/troops/dataset/data.yaml --device cpu --imgsz 640 --batch 1 --output runs/troops/evaluation.json
```

The JSON reports precision, recall, mAP50, and mAP50–95 overall and per class, along with each class's validation instance count. Classes absent from validation have null per-class metrics. Latency is the average milliseconds per image on the specified device and batch size. The report includes model inference alone and the combined preprocessing, inference, and postprocessing time. [Ultralytics validation metrics and timing](https://docs.ultralytics.com/guides/model-evaluation-insights/)

For the troop-only experiment, evaluate with its own checkpoint and dataset YAML. The `per_class` list has separate entries for `friendly_troop` and `enemy_troop`.

```sh
python -m clash_royale evaluate --weights runs/troops-only/weights/best.pt --data runs/troops-only/dataset/data.yaml --device cpu --imgsz 640 --batch 1 --output runs/troops-only/evaluation.json
```

### Benchmark troop detection on new matches

Record two fresh matches and annotate about 30 frames per match in `dataset/holdout`. Use a separate annotation directory for training. Label every visible friendly and enemy troop in each selected frame, and save a few frames with no troops to test false positives.

```sh
python -m clash_royale suggest-frames recordings/holdout01.mp4 --dataset dataset/holdout --limit 30
python -m clash_royale annotate recordings/holdout01.mp4 --dataset dataset/holdout --queue dataset/holdout/queues/holdout01.json
python -m clash_royale suggest-frames recordings/holdout02.mp4 --dataset dataset/holdout --limit 30
python -m clash_royale annotate recordings/holdout02.mp4 --dataset dataset/holdout --queue dataset/holdout/queues/holdout02.json
python -m clash_royale evaluate --holdout dataset/holdout --weights runs/troops-v2/weights/best.pt --output runs/troops-v2/holdout.json
```

`--holdout` scores the saved checkpoint on all annotated frames in that directory. It reports precision, recall, mAP50, mAP75 and mAP50–95, plus frame-level flags for misses, loose boxes, wrong teams and unmatched predictions. The report also includes per-class precision and recall at a fixed confidence cutoff (`0.25` by default) and IoU of at least `0.50`. The flags use the same cutoff and mark matching boxes below `0.75` IoU as loose. Open a flagged match in the annotator and use **P / N** to review its flagged frames.

```sh
python -m clash_royale annotate recordings/holdout01.mp4 --dataset dataset/holdout --review-report runs/troops-v2/holdout.json
```

Look for recurring errors and collect training examples from other matches that exhibit them. After training a candidate, compare it with the baseline on the same frames. Use the same evaluation settings and Ultralytics version for both runs. If you correct a held-out annotation, reevaluate both checkpoints on the corrected labels.

```sh
python -m clash_royale evaluate --holdout dataset/holdout --weights runs/troops-v3/weights/best.pt --baseline runs/troops-v2/holdout.json --output runs/troops-v3/holdout.json
```

`change_from_baseline` reports metric changes as decimals, so `0.05` is a gain of five percentage points. Check enemy recall and precision under `per_class_at_review_confidence` first. Use mAP75 to see whether performance also improved under tighter box matching. Before relying on a model tuned using these results, confirm it on another fresh match.

### Preview predictions on a recorded MP4

```sh
python -m clash_royale infer-video recordings/match02.mp4 --weights runs/troops/weights/best.pt --device cpu --imgsz 640 --conf 0.25
```

The OpenCV preview shows class names and confidence scores. Press **Q** or **Esc** to exit. Import `Detector` from `clash_royale.detection.detector` and call `Detector.detect(frame)` with a BGR NumPy frame to get boxes in the original pixel coordinates.

## Card hand and elixir recognition

### Calibrate and collect references

Record a short clip with the same game crop you intend to use live. Use `annotate` to find a clear frame index. Select the four card artwork crops from left to right, using about the same framing for each. Select the elixir bar and its displayed number. Include `--next-card` when the preview is visible. Press **Enter** after each selection or **C** to skip the optional preview. [OpenCV region selection controls](https://docs.opencv.org/4.x/d7/dfc/group__highgui.html)

```sh
python -m clash_royale hud calibrate recordings/reference.mp4 --frame 120 --next-card
python -m clash_royale hud collect recordings/reference.mp4 --frame 120 \
  --hand "First card name" "Second card name" - - --elixir 6
python -m clash_royale hud collect recordings/reference.mp4 --frame 180 --next-card "Visible card name"
```

Replace the example indices, names, and elixir value with the labels visible in your recording. Use `-` to skip a hand slot. Collect multiple examples of each card, including bright and dim artwork, and examples of each elixir level you want recognized. Next-card images have their own reference folder because the preview can look different. The tool saves normalized coordinates in `hud_data/calibration.json` and labelled PNG crops under `hud_data/references/`. Reuse the calibration for the same game layout and crop. Recalibrate if the HUD moves relative to the frame.

### Write game states from a recording or live capture

```sh
python -m clash_royale hud replay recordings/match02.mp4 \
  --weights runs/troops-v2/weights/best.pt --states hud_data/replay.jsonl --no-preview
python -m clash_royale capture --monitor 1 --region-file hud_data/game_region.json \
  --hud-calibration hud_data/calibration.json --weights runs/troops-v2/weights/best.pt \
  --states hud_data/live.jsonl --no-preview
```

Use the same game crop for the recording or live capture and the HUD calibration. `--weights` requires the training dependencies. Set `--device 0` for a compatible CUDA GPU. Omit `--weights` for HUD-only output. Omit `--no-preview` to see the HUD panel and any detector boxes. Use a new JSONL filename for each run.

Each JSONL line describes one frame. It includes the frame index, timestamp and clock source, elixir, four hand slots, optional next card, HUD confidence scores, and HUD recognition latency. Replay uses the capture CSV timestamps when available and video playback timestamps otherwise. An unknown HUD reading has a `null` value. A replacement seen in the same slot on three consecutive confident frames produces an inferred play/draw event in `hand_changes`.

With `--weights`, each line also includes `frame_width`, `frame_height`, and `detections`. Each detection has a `class_id`, `class_name`, `confidence`, and `xyxy` box in pixels relative to the top-left of the input frame. For the troop-only model, use `friendly_troop` and `enemy_troop` to tell the teams apart. An empty `detections` list means no boxes passed the confidence cutoff for that frame.

### Evaluate against labelled gameplay

Label frames from a separate recording using `label`. It writes ground truth without adding reference crops. Repeat the command for the frames you want to score.

```sh
python -m clash_royale hud label recordings/held-out.mp4 --frame 90 \
  --hand "First card name" "Second card name" - - --elixir 4
python -m clash_royale hud evaluate recordings/held-out.mp4
```

Evaluation reports per-slot hand accuracy, next-card accuracy when labelled, exact elixir accuracy, mean absolute elixir error for recognized values, unknown rates, and mean and p95 recognition latency. Unknown predictions count as incorrect for labelled fields. Use `--min-score` and `--min-margin` on replay or evaluation to adjust rejection thresholds. The live equivalents are `--hud-min-score` and `--hud-min-margin`.

## Source layout

- `clash_royale/capture/` contains screen capture, recording, and the live preview.
- `clash_royale/dataset/` contains replay annotation, YOLO export, and match-based dataset preparation.
- `clash_royale/detection/` contains model training, evaluation, and offline video inference.
- `clash_royale/hud/` contains calibration, template recognition, and HUD evaluation.
- `clash_royale/game_state.py` combines HUD observations and detections from the same frame.
- `clash_royale/__main__.py` exposes the commands under `python -m clash_royale`.
