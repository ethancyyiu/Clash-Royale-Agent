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

```sh
# List available monitor IDs, sizes and desktop offsets.
python main.py --list-monitors

# Preview the entire first monitor.
python main.py --monitor 1

# Preview a region and record it at a target of 30 FPS.
python main.py --monitor 1 --region 100 80 540 960 --fps 30 --record recordings/match01.mp4
```

`--region X Y WIDTH HEIGHT` uses pixels relative to the selected monitor's top-left corner. Replace the example numbers with the bounds of your game window. Keep the OpenCV preview outside the capture region to avoid capturing it, or add `--no-preview` to record without opening a window.

Press **Q** or **Esc** in the preview window, close the window, or press **Ctrl+C** in the terminal to stop and finalize the recording. A temporary capture error triggers a brief retry. Five consecutive failures end the run with an error.

## Recordings

`recordings/match01.mp4` contains video, and `recordings/match01.timestamps.csv` contains one row per frame submitted to the writer. `frame_index` starts at 0, and `capture_monotonic_ns` is sampled immediately after each successful screen grab. Subtract the first timestamp from a frame's timestamp and divide by 1 billion to get elapsed seconds. The MP4 uses a fixed playback rate, so use the CSV for actual capture intervals, especially when capture falls behind. If a recording has an odd width or height, the encoder input gets one black column on the right or row at the bottom as needed.

## Measure performance

The terminal reports three metrics about once per second.

- **Capture FPS** is the number of successful frames per elapsed second, including time spent previewing, recording, pacing and retrying.
- **Grab latency** is the average time spent in the MSS screen grab.
- **Processing latency** is the average and maximum time from grab completion through BGR conversion, recording and preview event handling when enabled. Pacing sleep happens after this measurement.

Run the same region for several seconds with preview only, then add `--record` to measure the recording overhead. Use `--fps 0` without recording to measure uncapped throughput, and add `--no-preview` to see how much the preview costs. For recorded footage, compare successive CSV timestamps against the target frame interval (about 33.3 ms at 30 FPS) to check spacing and spot gaps.

`capture.py` provides timestamped BGR NumPy frames for future recognition and tracking. `recording.py` handles video and CSV output, and `main.py` runs the preview and reports timing.

## Offline replay and annotation

The annotator uses the same Python environment and dependencies as capture. Open any MP4 from a desktop session. Keep a capture's `.timestamps.csv` next to the video to preserve its monotonic frame timestamps.

```sh
python annotate.py recordings/match01.mp4 --dataset dataset
python annotate.py recordings/match01.mp4 --dataset dataset --review
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
| **P / N** | Jump to the previous or next saved frame in this video |
| **Q / Esc** | Quit |

Save or revert pending edits before navigating. Use **S** on a frame with no boxes to save it as a background image. Reopen the same video with `--review` to go straight to its first saved frame. Saved boxes appear automatically when you visit an exported frame.

The default classes, in ID order, are Friendly troop, Enemy troop, Friendly tower, Enemy tower, Card, and Other / unknown. For a custom list, pass `--classes path/to/classes.txt` when starting a dataset, with one name per line. The dataset stores that order in its own `classes.txt`, which it reuses on later runs.

### Dataset layout

Each MP4 gets its own match group by default. If a match spans multiple clips, pass the same `--match match01` when first opening each clip. The dataset remembers each source's group. Record one match per MP4 when possible.

- `images/<match>/` contains full-resolution PNGs from the decoded video.
- `labels/<match>/` contains matching YOLO label files with zero-based class IDs and normalized center, width, and height coordinates computed from the original image pixels. [YOLO detection format](https://docs.ultralytics.com/datasets/detect/)
- `metadata/<match>/` contains matching JSON files with the original video filename, zero-based frame index, playback timestamp in nanoseconds, and the capture monotonic timestamp when available. The playback timestamp falls back to the frame index divided by nominal FPS when the decoder supplies no timestamp.

Saving the same video and frame index again updates its labels in place. The tool identifies source videos by their content, so reopening a renamed copy in the same dataset keeps the same exports.

### Use the data for troop detection

Label every visible instance of the classes you intend to train on in each selected image. When ready to train, assign whole match groups to either training or validation. A YOLO dataset YAML can point to separate train and validation image lists and use the class order in `classes.txt`. For a troop-only model, prepare a label copy that keeps Friendly troop and Enemy troop and maps them to the IDs in the training configuration. [YOLO dataset configuration](https://docs.ultralytics.com/datasets/detect/)

## YOLO training and evaluation

### Prepare the annotated dataset

Use the same environment as capture and annotation. Preparation only needs the existing dependencies. It expects the annotator's `classes.txt` and matching files under `images/`, `labels/`, and `metadata/`.

```sh
python prepare_dataset.py --dataset dataset --val-fraction 0.2 --seed 0
```

The fraction applies to **matches**, with at least one match in each split. Use `--val-match MATCH_ID` (repeat for multiple matches) to keep specific matches in validation as you collect more footage. Use the match IDs printed by the annotator or listed in the metadata.

The validator reports missing or unreadable exports, invalid class IDs and box coordinates, duplicate source video/frame IDs, and identical pixels across matches. It reports repeated pixels within a match and keeps those frames together. An empty label file represents a valid background image. Review the match lists, background counts, and per-class box counts in `dataset/splits/split.json`. That folder also contains `data.yaml`, `train.txt`, and `val.txt`. [Ultralytics dataset layout](https://docs.ultralytics.com/datasets/detect/)

### Train after collecting real labeled gameplay

```sh
python -m pip install -r requirements-training.txt
python train.py --dataset dataset --model yolo11n.pt --epochs 50 --imgsz 640 --batch 8 --device cpu --output runs/troops
```

Use `--device 0` for the first CUDA GPU with a compatible PyTorch installation. The default pretrained YOLO11 nano checkpoint can download on first use. You can also pass the path to a local pretrained detection checkpoint. Training requires annotated objects in both splits and checks that the exports still match preparation. Rerun preparation after editing or adding samples. [Ultralytics training options](https://docs.ultralytics.com/modes/train/)

The new run directory contains `results.csv`, `weights/best.pt`, `weights/last.pt`, `args.yaml`, `run_config.json`, and the split files under `dataset/`. The run's lists reference the original PNGs and labels. Keep those files unchanged to reproduce evaluation for that run.

### Evaluate the validation matches

```sh
python evaluate.py --weights runs/troops/weights/best.pt --data runs/troops/dataset/data.yaml --device cpu --imgsz 640 --batch 1 --output runs/troops/evaluation.json
```

The JSON reports precision, recall, mAP50, and mAP50–95 overall and per class, along with each class's validation instance count. Classes absent from validation have null per-class metrics. Latency is the average milliseconds per image on the specified device and batch size. The report includes model inference alone and the combined preprocessing, inference, and postprocessing time. [Ultralytics validation metrics and timing](https://docs.ultralytics.com/guides/model-evaluation-insights/)

### Preview predictions on a recorded MP4

```sh
python infer_video.py recordings/match02.mp4 --weights runs/troops/weights/best.pt --device cpu --imgsz 640 --conf 0.25
```

The OpenCV preview shows class names and confidence scores. Press **Q** or **Esc** to exit. In other code, call `Detector.detect(frame)` with a BGR NumPy frame to get boxes in the original pixel coordinates. This accepts the same frame representation as the live capture module.
