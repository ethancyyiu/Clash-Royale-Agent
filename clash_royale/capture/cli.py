import argparse
import json
import math
import os
import signal
import sys
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from statistics import fmean
from time import monotonic_ns, sleep

import cv2
import mss

from ..detection.detector import Detector, draw_detections
from ..hud.calibration import DEFAULT_REFERENCES, Calibration
from ..hud.recognition import DEFAULT_MARGIN, DEFAULT_SCORE, HudRecognizer, draw_hud
from .screen import draw_region, grab_frame, select_region
from .recording import record_video
from .region_file import load_region
from ..dataset.replay_dataset import write_file

WINDOW = "Clash Royale Capture"
NS_PER_SECOND = 1_000_000_000
MAX_CAPTURE_FAILURES = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview and record Clash Royale screen capture.")
    parser.add_argument("--list-monitors", action="store_true", help="show monitor IDs and bounds")
    parser.add_argument("--monitor", type=int, default=1, help="monitor ID (default: 1)")
    parser.add_argument("--select-region", type=Path, metavar="FILE.json",
                        help="draw a new capture box on a monitor screenshot and save it")
    region = parser.add_mutually_exclusive_group()
    region.add_argument(
        "--region", nargs=4, type=int, metavar=("X", "Y", "WIDTH", "HEIGHT"),
        help="crop in pixels relative to the selected monitor's top-left corner",
    )
    region.add_argument("--region-file", type=Path, help="load a monitor-relative crop from JSON")
    parser.add_argument("--record", type=Path, metavar="VIDEO.mp4", help="save video and frame timestamps")
    parser.add_argument(
        "--fps", type=float, default=30.0,
        help="target capture rate and MP4 playback rate (default: 30; 0 for uncapped capture)",
    )
    parser.add_argument("--no-preview", action="store_true", help="skip the OpenCV window")
    parser.add_argument("--hud-calibration", type=Path, help="enable HUD recognition with this calibration JSON")
    parser.add_argument("--hud-references", type=Path, default=DEFAULT_REFERENCES)
    parser.add_argument("--hud-min-score", type=float, default=DEFAULT_SCORE)
    parser.add_argument("--hud-min-margin", type=float, default=DEFAULT_MARGIN)
    parser.add_argument("--states", type=Path, help="write HUD observations to a new JSONL file")
    parser.add_argument("--weights", type=Path, help="optional trained YOLO checkpoint for preview boxes")
    parser.add_argument("--device", default="cpu", help="YOLO device, e.g. cpu or 0")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO input image size")
    parser.add_argument("--conf", type=float, default=0.25, help="minimum YOLO confidence")
    args = parser.parse_args()
    if not math.isfinite(args.fps) or (args.fps != 0 and args.fps < 1):
        parser.error("--fps must be 0 or at least 1")
    if args.record and args.record.suffix.lower() != ".mp4":
        parser.error("--record requires a .mp4 filename")
    if args.record and args.fps == 0:
        parser.error("--record requires a positive --fps")
    if args.states and not args.hud_calibration:
        parser.error("--states requires --hud-calibration")
    if args.weights and args.no_preview:
        parser.error("--weights draws boxes in the preview and requires it to be enabled")
    if args.imgsz <= 0 or not 0 <= args.conf <= 1:
        parser.error("--imgsz must be positive and --conf must be in [0, 1]")
    if args.select_region and (args.region or args.region_file or args.record or args.list_monitors):
        parser.error("--select-region cannot be combined with --region, --region-file, --record, or --list-monitors")
    if args.select_region and args.select_region.suffix.lower() != ".json":
        parser.error("--select-region requires a .json filename")
    return args


def preview_finished() -> bool:
    if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
        return True
    try:
        return cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        # Some backends remove the window before its visibility can be queried.
        return True


def capture_loop(
    screen: mss.MSS,
    region: dict[str, int],
    args: argparse.Namespace,
    stop_requested: Callable[[], bool],
) -> None:
    interval_ns = int(NS_PER_SECOND / args.fps) if args.fps else 0
    hud = None
    if args.hud_calibration:
        hud = HudRecognizer.from_directory(
            Calibration.load(args.hud_calibration), args.hud_references,
            args.hud_min_score, args.hud_min_margin,
        )
    detector = Detector(args.weights, device=args.device, imgsz=args.imgsz, conf=args.conf) if args.weights else None
    with ExitStack() as resources:
        write_frame = resources.enter_context(
            record_video(args.record, args.fps, region["width"], region["height"])
        ) if args.record else None
        state_file = None
        if args.states:
            args.states.parent.mkdir(parents=True, exist_ok=True)
            state_file = resources.enter_context(args.states.open("x", encoding="utf-8"))
        frame_index = 0
        report_started_ns = monotonic_ns()
        grab_times = []
        processing_times = []
        consecutive_failures = 0

        while not stop_requested():
            started_ns = monotonic_ns()
            try:
                frame = grab_frame(screen, region)
            except mss.ScreenShotError as error:
                consecutive_failures += 1
                if consecutive_failures >= MAX_CAPTURE_FAILURES:
                    raise RuntimeError(
                        f"Capture failed {MAX_CAPTURE_FAILURES} times in a row: {error}"
                    ) from error
                print(
                    f"Capture failed ({consecutive_failures}/{MAX_CAPTURE_FAILURES}): {error}. Retrying.",
                    file=sys.stderr,
                )
                sleep(0.1)
            else:
                consecutive_failures = 0
                if write_frame is not None:
                    write_frame(frame)
                state = hud.observe(frame.bgr, frame.captured_ns, "capture_monotonic", frame_index) if hud else None
                if state_file is not None and state is not None:
                    state.write(state_file)
                frame_index += 1
                stop = False
                if not args.no_preview:
                    preview = frame.bgr
                    if detector is not None:
                        preview = draw_detections(preview, detector.detect(frame.bgr))
                    if state is not None:
                        preview = draw_hud(preview, state)
                    cv2.imshow(WINDOW, preview)
                    stop = preview_finished()
                grab_times.append(frame.grab_ns)
                processing_times.append(monotonic_ns() - frame.captured_ns)
                if stop:
                    return

            now_ns = monotonic_ns()
            elapsed_ns = now_ns - report_started_ns
            if elapsed_ns >= NS_PER_SECOND:
                fps = len(grab_times) * NS_PER_SECOND / elapsed_ns
                if grab_times:
                    print(
                        f"Capture {fps:.1f} FPS | grab avg {fmean(grab_times) / 1e6:.2f} ms"
                        f" | processing avg {fmean(processing_times) / 1e6:.2f} ms"
                        f" / max {max(processing_times) / 1e6:.2f} ms",
                        flush=True,
                    )
                else:
                    print("Capture 0.0 FPS | no successful frames", flush=True)
                grab_times.clear()
                processing_times.clear()
                report_started_ns = now_ns

            remaining_ns = interval_ns - (monotonic_ns() - started_ns)
            if remaining_ns > 0:
                sleep(remaining_ns / NS_PER_SECOND)


def main() -> int:
    args = parse_args()
    window_created = False
    stopping = False

    def request_stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    previous_handler = signal.signal(signal.SIGINT, request_stop)
    try:
        if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
            raise RuntimeError("MSS needs an X11 display on Linux. Run from an X11 desktop with DISPLAY set.")
        with mss.MSS() as screen:
            if args.list_monitors:
                monitors = screen.monitors[1:]
                if not monitors:
                    raise RuntimeError("No monitors found.")
                for number, monitor in enumerate(monitors, start=1):
                    print(
                        f"Monitor {number}: {monitor['width']}x{monitor['height']} "
                        f"at ({monitor['left']}, {monitor['top']})"
                    )
                return 0

            if args.select_region:
                region_coordinates = draw_region(screen, args.monitor)
                write_file(args.select_region, (
                    json.dumps({"region": region_coordinates}, indent=2) + "\n"
                ).encode("utf-8"))
                print(f"Saved capture region {region_coordinates} to {args.select_region}")
                return 0

            coordinates = load_region(args.region_file) if args.region_file else args.region
            region = select_region(screen, args.monitor, coordinates)
            if not args.no_preview:
                cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
                window_created = True
            print(
                f"Capturing monitor {args.monitor}, {region['width']}x{region['height']} "
                f"at ({region['left']}, {region['top']}). Press Ctrl+C to stop.",
                flush=True,
            )
            # Finish the current video frame and CSV row before closing on Ctrl+C.
            capture_loop(screen, region, args, lambda: stopping)
    except (mss.ScreenShotError, cv2.error, OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    finally:
        if window_created:
            cv2.destroyAllWindows()
        signal.signal(signal.SIGINT, previous_handler)
    return 0


if __name__ == "__main__":
    sys.exit(main())
