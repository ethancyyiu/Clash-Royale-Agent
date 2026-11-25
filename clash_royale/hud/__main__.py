import argparse
import hashlib
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from time import monotonic

import cv2

from ..capture.video import frame_timestamp, open_video, read_frame
from ..dataset.replay_dataset import read_capture_timestamps, write_file
from ..detection.detector import Detector, add_detector_options
from ..game_state import observe_game
from .calibration import DEFAULT_CALIBRATION, DEFAULT_REFERENCES, Calibration, select_calibration
from .evaluation import evaluate, load_labels
from .recognition import DEFAULT_MARGIN, DEFAULT_SCORE, HudRecognizer, draw_hud

DEFAULT_LABELS = Path("hud_data/labels.jsonl")
WINDOW = "Gameplay HUD Replay"


def require_display():
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        raise OSError("OpenCV preview needs a desktop display. Run with DISPLAY set.")


def card_name(name: str) -> str | None:
    if name == "-":
        return None
    if not name or name != name.strip() or name.endswith(".") or any(
        character in '/\\:*?"<>|' or ord(character) < 32 for character in name
    ):
        raise argparse.ArgumentTypeError("Use a card name safe for a folder, or - to skip.")
    return name


def add_recognition_options(parser):
    parser.add_argument("--references", type=Path, default=DEFAULT_REFERENCES)
    parser.add_argument("--min-score", type=float, default=DEFAULT_SCORE)
    parser.add_argument("--min-margin", type=float, default=DEFAULT_MARGIN)


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate, label, recognize, and evaluate the gameplay HUD.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("calibrate", "select HUD regions on a recorded frame"),
        ("collect", "crop labelled reference images from a recording"),
        ("label", "save ground-truth labels for evaluation"),
        ("replay", "read the HUD and optionally detect troops in an MP4"),
        ("evaluate", "score HUD recognition on labelled frames"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("video", type=Path, help="source MP4")
        command.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
        if name in ("calibrate", "collect", "label"):
            command.add_argument("--frame", type=int, default=0, help="zero-based frame index")
        if name == "calibrate":
            command.add_argument("--next-card", action="store_true", help="also select the next-card preview")
            command.add_argument("--phase", action="store_true", help="also select the match timer and multiplier")
        if name in ("collect", "label"):
            command.add_argument("--hand", type=card_name, nargs=4, metavar="CARD", help="four labels left to right; - to skip")
            command.add_argument("--elixir", type=int, choices=range(11))
            command.add_argument("--next-card", type=card_name)
            command.add_argument("--phase", choices=("single", "double", "triple"))
        if name == "collect":
            command.add_argument("--references", type=Path, default=DEFAULT_REFERENCES)
        if name in ("label", "evaluate"):
            command.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
        if name in ("replay", "evaluate"):
            add_recognition_options(command)
        if name == "replay":
            command.add_argument("--states", type=Path, help="write game states to a new JSONL file")
            command.add_argument("--no-preview", action="store_true")
            add_detector_options(command)
    return parser.parse_args()


def label_frame(args, calibration: Calibration) -> None:
    hand = args.hand or [None] * 4
    if all(value is None for value in [*hand, args.elixir, args.next_card, args.phase]):
        raise ValueError("Provide at least one card or an elixir label.")
    image = read_frame(args.video, args.frame)
    calibration.check_frame(image)
    if args.command == "label":
        existing = load_labels(args.labels, args.video.name) if args.labels.exists() else {}
        if args.frame in existing:
            raise ValueError("This frame already has ground-truth labels in that file.")
        row = {
            "video_filename": args.video.name, "frame_index": args.frame,
            "hand": hand, "elixir": args.elixir, "next_card": args.next_card, "phase": args.phase,
        }
        args.labels.parent.mkdir(parents=True, exist_ok=True)
        with args.labels.open("a", encoding="utf-8") as file:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Saved ground truth to {args.labels}")
        return

    if args.next_card is not None and calibration.next_card is None:
        raise ValueError("Calibrate the next-card preview before collecting its references.")
    if args.phase is not None and calibration.phase is None:
        raise ValueError("Calibrate the timer/phase region before collecting phase references.")
    regions = [("cards", label, region) for label, region in zip(hand, calibration.hand)]
    regions += [("elixir", args.elixir, calibration.elixir), ("next", args.next_card, calibration.next_card)]
    regions += [("phase", args.phase, calibration.phase)]
    saved = 0
    for kind, label, region in regions:
        if label is None:
            continue
        ok, encoded = cv2.imencode(".png", region.crop(image))
        if not ok:
            raise OSError(f"Could not encode the {kind} crop.")
        data = encoded.tobytes()
        path = args.references / kind / str(label) / (hashlib.sha256(data).hexdigest() + ".png")
        if not path.exists():
            write_file(path, data)
            saved += 1
    print(f"Saved {saved} new reference image(s) in {args.references}")


def replay(args, recognizer: HudRecognizer) -> None:
    if not args.no_preview:
        require_display()
    elif args.states is None:
        raise ValueError("--no-preview requires --states.")
    detector = Detector(args.weights, device=args.device, imgsz=args.imgsz, conf=args.conf) if args.weights else None
    timestamps = read_capture_timestamps(args.video)
    if args.states is not None:
        args.states.parent.mkdir(parents=True, exist_ok=True)
    window_created = False
    try:
        with open_video(args.video) as (video, fps):
            ok, image = video.read()
            if not ok:
                raise OSError(f"Could not decode the first frame of {args.video}")
            recognizer.calibration.check_frame(image)
            with args.states.open("x", encoding="utf-8") if args.states else nullcontext() as state_file:
                if not args.no_preview:
                    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
                    window_created = True
                index = 0
                while ok:
                    started = monotonic()
                    timestamp, source = frame_timestamp(video, fps, index, timestamps)
                    state = observe_game(image, timestamp, source, index, recognizer, detector)
                    if state_file is not None:
                        state.write(state_file)
                    if window_created:
                        cv2.imshow(WINDOW, draw_hud(image, state))
                        delay = max(1, round((1 / fps - (monotonic() - started)) * 1000))
                        if cv2.waitKey(delay) & 0xFF in (ord("q"), 27):
                            break
                        try:
                            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                                break
                        except cv2.error:
                            break
                    index += 1
                    ok, image = video.read()
    finally:
        if window_created:
            cv2.destroyWindow(WINDOW)


def main() -> int:
    args = parse_args()
    try:
        if args.command == "calibrate":
            require_display()
            calibration = select_calibration(read_frame(args.video, args.frame), args.next_card, args.phase)
            calibration.save(args.calibration)
            print(f"Saved HUD calibration to {args.calibration}")
        else:
            calibration = Calibration.load(args.calibration)
            if args.command in ("collect", "label"):
                label_frame(args, calibration)
            else:
                recognizer = HudRecognizer.from_directory(
                    calibration, args.references, args.min_score, args.min_margin,
                )
                if args.command == "evaluate":
                    print(json.dumps(evaluate(args.video, args.labels, recognizer), indent=2, allow_nan=False))
                else:
                    replay(args, recognizer)
    except KeyboardInterrupt:
        return 0
    except (OSError, RuntimeError, ValueError, cv2.error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
