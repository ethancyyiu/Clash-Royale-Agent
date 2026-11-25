import argparse
import sys
from importlib import import_module

COMMANDS = {
    "capture": (".capture.cli", "capture the screen or write live game states"),
    "crop": (".capture.crop", "crop a recorded MP4"),
    "annotate": (".dataset.annotate", "label and review frames"),
    "suggest-frames": (".dataset.review_queue", "choose frames to label"),
    "check-dataset": (".dataset.check_dataset", "audit annotation quality"),
    "prepare-dataset": (".dataset.prepare_dataset", "prepare match-based training splits"),
    "train": (".detection.train", "train a troop detector"),
    "evaluate": (".detection.evaluate", "score a troop detector"),
    "infer-video": (".detection.infer_video", "preview detections in a recording"),
    "hud": (".hud.__main__", "calibrate, collect references, replay or evaluate the HUD"),
    "agent": (".agent.__main__", "learn from your play and run a Giant Graveyard bot"),
}


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m clash_royale", description="Clash Royale capture and analysis tools.")
    commands = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)
    for name, (_, description) in COMMANDS.items():
        commands.add_parser(name, add_help=False, help=description)
    command = parser.parse_args(sys.argv[1:2]).command
    sys.argv[0:2] = [f"{parser.prog} {command}"]
    return import_module(COMMANDS[command][0], __package__).main()


if __name__ == "__main__":
    raise SystemExit(main())
