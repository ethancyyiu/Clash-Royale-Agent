"""Learn from recorded play, review decisions, and run a saved policy."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .demonstrations import demos
from .policy import Policy
from .replay import replay as replay_policy
from .live import run as run_live
from ..detection.detector import add_detector_options


def parse_args():
    parser = argparse.ArgumentParser(description="Learn and run a Giant Graveyard imitation policy.")
    commands = parser.add_subparsers(dest="command", required=True)
    learn = commands.add_parser("learn", help="train from synchronized state and input event files")
    learn.add_argument("--demo", nargs=2, action="append", required=True, metavar=("STATES", "EVENTS"))
    learn.add_argument("--calibration", type=Path, default=Path("hud_data/calibration.json"))
    learn.add_argument("--output", type=Path, required=True)
    replay = commands.add_parser("replay", help="overlay policy decisions on a recorded match")
    replay.add_argument("video", type=Path)
    replay.add_argument("--states", type=Path, required=True)
    replay.add_argument("--policy", type=Path, required=True)
    replay.add_argument("--events", type=Path, help="recorded mouse events for human comparison")
    replay.add_argument("--calibration", type=Path, default=Path("hud_data/calibration.json"))
    replay.add_argument("--report", type=Path, required=True)
    replay.add_argument("--output-video", type=Path, help="save an MP4 with card and placement overlays")
    run = commands.add_parser("run", help="observe live gameplay; add --execute to send emulator clicks")
    run.add_argument("--policy", type=Path, required=True)
    run.add_argument("--region-file", type=Path, required=True)
    run.add_argument("--calibration", type=Path, default=Path("hud_data/calibration.json"))
    run.add_argument("--references", type=Path, default=Path("hud_data/references"))
    run.add_argument("--log", type=Path, required=True)
    run.add_argument("--monitor", type=int, default=1)
    run.add_argument("--execute", action="store_true")
    add_detector_options(run, required=True)
    return parser.parse_args()


def learn(args):
    if len(args.demo) < 2:
        raise ValueError("Provide at least two demonstration matches, reserving the last for evaluation.")
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    matches = []
    for states_path, events_path in args.demo:
        samples, actions = demos(Path(states_path), Path(events_path), calibration)
        matches.append(samples)
        print(f"{states_path}: {actions} confirmed plays, {len(samples) - actions} waits")
    training = [sample for match in matches[:-1] for sample in match]
    if sum(sample["action"] is not None for sample in training) < 10:
        raise ValueError("Need at least ten confirmed training plays. Check hand-change and phase labels.")
    policy = Policy(training, {"training_matches": [pair[0] for pair in args.demo[:-1]],
                               "held_out_match": args.demo[-1][0],
                               "confirmed_training_plays": sum(sample["action"] is not None for sample in training)})
    evaluated = correct = card_total = card_correct = 0
    for sample in matches[-1]:
        action, _ = policy.decide_query(sample, recent_giant=sample.get("recent_giant", False))
        predicted = action["card"] if action else None
        expected = sample["action"]["card"] if sample["action"] else None
        evaluated += 1
        correct += (predicted is None) == (expected is None)
        if expected is not None:
            card_total += 1
            card_correct += predicted == expected
    print(f"Held-out policy check: play/wait agreement {correct}/{evaluated}; card agreement {card_correct}/{card_total} plays")
    policy.save(args.output)
    print(f"Saved {len(training)} learned examples to {args.output}")


def main():
    args = parse_args()
    try:
        if args.command == "learn":
            learn(args)
        elif args.command == "replay":
            result = replay_policy(args.policy, args.states, args.video, args.report,
                                   events_path=args.events, calibration_path=args.calibration,
                                   output_video=args.output_video)
            print(f"Replayed {result['frames']} frames: {result['bot_plays']} bot plays, "
                  f"{result['human_plays']} confirmed human plays; report in {args.report}")
        elif args.command == "run":
            run_live(policy_path=args.policy, region_file=args.region_file,
                     calibration_path=args.calibration, references=args.references,
                     weights=args.weights, log_path=args.log, monitor=args.monitor,
                     execute=args.execute, device=args.device, imgsz=args.imgsz, conf=args.conf)
        else:
            raise ValueError(f"Unknown agent command: {args.command}")
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
