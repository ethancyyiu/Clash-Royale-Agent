"""Learn from demonstration recordings and run the resulting policy."""
from __future__ import annotations

import argparse
import json
import signal
import sys
from dataclasses import asdict
from pathlib import Path
from time import monotonic_ns, sleep

import mss

from ..capture.region_file import load_region
from ..capture.screen import grab_frame, select_region
from ..detection.detector import Detector, add_detector_options
from ..game_state import observe_game
from ..hud.calibration import Calibration
from ..hud.recognition import HudRecognizer
from .demonstrations import demos
from .mouse import click, require_windows
from .policy import Policy, card_name, state_hand
from .replay import replay as replay_policy
from .readiness import report as readiness_report
from .simulator import benchmark


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
    simulate = commands.add_parser("simulate", help="benchmark offline in an abstract match environment")
    simulate.add_argument("--policy", type=Path, help="learned policy; omit for the scripted baseline")
    simulate.add_argument("--matches", type=int, default=9)
    simulate.add_argument("--seed", type=int, default=0)
    simulate.add_argument("--output", type=Path, required=True)
    status = commands.add_parser("status", help="show missing offline-agent inputs and saved metrics")
    status.add_argument("--calibration", type=Path, default=Path("hud_data/calibration.json"))
    status.add_argument("--references", type=Path, default=Path("hud_data/references"))
    status.add_argument("--holdout", type=Path, default=Path("runs/troops-v2/holdout.json"))
    status.add_argument("--policy", type=Path)
    status.add_argument("--simulation", type=Path)
    run = commands.add_parser("run", help="run in dry mode, or act with --execute")
    run.add_argument("--policy", type=Path, required=True)
    run.add_argument("--region-file", type=Path, required=True)
    run.add_argument("--monitor", type=int, default=1)
    run.add_argument("--calibration", type=Path, default=Path("hud_data/calibration.json"))
    run.add_argument("--references", type=Path, default=Path("hud_data/references"))
    run.add_argument("--log", type=Path, required=True)
    run.add_argument("--execute", action="store_true", help="send mouse input to the emulator")
    add_detector_options(run)
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


def run(args):
    if args.execute:
        require_windows()
    policy = Policy.load(args.policy)
    calibration = Calibration.load(args.calibration)
    if calibration.phase is None:
        raise ValueError("Calibrate the match timer/phase before running the bot.")
    if args.weights is None:
        raise ValueError("Provide --weights so the policy can observe troops.")
    hud = HudRecognizer.from_directory(calibration, args.references)
    detector = Detector(args.weights, device=args.device, imgsz=args.imgsz, conf=args.conf) if args.weights else None
    stopped = False

    def stop(_signum, _frame):
        nonlocal stopped
        stopped = True

    old_handler = signal.signal(signal.SIGINT, stop)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    try:
        with mss.MSS() as screen, args.log.open("x", encoding="utf-8") as log:
            region = select_region(screen, args.monitor, load_region(args.region_file))
            last_action_ns = 0
            last_giant_ns = 0
            pending = None
            index = 0
            print("Bot running. Dry mode." if not args.execute else "Bot executing. Press Ctrl+C to stop.", flush=True)
            while not stopped:
                frame = grab_frame(screen, region)
                state = observe_game(frame.bgr, frame.captured_ns, "capture_monotonic", index, hud, detector)
                index += 1
                now = monotonic_ns()
                if pending is not None:
                    slot, card, sent_ns = pending
                    if any(change.slot == slot and card_name(change.played) == card for change in state.hand_changes):
                        pending = None
                    elif now - sent_ns > 3_000_000_000:
                        raise RuntimeError(f"Play of {card} was not confirmed; stopped before another input.")
                state_data = asdict(state)
                action, reason = policy.decide(
                    state_data,
                    recent_giant=now - last_giant_ns < 12_000_000_000,
                )
                if pending is not None or now - last_action_ns < 2_000_000_000:
                    action, reason = None, "waiting for confirmation or cooldown"
                if now - frame.captured_ns > 500_000_000:
                    action, reason = None, "stale observation"
                executed = False
                if action is not None:
                    hand = state_hand(state_data)
                    slot = hand.index(action["card"]) + 1
                    x, y = action["x"], action["y"]
                    if 0 <= x <= 1 and 0 <= y < 0.8:
                        if args.execute:
                            card_region = calibration.hand[slot - 1]
                            card_x = round(region["left"] + (card_region.left + card_region.right) / 2 * region["width"])
                            card_y = round(region["top"] + (card_region.top + card_region.bottom) / 2 * region["height"])
                            click(card_x, card_y)
                            click(round(region["left"] + x * region["width"]), round(region["top"] + y * region["height"]))
                            pending = (slot, action["card"], now)
                        if action["card"] == "Giant":
                            last_giant_ns = now
                        last_action_ns = now
                        executed = args.execute
                    else:
                        action, reason = None, "invalid placement"
                log.write(json.dumps({
                    "timestamp_ns": frame.captured_ns, "frame_index": state.frame_index,
                    "phase": state.phase.value, "elixir": state.elixir.value,
                    "hand": [reading.value for reading in state.hand],
                    "action": action, "reason": reason, "executed": executed,
                }) + "\n")
                sleep(0.05)
    finally:
        signal.signal(signal.SIGINT, old_handler)


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
        elif args.command == "simulate":
            result = benchmark(args.output, policy=Policy.load(args.policy) if args.policy else None,
                               matches=args.matches, seed=args.seed)
            print(f"Abstract simulator: {result['wins']} wins, {result['losses']} losses, "
                  f"{result['draws']} draws; {result['invalid_actions']} invalid player actions. "
                  f"Report in {args.output}")
        elif args.command == "status":
            print(json.dumps(readiness_report(args.calibration, args.references, args.holdout,
                                              args.policy, args.simulation), indent=2))
        else:
            run(args)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
