"""Live observation and optional emulator input using a saved policy."""
from __future__ import annotations

import json
import signal
from dataclasses import asdict
from pathlib import Path
from time import monotonic_ns, sleep

import mss

from ..capture.region_file import load_region
from ..capture.screen import grab_frame, select_region
from ..detection.detector import Detector
from ..game_state import observe_game
from ..hud.calibration import Calibration
from ..hud.recognition import HudRecognizer
from .mouse import click, require_windows
from .policy import Policy, card_name, state_hand

SPELLS = {"Graveyard", "Snowball", "Arrows"}


def valid_placement(card: str, x: float, y: float) -> bool:
    """Conservative crop-relative guard, not a complete game tile validator."""
    if not 0.05 <= x <= 0.95:
        return False
    if card == "Graveyard":
        return 0.08 <= y <= 0.42
    return 0.08 <= y <= 0.75 if card in SPELLS else 0.42 <= y <= 0.75


def run(*, policy_path: Path, region_file: Path, calibration_path: Path,
        references: Path, weights: Path, log_path: Path, monitor: int,
        execute: bool, device: str, imgsz: int, conf: float) -> None:
    if execute:
        require_windows()
    policy = Policy.load(policy_path)
    calibration = Calibration.load(calibration_path)
    if calibration.phase is None or len(calibration.timer_digits) != 3:
        raise ValueError("Calibrate the multiplier and timer digits before running the agent.")
    hud = HudRecognizer.from_directory(calibration, references)
    detector = Detector(weights, device=device, imgsz=imgsz, conf=conf)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stopped = False

    def stop(_signum, _frame):
        nonlocal stopped
        stopped = True

    previous_handler = signal.signal(signal.SIGINT, stop)
    try:
        with mss.MSS() as screen, log_path.open("x", encoding="utf-8") as log:
            region = select_region(screen, monitor, load_region(region_file))
            pending = None
            last_action_ns = last_giant_ns = 0
            frame_index = 0
            print("Agent running with clicks enabled." if execute else "Agent running in dry mode.", flush=True)
            while not stopped:
                frame = grab_frame(screen, region)
                state = observe_game(frame.bgr, frame.captured_ns, "capture_monotonic", frame_index, hud, detector)
                frame_index += 1
                now = monotonic_ns()
                if pending is not None:
                    slot, card, sent_ns = pending
                    if any(change.slot == slot and card_name(change.played) == card for change in state.hand_changes):
                        pending = None
                        if card == "Giant":
                            last_giant_ns = now
                    elif now - sent_ns > 3_000_000_000:
                        raise RuntimeError(f"Unconfirmed {card} play; stopped before another input")
                state_data = asdict(state)
                action, reason = policy.decide(state_data, recent_giant=now - last_giant_ns < 12_000_000_000)
                if pending is not None or now - last_action_ns < 2_000_000_000:
                    action, reason = None, "waiting for confirmation or cooldown"
                if now - frame.captured_ns > 500_000_000:
                    action, reason = None, "stale observation"
                sent = False
                if action is not None:
                    card = card_name(action.get("card"))
                    x, y = action.get("x"), action.get("y")
                    if (card is None or not isinstance(x, (int, float)) or not isinstance(y, (int, float))
                            or not valid_placement(card, x, y)):
                        action, reason = None, "invalid placement"
                    else:
                        hand = state_hand(state_data)
                        slot = hand.index(card) + 1
                        if execute:
                            slot_region = calibration.hand[slot - 1]
                            slot_x = round(region["left"] + (slot_region.left + slot_region.right) * region["width"] / 2)
                            slot_y = round(region["top"] + (slot_region.top + slot_region.bottom) * region["height"] / 2)
                            click(slot_x, slot_y)
                            click(round(region["left"] + x * region["width"]),
                                  round(region["top"] + y * region["height"]))
                            pending = (slot, card, now)
                            sent = True
                        elif card == "Giant":
                            last_giant_ns = now
                        last_action_ns = now
                log.write(json.dumps({"timestamp_ns": state.timestamp_ns, "frame_index": state.frame_index,
                                      "phase": state.phase.value, "elixir": state.elixir.value,
                                      "hand": [reading.value for reading in state.hand],
                                      "action": action, "reason": reason, "sent": sent}) + "\n")
                sleep(0.05)
    finally:
        signal.signal(signal.SIGINT, previous_handler)
