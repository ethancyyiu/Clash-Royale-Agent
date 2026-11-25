"""Recover confirmed card plays from mouse events and recorded game states."""
from __future__ import annotations

import json
from bisect import bisect_right
from pathlib import Path

from .policy import card_name, example, valid_state


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def inside(point, region):
    return region["left"] <= point["x"] <= region["right"] and region["top"] <= point["y"] <= region["bottom"]


def demos(states_path: Path, events_path: Path, calibration: dict, *, include_plays=False):
    states, events = read_jsonl(states_path), read_jsonl(events_path)
    if not states:
        raise ValueError(f"No states in {states_path}")
    if all(state.get("detections") is None for state in states):
        raise ValueError(f"{states_path} has no troop observations; record with --weights")
    timestamps = [state["timestamp_ns"] for state in states]
    plays = []
    selected = None
    for event in events:
        if event["event"] == "down":
            slot = next((i for i, region in enumerate(calibration["hand"]) if inside(event, region)), None)
            if slot is not None:
                selected = (slot, event["timestamp_ns"])
            continue
        if selected is None:
            continue
        slot, started = selected
        if event["timestamp_ns"] - started > 1_500_000_000:
            selected = None
            continue
        if event["y"] >= 0.80 or not 0 <= event["x"] <= 1 or not 0 <= event["y"] <= 1:
            continue
        index = bisect_right(timestamps, started) - 1
        if index < 0 or not valid_state(states[index]):
            selected = None
            continue
        card = card_name(states[index]["hand"][slot]["value"])
        confirmed = any(
            change["slot"] == slot + 1 and card_name(change["played"]) == card
            for state in states[index + 1:bisect_right(timestamps, event["timestamp_ns"] + 3_000_000_000)]
            for change in state.get("hand_changes", [])
        )
        if card and confirmed:
            plays.append((index, example(states[index], {
                "card": card, "x": round(event["x"], 3), "y": round(event["y"], 3),
            })))
        selected = None
    action_indices = [index for index, _ in plays]
    waits = []
    last = None
    for index, state in enumerate(states):
        stamp = state["timestamp_ns"]
        if not valid_state(state) or (last is not None and stamp - last < 2_000_000_000):
            continue
        if any(abs(stamp - timestamps[action_index]) < 1_500_000_000 for action_index in action_indices):
            continue
        waits.append(example(state))
        last = stamp
    samples = [item for _, item in plays] + waits
    return (samples, len(plays), plays) if include_plays else (samples, len(plays))
