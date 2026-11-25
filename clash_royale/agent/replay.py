"""Offline policy replay with optional annotated MP4 and human-action comparison."""
from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path

import cv2

from ..capture.recording import video_writer
from .demonstrations import demos, read_jsonl
from .policy import Policy, card_name


def overlay(frame, state, prediction, human):
    image = frame.copy()
    height, width = image.shape[:2]
    def mark(action, color, label):
        if action is None:
            return
        x, y = round(action["x"] * width), round(action["y"] * height)
        cv2.circle(image, (x, y), 14, color, 2)
        cv2.putText(image, label + ": " + action["card"], (max(3, x - 60), max(18, y - 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)
    mark(human, (0, 220, 0), "You")
    mark(prediction, (0, 190, 255), "Bot")
    phase = state.get("phase", {}).get("value") or "?"
    elixir = state.get("elixir", {}).get("value")
    line = f"Frame {state['frame_index']}  {phase}  elixir {elixir if elixir is not None else '?'}"
    cv2.rectangle(image, (0, 0), (min(width, 420), 27), (0, 0, 0), -1)
    cv2.putText(image, line, (5, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def replay(policy_path: Path, states_path: Path, video_path: Path, report_path: Path,
           *, events_path: Path | None = None, calibration_path: Path | None = None,
           output_video: Path | None = None):
    policy = Policy.load(policy_path)
    states = read_jsonl(states_path)
    if not states:
        raise ValueError("State file is empty")
    human = {}
    if events_path is not None:
        if calibration_path is None:
            raise ValueError("--events requires --calibration")
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        _, _, plays = demos(states_path, events_path, calibration, include_plays=True)
        human = {states[index]["frame_index"]: sample["action"] for index, sample in plays}
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise OSError(f"Could not open {video_path}")
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        width = round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not fps or width <= 0 or height <= 0:
            raise ValueError("Invalid video dimensions or FPS")
        if output_video is not None and output_video.exists():
            raise FileExistsError(output_video)
        if report_path.exists():
            raise FileExistsError(report_path)
        predictions = []
        last_play_ns = last_giant_ns = 0
        with ExitStack() as stack:
            writer = None
            if output_video is not None:
                output_video.parent.mkdir(parents=True, exist_ok=True)
                writer = stack.enter_context(video_writer(output_video, fps, width, height))
            state_by_frame = {state["frame_index"]: state for state in states}
            if len(state_by_frame) != len(states) or set(state_by_frame) != set(range(len(states))):
                raise ValueError("State file must have one row for each consecutive video frame")
            frame_index = 0
            shown_prediction = None
            shown_prediction_ns = 0
            shown_human = None
            shown_human_ns = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                state = state_by_frame.get(frame_index)
                if state is None:
                    raise ValueError(f"Missing game state for video frame {frame_index}")
                stamp = state["timestamp_ns"]
                action, reason = policy.decide(state, recent_giant=stamp - last_giant_ns < 12_000_000_000)
                if stamp - last_play_ns < 2_000_000_000:
                    action, reason = None, "cooldown"
                if action is not None:
                    predictions.append({"frame_index": frame_index, "timestamp_ns": stamp,
                                        "action": action, "reason": reason})
                    last_play_ns = stamp
                    if card_name(action["card"]) == "Giant":
                        last_giant_ns = stamp
                    shown_prediction, shown_prediction_ns = action, stamp
                if frame_index in human:
                    shown_human, shown_human_ns = human[frame_index], stamp
                if writer is not None:
                    writer(overlay(
                        frame, state,
                        shown_prediction if stamp - shown_prediction_ns < 1_000_000_000 else None,
                        shown_human if stamp - shown_human_ns < 1_000_000_000 else None,
                    ))
                frame_index += 1
            if frame_index != len(states):
                raise ValueError(f"Video has {frame_index} frames but state file has {len(states)} rows")
        matches = []
        used = set()
        human_actions = list(human.items())
        for prediction in predictions:
            candidate = min(
                ((index, frame, action) for index, (frame, action) in enumerate(human_actions)
                 if index not in used and abs(state_by_frame[frame]["timestamp_ns"] - prediction["timestamp_ns"]) <= 1_500_000_000),
                key=lambda item: abs(state_by_frame[item[1]]["timestamp_ns"] - prediction["timestamp_ns"]),
                default=None,
            )
            if candidate is not None:
                used.add(candidate[0])
                actual = candidate[2]
                predicted = prediction["action"]
                matches.append({"frame_index": prediction["frame_index"],
                                "human_card": actual["card"], "bot_card": predicted["card"],
                                "same_card": actual["card"] == predicted["card"],
                                "placement_distance": round(((actual["x"] - predicted["x"]) ** 2 +
                                                             (actual["y"] - predicted["y"]) ** 2) ** 0.5, 3)})
        report = {
            "video": str(video_path), "frames": frame_index,
            "bot_plays": len(predictions), "human_plays": len(human),
            "nearby_plays": len(matches), "same_card": sum(item["same_card"] for item in matches),
            "predictions": predictions, "comparisons": matches,
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
        return report
    finally:
        capture.release()
