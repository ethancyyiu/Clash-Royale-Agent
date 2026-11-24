import json
from collections import Counter
from pathlib import Path
from statistics import fmean

import numpy as np

from ..capture.video import frame_timestamp, open_video
from ..dataset.replay_dataset import read_capture_timestamps
from .recognition import HudRecognizer


def load_labels(path: Path, video_filename: str) -> dict[int, dict]:
    labels = {}
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                label = json.loads(line)
                if not isinstance(label, dict):
                    raise ValueError("expected a JSON object")
                if label["video_filename"] != video_filename:
                    continue
                index = label["frame_index"]
                if type(index) is not int or index < 0 or index in labels:
                    raise ValueError("invalid or duplicate frame index")
                hand = label["hand"]
                next_card = label["next_card"]
                elixir = label["elixir"]
                if not isinstance(hand, list) or len(hand) != 4:
                    raise ValueError("hand must contain exactly four slots")
                if any(value is not None and (not isinstance(value, str) or not value.strip())
                       for value in [*hand, next_card]):
                    raise ValueError("card labels must be nonempty names or null")
                if elixir is not None and (type(elixir) is not int or not 0 <= elixir <= 10):
                    raise ValueError("elixir must be an integer from 0 to 10 or null")
                if all(value is None for value in [*hand, next_card, elixir]):
                    raise ValueError("at least one ground-truth value is needed")
                labels[index] = label
            except (KeyError, ValueError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    return labels


def evaluate(video_path: Path, labels_path: Path, recognizer: HudRecognizer) -> dict:
    labels = load_labels(labels_path, video_path.name)
    if not labels:
        raise ValueError(f"No ground truth for {video_path.name} in {labels_path}")
    timestamps = read_capture_timestamps(video_path)
    counts = {field: Counter() for field in ("hand", "next_card", "elixir")}
    latencies = []
    elixir_errors = []

    def score(field, truth, predicted):
        if truth is None:
            return
        counts[field]["total"] += 1
        counts[field]["unknown"] += predicted is None
        counts[field]["correct"] += predicted == truth

    with open_video(video_path) as (video, fps):
        for index in range(max(labels) + 1):
            if index not in labels:
                if not video.grab():
                    raise ValueError(f"Ground truth refers to frames beyond the end of {video_path}")
                continue
            ok, image = video.read()
            if not ok:
                raise ValueError(f"Could not decode labelled frame {index}")
            timestamp, source = frame_timestamp(video, fps, index, timestamps)
            state = recognizer.observe(image, timestamp, source, index)
            truth = labels[index]
            latencies.append(state.recognition_latency_ms)
            for expected, reading in zip(truth["hand"], state.hand):
                score("hand", expected, reading.value)
            score("next_card", truth["next_card"], state.next_card.value if state.next_card is not None else None)
            score("elixir", truth["elixir"], state.elixir.value)
            if truth["elixir"] is not None and isinstance(state.elixir.value, int):
                elixir_errors.append(abs(truth["elixir"] - state.elixir.value))

    def report(counter: Counter) -> dict:
        total = counter["total"]
        return {
            "samples": total, "correct": counter["correct"], "unknown": counter["unknown"],
            "accuracy": counter["correct"] / total if total else None,
            "unknown_rate": counter["unknown"] / total if total else None,
        }

    return {
        "video_filename": video_path.name,
        "labelled_frames": len(latencies),
        "hand": report(counts["hand"]),
        "next_card": report(counts["next_card"]),
        "elixir": {
            **report(counts["elixir"]),
            "mean_absolute_error_when_recognized": fmean(elixir_errors) if elixir_errors else None,
        },
        "overall": report(sum(counts.values(), Counter())),
        "recognition_latency_ms": {"mean": fmean(latencies), "p95": float(np.percentile(latencies, 95))},
    }
