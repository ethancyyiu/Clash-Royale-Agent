from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, TextIO

if TYPE_CHECKING:
    import numpy as np

    from .detection.detector import Detection, Detector
    from .hud.recognition import HudRecognizer


@dataclass(frozen=True)
class Reading:
    value: str | int | None = None
    confidence: float = 0.0


@dataclass(frozen=True)
class HandChange:
    slot: int
    played: str
    drawn: str


@dataclass(frozen=True)
class GameState:
    timestamp_ns: int
    timestamp_source: str
    frame_index: int
    elixir: Reading
    hand: tuple[Reading, ...]
    next_card: Reading | None
    hand_changes: tuple[HandChange, ...]
    recognition_latency_ms: float
    detections: tuple[Detection, ...] | None = None
    frame_width: int | None = None
    frame_height: int | None = None
    phase: Reading = Reading()

    def write(self, stream: TextIO) -> None:
        payload = asdict(self)
        if self.detections is None:
            del payload["detections"], payload["frame_width"], payload["frame_height"]
        stream.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")


def observe_game(
    frame: np.ndarray, timestamp_ns: int, timestamp_source: str, frame_index: int,
    hud: HudRecognizer, detector: Detector | None = None,
) -> GameState:
    state = hud.observe(frame, timestamp_ns, timestamp_source, frame_index)
    if detector is None:
        return state
    height, width = frame.shape[:2]
    return replace(state, detections=tuple(detector.detect(frame)), frame_width=width, frame_height=height)
