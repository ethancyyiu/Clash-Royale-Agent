from dataclasses import dataclass
from pathlib import Path
from time import monotonic_ns
from typing import Protocol

import cv2
import numpy as np

from ..detection.detector import draw_detections
from ..game_state import GameState, HandChange, Reading
from .calibration import Calibration

DEFAULT_SCORE = 0.86
DEFAULT_MARGIN = 0.025
Label = str | int


class CropRecognizer(Protocol):
    def recognize(self, crop: np.ndarray) -> Reading: ...


class TemplateBank:
    """Classify a HUD crop by comparing it to user-labelled examples."""

    def __init__(
        self, directory: Path, size: tuple[int, int],
        min_score: float, min_margin: float, *, elixir: bool = False,
    ):
        self.size = size
        self.min_score = min_score
        self.min_margin = min_margin
        self.labels: list[Label] = []
        self.starts = []
        vectors = []
        for folder in sorted(directory.iterdir()) if directory.is_dir() else ():
            paths = sorted(folder.glob("*.png")) if folder.is_dir() else []
            if not paths:
                continue
            label = int(folder.name) if elixir else folder.name
            if elixir and not 0 <= label <= 10:
                raise ValueError(f"Elixir reference label must be between 0 and 10: {folder}")
            self.labels.append(label)
            self.starts.append(len(vectors))
            for path in paths:
                image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"Could not read reference image {path}")
                vectors.append(self.vector(image))
        self.vectors = np.asarray(vectors, dtype=np.int16)

    def vector(self, image: np.ndarray) -> np.ndarray:
        return cv2.resize(image, self.size, interpolation=cv2.INTER_AREA).ravel().astype(np.int16)

    def recognize(self, crop: np.ndarray) -> Reading:
        if not self.labels:
            return Reading()
        distance = np.abs(self.vectors - self.vector(crop)).mean(axis=1) / 255
        scores = 1 - np.minimum.reduceat(distance, self.starts)
        order = np.argsort(scores)
        best = order[-1]
        score = float(scores[best])
        ambiguous = len(order) > 1 and score - scores[order[-2]] <= self.min_margin
        return Reading(None if score < self.min_score or ambiguous else self.labels[best], score)


@dataclass
class SlotHistory:
    stable: str | None = None
    candidate: str | None = None
    sightings: int = 0
    last_seen_ns: int | None = None

    def update(self, value: str | None, timestamp_ns: int) -> tuple[str, str] | None:
        if self.last_seen_ns is not None and timestamp_ns - self.last_seen_ns > 1_000_000_000:
            self.stable = None
            self.candidate = None
            self.sightings = 0
        if value is None:
            self.candidate = None
            self.sightings = 0
            return None
        self.last_seen_ns = timestamp_ns
        self.sightings = self.sightings + 1 if value == self.candidate else 1
        self.candidate = value
        if self.sightings != 3:
            return None
        previous, self.stable = self.stable, value
        if previous is not None and previous != value:
            return previous, value
        return None


class HudRecognizer:
    def __init__(
        self, calibration: Calibration, hand: CropRecognizer, elixir: CropRecognizer,
        next_card: CropRecognizer, phase: CropRecognizer | None = None,
        timer_digit: CropRecognizer | None = None,
    ):
        self.calibration = calibration
        self.hand = hand
        self.elixir = elixir
        self.next_card = next_card
        self.phase = phase
        self.timer_digit = timer_digit
        self.stable_phase: str | None = None
        self.phase_candidate: str | None = None
        self.phase_sightings = 0
        self.phase_seen_ns: int | None = None
        self.history = [SlotHistory() for _ in calibration.hand]
        self.last_timestamp: tuple[str, int] | None = None

    @classmethod
    def from_directory(
        cls, calibration: Calibration, references: Path,
        min_score: float = DEFAULT_SCORE, min_margin: float = DEFAULT_MARGIN,
    ):
        if not (0 <= min_score <= 1 and 0 <= min_margin <= 1):
            raise ValueError("HUD score and margin must be in [0, 1].")
        return cls(
            calibration,
            TemplateBank(references / "cards", (40, 52), min_score, min_margin),
            TemplateBank(references / "elixir", (80, 24), min_score, min_margin, elixir=True),
            TemplateBank(references / "next", (36, 40), min_score, min_margin),
            TemplateBank(references / "phase", (80, 32), min_score, min_margin),
            TemplateBank(references / "timer_digits", (20, 28), min_score, min_margin),
        )

    def read_timer(self, frame: np.ndarray) -> Reading:
        if self.timer_digit is None or len(self.calibration.timer_digits) != 3:
            return Reading()
        readings = [self.timer_digit.recognize(region.crop(frame)) for region in self.calibration.timer_digits]
        digits = [reading.value for reading in readings]
        if any(not isinstance(digit, str) or len(digit) != 1 or not digit.isdigit() for digit in digits):
            return Reading()
        minute, tens, ones = map(int, digits)
        if tens > 5:
            return Reading()
        return Reading(minute * 60 + tens * 10 + ones, min(reading.confidence for reading in readings))

    def read_phase(self, frame: np.ndarray, timer: Reading, timestamp_ns: int) -> Reading:
        raw = self.phase.recognize(self.calibration.phase.crop(frame)) if self.phase is not None and self.calibration.phase is not None else Reading()
        value = raw.value if raw.value in ("single", "double", "triple") else None
        if self.calibration.timer_digits and timer.value is None:
            value = None
        if value is not None and self.stable_phase is not None:
            order = {"single": 0, "double": 1, "triple": 2}
            if order[value] < order[self.stable_phase] and self.phase_seen_ns is not None and timestamp_ns - self.phase_seen_ns < 5_000_000_000:
                value = None
        if value != self.phase_candidate:
            self.phase_candidate, self.phase_sightings = value, 1
        elif value is not None:
            self.phase_sightings += 1
        if value is not None and self.phase_sightings >= 2:
            self.stable_phase = value
            self.phase_seen_ns = timestamp_ns
        if self.stable_phase is not None and self.phase_seen_ns is not None and timestamp_ns - self.phase_seen_ns <= 1_000_000_000:
            return Reading(self.stable_phase, raw.confidence)
        return Reading()

    def observe(self, frame: np.ndarray, timestamp_ns: int, timestamp_source: str, frame_index: int) -> GameState:
        started = monotonic_ns()
        self.calibration.check_frame(frame)
        hand = tuple(self.hand.recognize(region.crop(frame)) for region in self.calibration.hand)
        elixir = self.elixir.recognize(self.calibration.elixir.crop(frame))
        region = self.calibration.next_card
        next_card = self.next_card.recognize(region.crop(frame)) if region is not None else None
        if self.last_timestamp is not None:
            source, stamp = self.last_timestamp
            if source != timestamp_source or timestamp_ns < stamp:
                self.history = [SlotHistory() for _ in hand]
                self.stable_phase = None
                self.phase_candidate = None
                self.phase_sightings = 0
                self.phase_seen_ns = None
        self.last_timestamp = timestamp_source, timestamp_ns
        timer = self.read_timer(frame)
        phase = self.read_phase(frame, timer, timestamp_ns)
        changes = []
        for slot, (history, reading) in enumerate(zip(self.history, hand), 1):
            change = history.update(reading.value if isinstance(reading.value, str) else None, timestamp_ns)
            if change is not None:
                changes.append(HandChange(slot, *change))
        return GameState(
            timestamp_ns, timestamp_source, frame_index, elixir, hand, next_card,
            tuple(changes), (monotonic_ns() - started) / 1e6,
            phase=phase,
            timer_seconds=timer,
        )


def draw_hud(frame: np.ndarray, state: GameState) -> np.ndarray:
    if state.detections:
        frame = draw_detections(frame, state.detections)

    def description(reading: Reading) -> str:
        return "?" if reading.value is None else f"{reading.value} ({reading.confidence:.2f})"

    lines = [f"Frame {state.frame_index}   Elixir: {description(state.elixir)}"]
    if state.phase.value is not None:
        lines.append(f"Phase: {description(state.phase)}")
    if state.timer_seconds.value is not None:
        minutes, seconds = divmod(state.timer_seconds.value, 60)
        lines.append(f"Timer: {minutes}:{seconds:02d}")
    lines.extend(f"{i}. {description(reading)}" for i, reading in enumerate(state.hand, 1))
    if state.next_card is not None:
        lines.append(f"Next: {description(state.next_card)}")
    lines.extend(f"Slot {change.slot}: {change.played} -> {change.drawn}" for change in state.hand_changes)
    scale = min(0.55, max(0.30, frame.shape[1] / 900))
    sizes = [cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0] for line in lines]
    line_height = max(size[1] for size in sizes) + 6
    panel_width = max(size[0] for size in sizes) + 12
    height, width = frame.shape[:2]
    annotated = cv2.copyMakeBorder(
        frame, 0, line_height * len(lines) + 5, 0, max(0, panel_width - width),
        cv2.BORDER_CONSTANT, value=(24, 24, 24),
    )
    for i, line in enumerate(lines, 1):
        cv2.putText(annotated, line, (6, height + i * line_height - 2), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (245, 245, 245), 1, cv2.LINE_AA)
    return annotated
