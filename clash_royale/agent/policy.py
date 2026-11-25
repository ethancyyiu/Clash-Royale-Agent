"""A small imitation policy trained from synchronized human demonstrations."""
from __future__ import annotations

import json
from pathlib import Path

DECK = {
    "Giant": 5, "Graveyard": 5, "Bowler": 5, "Evo Witch": 5,
    "Guards": 3, "Zappies": 4, "Snowball": 2, "Arrows": 3,
}
ALIASES = {
    "witch": "Evo Witch", "witch evolution": "Evo Witch",
    "giant snowball": "Snowball", "giant snowball evolution": "Snowball",
}
PHASES = ("single", "double", "triple")


def card_name(value):
    if not isinstance(value, str):
        return None
    normalized = value.strip().casefold()
    return ALIASES.get(normalized, next((name for name in DECK if name.casefold() == normalized), None))


def feature(state):
    """Fixed-size battlefield summary; detector coordinates are frame-relative."""
    width, height = state.get("frame_width") or 1, state.get("frame_height") or 1
    cells = [0.0] * 12  # 2 teams x 2 lanes x 3 arena bands
    for box in state.get("detections") or []:
        name = box.get("class_name", "").casefold().replace(" ", "_")
        if name not in ("friendly_troop", "enemy_troop"):
            continue
        x1, y1, x2, y2 = box["xyxy"]
        x, y = (x1 + x2) / (2 * width), (y1 + y2) / (2 * height)
        if not 0 <= x <= 1 or not 0 <= y <= 0.8:
            continue
        index = (0 if name == "friendly_troop" else 6) + (0 if x < 0.5 else 3) + min(2, int(y / 0.8 * 3))
        cells[index] = min(5.0, cells[index] + 1.0)
    return cells


def state_hand(state):
    return [card_name(reading.get("value")) for reading in state.get("hand", [])]


def valid_state(state):
    return (
        state.get("phase", {}).get("value") in PHASES
        and type(state.get("elixir", {}).get("value")) is int
        and len(state_hand(state)) == 4
        and all(state_hand(state))
    )


def example(state, action=None):
    return {
        "phase": state["phase"]["value"],
        "elixir": state["elixir"]["value"],
        "hand": state_hand(state),
        "board": feature(state),
        "action": action,
    }


def distance(left, right):
    score = 0 if left["phase"] == right["phase"] else 8
    score += abs(left["elixir"] - right["elixir"]) * 0.6
    score += sum(abs(a - b) for a, b in zip(left["board"], right["board"])) * 0.35
    score += len(set(left["hand"]) ^ set(right["hand"])) * 0.45
    return score


class Policy:
    def __init__(self, samples, metadata=None):
        self.samples = samples
        self.metadata = metadata or {}

    @classmethod
    def load(cls, path: Path):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("version") != 1 or not isinstance(data.get("samples"), list):
            raise ValueError("Invalid policy file")
        return cls(data["samples"], data.get("metadata"))

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump({"version": 1, "deck": DECK, "metadata": self.metadata,
                       "samples": self.samples}, stream)

    def decide(self, state, *, recent_giant=False):
        if not valid_state(state):
            return None, "unknown game state"
        return self.decide_query(example(state), recent_giant=recent_giant)

    def decide_query(self, query, *, recent_giant=False):
        elixir, phase = query["elixir"], query["phase"]
        candidates = []
        for sample in self.samples:
            action = sample["action"]
            if action is not None:
                card = action["card"]
                if card not in query["hand"] or DECK[card] > elixir:
                    continue
                if phase == "single" and card in ("Giant", "Graveyard"):
                    continue
                if card == "Graveyard" and not recent_giant:
                    continue
            candidates.append((distance(query, sample), sample))
        if not candidates:
            return None, "no applicable demonstrations"
        candidates.sort(key=lambda item: item[0])
        closest = candidates[:7]
        votes = {}
        for gap, sample in closest:
            action = sample["action"]
            key = action["card"] if action else "wait"
            votes[key] = votes.get(key, 0) + 1 / (0.5 + gap)
        choice = max(votes, key=votes.get)
        if choice == "wait":
            return None, "learned wait"
        matching = [(gap, sample["action"]) for gap, sample in closest if sample["action"] and sample["action"]["card"] == choice]
        action = min(matching, key=lambda item: item[0])[1]
        return action, f"learned {choice} ({len(matching)} nearby examples)"
