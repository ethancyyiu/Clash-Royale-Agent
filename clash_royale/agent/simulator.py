"""Deterministic toy arena for offline policy development, not Clash Royale rules."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from .policy import DECK, Policy, card_name

MATCH_SECONDS = 300
SPELLS = {"Graveyard", "Snowball", "Arrows"}
STATS = {
    "Giant": (12, 2.0, 0.013), "Bowler": (7, 1.5, 0.017),
    "Evo Witch": (6, 1.7, 0.015), "Guards": (5, 1.1, 0.026),
    "Zappies": (5, 1.2, 0.017),
}


@dataclass
class Unit:
    side: int  # 0: learned policy, 1: scripted opponent
    lane: int
    y: float
    hp: float
    attack: float
    speed: float
    name: str


class Arena:
    """Small abstract environment with elixir, deck cycle, lanes, towers, and units."""

    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.seed = seed
        self.second = 0
        self.elixir = [5.0, 5.0]
        self.towers = [[100.0, 100.0], [100.0, 100.0]]
        self.hands = []
        self.draws = []
        for _ in range(2):
            shuffled = list(DECK)
            self.rng.shuffle(shuffled)
            self.hands.append(shuffled[:4])
            self.draws.append(shuffled[4:])
        self.units: list[Unit] = []
        self.last_giant = [-100, -100]
        self.invalid = [0, 0]
        self.plays = [0, 0]

    @property
    def phase(self):
        return "single" if self.second < 120 else "double" if self.second < 240 else "triple"

    def state(self, side: int):
        # Reflect the field so both agents see their own tower at the bottom.
        detections = []
        for unit in self.units:
            x = 0.27 if unit.lane == 0 else 0.73
            y = unit.y if side == 0 else 0.8 - unit.y
            detections.append({
                "class_name": "friendly_troop" if unit.side == side else "enemy_troop",
                "confidence": 1.0,
                "xyxy": [max(0, (x - 0.025) * 512), max(0, (y - 0.02) * 902),
                         min(512, (x + 0.025) * 512), min(902, (y + 0.02) * 902)],
            })
        return {
            "timestamp_ns": self.second * 1_000_000_000,
            "frame_index": self.second,
            "phase": {"value": self.phase, "confidence": 1.0},
            "elixir": {"value": int(self.elixir[side]), "confidence": 1.0},
            "hand": [{"value": card, "confidence": 1.0} for card in self.hands[side]],
            "next_card": {"value": self.draws[side][0], "confidence": 1.0},
            "detections": detections, "frame_width": 512, "frame_height": 902,
        }

    def legal(self, side: int, action):
        if action is None:
            return True, "wait"
        if not isinstance(action, dict) or card_name(action.get("card")) not in self.hands[side]:
            return False, "card unavailable"
        card = card_name(action["card"])
        if DECK[card] > int(self.elixir[side]):
            return False, "insufficient elixir"
        x, y = action.get("x"), action.get("y")
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)) or not 0.05 <= x <= 0.95:
            return False, "invalid coordinates"
        if card in SPELLS:
            if not 0.05 <= y <= 0.75:
                return False, "spell outside arena"
        elif not 0.42 <= y <= 0.75:
            return False, "troop outside friendly side"
        if card in ("Giant", "Graveyard") and self.phase == "single" and side == 0:
            return False, "single-elixir survival rule"
        return True, "legal"

    def play(self, side: int, action):
        legal, reason = self.legal(side, action)
        if not legal:
            self.invalid[side] += 1
            return False, reason
        if action is None:
            return False, "wait"
        card = card_name(action["card"])
        lane = 0 if action["x"] < 0.5 else 1
        self.elixir[side] -= DECK[card]
        self.hands[side].remove(card)
        self.hands[side].append(self.draws[side].pop(0))
        self.draws[side].append(card)
        self.plays[side] += 1
        if card in STATS:
            hp, attack, speed = STATS[card]
            start_y = action["y"] if side == 0 else 0.8 - action["y"]
            self.units.append(Unit(side, lane, start_y, hp, attack, speed, card))
            if card == "Giant":
                self.last_giant[side] = self.second
        elif card == "Graveyard":
            # Tanked Graveyards are stronger; still only a coarse proxy.
            damage = 8 if self.second - self.last_giant[side] <= 15 else 3
            self.towers[1 - side][lane] -= damage
        else:
            center = action["y"] if side == 0 else 0.8 - action["y"]
            for unit in self.units:
                if unit.side != side and unit.lane == lane and abs(unit.y - center) < 0.18:
                    unit.hp -= 5 if card == "Arrows" else 3
            if (center < 0.18 if side == 0 else center > 0.62):
                self.towers[1 - side][lane] -= 2 if card == "Arrows" else 1
        return True, "played"

    def tick(self):
        rate = {"single": 1 / 2.8, "double": 1 / 1.4, "triple": 1 / 0.93}[self.phase]
        self.elixir = [min(10.0, value + rate) for value in self.elixir]
        for unit in self.units:
            enemies = [other for other in self.units if other.side != unit.side and other.lane == unit.lane
                       and abs(other.y - unit.y) < 0.075 and other.hp > 0]
            if enemies:
                enemies[0].hp -= unit.attack
            else:
                unit.y += -unit.speed if unit.side == 0 else unit.speed
                if (unit.side == 0 and unit.y <= 0.10) or (unit.side == 1 and unit.y >= 0.70):
                    self.towers[1 - unit.side][unit.lane] -= unit.attack
                    unit.hp -= 1.5
        self.units = [unit for unit in self.units if unit.hp > 0 and 0 <= unit.y <= 0.8]
        self.second += 1

    def result(self):
        own, enemy = sum(max(0, hp) for hp in self.towers[0]), sum(max(0, hp) for hp in self.towers[1])
        return "win" if own > enemy else "loss" if own < enemy else "draw"


class ScriptedOpponent:
    def __init__(self, style: str, seed: int):
        self.style = style
        self.rng = random.Random(seed)

    def decide(self, arena: Arena, side: int):
        hand = arena.hands[side]
        affordable = [card for card in hand if DECK[card] <= int(arena.elixir[side])
                      and not (side == 0 and arena.phase == "single" and card in ("Giant", "Graveyard"))]
        if not affordable:
            return None
        if self.style == "rush":
            preferred = ["Giant", "Graveyard", "Evo Witch", "Bowler", "Zappies", "Guards"]
        elif self.style == "defensive":
            preferred = ["Bowler", "Guards", "Zappies", "Evo Witch", "Giant", "Graveyard"]
        else:
            preferred = self.rng.sample(list(DECK), len(DECK))
        card = next((name for name in preferred if name in affordable), affordable[0])
        lane = self.rng.randrange(2)
        x = 0.27 if lane == 0 else 0.73
        y = 0.13 if card == "Graveyard" else 0.55
        if card in ("Arrows", "Snowball"):
            y = 0.32
        return {"card": card, "x": x, "y": y}


def benchmark(output: Path, *, policy: Policy | None, matches: int, seed: int):
    if matches < 1:
        raise ValueError("--matches must be positive")
    if output.exists():
        raise FileExistsError(output)
    results = []
    for number in range(matches):
        arena = Arena(seed + number)
        opponent_style = ("rush", "defensive", "random")[number % 3]
        player_script = ScriptedOpponent("defensive", seed + number * 2)
        opponent = ScriptedOpponent(opponent_style, seed + number * 2 + 1)
        recent_action = -100
        trajectory = []
        while arena.second < MATCH_SECONDS and all(hp > 0 for towers in arena.towers for hp in towers):
            state = arena.state(0)
            if policy is None:
                action = player_script.decide(arena, 0) if arena.second % 7 == 0 else None
                source = "scripted"
            else:
                action, source = policy.decide(state, recent_giant=arena.second - arena.last_giant[0] < 12)
                if arena.second - recent_action < 2:
                    action, source = None, "cooldown"
            played, reason = arena.play(0, action)
            if played:
                recent_action = arena.second
            if arena.second % 7 == 0:
                arena.play(1, opponent.decide(arena, 1))
            trajectory.append({"second": arena.second, "state": state, "action": action,
                               "played": played, "reason": reason, "source": source})
            arena.tick()
        results.append({"seed": seed + number, "opponent": opponent_style, "result": arena.result(),
                        "seconds": arena.second, "tower_hp": arena.towers, "plays": arena.plays,
                        "invalid_actions": arena.invalid, "trajectory": trajectory})
    summary = {"simulator": "abstract-v1", "policy": "learned" if policy else "scripted baseline",
               "policy_metadata": policy.metadata if policy else None,
               "matches": matches, "seed": seed,
               "wins": sum(match["result"] == "win" for match in results),
               "losses": sum(match["result"] == "loss" for match in results),
               "draws": sum(match["result"] == "draw" for match in results),
               "completed": sum(match["seconds"] == MATCH_SECONDS or any(hp <= 0 for towers in match["tower_hp"] for hp in towers) for match in results),
               "invalid_actions": sum(match["invalid_actions"][0] for match in results),
               "results": results}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    return summary
