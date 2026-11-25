"""Read-only inventory of the offline agent's current inputs and evidence."""
from __future__ import annotations

import json
from pathlib import Path


def report(calibration: Path, references: Path, holdout: Path, policy: Path | None,
           simulation: Path | None):
    result = {"calibration": calibration.is_file(), "references": {},
              "holdout": None, "policy": None, "simulation": None, "missing": []}
    if not result["calibration"]:
        result["missing"].append("HUD calibration")
    else:
        data = json.loads(calibration.read_text(encoding="utf-8"))
        if not data.get("phase") or len(data.get("timer_digits", [])) != 3:
            result["missing"].append("multiplier and timer-digit calibration")
    for category in ("cards", "elixir", "phase", "timer_digits"):
        labels = [folder.name for folder in (references / category).iterdir()
                  if folder.is_dir() and any(folder.glob("*.png"))] if (references / category).is_dir() else []
        result["references"][category] = labels
        if not labels:
            result["missing"].append(f"{category} reference crops")
    if holdout.is_file():
        data = json.loads(holdout.read_text(encoding="utf-8"))
        result["holdout"] = {key: data.get("overall", {}).get(key) for key in ("mAP50", "mAP75", "precision", "recall")}
    else:
        result["missing"].append("troop holdout report")
    if policy is not None and policy.is_file():
        data = json.loads(policy.read_text(encoding="utf-8"))
        result["policy"] = {"samples": len(data.get("samples", [])), "metadata": data.get("metadata", {})}
    else:
        result["missing"].append("trained demonstration policy")
    if simulation is not None and simulation.is_file():
        data = json.loads(simulation.read_text(encoding="utf-8"))
        result["simulation"] = {key: data.get(key) for key in ("matches", "completed", "wins", "losses", "draws", "invalid_actions")}
    else:
        result["missing"].append("offline simulation report")
    return result
