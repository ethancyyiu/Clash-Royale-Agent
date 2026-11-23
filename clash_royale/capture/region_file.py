import json
from pathlib import Path


def load_region(path: Path) -> list[int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    region = data.get("region") if isinstance(data, dict) else None
    if not isinstance(region, list) or len(region) != 4 or any(type(value) is not int for value in region):
        raise ValueError(f"{path} must contain a region array with four integers.")
    return region
