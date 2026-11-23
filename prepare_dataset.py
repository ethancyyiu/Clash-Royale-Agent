import argparse
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from replay_dataset import read_classes, write_file


@dataclass(frozen=True)
class Sample:
    key: Path
    class_counts: Counter

    @property
    def match(self) -> str:
        return self.key.parts[0]


@dataclass(frozen=True)
class Dataset:
    root: Path
    classes: list[str]
    samples: list[Sample]
    fingerprint: str
    repeated_pixels: int


def validate_labels(content: bytes, class_count: int) -> Counter:
    counts = Counter()
    for line_number, line in enumerate(content.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            class_text, cx_text, cy_text, w_text, h_text = line.split()
            class_id = int(class_text)
            cx, cy, width, height = map(float, (cx_text, cy_text, w_text, h_text))
        except ValueError as error:
            raise ValueError(f"line {line_number}: expected class_id x_center y_center width height") from error
        if not 0 <= class_id < class_count:
            raise ValueError(f"line {line_number}: class ID {class_id} is out of range")
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in (cx, cy, width, height)):
            raise ValueError(f"line {line_number}: coordinates must be finite and normalized to [0, 1]")
        # Exported labels round center coordinates and sizes independently.
        tolerance = 1e-9
        if (
            width <= 0 or height <= 0
            or cx - width / 2 < -tolerance or cx + width / 2 > 1 + tolerance
            or cy - height / 2 < -tolerance or cy + height / 2 > 1 + tolerance
        ):
            raise ValueError(f"line {line_number}: box is empty or extends beyond the image")
        counts[class_id] += 1
    return counts


def read_sample(root: Path, key: Path, class_count: int):
    if len(key.parts) != 2:
        raise ValueError("expected a file directly inside a match folder")
    image_path = root / "images" / key.with_suffix(".png")
    label_path = root / "labels" / key.with_suffix(".txt")
    metadata_path = root / "metadata" / key.with_suffix(".json")
    missing = [str(path.relative_to(root)) for path in (image_path, label_path, metadata_path) if not path.is_file()]
    if missing:
        raise ValueError("missing " + ", ".join(missing))

    metadata_bytes = metadata_path.read_bytes()
    info = json.loads(metadata_bytes)
    if not isinstance(info, dict) or info.get("match_id") != key.parts[0]:
        raise ValueError("metadata match_id must match the folder")
    source = info.get("video_sha256")
    index = info.get("frame_index")
    if not isinstance(source, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", source):
        raise ValueError("metadata video_sha256 must be a SHA-256 digest")
    if type(index) is not int or index < 0:
        raise ValueError("metadata frame_index must be a nonnegative integer")

    image = cv2.imdecode(np.frombuffer(image_path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or min(image.shape[:2]) < 10:
        raise ValueError("PNG is unreadable or smaller than 10 pixels on one side")
    if (info.get("height"), info.get("width")) != image.shape[:2]:
        raise ValueError("metadata dimensions disagree with the PNG")
    pixels = hashlib.sha256(str(image.shape).encode("ascii"))
    pixels.update(image)
    label_bytes = label_path.read_bytes()
    counts = validate_labels(label_bytes, class_count)
    fingerprint = hashlib.sha256(metadata_bytes).digest() + hashlib.sha256(label_bytes).digest()
    return Sample(key, counts), (source.lower(), index), pixels.digest(), fingerprint


def inspect_dataset(root: Path) -> Dataset:
    root = root.resolve()
    classes = read_classes(root / "classes.txt")
    keys = set()
    for folder, suffix in (("images", ".png"), ("labels", ".txt"), ("metadata", ".json")):
        directory = root / folder
        keys.update(path.relative_to(directory).with_suffix("") for path in directory.rglob(f"*{suffix}") if path.is_file())
    if not keys:
        raise ValueError(f"No exported samples found in {root}")

    samples = []
    errors = []
    source_matches = {}
    source_frames = {}
    pixel_frames = {}
    repeated_pixels = 0
    fingerprint = hashlib.sha256(json.dumps(classes).encode("utf-8"))
    for key in sorted(keys):
        try:
            sample, identity, pixels, annotation_hash = read_sample(root, key, len(classes))
            if identity in source_frames:
                raise ValueError(f"duplicate source video and frame index also found at {source_frames[identity]}")
            source = identity[0]
            if source_matches.setdefault(source, sample.match) != sample.match:
                raise ValueError(f"source video is also assigned to match {source_matches[source]}")
            source_frames[identity] = key
            previous = pixel_frames.setdefault(pixels, sample)
            if previous.match != sample.match:
                raise ValueError(f"identical pixels also found in another match at {previous.key}")
            repeated_pixels += previous is not sample
            samples.append(sample)
            fingerprint.update(json.dumps(key.as_posix()).encode("utf-8"))
            fingerprint.update(pixels + annotation_hash)
        except (OSError, ValueError, cv2.error) as error:
            errors.append(f"{key.as_posix()}: {error}")
    if errors:
        omitted = f"\n... {len(errors) - 30} more" if len(errors) > 30 else ""
        raise ValueError(f"{len(errors)} invalid sample(s)\n" + "\n".join(errors[:30]) + omitted)
    return Dataset(root, classes, samples, fingerprint.hexdigest(), repeated_pixels)


def choose_validation_matches(
    dataset: Dataset, *, fraction: float = 0.2, seed: int = 0, explicit: list[str] | None = None,
) -> set[str]:
    matches = sorted({sample.match for sample in dataset.samples})
    if len(matches) < 2:
        raise ValueError("At least two distinct matches are required for a train/validation split")
    if explicit is not None:
        validation = set(explicit)
        unknown = validation.difference(matches)
        if unknown:
            raise ValueError("Unknown validation match IDs: " + ", ".join(sorted(unknown)))
        if not validation or len(validation) == len(matches):
            raise ValueError("Select at least one match for each split")
        return validation
    if not 0 < fraction < 1:
        raise ValueError("--val-fraction must be between 0 and 1")
    count = min(len(matches) - 1, max(1, round(len(matches) * fraction)))
    return set(random.Random(seed).sample(matches, count))


def write_split_bundle(dataset: Dataset, output: Path, validation: set[str]) -> Path:
    output = output.resolve()
    summary = {
        "dataset_root": str(dataset.root),
        "dataset_sha256": dataset.fingerprint,
        "classes": dataset.classes,
        "repeated_pixel_frames_within_match": dataset.repeated_pixels,
    }
    for split, is_val in (("train", False), ("val", True)):
        samples = [sample for sample in dataset.samples if (sample.match in validation) == is_val]
        if not samples:
            raise ValueError(f"The {split} split has no images")
        paths = [dataset.root / "images" / sample.key.with_suffix(".png") for sample in samples]
        write_file(output / f"{split}.txt", "".join(f"{path}\n" for path in paths).encode("utf-8"))
        counts = Counter()
        for sample in samples:
            counts.update(sample.class_counts)
        summary[split] = {
            "matches": sorted({sample.match for sample in samples}),
            "images": len(samples),
            "background_images": sum(not sample.class_counts for sample in samples),
            "boxes_per_class": [counts[i] for i in range(len(dataset.classes))],
        }
    yaml_path = output / "data.yaml"
    yaml_text = (
        f"path: {json.dumps(str(output), ensure_ascii=False)}\n"
        "train: train.txt\nval: val.txt\n"
        f"names: {json.dumps(dataset.classes, ensure_ascii=False)}\n"
    )
    write_file(yaml_path, yaml_text.encode("utf-8"))
    write_file(output / "split.json", (json.dumps(summary, indent=2) + "\n").encode("utf-8"))
    return yaml_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate annotations and split the dataset by match.")
    parser.add_argument("--dataset", type=Path, default=Path("dataset"), help="annotation dataset root")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="fraction of matches for validation")
    parser.add_argument("--seed", type=int, default=0, help="seed for random selection of whole matches")
    parser.add_argument("--val-match", action="append", help="explicit validation match ID; repeat as needed")
    args = parser.parse_args()
    try:
        dataset = inspect_dataset(args.dataset)
        validation = choose_validation_matches(
            dataset, fraction=args.val_fraction, seed=args.seed, explicit=args.val_match,
        )
        yaml_path = write_split_bundle(dataset, dataset.root / "splits", validation)
        if dataset.repeated_pixels:
            print(f"Found {dataset.repeated_pixels} repeated-pixel frame(s) within matches. Kept in the same split.")
        print(f"Prepared {len(dataset.samples)} images. Dataset YAML is at {yaml_path}.")
        print("Validation matches: " + ", ".join(sorted(validation)))
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
