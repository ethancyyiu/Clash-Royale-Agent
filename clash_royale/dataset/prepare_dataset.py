import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .replay_dataset import TROOP_CLASSES, read_boxes, read_classes, training_class_ids, write_file


@dataclass(frozen=True)
class Sample:
    key: Path
    class_counts: Counter
    labels: bytes

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


def read_metadata(path: Path, match: str) -> tuple[dict, bytes]:
    metadata_bytes = path.read_bytes()
    info = json.loads(metadata_bytes)
    if not isinstance(info, dict) or info.get("match_id") != match:
        raise ValueError("metadata match_id must match the folder")
    source = info.get("video_sha256")
    index = info.get("frame_index")
    if not isinstance(source, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", source):
        raise ValueError("metadata video_sha256 must be a SHA-256 digest")
    if type(index) is not int or index < 0:
        raise ValueError("metadata frame_index must be a nonnegative integer")
    return info, metadata_bytes


def read_image(path: Path) -> np.ndarray:
    content = path.read_bytes()
    image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR) if content else None
    if image is None or min(image.shape[:2]) < 10:
        raise ValueError("PNG is unreadable or smaller than 10 pixels on one side")
    return image


def pixel_digest(image: np.ndarray) -> bytes:
    pixels = hashlib.sha256(str(image.shape).encode("ascii"))
    pixels.update(image)
    return pixels.digest()


def read_sample(root: Path, key: Path, class_count: int):
    if len(key.parts) != 2:
        raise ValueError("expected a file directly inside a match folder")
    image_path = root / "images" / key.with_suffix(".png")
    label_path = root / "labels" / key.with_suffix(".txt")
    metadata_path = root / "metadata" / key.with_suffix(".json")
    missing = [str(path.relative_to(root)) for path in (image_path, label_path, metadata_path) if not path.is_file()]
    if missing:
        raise ValueError("missing " + ", ".join(missing))

    info, metadata_bytes = read_metadata(metadata_path, key.parts[0])
    image = read_image(image_path)
    height, width = image.shape[:2]
    if (info.get("height"), info.get("width")) != (height, width):
        raise ValueError("metadata dimensions disagree with the PNG")
    label_bytes = label_path.read_bytes()
    boxes, errors = read_boxes(label_bytes, class_count, width, height)
    if errors:
        raise ValueError(errors[0])
    counts = Counter(box.class_id for box in boxes)
    fingerprint = hashlib.sha256(metadata_bytes).digest() + hashlib.sha256(label_bytes).digest()
    identity = (info["video_sha256"].lower(), info["frame_index"])
    return Sample(key, counts, label_bytes), identity, pixel_digest(image), fingerprint


def sample_keys(root: Path) -> list[Path]:
    keys = set()
    for folder, suffix in (("images", ".png"), ("labels", ".txt"), ("metadata", ".json")):
        directory = root / folder
        keys.update(path.relative_to(directory).with_suffix("") for path in directory.rglob(f"*{suffix}") if path.is_file())
    if not keys:
        raise ValueError(f"No exported samples found in {root}")
    return sorted(keys)


def inspect_dataset(root: Path) -> Dataset:
    root = root.resolve()
    classes = read_classes(root / "classes.txt")
    samples = []
    errors = []
    source_matches = {}
    source_frames = {}
    pixel_frames = {}
    repeated_pixels = 0
    fingerprint = hashlib.sha256(json.dumps(classes).encode("utf-8"))
    for key in sample_keys(root):
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


def prepared_split_dir(root: Path, troop_only: bool) -> Path:
    return root / "splits" / "troops" if troop_only else root / "splits"


def write_box_preview(image_path: Path, labels: bytes, classes: list[str], output: Path) -> None:
    image = read_image(image_path)
    height, width = image.shape[:2]
    boxes, errors = read_boxes(labels, len(classes), width, height)
    if errors:
        raise ValueError(f"{image_path}: {errors[0]}")
    for box in boxes:
        color = (80, 220, 80) if box.class_id % 2 == 0 else (80, 125, 255)
        cv2.rectangle(image, (box.left, box.top), (box.right - 1, box.bottom - 1), color, 2)
        cv2.putText(image, classes[box.class_id], (box.left, max(15, box.top - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    ok, preview = cv2.imencode(".jpg", image)
    if not ok:
        raise OSError(f"Could not encode {output}")
    write_file(output, preview.tobytes())


def write_split_bundle(
    dataset: Dataset, output: Path, validation: set[str], *, troop_only: bool = False, preview: bool = False,
) -> Path:
    output = output.resolve()
    class_ids = training_class_ids(dataset.classes, troop_only)
    class_map = {old: new for new, old in enumerate(class_ids)}
    classes = list(TROOP_CLASSES) if troop_only else dataset.classes
    # Key generated files by the source dataset so a new revision never mixes in old previews or labels.
    export = output / "export" / dataset.fingerprint
    previews = output / "previews" / dataset.fingerprint
    summary = {
        "dataset_root": str(dataset.root),
        "dataset_sha256": dataset.fingerprint,
        "classes": classes,
        "troop_only": troop_only,
        "repeated_pixel_frames_within_match": dataset.repeated_pixels,
    }
    for split, is_val in (("train", False), ("val", True)):
        samples = [sample for sample in dataset.samples if (sample.match in validation) == is_val]
        if not samples:
            raise ValueError(f"The {split} split has no images")
        paths = []
        counts = Counter()
        backgrounds = 0
        for sample in samples:
            image_key = sample.key.with_suffix(".png")
            image_path = dataset.root / "images" / image_key
            labels = sample.labels
            if troop_only:
                rows = []
                for line in labels.splitlines():
                    if line.strip():
                        class_id, coordinates = line.split(maxsplit=1)
                        if int(class_id) in class_map:
                            rows.append(f"{class_map[int(class_id)]} ".encode("ascii") + coordinates + b"\n")
                labels = b"".join(rows)
                write_file(export / "labels" / sample.key.with_suffix(".txt"), labels)
                exported_image = export / "images" / image_key
                write_file(exported_image, image_path.read_bytes())
                image_path = exported_image
            if preview:
                write_box_preview(image_path, labels, classes, previews / sample.key.with_suffix(".jpg"))
            paths.append(image_path)
            counts.update(sample.class_counts)
            backgrounds += not any(sample.class_counts[class_id] for class_id in class_ids)
        write_file(output / f"{split}.txt", "".join(f"{path}\n" for path in paths).encode("utf-8"))
        summary[split] = {
            "matches": sorted({sample.match for sample in samples}),
            "images": len(samples),
            "background_images": backgrounds,
            "boxes_per_class": [counts[class_id] for class_id in class_ids],
        }
    if preview:
        print(f"Label previews are at {previews}.")
    yaml_path = output / "data.yaml"
    yaml_text = (
        f"path: {json.dumps(str(output), ensure_ascii=False)}\n"
        "train: train.txt\nval: val.txt\n"
        f"names: {json.dumps(classes, ensure_ascii=False)}\n"
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
    parser.add_argument("--troop-only", action="store_true", help="export a two-class copy for troop detection")
    parser.add_argument("--preview", action="store_true", help="save JPEGs showing the exported boxes")
    args = parser.parse_args()
    try:
        dataset = inspect_dataset(args.dataset)
        validation = choose_validation_matches(
            dataset, fraction=args.val_fraction, seed=args.seed, explicit=args.val_match,
        )
        yaml_path = write_split_bundle(
            dataset, prepared_split_dir(dataset.root, args.troop_only), validation,
            troop_only=args.troop_only, preview=args.preview,
        )
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
