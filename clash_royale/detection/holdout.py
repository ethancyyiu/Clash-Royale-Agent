import json
from collections import Counter
from pathlib import Path

from ..dataset.prepare_dataset import Dataset, read_metadata, remap_labels
from ..dataset.replay_dataset import TROOP_CLASSES, read_boxes, training_class_ids, write_file
from .detector import Detection


def export_holdout(dataset: Dataset, output: Path) -> Path:
    class_ids = training_class_ids(dataset.classes, troop_only=True)
    if not any(sample.class_counts[class_id] for sample in dataset.samples for class_id in class_ids):
        raise ValueError("The held-out dataset needs at least one annotated troop.")
    class_map = {old: new for new, old in enumerate(class_ids)}
    for index, sample in enumerate(dataset.samples):
        # Flat, unique stems keep validator prediction files unambiguous across matches.
        stem = f"{index:08d}"
        image = dataset.root / "images" / sample.key.with_suffix(".png")
        write_file(output / "images" / f"{stem}.png", image.read_bytes())
        write_file(output / "labels" / f"{stem}.txt", remap_labels(sample.labels, class_map))

    data = {"path": str(output), "train": None, "val": "images", "names": TROOP_CLASSES}
    path = output / "data.yaml"
    write_file(path, json.dumps(data).encode("utf-8"))
    return path


def box_iou(a: tuple, b: tuple) -> float:
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def read_predictions(path: Path, width: int, height: int, confidence: float) -> list[Detection]:
    if not path.is_file():
        return []
    predictions = []
    for line in path.read_text(encoding="utf-8").splitlines():
        class_id, *values = line.split()
        cx, cy, w, h, score = map(float, values)
        if score >= confidence:
            xyxy = ((cx - w / 2) * width, (cy - h / 2) * height,
                    (cx + w / 2) * width, (cy + h / 2) * height)
            team = int(class_id)
            predictions.append(Detection(team, TROOP_CLASSES[team], score, xyxy))
    return predictions


def review_frame(labels: list[Detection], predictions: list[Detection]) -> tuple[list[dict], Counter]:
    pairs = sorted((box_iou(label.xyxy, prediction.xyxy), i, j)
                   for i, label in enumerate(labels) for j, prediction in enumerate(predictions))
    unmatched_labels = set(range(len(labels)))
    unmatched_predictions = set(range(len(predictions)))
    issues = []
    true_positives = Counter()

    for same_team in (True, False):
        for iou, i, j in reversed(pairs):
            if iou < 0.5:
                break
            if i not in unmatched_labels or j not in unmatched_predictions:
                continue
            label, prediction = labels[i], predictions[j]
            if (label.class_id == prediction.class_id) != same_team:
                continue
            unmatched_labels.remove(i)
            unmatched_predictions.remove(j)
            if not same_team:
                issues.append({"kind": "wrong_team", "message": (
                    f"{label.class_name} labelled at {label.xyxy}, predicted as {prediction.class_name} "
                    f"(IoU {iou:.2f}, confidence {prediction.confidence:.2f})."
                )})
            else:
                true_positives[label.class_id] += 1
                if iou < 0.75:
                    issues.append({"kind": "loose_box", "message": (
                        f"{label.class_name} box at {label.xyxy} has IoU {iou:.2f} with the prediction."
                    )})

    for i in sorted(unmatched_labels):
        label = labels[i]
        issues.append({"kind": "missed", "message": f"No match for labelled {label.class_name} at {label.xyxy}."})
    for j in sorted(unmatched_predictions):
        prediction = predictions[j]
        xyxy = tuple(map(round, prediction.xyxy))
        issues.append({"kind": "extra", "message": (
            f"Unmatched {prediction.class_name} prediction at {xyxy} (confidence {prediction.confidence:.2f})."
        )})
    return issues, true_positives


def review_holdout(dataset: Dataset, export: Path, predictions: Path, confidence: float) -> dict:
    match_images = Counter(sample.match for sample in dataset.samples)
    match_issues = {match: Counter() for match in match_images}
    label_counts, prediction_counts, true_positives = Counter(), Counter(), Counter()
    flagged_frames = []
    for index, sample in enumerate(dataset.samples):
        info, _ = read_metadata(dataset.root / "metadata" / sample.key.with_suffix(".json"), sample.match)
        width, height = info["width"], info["height"]
        stem = f"{index:08d}"
        boxes, errors = read_boxes((export / "labels" / f"{stem}.txt").read_bytes(), len(TROOP_CLASSES), width, height)
        if errors:
            raise ValueError(f"{sample.key}: {errors[0]}")
        labels = [Detection(box.class_id, TROOP_CLASSES[box.class_id], 1.0,
                            (box.left, box.top, box.right, box.bottom)) for box in boxes]
        detections = read_predictions(predictions / f"{stem}.txt", width, height, confidence)
        issues, matches = review_frame(labels, detections)
        label_counts.update(label.class_id for label in labels)
        prediction_counts.update(detection.class_id for detection in detections)
        true_positives.update(matches)
        if issues:
            flagged_frames.append({**info, "video_sha256": info["video_sha256"].lower(), "issues": issues})
            match_issues[sample.match].update(issue["kind"] for issue in issues)

    return {
        "holdout": {
            "dataset_sha256": dataset.fingerprint,
            "matches": {match: {"images": images, "issues": dict(match_issues[match])}
                        for match, images in match_images.items()},
            "review_confidence": confidence,
            "per_class_at_review_confidence": {
                name: {
                    "labels": label_counts[i],
                    "predictions": prediction_counts[i],
                    "true_positives": true_positives[i],
                    "precision": true_positives[i] / prediction_counts[i] if prediction_counts[i] else None,
                    "recall": true_positives[i] / label_counts[i] if label_counts[i] else None,
                }
                for i, name in enumerate(TROOP_CLASSES)
            },
        },
        "flagged_frames": flagged_frames,
    }
