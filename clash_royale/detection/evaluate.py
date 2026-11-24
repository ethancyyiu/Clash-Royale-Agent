import argparse
import json
import sys
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2

from ..dataset.prepare_dataset import inspect_dataset
from ..dataset.replay_dataset import write_file
from .detector import load_detection_model
from .holdout import export_holdout, review_holdout

METRIC_NAMES = ("precision", "recall", "mAP50", "mAP50-95")
HOLDOUT_MIN_CONFIDENCE = 0.001


def summarize(metrics, names: dict[int, str]) -> dict:
    box = metrics.box
    by_class = {}
    for row, class_id in enumerate(box.ap_class_index):
        scores = dict(zip(METRIC_NAMES, map(float, box.class_result(row))))
        scores["mAP75"] = float(box.all_ap[row, 5])
        by_class[int(class_id)] = scores
    per_class = []
    for class_id, name in sorted(names.items()):
        instances = int(metrics.nt_per_class[class_id]) if class_id < len(metrics.nt_per_class) else 0
        scores = by_class.get(class_id, dict.fromkeys((*METRIC_NAMES, "mAP75"), 0.0 if instances else None))
        per_class.append({"class_id": class_id, "name": name, "instances": instances, **scores})
    overall = dict(zip(METRIC_NAMES, map(float, (box.mp, box.mr, box.map50, box.map))))
    overall["mAP75"] = float(box.map75)
    timing = {stage: float(ms) for stage, ms in metrics.speed.items()}
    return {
        "overall": overall,
        "per_class": per_class,
        "latency_ms_per_image": {
            **timing,
            "preprocess_inference_postprocess": sum(
                timing[stage] for stage in ("preprocess", "inference", "postprocess")
            ),
        },
    }


def validate(model, data: Path, settings: dict) -> dict:
    from yaml import safe_load

    classes = safe_load(data.read_text(encoding="utf-8"))["names"]
    names = dict(enumerate(classes)) if isinstance(classes, list) else {int(k): v for k, v in classes.items()}
    if model.names != names:
        raise ValueError("Checkpoint class names and IDs differ from the dataset YAML.")
    return summarize(model.val(data=str(data.resolve()), **settings), names)


def compare_holdout(report: dict, baseline: dict) -> dict:
    if not isinstance(baseline, dict) or not isinstance(baseline.get("holdout"), dict):
        raise ValueError("Use an evaluation report created with --holdout as the baseline.")
    holdout, previous_holdout = report["holdout"], baseline["holdout"]
    if holdout["dataset_sha256"] != previous_holdout.get("dataset_sha256"):
        raise ValueError("The held-out annotations differ from the baseline. Reevaluate both checkpoints on the same dataset.")
    if report["ultralytics"] != baseline.get("ultralytics") or report["validation"] != baseline.get("validation"):
        raise ValueError("Use the same Ultralytics version and validation settings as the baseline.")
    if holdout["review_confidence"] != previous_holdout.get("review_confidence"):
        raise ValueError("Use the same --review-conf as the baseline.")

    def deltas(current: dict, previous: dict, metrics: tuple[str, ...]) -> dict:
        return {
            metric: None if None in (current[metric], previous[metric]) else round(current[metric] - previous[metric], 6)
            for metric in metrics
        }

    metrics = (*METRIC_NAMES, "mAP75")
    previous_classes = {row["name"]: row for row in baseline["per_class"]}
    return {
        "baseline_weights": baseline["weights"],
        "overall": deltas(report["overall"], baseline["overall"], metrics),
        "per_class": {row["name"]: deltas(row, previous_classes[row["name"]], metrics) for row in report["per_class"]},
        "per_class_at_review_confidence": {
            name: deltas(scores, previous_holdout["per_class_at_review_confidence"][name], ("precision", "recall"))
            for name, scores in holdout["per_class_at_review_confidence"].items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a YOLO checkpoint on labelled matches.")
    parser.add_argument("--weights", type=Path, required=True, help="trained .pt checkpoint")
    data_source = parser.add_mutually_exclusive_group(required=True)
    data_source.add_argument("--data", type=Path, help="dataset YAML saved with the run")
    data_source.add_argument("--holdout", type=Path, help="separate annotation dataset for the two troop classes")
    parser.add_argument("--baseline", type=Path, help="compare with a prior --holdout evaluation JSON")
    parser.add_argument("--review-conf", type=float, default=0.25,
                        help="confidence cutoff for held-out review flags (default: 0.25)")
    parser.add_argument("--device", default="cpu", help="cpu or GPU index such as 0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("runs/evaluation.json"))
    args = parser.parse_args()
    if min(args.imgsz, args.batch) <= 0:
        parser.error("--imgsz and --batch must be positive")
    if args.baseline and args.holdout is None:
        parser.error("--baseline requires --holdout")
    if not HOLDOUT_MIN_CONFIDENCE <= args.review_conf <= 1:
        parser.error(f"--review-conf must be in [{HOLDOUT_MIN_CONFIDENCE}, 1]")

    try:
        for path in (args.weights, args.data, args.baseline):
            if path is not None and not path.is_file():
                raise FileNotFoundError(f"File does not exist: {path}")
        if args.baseline and args.baseline.resolve() == args.output.resolve():
            raise ValueError("Choose a separate output path to preserve the baseline.")
        model = load_detection_model(args.weights)
        settings = {
            "device": args.device, "imgsz": args.imgsz, "batch": args.batch, "split": "val", "plots": False,
        }
        report = {"weights": str(args.weights.resolve()), "ultralytics": version("ultralytics")}
        if args.holdout is None:
            settings.update(project=str(args.output.resolve().parent), name="validation", exist_ok=True)
            report.update(validation={"data": str(args.data.resolve()), **settings})
            report.update(validate(model, args.data, settings))
        else:
            dataset = inspect_dataset(args.holdout)
            settings.update(conf=HOLDOUT_MIN_CONFIDENCE, workers=0)
            report.update(validation=settings)
            with TemporaryDirectory(prefix="clash-holdout-") as directory:
                root = Path(directory).resolve()
                data = export_holdout(dataset, root / "dataset")
                report.update(validate(model, data, {
                    **settings, "project": str(root), "name": "predictions", "exist_ok": True,
                    "save_txt": True, "save_conf": True,
                }))
                report.update(review_holdout(dataset, data.parent, root / "predictions" / "labels", args.review_conf))
            if args.baseline:
                baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
                report["change_from_baseline"] = compare_holdout(report, baseline)
        output = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf-8")
        write_file(args.output, output)
        if args.holdout is None:
            print(output.decode("utf-8"))
        else:
            summary = {key: report[key] for key in ("overall", "per_class", "change_from_baseline") if key in report}
            summary["holdout"] = report["holdout"]
            print(json.dumps(summary, indent=2))
            print(f"Flagged {len(report['flagged_frames'])} frame(s) for review.")
        print(f"Evaluation saved to {args.output}")
    except (OSError, RuntimeError, ValueError, KeyError, cv2.error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
