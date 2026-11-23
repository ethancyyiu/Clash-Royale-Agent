import argparse
import json
import sys
from importlib.metadata import version
from pathlib import Path

from .detector import load_detection_model
from ..dataset.replay_dataset import write_file

METRIC_NAMES = ("precision", "recall", "mAP50", "mAP50-95")


def summarize(metrics, names: dict[int, str]) -> dict:
    box = metrics.box
    by_class = {
        int(class_id): dict(zip(METRIC_NAMES, map(float, box.class_result(row))))
        for row, class_id in enumerate(box.ap_class_index)
    }
    per_class = [
        {
            "class_id": class_id,
            "name": name,
            "instances": int(metrics.nt_per_class[class_id]) if class_id < len(metrics.nt_per_class) else 0,
            **by_class.get(class_id, dict.fromkeys(METRIC_NAMES)),
        }
        for class_id, name in sorted(names.items())
    ]
    timing = {stage: float(ms) for stage, ms in metrics.speed.items()}
    return {
        "overall": dict(zip(METRIC_NAMES, map(float, (box.mp, box.mr, box.map50, box.map)))),
        "per_class": per_class,
        "latency_ms_per_image": {
            **timing,
            "preprocess_inference_postprocess": sum(
                timing[stage] for stage in ("preprocess", "inference", "postprocess")
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a YOLO checkpoint on the validation matches.")
    parser.add_argument("--weights", type=Path, required=True, help="trained .pt checkpoint")
    parser.add_argument("--data", type=Path, required=True, help="dataset YAML saved with the run")
    parser.add_argument("--device", default="cpu", help="cpu or GPU index such as 0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("runs/evaluation.json"))
    args = parser.parse_args()
    if min(args.imgsz, args.batch) <= 0:
        parser.error("--imgsz and --batch must be positive")

    try:
        for path in (args.weights, args.data):
            if not path.is_file():
                raise FileNotFoundError(f"File does not exist: {path}")
        model = load_detection_model(args.weights)

        from yaml import safe_load

        classes = safe_load(args.data.read_text(encoding="utf-8"))["names"]
        names = dict(enumerate(classes)) if isinstance(classes, list) else {int(k): v for k, v in classes.items()}
        if model.names != names:
            raise ValueError("Checkpoint class names and IDs differ from the dataset YAML.")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        settings = {
            "data": str(args.data.resolve()), "device": args.device,
            "imgsz": args.imgsz, "batch": args.batch, "split": "val", "plots": False,
            "project": str(args.output.resolve().parent), "name": "validation", "exist_ok": True,
        }
        metrics = model.val(**settings)
        report = {
            "weights": str(args.weights.resolve()), "ultralytics": version("ultralytics"),
            "validation": settings, **summarize(metrics, names),
        }
        output = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf-8")
        write_file(args.output, output)
        print(output.decode("utf-8"), end="")
        print(f"Evaluation saved to {args.output}")
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
