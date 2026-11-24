import argparse
import json
import sys
from importlib.metadata import version
from pathlib import Path

from .detector import load_detection_model
from ..dataset.prepare_dataset import choose_validation_matches, inspect_dataset, prepared_split_dir, write_split_bundle
from ..dataset.replay_dataset import training_class_ids, write_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune a pretrained YOLO detector on prepared match splits.")
    parser.add_argument("--dataset", type=Path, default=Path("dataset"), help="prepared annotation dataset root")
    parser.add_argument("--troop-only", action="store_true", help="use the prepared two-class troop split")
    parser.add_argument("--model", default="yolo11n.pt", help="pretrained detection model name or local .pt path")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640, help="training image size in pixels")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0 if sys.platform == "win32" else 8,
                        help="data loader processes (default: 0 on Windows, 8 elsewhere)")
    parser.add_argument("--device", default="cpu", help="cpu or GPU index such as 0")
    parser.add_argument("--output", type=Path, default=Path("runs/train"), help="new run directory")
    args = parser.parse_args()
    if min(args.epochs, args.imgsz, args.batch) <= 0 or args.workers < 0:
        parser.error("--epochs, --imgsz, and --batch must be positive; --workers must be nonnegative")
    if Path(args.model).suffix.lower() != ".pt":
        parser.error("--model must be a pretrained .pt checkpoint")

    try:
        manifest = prepared_split_dir(args.dataset, args.troop_only) / "split.json"
        if not manifest.is_file():
            prepare = "python -m clash_royale prepare-dataset"
            if args.troop_only:
                prepare += " --troop-only"
            raise ValueError(f"Run {prepare} for this dataset first.")
        prepared = json.loads(manifest.read_text(encoding="utf-8"))
        dataset = inspect_dataset(args.dataset)
        if dataset.fingerprint != prepared["dataset_sha256"]:
            raise ValueError("The dataset has changed. Run prepare-dataset again before training.")
        validation = choose_validation_matches(dataset, explicit=prepared["val"]["matches"])
        class_ids = training_class_ids(dataset.classes, args.troop_only)
        for split, select_val in (("train", False), ("val", True)):
            if not any(
                sample.class_counts[class_id]
                for sample in dataset.samples if (sample.match in validation) == select_val
                for class_id in class_ids
            ):
                raise ValueError(f"The {split} split must contain at least one annotated object in the selected classes.")

        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=False)
        data = write_split_bundle(dataset, output / "dataset", validation, troop_only=args.troop_only)
        model = load_detection_model(args.model)
        settings = {
            "data": str(data),
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": args.device,
            "project": str(output.parent),
            "name": output.name,
            "exist_ok": True,
            "pretrained": True,
            "save": True,
            "workers": args.workers,
        }
        config = {"model": args.model, "ultralytics": version("ultralytics"), "training": settings}
        write_file(output / "run_config.json", (json.dumps(config, indent=2) + "\n").encode("utf-8"))
        model.train(**settings)
        print(f"Training results and checkpoints saved in {output}")
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
