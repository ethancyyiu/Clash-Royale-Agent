import argparse
import json
import sys
from importlib.metadata import version
from pathlib import Path

from detector import load_detection_model
from prepare_dataset import choose_validation_matches, inspect_dataset, write_split_bundle
from replay_dataset import write_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune a pretrained YOLO detector on prepared match splits.")
    parser.add_argument("--dataset", type=Path, default=Path("dataset"), help="prepared annotation dataset root")
    parser.add_argument("--model", default="yolo11n.pt", help="pretrained detection model name or local .pt path")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640, help="training image size in pixels")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu", help="cpu or GPU index such as 0")
    parser.add_argument("--output", type=Path, default=Path("runs/train"), help="new run directory")
    args = parser.parse_args()
    if min(args.epochs, args.imgsz, args.batch) <= 0:
        parser.error("--epochs, --imgsz, and --batch must be positive")
    if Path(args.model).suffix.lower() != ".pt":
        parser.error("--model must be a pretrained .pt checkpoint")

    try:
        manifest = args.dataset / "splits" / "split.json"
        if not manifest.is_file():
            raise ValueError("Run prepare_dataset.py for this dataset first.")
        prepared = json.loads(manifest.read_text(encoding="utf-8"))
        dataset = inspect_dataset(args.dataset)
        if dataset.fingerprint != prepared["dataset_sha256"]:
            raise ValueError("The dataset has changed. Run prepare_dataset.py again before training.")
        validation = choose_validation_matches(dataset, explicit=prepared["val"]["matches"])
        for split, select_val in (("train", False), ("val", True)):
            if not any(
                sample.class_counts for sample in dataset.samples if (sample.match in validation) == select_val
            ):
                raise ValueError(f"The {split} split must contain at least one annotated object.")

        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=False)
        data = write_split_bundle(dataset, output / "dataset", validation)
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
