from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


def add_detector_options(parser, *, required: bool = False) -> None:
    parser.add_argument("--weights", type=Path, required=required, help="trained YOLO checkpoint")
    parser.add_argument("--device", default="cpu", help="YOLO device, e.g. cpu or 0")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO input image size")
    parser.add_argument("--conf", type=float, default=0.25, help="minimum YOLO confidence")


def load_detection_model(weights: str | Path):
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("Install the training dependencies with pip install -r requirements-training.txt") from error
    model = YOLO(str(weights))
    if model.task != "detect":
        raise ValueError("Use an object detection checkpoint.")
    return model


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]


class Detector:
    def __init__(self, weights: Path, *, device: str = "cpu", imgsz: int = 640, conf: float = 0.25):
        if imgsz <= 0 or not 0 <= conf <= 1:
            raise ValueError("--imgsz must be positive and --conf must be in [0, 1]")
        if not weights.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {weights}")
        self.model = load_detection_model(weights)
        self.device = device
        self.imgsz = imgsz
        self.conf = conf

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Return detections in original pixel coordinates for one BGR image."""
        result = self.model.predict(
            source=frame, device=self.device, imgsz=self.imgsz, conf=self.conf, verbose=False,
        )[0]
        boxes = result.boxes.cpu().numpy()
        return [
            Detection(int(class_id), result.names[int(class_id)], float(confidence), tuple(map(float, coords)))
            for coords, confidence, class_id in zip(boxes.xyxy, boxes.conf, boxes.cls)
        ]


def draw_detections(frame: np.ndarray, detections: Sequence[Detection]) -> np.ndarray:
    annotated = frame.copy()
    for detection in detections:
        left, top, right, bottom = map(round, detection.xyxy)
        color = (40, 220, 80)
        cv2.rectangle(annotated, (left, top), (right, bottom), color, 2)
        label = f"{detection.class_name} {detection.confidence:.2f}"
        cv2.putText(annotated, label, (left, max(15, top - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    return annotated
