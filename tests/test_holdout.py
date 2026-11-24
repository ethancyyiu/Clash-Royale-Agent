import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from clash_royale.dataset.annotate import load_review_flags
from clash_royale.dataset.prepare_dataset import inspect_dataset
from clash_royale.dataset.replay_dataset import TROOP_CLASSES, Box, Dataset
from clash_royale.detection.detector import Detection
from clash_royale.detection.evaluate import compare_holdout
from clash_royale.detection.holdout import export_holdout, review_frame, review_holdout


def detection(team, xyxy):
    return Detection(team, TROOP_CLASSES[team], 0.5, xyxy)


class HoldoutTests(unittest.TestCase):
    def test_review_matches_boxes_once_and_checks_team(self):
        labels = [detection(0, (0, 0, 20, 20)), detection(1, (25, 0, 45, 20)),
                  detection(1, (50, 0, 70, 20))]
        predictions = [detection(0, (4, 0, 24, 20)), detection(0, (25, 0, 45, 20)),
                       detection(0, (5, 0, 25, 20))]

        issues, true_positives = review_frame(labels, predictions)

        self.assertCountEqual([issue["kind"] for issue in issues], ["loose_box", "wrong_team", "missed", "extra"])
        self.assertEqual(true_positives, {0: 1})

    def test_export_remaps_labels_and_review_keeps_original_frame_indices(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            classes = root / "classes.txt"
            classes.write_text("Enemy troop\nEnemy tower\nFriendly troop\n", encoding="utf-8")
            video = root / "first.mp4"
            video.write_bytes(b"first match")
            first = Dataset(root / "holdout", video, classes, "first")
            image = np.zeros((100, 100, 3), dtype=np.uint8)
            first.save(81, image, 0, [Box(0, 10, 10, 30, 30), Box(1, 70, 70, 95, 95)])
            video = root / "second.mp4"
            video.write_bytes(b"second match")
            second = Dataset(root / "holdout", video, classes, "second")
            second.save(25, image + 1, 0, [])

            dataset = inspect_dataset(root / "holdout")
            export = root / "export"
            data = export_holdout(dataset, export)
            self.assertIsNone(json.loads(data.read_text())["train"])
            labels = (export / "labels" / "00000000.txt").read_text().splitlines()
            self.assertEqual([line.split()[0] for line in labels], ["1"])
            predictions = root / "predictions"
            predictions.mkdir()
            (predictions / "00000001.txt").write_text("1 .2 .2 .2 .2 .7\n0 .5 .5 .2 .2 .1\n")

            report = review_holdout(dataset, export, predictions, confidence=0.25)
            self.assertEqual(report["holdout"]["matches"]["second"]["issues"], {"extra": 1})
            scores = report["holdout"]["per_class_at_review_confidence"]
            self.assertEqual(scores["enemy_troop"], {
                "labels": 1, "predictions": 1, "true_positives": 0, "precision": 0.0, "recall": 0.0,
            })
            path = root / "report.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            self.assertEqual(list(load_review_flags(path, first.source_id, 100)), [81])
            self.assertEqual(list(load_review_flags(path, second.source_id, 100)), [25])

    def test_comparison_rejects_changed_annotations(self):
        current = {"holdout": {"dataset_sha256": "new"}}
        baseline = {"holdout": {"dataset_sha256": "old"}}
        with self.assertRaisesRegex(ValueError, "annotations differ"):
            compare_holdout(current, baseline)


if __name__ == "__main__":
    unittest.main()
