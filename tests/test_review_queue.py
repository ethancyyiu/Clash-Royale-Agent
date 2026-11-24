import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from clash_royale.dataset.annotate import Annotator, load_queue
from clash_royale.dataset.review_queue import Candidate, select_frames
from clash_royale.dataset.replay_dataset import Box


class ReviewQueueTests(unittest.TestCase):
    def test_selection_skips_nearby_labels_and_spreads_across_clip(self):
        candidates = [Candidate(index, np.full(16, index / 100, dtype=np.float32))
                      for index in range(0, 100, 10)]
        selected = select_frames(candidates, {0, 50}, 4, 11, 100, 10)
        self.assertEqual(len(selected), 4)
        self.assertEqual(selected, sorted(set(selected)))
        self.assertTrue(all(min(abs(index), abs(index - 50)) >= 11 for index in selected))
        self.assertGreater(max(selected) - min(selected), 40)

    def test_queue_rejects_wrong_video_and_bad_frames(self):
        path = Path("queue.json")
        data = {"version": 1, "video_sha256": "source", "frames": [2, 8]}
        with patch.object(Path, "read_text", return_value=json.dumps(data)):
            self.assertEqual(load_queue(path, "source", 10), [2, 8])
            with self.assertRaises(ValueError):
                load_queue(path, "other", 10)
        data["frames"] = [2, 11]
        with patch.object(Path, "read_text", return_value=json.dumps(data)):
            with self.assertRaises(ValueError):
                load_queue(path, "source", 10)
        with patch.object(Path, "read_text", return_value="[]"):
            with self.assertRaises(ValueError):
                load_queue(path, "source", 10)

    def test_saved_boxes_load_as_box_list(self):
        class DatasetStub:
            classes = ["Troop"]
            saved_frames = {4}

            def load_boxes(self, index, width, height):
                return [Box(0, 1, 2, 3, 4)], []

        annotator = Annotator(None, DatasetStub(), 10, 30)
        annotator.index = 4
        annotator.width = 20
        annotator.height = 20
        annotator.reload_boxes()
        self.assertEqual(annotator.boxes, [Box(0, 1, 2, 3, 4)])


if __name__ == "__main__":
    unittest.main()
