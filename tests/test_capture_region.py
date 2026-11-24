import unittest
from unittest.mock import patch

import numpy as np

from clash_royale.capture.screen import draw_region


class FakeScreen:
    monitors = [{}, {"left": 100, "top": 50, "width": 1920, "height": 1080}]

    def grab(self, bounds):
        assert bounds == {"left": 100, "top": 50, "width": 1920, "height": 1080}
        return np.zeros((1080, 1920, 4), dtype=np.uint8)


class CaptureRegionTests(unittest.TestCase):
    @patch("clash_royale.capture.screen.cv2.destroyWindow")
    @patch("clash_royale.capture.screen.cv2.selectROI", return_value=(110, 62, 550, 310))
    @patch("clash_royale.capture.screen.cv2.namedWindow")
    def test_draw_region_returns_monitor_relative_coordinates(self, _window, _roi, _destroy):
        self.assertEqual(draw_region(FakeScreen(), 1), [192, 108, 960, 541])

    @patch("clash_royale.capture.screen.cv2.destroyWindow")
    @patch("clash_royale.capture.screen.cv2.selectROI", return_value=(0, 0, 0, 0))
    @patch("clash_royale.capture.screen.cv2.namedWindow")
    def test_cancelled_selection_raises(self, _window, _roi, _destroy):
        with self.assertRaisesRegex(ValueError, "cancelled"):
            draw_region(FakeScreen(), 1)


if __name__ == "__main__":
    unittest.main()
