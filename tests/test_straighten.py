import unittest

import cv2
import numpy as np

from app.straighten import LineSegment, auto_straighten_verticals, detect_vertical_segments, estimate_roll
from tools.compare_ground_truth import normalized_match_key


def architectural_grid(width: int = 1200, height: int = 900) -> np.ndarray:
    image = np.full((height, width, 3), 235, dtype=np.uint8)
    for x in range(120, width, 150):
        cv2.line(image, (x, 60), (x, height - 60), (25, 25, 25), 8)
    for y in range(100, height, 140):
        cv2.line(image, (60, y), (width - 60, y), (80, 80, 80), 5)
    return image


class StraightenTests(unittest.TestCase):
    def test_manual_export_filename_matches_original(self):
        self.assertEqual(
            normalized_match_key("Manual-correct-N 1163 Michigan Ave 001-.jpg"),
            normalized_match_key("N 1163 Michigan Ave 001.jpg"),
        )

    def test_manual_export_filename_normalization_is_case_insensitive(self):
        self.assertEqual(
            normalized_match_key("MANUALLY-CORRECTED-P1269747-.JPG"),
            normalized_match_key("P1269747.jpg"),
        )

    def test_detection_resize_does_not_reduce_delivery_resolution(self):
        image = architectural_grid(1800, 1200)
        matrix = cv2.getRotationMatrix2D((900, 600), 3.0, 1.0)
        tilted = cv2.warpAffine(image, matrix, (1800, 1200), borderMode=cv2.BORDER_REPLICATE)

        result = auto_straighten_verticals(tilted, max_dimension=500, mode="level")

        self.assertGreater(result.corrected_bgr.shape[1], 1200)
        self.assertGreater(abs(result.correction_angle_deg), 1.5)
        self.assertEqual(result.applied_mode, "level")

        before_segments, _, _ = detect_vertical_segments(tilted, 50, 150, 80, 0.12, 20, 15.0)
        after_segments, _, _ = detect_vertical_segments(result.corrected_bgr, 50, 150, 80, 0.12, 20, 15.0)
        before_roll, _, _ = estimate_roll(before_segments, 8.0)
        after_roll, _, _ = estimate_roll(after_segments, 8.0)
        self.assertLess(abs(after_roll), abs(before_roll))

    def test_converging_verticals_trigger_perspective_correction(self):
        width, height = 1200, 900
        image = np.full((height, width, 3), 240, dtype=np.uint8)
        center = width / 2
        for x in range(120, width, 120):
            top_x = round(center + (x - center) * 0.72)
            bottom_x = round(center + (x - center) * 1.08)
            cv2.line(image, (top_x, 40), (bottom_x, height - 40), (20, 20, 20), 7)

        result = auto_straighten_verticals(
            image,
            max_dimension=900,
            mode="perspective",
            minimum_confidence=0.25,
            max_perspective_ratio=0.5,
            max_crop_fraction=0.5,
        )

        self.assertTrue(result.perspective_applied, result.debug)
        self.assertEqual(result.applied_mode, "perspective")
        self.assertGreater(result.corrected_bgr.size, 0)

        before_segments, _, _ = detect_vertical_segments(image, 50, 150, 80, 0.12, 20, 25.0)
        after_segments, _, _ = detect_vertical_segments(result.corrected_bgr, 50, 150, 80, 0.12, 20, 25.0)
        before_mean = np.mean([abs(segment.offset_from_vertical_deg) for segment in before_segments])
        after_mean = np.mean([abs(segment.offset_from_vertical_deg) for segment in after_segments])
        self.assertLess(after_mean, before_mean * 0.55)

    def test_blank_or_dark_edges_are_not_mistaken_for_rotation_padding(self):
        image = np.zeros((700, 1000, 3), dtype=np.uint8)
        cv2.rectangle(image, (250, 180), (750, 520), (230, 230, 230), -1)

        result = auto_straighten_verticals(image, mode="auto")

        self.assertEqual(result.applied_mode, "none")
        self.assertEqual(result.corrected_bgr.shape, image.shape)
        self.assertTrue(np.array_equal(result.corrected_bgr, image))

    def test_one_ambiguous_line_does_not_rotate_the_photo(self):
        image = np.full((800, 800, 3), 230, dtype=np.uint8)
        cv2.line(image, (360, 60), (430, 740), (20, 20, 20), 6)

        result = auto_straighten_verticals(image, mode="auto")

        self.assertEqual(result.applied_mode, "none", result.debug)
        self.assertEqual(result.correction_angle_deg, 0.0)
        self.assertTrue(any("Leveling skipped" in warning for warning in result.debug["warnings"]))

    def test_repeated_lines_on_one_object_do_not_outvote_room_geometry(self):
        segments = []
        for x in (80, 220, 760, 920):
            segments.append(LineSegment((x, 80), (x, 720), 640.0, 0.0))
        for index in range(30):
            x = 380 + index * 4
            segments.append(LineSegment((x, 100), (x + 86, 800), 705.0, 7.0))

        correction, confidence, debug = estimate_roll(segments, 8.0, image_width=1000)

        self.assertAlmostEqual(correction, 0.0, delta=0.2)
        self.assertGreater(confidence, 0.45)
        self.assertGreaterEqual(debug["occupied_horizontal_buckets"], 4)


if __name__ == "__main__":
    unittest.main()
