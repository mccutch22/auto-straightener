import math
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np


CorrectionMode = Literal["auto", "level", "perspective"]
CropMode = Literal["crop", "keep_all"]


@dataclass
class StraightenResult:
    corrected_bgr: np.ndarray
    correction_angle_deg: float
    perspective_applied: bool
    confidence: float
    crop_fraction: float
    applied_mode: str
    debug: dict
    debug_bgr: np.ndarray | None = None


@dataclass
class LineSegment:
    p1: tuple[float, float]
    p2: tuple[float, float]
    length: float
    offset_from_vertical_deg: float


def resize_for_detection(image_bgr: np.ndarray, max_dimension: int) -> tuple[np.ndarray, float]:
    """Return an analysis copy and its scale; never resize the delivered image."""
    h, w = image_bgr.shape[:2]
    largest = max(h, w)
    if largest <= max_dimension:
        return image_bgr.copy(), 1.0

    scale = max_dimension / largest
    resized = cv2.resize(
        image_bgr,
        (max(1, round(w * scale)), max(1, round(h * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def weighted_median(values: list[float], weights: list[float]) -> float:
    if not values:
        raise ValueError("No values for weighted median")
    pairs = sorted(zip(values, weights), key=lambda pair: pair[0])
    midpoint = sum(weights) / 2.0
    cumulative = 0.0
    for value, weight in pairs:
        cumulative += weight
        if cumulative >= midpoint:
            return value
    return pairs[-1][0]


def _angle_from_vertical(dx: float, dy: float) -> float:
    theta = (math.degrees(math.atan2(dy, dx)) + 180.0) % 180.0
    return theta - 90.0


def detect_vertical_segments(
    image_bgr: np.ndarray,
    canny_threshold1: int,
    canny_threshold2: int,
    hough_threshold: int,
    min_line_length_ratio: float,
    max_line_gap: int,
    vertical_tolerance_deg: float,
) -> tuple[list[LineSegment], int, np.ndarray]:
    """Detect plausible architectural verticals on an analysis-sized image."""
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # CLAHE is used only for geometry detection. It does not change the photo.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, canny_threshold1, canny_threshold2)

    min_line_length = max(12, round(min(h, w) * min_line_length_ratio))
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 360.0,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )

    debug_img = image_bgr.copy()
    segments: list[LineSegment] = []
    total_lines = 0
    if lines is None:
        return segments, total_lines, debug_img

    for raw_line in lines[:, 0]:
        x1, y1, x2, y2 = (float(value) for value in raw_line)
        total_lines += 1
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        offset = _angle_from_vertical(dx, dy)
        accepted = length >= min_line_length and abs(offset) <= vertical_tolerance_deg
        color = (0, 210, 0) if accepted else (0, 0, 180)
        cv2.line(debug_img, (round(x1), round(y1)), (round(x2), round(y2)), color, 2 if accepted else 1)
        if accepted:
            segments.append(LineSegment((x1, y1), (x2, y2), length, offset))

    # Pairwise vanishing-point work grows quadratically. Keep strong lines, but
    # distribute them across the frame so one repetitive object (cabinet doors,
    # shelving, stacked panels) cannot overwhelm the room geometry.
    bucket_count = 8
    bucket_limit = 12
    buckets: list[list[LineSegment]] = [[] for _ in range(bucket_count)]
    for segment in segments:
        midpoint_x = (segment.p1[0] + segment.p2[0]) / 2.0
        bucket_index = min(bucket_count - 1, max(0, int(midpoint_x / max(w, 1) * bucket_count)))
        buckets[bucket_index].append(segment)

    balanced_segments: list[LineSegment] = []
    for bucket in buckets:
        bucket.sort(key=lambda segment: segment.length, reverse=True)
        balanced_segments.extend(bucket[:bucket_limit])
    balanced_segments.sort(key=lambda segment: segment.length, reverse=True)
    return balanced_segments, total_lines, debug_img


def estimate_roll(
    segments: list[LineSegment], max_correction_deg: float, image_width: float | None = None
) -> tuple[float, float, dict]:
    if not segments:
        return 0.0, 0.0, {"reason": "no_vertical_candidates"}

    if image_width is None:
        image_width = max(max(segment.p1[0], segment.p2[0]) for segment in segments)
    image_width = max(float(image_width), 1.0)

    # First summarize evidence within vertical strips, then combine the strips
    # with capped support. This preserves the precision of repeated edges while
    # preventing one localized subject from receiving dozens of votes.
    bucket_count = 8
    buckets: list[list[LineSegment]] = [[] for _ in range(bucket_count)]
    for segment in segments:
        midpoint_x = (segment.p1[0] + segment.p2[0]) / 2.0
        bucket_index = min(bucket_count - 1, max(0, int(midpoint_x / image_width * bucket_count)))
        buckets[bucket_index].append(segment)

    bucket_offsets: list[float] = []
    bucket_support: list[float] = []
    occupied_midpoints: list[float] = []
    for bucket in buckets:
        if not bucket:
            continue
        weights = [segment.length for segment in bucket]
        bucket_offsets.append(
            weighted_median([segment.offset_from_vertical_deg for segment in bucket], weights)
        )
        bucket_support.append(sum(weights))
        occupied_midpoints.extend((segment.p1[0] + segment.p2[0]) / 2.0 for segment in bucket)

    support_cap = float(np.median(bucket_support))
    balanced_weights = [min(support, support_cap) for support in bucket_support]
    median = weighted_median(bucket_offsets, balanced_weights)
    # OpenCV's image-coordinate rotation convention corrects a detected line by
    # applying the measured offset itself (not its arithmetic opposite).
    correction = float(np.clip(median, -max_correction_deg, max_correction_deg))
    deviations = [abs(value - median) for value in bucket_offsets]
    mad = weighted_median(deviations, balanced_weights)

    line_support = min(1.0, len(segments) / 10.0)
    bucket_support_score = min(1.0, len(bucket_offsets) / 4.0)
    horizontal_span = (max(occupied_midpoints) - min(occupied_midpoints)) / image_width
    spatial_support = min(1.0, horizontal_span / 0.45)
    consistency = math.exp(-mad / 4.0)
    confidence = float(
        np.clip(line_support * consistency * math.sqrt(bucket_support_score * spatial_support), 0.0, 1.0)
    )
    return correction, confidence, {
        "median_vertical_offset_deg": round(median, 4),
        "angle_mad_deg": round(mad, 4),
        "occupied_horizontal_buckets": len(bucket_offsets),
        "horizontal_span_fraction": round(horizontal_span, 4),
    }


def _segment_errors(
    segments: list[LineSegment], image_width: float | None = None
) -> tuple[float | None, float | None]:
    """Return roll error and total vertical error for quality validation."""
    if not segments:
        return None, None
    if image_width is None:
        image_width = max(max(segment.p1[0], segment.p2[0]) for segment in segments)
    image_width = max(float(image_width), 1.0)
    roll_error = abs(estimate_roll(segments, 90.0, image_width)[0])

    buckets: list[list[LineSegment]] = [[] for _ in range(8)]
    for segment in segments:
        midpoint_x = (segment.p1[0] + segment.p2[0]) / 2.0
        bucket_index = min(7, max(0, int(midpoint_x / image_width * 8)))
        buckets[bucket_index].append(segment)
    bucket_errors: list[float] = []
    bucket_support: list[float] = []
    for bucket in buckets:
        if not bucket:
            continue
        support = sum(segment.length for segment in bucket)
        bucket_errors.append(
            sum(abs(segment.offset_from_vertical_deg) * segment.length for segment in bucket)
            / support
        )
        bucket_support.append(support)
    support_cap = float(np.median(bucket_support))
    balanced_weights = [min(support, support_cap) for support in bucket_support]
    vertical_error = sum(
        error * weight for error, weight in zip(bucket_errors, balanced_weights)
    ) / sum(balanced_weights)
    return roll_error, vertical_error


def _measure_output_errors(
    image_bgr: np.ndarray,
    max_dimension: int,
    canny_threshold1: int,
    canny_threshold2: int,
    hough_threshold: int,
    min_line_length_ratio: float,
    max_line_gap: int,
) -> tuple[float | None, float | None, int]:
    analysis, _ = resize_for_detection(image_bgr, max_dimension)
    segments, _, _ = detect_vertical_segments(
        analysis,
        canny_threshold1,
        canny_threshold2,
        hough_threshold,
        min_line_length_ratio,
        max_line_gap,
        vertical_tolerance_deg=25.0,
    )
    roll_error, vertical_error = _segment_errors(segments, analysis.shape[1])
    return roll_error, vertical_error, len(segments)


def _rotation_homography(width: int, height: int, angle_deg: float) -> np.ndarray:
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle_deg, 1.0)
    return np.vstack([matrix, [0.0, 0.0, 1.0]]).astype(np.float64)


def _line_equation(segment: LineSegment) -> np.ndarray:
    p1 = np.array([segment.p1[0], segment.p1[1], 1.0], dtype=np.float64)
    p2 = np.array([segment.p2[0], segment.p2[1], 1.0], dtype=np.float64)
    line = np.cross(p1, p2)
    norm = math.hypot(line[0], line[1])
    return line / max(norm, 1e-9)


def _angular_residual_deg(segment: LineSegment, point: np.ndarray) -> float:
    midpoint = np.array(
        [(segment.p1[0] + segment.p2[0]) / 2.0, (segment.p1[1] + segment.p2[1]) / 2.0],
        dtype=np.float64,
    )
    segment_direction = np.array(
        [segment.p2[0] - segment.p1[0], segment.p2[1] - segment.p1[1]], dtype=np.float64
    )
    point_direction = point[:2] - midpoint
    denominator = np.linalg.norm(segment_direction) * np.linalg.norm(point_direction)
    if denominator < 1e-9:
        return 90.0
    cross_product = segment_direction[0] * point_direction[1] - segment_direction[1] * point_direction[0]
    sine = abs(cross_product) / denominator
    return math.degrees(math.asin(float(np.clip(sine, 0.0, 1.0))))


def estimate_vertical_vanishing_point(
    segments: list[LineSegment], width: int, height: int, inlier_angle_deg: float = 2.5
) -> tuple[np.ndarray | None, float, dict]:
    """Robustly estimate the shared vanishing point of architectural verticals."""
    if len(segments) < 3:
        return None, 0.0, {"reason": "not_enough_vertical_lines"}

    equations = [_line_equation(segment) for segment in segments]
    total_weight = sum(segment.length for segment in segments)
    best_inliers: list[int] = []
    best_score = 0.0
    center = np.array([width / 2.0, height / 2.0])

    for first in range(len(segments)):
        for second in range(first + 1, len(segments)):
            intersection = np.cross(equations[first], equations[second])
            if abs(intersection[2]) < 1e-8:
                continue
            point = intersection[:2] / intersection[2]
            distance = float(np.linalg.norm(point - center))
            if not np.all(np.isfinite(point)) or distance < height * 0.75 or distance > height * 100.0:
                continue

            inliers = [
                index
                for index, segment in enumerate(segments)
                if _angular_residual_deg(segment, np.array([point[0], point[1], 1.0])) <= inlier_angle_deg
            ]
            score = sum(segments[index].length for index in inliers)
            if score > best_score:
                best_score = score
                best_inliers = inliers

    if len(best_inliers) < 3:
        return None, 0.0, {"reason": "no_consistent_vanishing_point"}

    weighted_lines = np.stack(
        [equations[index] * math.sqrt(segments[index].length) for index in best_inliers]
    )
    _, _, vh = np.linalg.svd(weighted_lines)
    homogeneous_point = vh[-1]
    if abs(homogeneous_point[2]) < 1e-8:
        return None, 0.0, {"reason": "verticals_already_parallel"}
    point = homogeneous_point[:2] / homogeneous_point[2]
    if not np.all(np.isfinite(point)):
        return None, 0.0, {"reason": "invalid_vanishing_point"}

    residuals = [_angular_residual_deg(segments[index], np.r_[point, 1.0]) for index in best_inliers]
    support = best_score / max(total_weight, 1e-9)
    count_score = min(1.0, len(best_inliers) / 8.0)
    residual_score = math.exp(-float(np.median(residuals)) / inlier_angle_deg)
    confidence = float(np.clip(support * count_score * residual_score, 0.0, 1.0))
    return point, confidence, {
        "inlier_lines": len(best_inliers),
        "support_fraction": round(support, 4),
        "median_residual_deg": round(float(np.median(residuals)), 4),
    }


def _vertical_rectification_homography(
    width: int, height: int, vanishing_point: np.ndarray, strength: float
) -> np.ndarray:
    cx, cy = width / 2.0, height / 2.0
    vertical_distance = float(vanishing_point[1] - cy)
    q = (-1.0 / vertical_distance) * strength
    centered = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, q, 1.0]], dtype=np.float64)
    to_origin = np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]])
    from_origin = np.array([[1.0, 0.0, cx], [0.0, 1.0, cy], [0.0, 0.0, 1.0]])
    return from_origin @ centered @ to_origin


def _largest_rectangle_in_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Find the largest all-valid axis-aligned rectangle in a binary mask."""
    height, width = mask.shape
    hist = np.zeros(width, dtype=np.int32)
    best_area = 0
    best = (0, 0, width, height)

    for y in range(height):
        hist = np.where(mask[y], hist + 1, 0)
        stack: list[int] = []
        for x in range(width + 1):
            current = int(hist[x]) if x < width else 0
            while stack and current < hist[stack[-1]]:
                index = stack.pop()
                rect_height = int(hist[index])
                left = stack[-1] + 1 if stack else 0
                rect_width = x - left
                area = rect_height * rect_width
                if area > best_area:
                    best_area = area
                    best = (left, y - rect_height + 1, x, y + 1)
            stack.append(x)
    return best


def _warp_and_crop(
    image_bgr: np.ndarray,
    homography: np.ndarray,
    crop_mode: CropMode,
) -> tuple[np.ndarray, float, dict]:
    height, width = image_bgr.shape[:2]
    corners = np.array([[[0, 0], [width, 0], [width, height], [0, height]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(corners, homography.astype(np.float64))[0]
    if not np.all(np.isfinite(transformed)):
        raise ValueError("Correction produced invalid image bounds")

    min_x, min_y = np.floor(transformed.min(axis=0)).astype(int)
    max_x, max_y = np.ceil(transformed.max(axis=0)).astype(int)
    output_width, output_height = int(max_x - min_x), int(max_y - min_y)
    if output_width <= 0 or output_height <= 0 or output_width > width * 4 or output_height > height * 4:
        raise ValueError("Correction would create unsafe image dimensions")

    translation = np.array([[1.0, 0.0, -min_x], [0.0, 1.0, -min_y], [0.0, 0.0, 1.0]])
    matrix = translation @ homography
    warped = cv2.warpPerspective(
        image_bgr,
        matrix,
        (output_width, output_height),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REPLICATE,
    )

    if crop_mode == "keep_all":
        return warped, 0.0, {
            "uncropped_dimensions": [output_width, output_height],
            "effective_homography": matrix.tolist(),
        }

    # Build a low-resolution validity mask, avoiding assumptions about whether
    # real pixels at the edge are black or dark.
    mask_scale = min(1.0, 1200.0 / max(output_width, output_height))
    mask_width = max(1, round(output_width * mask_scale))
    mask_height = max(1, round(output_height * mask_scale))
    scaled_matrix = np.diag([mask_scale, mask_scale, 1.0]) @ matrix
    source_mask = np.full((height, width), 255, dtype=np.uint8)
    mask = cv2.warpPerspective(
        source_mask,
        scaled_matrix,
        (mask_width, mask_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    binary_mask = mask >= 250
    x0, y0, x1, y1 = _largest_rectangle_in_mask(binary_mask)
    valid_area = int(binary_mask.sum())
    retained_area = max(0, x1 - x0) * max(0, y1 - y0)
    crop_fraction = float(np.clip(1.0 - retained_area / max(valid_area, 1), 0.0, 1.0))

    inverse_scale = 1.0 / mask_scale
    full_x0 = min(output_width - 1, max(0, math.ceil(x0 * inverse_scale) + 1))
    full_y0 = min(output_height - 1, max(0, math.ceil(y0 * inverse_scale) + 1))
    full_x1 = min(output_width, max(full_x0 + 1, math.floor(x1 * inverse_scale) - 1))
    full_y1 = min(output_height, max(full_y0 + 1, math.floor(y1 * inverse_scale) - 1))
    cropped = warped[full_y0:full_y1, full_x0:full_x1]
    crop_translation = np.array(
        [[1.0, 0.0, -full_x0], [0.0, 1.0, -full_y0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return cropped, crop_fraction, {
        "uncropped_dimensions": [output_width, output_height],
        "crop_box": [full_x0, full_y0, full_x1, full_y1],
        "effective_homography": (crop_translation @ matrix).tolist(),
    }


def auto_straighten_verticals(
    original_bgr: np.ndarray,
    max_dimension: int = 2400,
    canny_threshold1: int = 50,
    canny_threshold2: int = 150,
    hough_threshold: int = 80,
    min_line_length_ratio: float = 0.12,
    max_line_gap: int = 20,
    max_correction_deg: float = 5.0,
    mode: CorrectionMode = "auto",
    crop_mode: CropMode = "crop",
    perspective_strength: float = 0.5,
    minimum_confidence: float = 0.45,
    max_perspective_ratio: float = 0.28,
    max_crop_fraction: float = 0.18,
    return_debug: bool = False,
) -> StraightenResult:
    if original_bgr is None or original_bgr.size == 0:
        raise ValueError("Image is empty")

    working_bgr, detection_scale = resize_for_detection(original_bgr, max_dimension)
    detection_height, detection_width = working_bgr.shape[:2]
    original_height, original_width = original_bgr.shape[:2]
    warnings: list[str] = []

    initial_segments, total_lines, _ = detect_vertical_segments(
        working_bgr,
        canny_threshold1,
        canny_threshold2,
        hough_threshold,
        min_line_length_ratio,
        max_line_gap,
        vertical_tolerance_deg=max(15.0, max_correction_deg + 8.0),
    )
    detected_correction_angle, level_confidence, level_debug = estimate_roll(
        initial_segments, max_correction_deg, detection_width
    )
    correction_angle = detected_correction_angle
    strong_spatial_evidence = (
        level_debug.get("occupied_horizontal_buckets", 0) >= 4
        and level_debug.get("angle_mad_deg", 99.0) <= 1.0
    )
    moderate_rotation_needs_more_evidence = abs(correction_angle) > 2.25 and (
        level_confidence < 0.72
        or level_debug.get("occupied_horizontal_buckets", 0) < 4
        or level_debug.get("angle_mad_deg", 99.0) > 1.5
    )
    large_rotation_needs_more_evidence = abs(correction_angle) > 2.75 and (
        level_confidence < 0.72 or not strong_spatial_evidence
    )
    if (
        level_confidence < minimum_confidence
        or moderate_rotation_needs_more_evidence
        or large_rotation_needs_more_evidence
    ):
        reasons = []
        if level_confidence < minimum_confidence:
            reasons.append("low confidence")
        if moderate_rotation_needs_more_evidence or large_rotation_needs_more_evidence:
            reasons.append("large rotation lacks enough line agreement")
        warnings.append("Leveling skipped: " + ", ".join(reasons))
        correction_angle = 0.0
    elif abs(correction_angle) < 0.15:
        correction_angle = 0.0

    detect_rotation = _rotation_homography(detection_width, detection_height, correction_angle)
    leveled_detection = cv2.warpPerspective(
        working_bgr,
        detect_rotation,
        (detection_width, detection_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    leveled_segments, _, debug_bgr = detect_vertical_segments(
        leveled_detection,
        canny_threshold1,
        canny_threshold2,
        hough_threshold,
        min_line_length_ratio,
        max_line_gap,
        vertical_tolerance_deg=25.0,
    )

    original_rotation = _rotation_homography(original_width, original_height, correction_angle)
    combined = original_rotation
    perspective_applied = False
    perspective_confidence = 0.0
    perspective_debug: dict = {"reason": "level_mode_requested"}
    vanishing_point = None
    perspective_ratio = 0.0

    if mode in ("auto", "perspective"):
        vanishing_point, perspective_confidence, perspective_debug = estimate_vertical_vanishing_point(
            leveled_segments, detection_width, detection_height
        )
        if vanishing_point is not None:
            vp_x_from_center = abs(float(vanishing_point[0]) - detection_width / 2.0) / detection_width
            vp_vertical_distance = abs(float(vanishing_point[1]) - detection_height / 2.0)
            perspective_ratio = detection_height / max(vp_vertical_distance, 1e-9)
            correction_is_meaningful = perspective_ratio >= 0.012
            correction_is_safe = perspective_ratio <= max_perspective_ratio
            centered_enough = vp_x_from_center <= 0.35
            confident_enough = perspective_confidence >= minimum_confidence

            if confident_enough and correction_is_meaningful and correction_is_safe and centered_enough:
                vp_original = vanishing_point / detection_scale
                perspective = _vertical_rectification_homography(
                    original_width,
                    original_height,
                    vp_original,
                    perspective_strength,
                )
                combined = perspective @ original_rotation
                perspective_applied = True
            else:
                reasons = []
                if not confident_enough:
                    reasons.append("low confidence")
                if not correction_is_meaningful:
                    reasons.append("verticals already effectively parallel")
                if not correction_is_safe:
                    reasons.append("correction would be too aggressive")
                if not centered_enough:
                    reasons.append("vertical vanishing point is too far off-center")
                warnings.append("Perspective skipped: " + ", ".join(reasons))

    no_change = abs(correction_angle) < 0.01 and not perspective_applied
    if no_change:
        corrected = original_bgr.copy()
        crop_fraction = 0.0
        warp_debug = {"uncropped_dimensions": [original_width, original_height]}
    else:
        corrected, crop_fraction, warp_debug = _warp_and_crop(original_bgr, combined, crop_mode)

    initial_roll_error, initial_vertical_error = _segment_errors(initial_segments, detection_width)
    leveled_roll_error, leveled_vertical_error = _segment_errors(leveled_segments, detection_width)
    validation_debug: dict = {
        "initial_roll_error_deg": round(initial_roll_error, 4) if initial_roll_error is not None else None,
        "initial_vertical_error_deg": round(initial_vertical_error, 4) if initial_vertical_error is not None else None,
        "leveled_roll_error_deg": round(leveled_roll_error, 4) if leveled_roll_error is not None else None,
        "leveled_vertical_error_deg": round(leveled_vertical_error, 4) if leveled_vertical_error is not None else None,
    }

    if perspective_applied:
        output_roll_error, output_vertical_error, output_line_count = _measure_output_errors(
            corrected,
            max_dimension,
            canny_threshold1,
            canny_threshold2,
            hough_threshold,
            min_line_length_ratio,
            max_line_gap,
        )
        validation_debug.update(
            {
                "candidate_output_roll_error_deg": round(output_roll_error, 4)
                if output_roll_error is not None
                else None,
                "candidate_output_vertical_error_deg": round(output_vertical_error, 4)
                if output_vertical_error is not None
                else None,
                "candidate_output_vertical_lines": output_line_count,
            }
        )
        required_improvement = max(0.12, (leveled_vertical_error or 0.0) * 0.05)
        initial_required_improvement = max(0.12, (initial_vertical_error or 0.0) * 0.05)
        perspective_improved = (
            leveled_vertical_error is not None
            and initial_vertical_error is not None
            and output_vertical_error is not None
            and output_vertical_error <= leveled_vertical_error - required_improvement
            and output_vertical_error <= initial_vertical_error - initial_required_improvement
        )
        if not perspective_improved:
            warnings.append("Perspective skipped because output line geometry did not improve")
            perspective_applied = False
            combined = original_rotation
            if abs(correction_angle) < 0.01:
                corrected = original_bgr.copy()
                crop_fraction = 0.0
                warp_debug = {"uncropped_dimensions": [original_width, original_height]}
            else:
                corrected, crop_fraction, warp_debug = _warp_and_crop(original_bgr, combined, crop_mode)

    # An aggressive crop is worse than a slightly converging room. Fall back to
    # level-only correction and report why.
    if perspective_applied and crop_mode == "crop" and crop_fraction > max_crop_fraction:
        warnings.append(
            f"Perspective skipped because it would discard {crop_fraction:.1%} of valid image area"
        )
        perspective_applied = False
        combined = original_rotation
        if abs(correction_angle) < 0.01:
            corrected = original_bgr.copy()
            crop_fraction = 0.0
            warp_debug = {"uncropped_dimensions": [original_width, original_height]}
        else:
            corrected, crop_fraction, warp_debug = _warp_and_crop(original_bgr, combined, crop_mode)

    if not perspective_applied and abs(correction_angle) >= 0.01:
        output_roll_error, output_vertical_error, output_line_count = _measure_output_errors(
            corrected,
            max_dimension,
            canny_threshold1,
            canny_threshold2,
            hough_threshold,
            min_line_length_ratio,
            max_line_gap,
        )
        validation_debug.update(
            {
                "level_output_roll_error_deg": round(output_roll_error, 4)
                if output_roll_error is not None
                else None,
                "level_output_vertical_error_deg": round(output_vertical_error, 4)
                if output_vertical_error is not None
                else None,
                "level_output_vertical_lines": output_line_count,
            }
        )
        required_improvement = max(0.08, (initial_roll_error or 0.0) * 0.25)
        leveling_improved = (
            initial_roll_error is not None
            and output_roll_error is not None
            and output_roll_error <= initial_roll_error - required_improvement
        )
        if not leveling_improved:
            warnings.append("Leveling skipped because output line geometry did not improve")
            corrected = original_bgr.copy()
            correction_angle = 0.0
            crop_fraction = 0.0
            warp_debug = {"uncropped_dimensions": [original_width, original_height]}

    if crop_mode == "crop" and crop_fraction > max_crop_fraction:
        warnings.append(
            f"Leveling skipped because it would discard {crop_fraction:.1%} of valid image area"
        )
        corrected = original_bgr.copy()
        correction_angle = 0.0
        crop_fraction = 0.0
        warp_debug = {"uncropped_dimensions": [original_width, original_height]}

    confidence = perspective_confidence if perspective_applied else level_confidence
    applied_mode = "perspective" if perspective_applied else ("level" if abs(correction_angle) >= 0.01 else "none")
    debug = {
        "requested_mode": mode,
        "applied_mode": applied_mode,
        "original_dimensions": [original_width, original_height],
        "detection_dimensions": [detection_width, detection_height],
        "detection_scale": round(detection_scale, 6),
        "total_hough_lines": total_lines,
        "initial_vertical_lines": len(initial_segments),
        "leveled_vertical_lines": len(leveled_segments),
        "level": level_debug,
        "detected_correction_angle_deg": round(detected_correction_angle, 4),
        "perspective": {
            **perspective_debug,
            "vanishing_point_detection_px": vanishing_point.tolist() if vanishing_point is not None else None,
            "perspective_ratio": round(perspective_ratio, 6),
            "confidence": round(perspective_confidence, 4),
        },
        "output_dimensions": [corrected.shape[1], corrected.shape[0]],
        "crop_fraction": round(crop_fraction, 6),
        "validation": validation_debug,
        "warp": warp_debug,
        "warnings": warnings,
    }

    if return_debug:
        cv2.putText(
            debug_bgr,
            f"mode={applied_mode} roll={correction_angle:+.2f} conf={confidence:.2f}",
            (18, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (30, 220, 255),
            2,
            cv2.LINE_AA,
        )
        if vanishing_point is not None:
            direction = vanishing_point - np.array([detection_width / 2.0, detection_height / 2.0])
            norm = np.linalg.norm(direction)
            if norm > 0:
                endpoint = np.array([detection_width / 2.0, detection_height / 2.0]) + direction / norm * min(
                    detection_width, detection_height
                ) * 0.4
                cv2.arrowedLine(
                    debug_bgr,
                    (detection_width // 2, detection_height // 2),
                    tuple(np.round(endpoint).astype(int)),
                    (255, 180, 0),
                    2,
                    tipLength=0.12,
                )
    else:
        debug_bgr = None

    return StraightenResult(
        corrected_bgr=corrected,
        correction_angle_deg=correction_angle,
        perspective_applied=perspective_applied,
        confidence=confidence,
        crop_fraction=crop_fraction,
        applied_mode=applied_mode,
        debug=debug,
        debug_bgr=debug_bgr,
    )
