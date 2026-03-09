import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class StraightenResult:
    corrected_bgr: np.ndarray
    correction_angle_deg: float
    debug: dict
    debug_bgr: np.ndarray | None = None


def resize_for_detection(image_bgr: np.ndarray, max_dimension: int) -> tuple[np.ndarray, float]:
    h, w = image_bgr.shape[:2]
    largest = max(h, w)
    if largest <= max_dimension:
        return image_bgr.copy(), 1.0

    scale = max_dimension / largest
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def rotate_bound(image: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = image.shape[:2]
    center = (w / 2, h / 2)

    m = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos = abs(m[0, 0])
    sin = abs(m[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    m[0, 2] += (new_w / 2) - center[0]
    m[1, 2] += (new_h / 2) - center[1]

    return cv2.warpAffine(
        image,
        m,
        (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def crop_largest_rect_from_rotated(image: np.ndarray, bg_threshold: int = 3) -> np.ndarray:
    """
    Simple trim of edge padding after rotate_bound.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    mask = gray > bg_threshold
    coords = np.column_stack(np.where(mask))
    if coords.size == 0:
        return image

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    return image[y0:y1 + 1, x0:x1 + 1]


def weighted_median(values: list[float], weights: list[float]) -> float:
    if not values:
        raise ValueError("No values for weighted median")
    pairs = sorted(zip(values, weights), key=lambda x: x[0])
    total = sum(weights)
    cumulative = 0.0
    for value, weight in pairs:
        cumulative += weight
        if cumulative >= total / 2:
            return value
    return pairs[-1][0]


def detect_vertical_correction(
    image_bgr: np.ndarray,
    canny_threshold1: int = 50,
    canny_threshold2: int = 150,
    hough_threshold: int = 80,
    min_line_length_ratio: float = 0.12,
    max_line_gap: int = 20,
    max_correction_deg: float = 8.0,
) -> tuple[float, dict, np.ndarray]:
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, canny_threshold1, canny_threshold2)

    min_line_length = int(min(h, w) * min_line_length_ratio)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )

    debug_img = image_bgr.copy()
    all_lines = 0
    vertical_candidates = 0
    angle_offsets = []
    weights = []

    if lines is not None:
        for line in lines[:, 0]:
            x1, y1, x2, y2 = line.tolist()
            all_lines += 1

            dx = x2 - x1
            dy = y2 - y1
            length = math.hypot(dx, dy)
            if length < min_line_length:
                continue

            angle_deg = math.degrees(math.atan2(dy, dx))
            # Convert line angle into offset from true vertical.
            # Vertical is +/-90 degrees.
            offset_from_vertical = angle_deg - 90 if angle_deg >= 0 else angle_deg + 90

            # Keep only near-vertical lines.
            if abs(offset_from_vertical) <= max_correction_deg:
                vertical_candidates += 1
                angle_offsets.append(offset_from_vertical)
                weights.append(length)
                cv2.line(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            else:
                cv2.line(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 1)

    if not angle_offsets:
        return 0.0, {
            "total_lines": all_lines,
            "vertical_lines": 0,
            "used_method": "hough_lines_p",
            "reason": "no_vertical_candidates",
        }, debug_img

    correction = weighted_median(angle_offsets, weights)

    # To make lines vertical, rotate opposite the measured offset.
    correction = float(np.clip(correction, -max_correction_deg, max_correction_deg))

    return correction, {
        "total_lines": all_lines,
        "vertical_lines": vertical_candidates,
        "used_method": "weighted_median_of_vertical_line_offsets",
    }, debug_img


def auto_straighten_verticals(
    original_bgr: np.ndarray,
    max_dimension: int = 2400,
    canny_threshold1: int = 50,
    canny_threshold2: int = 150,
    hough_threshold: int = 80,
    min_line_length_ratio: float = 0.12,
    max_line_gap: int = 20,
    max_correction_deg: float = 8.0,
    return_debug: bool = False,
) -> StraightenResult:
    working_bgr, _ = resize_for_detection(original_bgr, max_dimension)

    correction_angle_deg, debug, debug_bgr = detect_vertical_correction(
        working_bgr,
        canny_threshold1=canny_threshold1,
        canny_threshold2=canny_threshold2,
        hough_threshold=hough_threshold,
        min_line_length_ratio=min_line_length_ratio,
        max_line_gap=max_line_gap,
        max_correction_deg=max_correction_deg,
    )

    corrected = rotate_bound(working_bgr, -correction_angle_deg)
    corrected = crop_largest_rect_from_rotated(corrected)

    return StraightenResult(
        corrected_bgr=corrected,
        correction_angle_deg=correction_angle_deg,
        debug=debug,
        debug_bgr=debug_bgr if return_debug else None,
    )