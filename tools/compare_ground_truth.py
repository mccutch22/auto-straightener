import argparse
import csv
import html
import json
import math
import re
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.straighten import auto_straighten_verticals
from tools.evaluate_batch import fit_tile, resize_to_limit, vertical_error, write_jpeg


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def normalized_match_key(path_or_name: Path | str) -> str:
    """Normalize the filename wrappers used by manual-correction exports."""
    stem = Path(path_or_name).stem.casefold().strip()
    prefixes = (
        "manual-correct-",
        "manual-corrected-",
        "manually-correct-",
        "manually-corrected-",
        "manual correct ",
        "manual corrected ",
        "manually correct ",
        "manually corrected ",
    )
    for prefix in prefixes:
        if stem.startswith(prefix):
            stem = stem[len(prefix) :]
            break
    stem = re.sub(r"[-_\s]+$", "", stem)
    return re.sub(r"\s+", " ", stem)


def index_images(folder: Path) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    for path in folder.iterdir():
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        key = normalized_match_key(path)
        if key in indexed:
            raise ValueError(f"Duplicate normalized filename in {folder}: {path.name}")
        indexed[key] = path
    return indexed


def _analysis_copy(image: np.ndarray, max_dimension: int = 1400) -> tuple[np.ndarray, float]:
    largest = max(image.shape[:2])
    scale = min(1.0, max_dimension / largest)
    if scale == 1.0:
        return image.copy(), scale
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), scale


def estimate_pair_homography(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray | None, dict]:
    source_small, source_scale = _analysis_copy(source)
    target_small, target_scale = _analysis_copy(target)
    sift = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.018, edgeThreshold=12)
    source_gray = cv2.cvtColor(source_small, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target_small, cv2.COLOR_BGR2GRAY)
    source_points, source_descriptors = sift.detectAndCompute(source_gray, None)
    target_points, target_descriptors = sift.detectAndCompute(target_gray, None)
    if source_descriptors is None or target_descriptors is None:
        return None, {"reason": "no_descriptors"}

    matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(source_descriptors, target_descriptors, k=2)
    good = [first for first, second in matches if first.distance < 0.72 * second.distance]
    if len(good) < 12:
        return None, {"reason": "not_enough_matches", "matches": len(good)}

    source_xy = np.float32([source_points[match.queryIdx].pt for match in good]).reshape(-1, 1, 2)
    target_xy = np.float32([target_points[match.trainIdx].pt for match in good]).reshape(-1, 1, 2)
    analysis_h, mask = cv2.findHomography(source_xy, target_xy, cv2.RANSAC, 3.0)
    if analysis_h is None or mask is None:
        return None, {"reason": "homography_failed", "matches": len(good)}

    source_scale_matrix = np.diag([source_scale, source_scale, 1.0])
    target_scale_matrix = np.diag([target_scale, target_scale, 1.0])
    full_h = np.linalg.inv(target_scale_matrix) @ analysis_h @ source_scale_matrix
    full_h /= full_h[2, 2]

    inliers = mask.ravel().astype(bool)
    projected = cv2.perspectiveTransform(source_xy[inliers], analysis_h)
    errors = np.linalg.norm(projected.reshape(-1, 2) - target_xy[inliers].reshape(-1, 2), axis=1)
    return full_h, {
        "reason": "ok",
        "matches": len(good),
        "inliers": int(inliers.sum()),
        "inlier_ratio": round(float(inliers.mean()), 4),
        "median_reprojection_error": round(float(np.median(errors)), 4),
    }


def _transform_point(homography: np.ndarray, x: float, y: float) -> np.ndarray:
    point = np.array([[[x, y]]], dtype=np.float64)
    return cv2.perspectiveTransform(point, homography)[0, 0]


def describe_transform(homography: np.ndarray, width: int, height: int) -> dict:
    cx, cy = width / 2.0, height / 2.0
    delta = max(20.0, min(width, height) * 0.05)
    center = _transform_point(homography, cx, cy)
    across = _transform_point(homography, cx + delta, cy) - center
    down = _transform_point(homography, cx, cy + delta) - center
    rotation = math.degrees(math.atan2(float(across[1]), float(across[0])))

    def vertical_angle(x: float) -> float:
        upper = _transform_point(homography, x, height * 0.2)
        lower = _transform_point(homography, x, height * 0.8)
        vector = lower - upper
        return math.degrees(math.atan2(float(vector[0]), float(vector[1])))

    left_angle = vertical_angle(width * 0.15)
    right_angle = vertical_angle(width * 0.85)
    top_left = _transform_point(homography, width * 0.15, height * 0.2)
    top_right = _transform_point(homography, width * 0.85, height * 0.2)
    bottom_left = _transform_point(homography, width * 0.15, height * 0.8)
    bottom_right = _transform_point(homography, width * 0.85, height * 0.8)
    top_width = float(np.linalg.norm(top_right - top_left))
    bottom_width = float(np.linalg.norm(bottom_right - bottom_left))
    width_taper = (top_width - bottom_width) / max((top_width + bottom_width) / 2.0, 1e-9)
    shear = math.degrees(math.atan2(float(down[0]), float(down[1])))
    return {
        "manual_rotation_deg": round(rotation, 4),
        "manual_vertical_left_deg": round(left_angle, 4),
        "manual_vertical_right_deg": round(right_angle, 4),
        "manual_convergence_change_deg": round(right_angle - left_angle, 4),
        "manual_width_taper": round(width_taper, 6),
        "manual_vertical_shear_deg": round(shear, 4),
    }


def build_sheet(records: list[dict], output_dir: Path) -> None:
    page_size = 9
    tile_width, image_height, label_height = 1050, 260, 68
    tile_height = image_height + label_height
    for page_index in range(math.ceil(len(records) / page_size)):
        page_records = records[page_index * page_size : (page_index + 1) * page_size]
        sheet = np.full((tile_height * 3, tile_width * 3, 3), 246, dtype=np.uint8)
        for index, record in enumerate(page_records):
            row, column = divmod(index, 3)
            x, y = column * tile_width, row * tile_height
            images = [cv2.imread(str(output_dir / record[key])) for key in ("original_preview", "manual_preview", "auto_preview")]
            strip = np.hstack([fit_tile(image, tile_width // 3, image_height) for image in images])
            sheet[y : y + image_height, x : x + tile_width] = strip
            label = (
                f"{record['name']} | manual rot {record.get('manual_rotation_deg', 0):+.2f} | "
                f"auto {record['auto_mode']} {record['auto_rotation_deg']:+.2f} | "
                f"manual/auto error {record['manual_vertical_error']}/{record['auto_vertical_error']}"
            )
            cv2.putText(sheet, label[:125], (x + 10, y + image_height + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
            cv2.putText(sheet, "ORIGINAL                         MANUAL                          PHOTODASH", (x + 10, y + image_height + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (95, 95, 95), 1, cv2.LINE_AA)
        write_jpeg(output_dir / f"comparison-{page_index + 1:02d}.jpg", sheet, quality=89)


def build_html(records: list[dict], summary: dict, output_dir: Path) -> None:
    cards = []
    for record in records:
        images = "".join(
            f'<figure><img loading="lazy" src="{html.escape(record[key])}" alt=""><figcaption>{label}</figcaption></figure>'
            for key, label in (
                ("original_preview", "Original"),
                ("manual_preview", "Manual reference"),
                ("auto_preview", "PhotoDash"),
            )
        )
        cards.append(
            f'''<article>
            <h2>{html.escape(record["name"])}</h2>
            <div class="images">{images}</div>
            <p><strong>{record["auto_mode"]}</strong> · rotation error {abs(record["auto_transform_rotation_deg"] - record["manual_rotation_deg"]):.2f}° · perspective error {abs(record["auto_transform_convergence_change_deg"] - record["manual_convergence_change_deg"]):.2f}° · crop {record["auto_crop_fraction"]:.1%}</p>
            </article>'''
        )
    page = f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>PhotoDash ground-truth comparison</title><style>
    *{{box-sizing:border-box}}body{{margin:0;background:#eef2f7;color:#142033;font:15px/1.45 system-ui,sans-serif}}header{{position:sticky;top:0;z-index:2;padding:18px 28px;background:#111b2dcc;color:white}}header h1{{margin:0 0 6px;font-size:23px}}header p{{margin:0;color:#c9d5e8}}main{{max-width:1540px;margin:auto;padding:24px}}article{{margin:0 0 24px;padding:18px;background:white;box-shadow:0 5px 24px #18253b18}}h2{{margin:0 0 12px;font-size:15px;font-weight:650}}.images{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}}figure{{margin:0;background:#eef1f5}}img{{display:block;width:100%;height:340px;object-fit:contain}}figcaption{{padding:7px 10px;background:#1b2638;color:white;text-transform:uppercase;letter-spacing:.08em;font-size:11px}}article p{{margin:12px 0 0;color:#526176}}@media(max-width:760px){{main{{padding:12px}}.images{{grid-template-columns:1fr}}img{{height:auto}}header{{position:static}}}}
    </style></head><body><header><h1>PhotoDash vs. manual corrections</h1><p>{summary["evaluated_files"]} matched images · {summary["auto_rotation_within_one_degree"]} within 1° of manual rotation · median rotation error {summary["auto_vs_manual_rotation_median_error_deg"]:.2f}°</p></header><main>{''.join(cards)}</main></body></html>'''
    (output_dir / "index.html").write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare PhotoDash corrections with manually corrected ground truth.")
    parser.add_argument("original_dir", type=Path)
    parser.add_argument("manual_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--processing-max", type=int, default=2200)
    parser.add_argument("--remeasure-registration", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    prior_records: dict[str, dict] = {}
    prior_manifest = args.output_dir / "manifest.json"
    if prior_manifest.exists() and not args.remeasure_registration:
        prior_payload = json.loads(prior_manifest.read_text(encoding="utf-8"))
        prior_records = {
            normalized_match_key(record["name"]): record
            for record in prior_payload.get("images", [])
        }

    originals = index_images(args.original_dir)
    manuals = index_images(args.manual_dir)
    common = sorted(originals.keys() & manuals.keys())
    if not common:
        raise SystemExit("No matching image filenames found")

    records: list[dict] = []
    for index, key in enumerate(common, start=1):
        source_path, manual_path = originals[key], manuals[key]
        source = cv2.imread(str(source_path))
        manual = cv2.imread(str(manual_path))
        if source is None or manual is None:
            continue
        prior = prior_records.get(key)
        if prior and prior.get("reason") == "ok":
            match_info = {
                field: prior[field]
                for field in ("reason", "matches", "inliers", "inlier_ratio", "median_reprojection_error")
                if field in prior
            }
            transform = {
                field: prior[field]
                for field in (
                    "manual_rotation_deg",
                    "manual_vertical_left_deg",
                    "manual_vertical_right_deg",
                    "manual_convergence_change_deg",
                    "manual_width_taper",
                    "manual_vertical_shear_deg",
                )
                if field in prior
            }
        else:
            homography, match_info = estimate_pair_homography(source, manual)
            transform = describe_transform(homography, source.shape[1], source.shape[0]) if homography is not None else {}

        processing_source = source
        auto = auto_straighten_verticals(
            processing_source, max_dimension=args.processing_max, return_debug=False
        )
        auto_homography = np.asarray(
            auto.debug.get("warp", {}).get("effective_homography", np.eye(3)), dtype=np.float64
        )
        auto_transform = {
            key.replace("manual_", "auto_transform_"): value
            for key, value in describe_transform(
                auto_homography, processing_source.shape[1], processing_source.shape[0]
            ).items()
        }
        stem = f"{index:03d}"
        original_preview = f"previews/{stem}-original.jpg"
        manual_preview = f"previews/{stem}-manual.jpg"
        auto_preview = f"previews/{stem}-auto.jpg"
        write_jpeg(args.output_dir / original_preview, resize_to_limit(source, 1200))
        write_jpeg(args.output_dir / manual_preview, resize_to_limit(manual, 1200))
        write_jpeg(args.output_dir / auto_preview, resize_to_limit(auto.corrected_bgr, 1200))
        record = {
            "name": source_path.name,
            "source_width": source.shape[1],
            "source_height": source.shape[0],
            "manual_width": manual.shape[1],
            "manual_height": manual.shape[0],
            "manual_vertical_error": None if (value := vertical_error(manual)) is None else round(value, 4),
            "auto_vertical_error": None if (value := vertical_error(auto.corrected_bgr)) is None else round(value, 4),
            "original_vertical_error": None if (value := vertical_error(source)) is None else round(value, 4),
            "auto_mode": auto.applied_mode,
            "auto_rotation_deg": round(auto.correction_angle_deg, 4),
            "auto_confidence": round(auto.confidence, 4),
            "auto_crop_fraction": round(auto.crop_fraction, 6),
            "original_preview": original_preview,
            "manual_preview": manual_preview,
            "auto_preview": auto_preview,
            **match_info,
            **transform,
            **auto_transform,
        }
        records.append(record)
        print(f"[{index:03d}/{len(common):03d}] {record['auto_mode']:11s} {source_path.name}", flush=True)

    build_sheet(records, args.output_dir)
    usable = [record for record in records if record.get("reason") == "ok"]
    rotation_errors = [
        abs(record["auto_transform_rotation_deg"] - record["manual_rotation_deg"])
        for record in usable
    ]
    convergence_errors = [
        abs(
            record["auto_transform_convergence_change_deg"]
            - record["manual_convergence_change_deg"]
        )
        for record in usable
    ]
    summary = {
        "original_files": len(originals),
        "manual_files": len(manuals),
        "matched_files": len(common),
        "unmatched_original_files": len(originals.keys() - manuals.keys()),
        "unmatched_manual_files": len(manuals.keys() - originals.keys()),
        "evaluated_files": len(records),
        "registered_files": len(usable),
        "manual_rotation_median_abs_deg": round(float(np.median([abs(r["manual_rotation_deg"]) for r in usable])), 4),
        "manual_rotation_p90_abs_deg": round(float(np.percentile([abs(r["manual_rotation_deg"]) for r in usable], 90)), 4),
        "manual_convergence_median_abs_deg": round(float(np.median([abs(r["manual_convergence_change_deg"]) for r in usable])), 4),
        "manual_convergence_p90_abs_deg": round(float(np.percentile([abs(r["manual_convergence_change_deg"]) for r in usable], 90)), 4),
        "auto_vs_manual_rotation_median_error_deg": round(float(np.median(rotation_errors)), 4),
        "auto_vs_manual_rotation_p90_error_deg": round(float(np.percentile(rotation_errors, 90)), 4),
        "auto_rotation_within_one_degree": sum(error <= 1.0 for error in rotation_errors),
        "auto_vs_manual_convergence_median_error_deg": round(float(np.median(convergence_errors)), 4),
        "auto_modes": {mode: sum(r["auto_mode"] == mode for r in records) for mode in ("perspective", "level", "none")},
    }
    with (args.output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "images": records}, handle, indent=2)
    with (args.output_dir / "unmatched.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "originals_without_manual_reference": [
                    originals[key].name for key in sorted(originals.keys() - manuals.keys())
                ],
                "manual_references_without_original": [
                    manuals[key].name for key in sorted(manuals.keys() - originals.keys())
                ],
            },
            handle,
            indent=2,
        )
    fieldnames = sorted({key for record in records for key in record if not key.endswith("_preview")})
    with (args.output_dir / "measurements.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: value for key, value in record.items() if key in fieldnames} for record in records)
    build_html(records, summary, args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
