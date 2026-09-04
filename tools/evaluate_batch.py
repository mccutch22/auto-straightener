import argparse
import html
import json
import math
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.straighten import auto_straighten_verticals, detect_vertical_segments


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def resize_to_limit(image: np.ndarray, max_dimension: int) -> np.ndarray:
    height, width = image.shape[:2]
    if max(height, width) <= max_dimension:
        return image.copy()
    scale = max_dimension / max(height, width)
    return cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def vertical_error(image: np.ndarray) -> float | None:
    analysis = resize_to_limit(image, 1000)
    segments, _, _ = detect_vertical_segments(analysis, 50, 150, 80, 0.12, 20, 25.0)
    if not segments:
        return None
    total_weight = sum(segment.length for segment in segments)
    return sum(abs(segment.offset_from_vertical_deg) * segment.length for segment in segments) / total_weight


def write_jpeg(path: Path, image: np.ndarray, quality: int = 88) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), quality]):
        raise RuntimeError(f"Could not write {path}")


def fit_tile(image: np.ndarray, width: int, height: int) -> np.ndarray:
    canvas = np.full((height, width, 3), 242, dtype=np.uint8)
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def build_contact_sheets(records: list[dict], output_dir: Path, mode: str) -> list[str]:
    selected = [record for record in records if record["applied_mode"] == mode]
    if mode == "none":
        selected = selected[:24]
    if not selected:
        return []

    page_size = 12
    page_paths: list[str] = []
    tile_width, image_height, label_height = 900, 300, 78
    tile_height = image_height + label_height

    for page_index in range(math.ceil(len(selected) / page_size)):
        page_records = selected[page_index * page_size : (page_index + 1) * page_size]
        sheet = np.full((tile_height * 4, tile_width * 3, 3), 248, dtype=np.uint8)
        for index, record in enumerate(page_records):
            row, column = divmod(index, 3)
            x, y = column * tile_width, row * tile_height
            baseline = cv2.imread(str(output_dir / record["baseline_preview"]))
            corrected = cv2.imread(str(output_dir / record["corrected_preview"]))
            pair = np.hstack(
                [fit_tile(baseline, tile_width // 2, image_height), fit_tile(corrected, tile_width // 2, image_height)]
            )
            sheet[y : y + image_height, x : x + tile_width] = pair
            cv2.line(sheet, (x + tile_width // 2, y), (x + tile_width // 2, y + image_height), (0, 145, 255), 3)
            label = (
                f"{record['name']} | {record['applied_mode']} | "
                f"roll {record['correction_angle_deg']:+.2f} | conf {record['confidence']:.2f} | "
                f"crop {record['crop_fraction']:.1%}"
            )
            cv2.putText(
                sheet,
                label[:108],
                (x + 12, y + image_height + 29),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.54,
                (25, 25, 25),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                sheet,
                "BASELINE",
                (x + 12, y + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                sheet,
                "PHOTODASH",
                (x + tile_width // 2 + 12, y + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        filename = f"contact-{mode}-{page_index + 1}.jpg"
        write_jpeg(output_dir / filename, sheet, 90)
        page_paths.append(filename)
    return page_paths


def render_html(records: list[dict], summary: dict, output_dir: Path) -> None:
    cards = []
    for record in sorted(records, key=lambda item: (item["applied_mode"] == "none", -item["crop_fraction"])):
        warnings = " · ".join(record["warnings"]) or "No warnings"
        cards.append(
            f"""
            <article class="card" data-mode="{record['applied_mode']}">
              <header><strong>{html.escape(record['name'])}</strong><span class="badge {record['applied_mode']}">{record['applied_mode']}</span></header>
              <div class="pair">
                <button onclick="openImage('{record['baseline_preview']}')"><img loading="lazy" src="{record['baseline_preview']}" alt="Baseline"><small>Baseline</small></button>
                <button onclick="openImage('{record['corrected_preview']}')"><img loading="lazy" src="{record['corrected_preview']}" alt="PhotoDash"><small>PhotoDash</small></button>
              </div>
              <dl>
                <div><dt>Rotation</dt><dd>{record['correction_angle_deg']:+.2f}°</dd></div>
                <div><dt>Confidence</dt><dd>{record['confidence']:.2f}</dd></div>
                <div><dt>Crop</dt><dd>{record['crop_fraction']:.1%}</dd></div>
                <div><dt>Vertical error</dt><dd>{record['vertical_error_before_deg']}° → {record['vertical_error_after_deg']}°</dd></div>
              </dl>
              <p>{html.escape(warnings)}</p>
            </article>
            """
        )

    output_dir.joinpath("index.html").write_text(
        f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PhotoDash Straightening Evaluation</title>
<style>
:root{{--ink:#15202b;--muted:#687383;--line:#dfe4ea;--blue:#0877ff;--paper:#f4f6f8}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font:14px/1.45 Inter,Arial,sans-serif}}
.top{{position:sticky;top:0;z-index:3;background:#fff;border-bottom:1px solid var(--line);padding:18px 24px}}
h1{{font-size:22px;margin:0 0 10px}} .summary,.filters{{display:flex;gap:10px;flex-wrap:wrap}}
.summary span,.filters button{{border:1px solid var(--line);background:#fff;padding:7px 11px}}
.filters button{{cursor:pointer}} .filters button.active{{background:var(--ink);color:white}}
main{{display:grid;grid-template-columns:repeat(auto-fill,minmax(430px,1fr));gap:16px;padding:18px}}
.card{{background:#fff;border:1px solid var(--line);padding:13px}} header{{display:flex;justify-content:space-between;gap:12px;margin-bottom:10px}}
.badge{{text-transform:uppercase;font-size:11px;letter-spacing:.08em}} .perspective{{color:#7c3aed}} .level{{color:#0877ff}} .none{{color:#687383}}
.pair{{display:grid;grid-template-columns:1fr 1fr;gap:3px}} .pair button{{border:0;padding:0;background:#eee;cursor:zoom-in;position:relative;min-height:220px}}
.pair img{{width:100%;height:250px;object-fit:contain;display:block}} small{{position:absolute;left:8px;bottom:8px;background:#111c;color:#fff;padding:4px 7px}}
dl{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:10px 0 0}} dl div{{border-top:1px solid var(--line);padding-top:7px}} dt{{font-size:11px;color:var(--muted)}} dd{{margin:2px 0 0}}
.card p{{font-size:12px;color:var(--muted);margin:8px 0 0}} dialog{{border:0;background:#111;padding:12px;max-width:96vw;max-height:96vh}} dialog img{{max-width:92vw;max-height:90vh;display:block}}
@media(max-width:600px){{main{{grid-template-columns:1fr;padding:8px}}.pair img{{height:170px}}dl{{grid-template-columns:1fr 1fr}}}}
</style></head><body>
<section class="top"><h1>PhotoDash Straightening Evaluation</h1>
<div class="summary"><span>{summary['total']} images</span><span>{summary['perspective']} perspective</span><span>{summary['level']} level</span><span>{summary['none']} unchanged</span></div>
<div class="filters"><button class="active" onclick="filterCards('all',this)">All</button><button onclick="filterCards('perspective',this)">Perspective</button><button onclick="filterCards('level',this)">Level</button><button onclick="filterCards('none',this)">Unchanged</button></div></section>
<main>{''.join(cards)}</main><dialog id="viewer" onclick="this.close()"><img id="large" alt="Full preview"></dialog>
<script>function filterCards(mode,b){{document.querySelectorAll('.filters button').forEach(x=>x.classList.remove('active'));b.classList.add('active');document.querySelectorAll('.card').forEach(c=>c.hidden=mode!=='all'&&c.dataset.mode!==mode)}}function openImage(src){{document.getElementById('large').src=src;document.getElementById('viewer').showModal()}}</script>
</body></html>""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate automatic straightening across a folder of photos")
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--processing-max", type=int, default=2200)
    parser.add_argument("--detection-max", type=int, default=1200)
    args = parser.parse_args()

    files = sorted(
        path for path in args.input_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not files:
        raise SystemExit("No supported images found")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for index, path in enumerate(files, start=1):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"[{index}/{len(files)}] skipped unreadable {path.name}", flush=True)
            continue
        evaluation_image = resize_to_limit(image, args.processing_max)
        result = auto_straighten_verticals(
            evaluation_image,
            max_dimension=args.detection_max,
            mode="auto",
            crop_mode="crop",
            return_debug=False,
        )

        stem = f"{index:03d}-{path.stem}"
        baseline_path = Path("previews") / f"{stem}-baseline.jpg"
        corrected_path = Path("previews") / f"{stem}-photodash.jpg"
        write_jpeg(args.output_dir / baseline_path, resize_to_limit(evaluation_image, 1200))
        write_jpeg(args.output_dir / corrected_path, resize_to_limit(result.corrected_bgr, 1200))

        before_error = vertical_error(evaluation_image)
        after_error = vertical_error(result.corrected_bgr)
        record = {
            "name": path.name,
            "source": str(path),
            "applied_mode": result.applied_mode,
            "correction_angle_deg": round(result.correction_angle_deg, 4),
            "confidence": round(result.confidence, 4),
            "crop_fraction": round(result.crop_fraction, 6),
            "vertical_error_before_deg": round(before_error, 3) if before_error is not None else None,
            "vertical_error_after_deg": round(after_error, 3) if after_error is not None else None,
            "warnings": result.debug["warnings"],
            "baseline_preview": baseline_path.as_posix(),
            "corrected_preview": corrected_path.as_posix(),
            "debug": result.debug,
        }
        records.append(record)
        print(
            f"[{index}/{len(files)}] {path.name}: {result.applied_mode}, "
            f"roll={result.correction_angle_deg:+.2f}, confidence={result.confidence:.2f}, "
            f"crop={result.crop_fraction:.1%}",
            flush=True,
        )

    counts = Counter(record["applied_mode"] for record in records)
    summary = {
        "total": len(records),
        "perspective": counts["perspective"],
        "level": counts["level"],
        "none": counts["none"],
    }
    contact_sheets = {
        mode: build_contact_sheets(records, args.output_dir, mode)
        for mode in ("perspective", "level", "none")
    }
    args.output_dir.joinpath("manifest.json").write_text(
        json.dumps({"summary": summary, "contact_sheets": contact_sheets, "images": records}, indent=2),
        encoding="utf-8",
    )
    render_html(records, summary, args.output_dir)
    print("SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
