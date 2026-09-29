# Auto Straightener

A conservative FastAPI service for real-estate photo geometry correction. It can:

- level a rotated photograph;
- detect converging architectural verticals and correct vertical perspective;
- perform detection on a smaller working copy while delivering the full-resolution image;
- measure crop loss and fall back to level-only correction when perspective would be destructive;
- preserve the untouched photo when the detector lacks confidence;
- optionally save a line-detection debug image.

The service transforms existing pixels only. It does not generate or replace property details.

## API

### PhotoDash integration (recommended)

`POST /v1/straighten`, `Authorization: Bearer <API_TOKEN>`, raw JPEG/PNG/WebP request body.
Authentication is required, including when the worker is misconfigured. The API accepts
up to 32 MB / 24 megapixels, one image at a time. It normalizes EXIF orientation and ICC
color to sRGB. It does not download user-supplied URLs or need AWS credentials.

The JSON response includes `outcome` (`corrected` or `unchanged`), `mode`, `rotation`,
`confidence`, `cropFraction`, dimensions, warnings, geometry and algorithm version.
Corrected responses also include `imageBase64` (JPEG quality 96); unchanged responses
contain no image, so PhotoDash keeps the original bytes. Concurrent requests receive 429.
PhotoDash saves outputs privately, owns the durable queue, and requires review before
replacing a gallery image. Closing its review window pauses the remaining queue.

The source aspect ratio is preserved. Crops include interpolation margins and an inverse
geometry check; they contain only source pixels. No replicated or invented scenery is
delivered by this endpoint. Detection runs at a maximum 1400px; the final warp uses the
full-resolution original. Corrections that discard more than 18% of valid area are skipped.

Render: Docker runtime, one worker, `/health` health check. Set `API_TOKEN`; the Dockerfile
uses Render's `PORT`. PhotoDash needs `STRAIGHTENER_URL` and the matching
`STRAIGHTENER_API_TOKEN`. Keep both tokens out of source control. No GPU is needed.

### Legacy URL API

The following endpoint is disabled by default. Explicit `ENABLE_LEGACY_URL_API=true`
opts into its original URL/S3 behavior; it is not used by PhotoDash.

`POST /straighten`

```json
{
  "image_url": "https://example.com/input.jpg",
  "mode": "auto",
  "crop_mode": "crop",
  "perspective_strength": 0.5,
  "minimum_confidence": 0.45,
  "max_crop_fraction": 0.18,
  "save_debug": true
}
```

Modes:

- `auto`: level first, then apply vertical perspective only when it is meaningful, safe, and confident.
- `level`: correct camera roll only.
- `perspective`: explicitly attempt automatic vertical-perspective correction, with the same safeguards.

Crop modes:

- `crop`: return the largest valid rectangle and report `crop_fraction`.
- `keep_all`: retain the complete transformed frame and fill exposed borders from nearby edge pixels.

The response reports the applied mode, confidence, crop loss, correction angle, warnings, and output URL. Keep the original image in storage so every correction remains reversible.

Automatic leveling defaults to a five-degree ceiling. Corrections over 2.25 degrees require stronger spatial evidence, with an additional consistency check over 2.75 degrees. Any operation that exceeds the allowed crop loss returns the untouched original. Repeated lines are balanced by their horizontal position so one cabinet, shelf, or stack of panels cannot rotate the entire room.

Perspective correction defaults to half strength. The setting was calibrated against manually corrected real-estate photos to avoid the over-stretched look produced by full geometric rectification; callers can still request any strength from `0.0` through `1.0`.

Every candidate correction is analyzed a second time after transformation. If the measured line geometry did not improve, the service falls back from perspective to level-only correction, or from level-only correction to the untouched original.

## Run locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 10000
```

Set `API_TOKEN`. Only the optional legacy API additionally needs `S3_BUCKET`, `AWS_REGION`, and optionally `S3_PUBLIC_BASE_URL`.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Evaluate a photo batch

The evaluation tool creates baseline/corrected previews, contact sheets, an interactive HTML review gallery, and a JSON metrics manifest without overwriting source photos:

```bash
python tools/evaluate_batch.py /path/to/photos evaluation/my-batch
```

To compare a batch with manually corrected reference files that have matching names:

```bash
python tools/compare_ground_truth.py /path/to/originals /path/to/manual-corrections evaluation/ground-truth
```

This produces three-way previews, contact sheets, registration measurements, and a JSON/CSV report. Existing feature registrations are reused on subsequent runs unless `--remeasure-registration` is supplied.
