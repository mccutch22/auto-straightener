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
PhotoDash saves outputs privately and owns the durable queue. Manual corrections have
a review step; new GPT bracket results are straightened automatically after quality
review. Esoft results bypass automatic straightening. Saved original and corrected
versions can be switched without another processing request.

The source aspect ratio is preserved. Crops include interpolation margins and an inverse
geometry check; they contain only source pixels. No replicated or invented scenery is
delivered by this endpoint. Detection runs at a maximum 1400px; the final warp uses the
full-resolution original. The PhotoDash handheld preset uses 0.65 perspective
strength, 0.315 minimum confidence, 0.364 maximum perspective ratio, and 0.234 maximum
crop fraction. These are 30% more aggressive than v1 (confidence is reduced by 30%).
Corrections that discard more than 23.4% of valid area are skipped. Large-rotation
evidence checks and post-transform improvement checks are unchanged. Legacy API
and direct algorithm defaults remain unchanged.

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

Version `photodash-verticals-3` estimates camera roll and vertical convergence jointly
from source architectural lines. This prevents an asymmetric set of converging window
edges from being mistaken for a tilted camera. The fitted vanishing point is carried
through rotation instead of being fitted again to a different set of edges.

Validation follows the same source lines through the proposed transformation, using
fixed source weights and horizontal groups. Cropping or resampling cannot improve a
score merely by changing which edges are detected. Perspective must improve vertical
alignment without worsening the modeled center direction; a rotation-only fallback
must improve roll without worsening overall vertical alignment. Otherwise the service
preserves the original. This does not correct lens curvature or local AI distortions
that cannot be represented by a single projective transform.

## Run locally

Version `photodash-verticals-4` also provides authenticated `POST /v1/adjust`.
Send the source image as the raw request body, with query parameters `rotation`
(-15 to 15 degrees, positive clockwise), `vertical` (-100 to 100), and
`horizontal` (-100 to 100). The response uses the same image/metadata envelope
as `/v1/straighten`. Manual adjustments bypass line-detection confidence checks;
a deterministic projective transform and centered safe crop avoid exposed
borders. PhotoDash mirrors this geometry in its live preview, saves at source
resolution after cropping, and retains the original for reversible changes.

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
