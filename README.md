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

`POST /straighten`

```json
{
  "image_url": "https://example.com/input.jpg",
  "mode": "auto",
  "crop_mode": "crop",
  "perspective_strength": 1.0,
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

Automatic leveling defaults to a five-degree ceiling. Corrections over three degrees require stronger line agreement, and any operation that exceeds the allowed crop loss returns the untouched original.

## Run locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 10000
```

Set `S3_BUCKET`, `AWS_REGION`, and optionally `S3_PUBLIC_BASE_URL`. Set `API_TOKEN` to require a bearer token.

## Tests

```bash
python -m unittest discover -s tests -v
```
