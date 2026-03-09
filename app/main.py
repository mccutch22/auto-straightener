import os
from io import BytesIO

import cv2
import numpy as np
import requests
from fastapi import FastAPI, Header, HTTPException

from .models import StraightenRequest, StraightenResponse
from .storage import make_output_key, upload_bytes
from .straighten import auto_straighten_verticals


API_TOKEN = os.getenv("API_TOKEN")

app = FastAPI(title="Auto Straightener API")


@app.get("/health")
def health():
    return {"ok": True}


def require_auth(authorization: str | None):
    if not API_TOKEN:
        return

    expected = f"Bearer {API_TOKEN}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


def download_image(url: str, timeout: int = 30) -> bytes:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.content


def decode_image(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image")
    return image


def encode_jpeg(image_bgr: np.ndarray, jpeg_quality: int) -> bytes:
    ok, encoded = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    if not ok:
        raise ValueError("Could not encode jpeg")
    return encoded.tobytes()


@app.post("/straighten", response_model=StraightenResponse)
def straighten(
    payload: StraightenRequest,
    authorization: str | None = Header(default=None),
):
    require_auth(authorization)

    try:
        image_bytes = download_image(str(payload.image_url))
        image_bgr = decode_image(image_bytes)

        result = auto_straighten_verticals(
            original_bgr=image_bgr,
            max_dimension=payload.max_dimension,
            canny_threshold1=payload.canny_threshold1,
            canny_threshold2=payload.canny_threshold2,
            hough_threshold=payload.hough_threshold,
            min_line_length_ratio=payload.min_line_length_ratio,
            max_line_gap=payload.max_line_gap,
            max_correction_deg=payload.max_correction_deg,
            return_debug=payload.save_debug,
        )

        corrected_bytes = encode_jpeg(result.corrected_bgr, payload.jpeg_quality)
        corrected_key = make_output_key(prefix="straightened", suffix=".jpg")
        corrected_url = upload_bytes(corrected_bytes, corrected_key, content_type="image/jpeg")

        debug_url = None
        if payload.save_debug and result.debug_bgr is not None:
            debug_bytes = encode_jpeg(result.debug_bgr, 90)
            debug_key = make_output_key(prefix="straightened-debug", suffix=".jpg")
            debug_url = upload_bytes(debug_bytes, debug_key, content_type="image/jpeg")

        return StraightenResponse(
            success=True,
            corrected_url=corrected_url,
            correction_angle_deg=result.correction_angle_deg,
            debug_url=debug_url,
            debug=result.debug,
        )

    except requests.HTTPError as e:
        return StraightenResponse(success=False, error=f"Failed to download image: {e}")
    except Exception as e:
        return StraightenResponse(success=False, error=str(e))