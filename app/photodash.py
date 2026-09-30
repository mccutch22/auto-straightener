"""Private binary API: no remote URL downloads and no storage credentials required."""
import base64
import json
import threading
from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageCms, ImageOps, UnidentifiedImageError
from fastapi import APIRouter, Header, HTTPException, Request, Query
from fastapi.concurrency import run_in_threadpool

from .straighten import auto_straighten_verticals

router = APIRouter()
busy = threading.Lock()
VERSION = "photodash-verticals-4"
MAX_BYTES = 32 * 1024 * 1024
MAX_PIXELS = 24_000_000
cv2.setNumThreads(1)


def process(data: bytes, manual: tuple | None = None):
    try:
        with Image.open(BytesIO(data)) as source:
            # iPhone JPEGs can be MPO containers with an auxiliary HDR gain map.
            # Process the primary photograph, never the auxiliary image.
            if source.format not in ("JPEG", "MPO", "PNG", "WEBP") or source.width * source.height > MAX_PIXELS or (source.format != "MPO" and getattr(source, "n_frames", 1) != 1):
                raise ValueError("Use a single JPEG, PNG or WebP up to 24 megapixels")
            source.seek(0)
            image = ImageOps.exif_transpose(source).convert("RGB")
            if source.info.get("icc_profile"):
                image = ImageCms.profileToProfile(image, ImageCms.ImageCmsProfile(BytesIO(source.info["icc_profile"])), ImageCms.createProfile("sRGB"), outputMode="RGB")
            original = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            image.close()
    except (ValueError, OSError, UnidentifiedImageError, Image.DecompressionBombError) as error:
        raise HTTPException(422, "Unsupported or damaged photo") from error
    if manual is not None:
        from .manual import adjust
        try:
            result = adjust(original, *manual)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
    else:
        result = auto_straighten_verticals(
            original, max_dimension=1400, crop_mode="crop",
            perspective_strength=0.65, minimum_confidence=0.315,
            max_perspective_ratio=0.364, max_crop_fraction=0.234,
        )
    metadata = dict(version=VERSION, outcome="corrected" if result.applied_mode != "none" else "unchanged",
                    mode=result.applied_mode, rotation=result.correction_angle_deg, confidence=result.confidence,
                    cropFraction=result.crop_fraction, width=result.corrected_bgr.shape[1], height=result.corrected_bgr.shape[0],
                    warnings=result.debug.get("warnings", []), geometry=result.debug)
    if result.applied_mode != "none":
        ok, encoded = cv2.imencode(".jpg", result.corrected_bgr, [cv2.IMWRITE_JPEG_QUALITY, 96])
        if not ok:
            raise RuntimeError("Could not encode correction")
        metadata["imageBase64"] = base64.b64encode(encoded).decode("ascii")
    json.dumps(metadata, allow_nan=False)
    return metadata


@router.post("/v1/straighten")
async def straighten_binary(request: Request, authorization: str | None = Header(default=None)):
    return await handle_binary(request, authorization)


@router.post("/v1/adjust")
async def adjust_binary(request: Request, authorization: str | None = Header(default=None),
                        rotation: float = Query(0, ge=-15, le=15),
                        vertical: float = Query(0, ge=-100, le=100),
                        horizontal: float = Query(0, ge=-100, le=100)):
    return await handle_binary(request, authorization, (rotation, vertical, horizontal))


async def handle_binary(request, authorization, manual=None):
    from .main import require_auth
    require_auth(authorization)
    if not busy.acquire(blocking=False):
        raise HTTPException(429, "Worker is busy; retry shortly", headers={"Retry-After": "5"})
    try:
        data = bytearray()
        async for chunk in request.stream():
            data.extend(chunk)
            if len(data) > MAX_BYTES:
                raise HTTPException(413, "Photo must be smaller than 32 MB")
        return await run_in_threadpool(process, bytes(data), manual)
    finally:
        busy.release()
