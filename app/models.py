from typing import Literal

from pydantic import BaseModel, Field, HttpUrl


class StraightenRequest(BaseModel):
    image_url: HttpUrl
    mode: Literal["auto", "level", "perspective"] = "auto"
    crop_mode: Literal["crop", "keep_all"] = "crop"
    max_dimension: int = Field(default=2400, ge=500, le=6000)
    max_correction_deg: float = Field(default=5.0, ge=0.1, le=20.0)
    perspective_strength: float = Field(default=1.0, ge=0.0, le=1.0)
    minimum_confidence: float = Field(default=0.45, ge=0.0, le=1.0)
    max_perspective_ratio: float = Field(default=0.28, ge=0.01, le=0.75)
    max_crop_fraction: float = Field(default=0.18, ge=0.0, le=0.75)
    canny_threshold1: int = Field(default=50, ge=0, le=500)
    canny_threshold2: int = Field(default=150, ge=0, le=500)
    hough_threshold: int = Field(default=80, ge=10, le=500)
    min_line_length_ratio: float = Field(default=0.12, ge=0.01, le=1.0)
    max_line_gap: int = Field(default=20, ge=0, le=500)
    jpeg_quality: int = Field(default=92, ge=60, le=100)
    save_debug: bool = False


class StraightenResponse(BaseModel):
    success: bool
    corrected_url: str | None = None
    correction_angle_deg: float | None = None
    perspective_applied: bool | None = None
    confidence: float | None = None
    crop_fraction: float | None = None
    applied_mode: str | None = None
    debug_url: str | None = None
    debug: dict | None = None
    error: str | None = None
