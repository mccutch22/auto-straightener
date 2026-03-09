import os
import uuid
from io import BytesIO

import boto3


AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET = os.getenv("S3_BUCKET")
S3_PUBLIC_BASE_URL = os.getenv("S3_PUBLIC_BASE_URL")  # optional, e.g. https://my-bucket.s3.amazonaws.com

s3 = boto3.client("s3", region_name=AWS_REGION)


def upload_bytes(
    file_bytes: bytes,
    key: str,
    content_type: str = "image/jpeg",
    cache_control: str = "public, max-age=31536000, immutable",
) -> str:
    if not S3_BUCKET:
        raise RuntimeError("S3_BUCKET is not configured")

    s3.put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=file_bytes,
        ContentType=content_type,
        CacheControl=cache_control,
    )

    if S3_PUBLIC_BASE_URL:
        return f"{S3_PUBLIC_BASE_URL.rstrip('/')}/{key}"

    return f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{key}"


def make_output_key(prefix: str = "straightened", suffix: str = ".jpg") -> str:
    return f"{prefix}/{uuid.uuid4().hex}{suffix}"