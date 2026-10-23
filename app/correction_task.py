"""Disposable local correction process; invoked by the private binary API."""
import json
import sys
from pathlib import Path
from fastapi import HTTPException
from .photodash import process

if __name__ == "__main__":
    try:
        result = process(Path(sys.argv[1]).read_bytes(), json.loads(sys.argv[3]))
        Path(sys.argv[2]).write_text(json.dumps(result, allow_nan=False))
    except HTTPException as error:
        sys.exit(22 if error.status_code == 422 else 1)
