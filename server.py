from __future__ import annotations

import base64
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


STORAGE_ROOT = Path("data")


class FrameUpload(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    sequence: int = Field(ge=0)
    timestamp_utc: str
    window_title: str = Field(min_length=1, max_length=512)
    screen_size: dict[str, int]
    changed_region: dict[str, int]
    image_jpeg_base64: str = Field(min_length=1)


app = FastAPI(title="Visible Capture Receiver", version="0.1.0")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitize_session_id(value: str) -> str:
    cleaned = "".join(ch for ch in value if ch.isalnum() or ch in ("-", "_"))
    return cleaned[:128] or "session"


def _decode_image(payload: str) -> bytes:
    try:
        return base64.b64decode(payload, validate=True)
    except Exception as exc:  # pragma: no cover - defensive guard
        raise HTTPException(status_code=400, detail="invalid image payload") from exc


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


@app.on_event("startup")
def startup() -> None:
    STORAGE_ROOT.mkdir(parents=True, exist_ok=True)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "timestamp_utc": _utc_now()}


@app.post("/api/frame")
def upload_frame(frame: FrameUpload) -> dict[str, Any]:
    session_id = _sanitize_session_id(frame.session_id)
    session_dir = STORAGE_ROOT / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    image_bytes = _decode_image(frame.image_jpeg_base64)
    sha256 = hashlib.sha256(image_bytes).hexdigest()
    stem = f"{frame.sequence:08d}_{sha256[:12]}"

    image_path = session_dir / f"{stem}.jpg"
    meta_path = session_dir / f"{stem}.json"

    if image_path.exists():
        raise HTTPException(status_code=409, detail="duplicate frame upload")

    image_path.write_bytes(image_bytes)
    _write_json(
        meta_path,
        {
            "session_id": session_id,
            "sequence": frame.sequence,
            "timestamp_utc": frame.timestamp_utc,
            "received_at_utc": _utc_now(),
            "window_title": frame.window_title,
            "screen_size": frame.screen_size,
            "changed_region": frame.changed_region,
            "image_sha256": sha256,
            "image_size_bytes": len(image_bytes),
        },
    )

    return {
        "status": "stored",
        "session_id": session_id,
        "sequence": frame.sequence,
        "image_path": os.fspath(image_path),
        "meta_path": os.fspath(meta_path),
    }
