from __future__ import annotations

import base64
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


STORAGE_ROOT = Path("data")
CONFIG_PATH = Path("config.json")

DEFAULT_REMOTE_CONFIG = {
    "poll_interval": 0.35,
    "jpeg_quality": 60,
    "diff_threshold": 8.0,
    "chat_top_ratio": 0.10,
    "chat_bottom_ratio": 0.22,
    "chat_side_margin_ratio": 0.04,
    "area_threshold": 30000,
    "time_threshold": 3.0,
    "accumulation_ratio": 0.8,
    "dynamic_frames": 3,
    "dynamic_cooldown": 0.8,
    "dynamic_block_size": 64,
    "cursor_mask_size": 32,
    "text_cursor_max_area": 64,
    "dirty_rect_merge_gap": 16,
    "dirty_rect_min_size": 32,
    "dirty_rect_merge_window_ms": 120,
    "config_poll_interval": 30,
    "windows": [],
}


class FrameUpload(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    sequence: int = Field(ge=0)
    timestamp_utc: str
    window_title: str = Field(min_length=1, max_length=512)
    window_rect: Optional[dict[str, int]] = None
    screen_size: dict[str, int]
    changed_region: dict[str, int]
    image_jpeg_base64: str = Field(min_length=0)
    event: Optional[str] = None


class ConfigUpdate(BaseModel):
    poll_interval: Optional[float] = None
    jpeg_quality: Optional[int] = None
    diff_threshold: Optional[float] = None
    chat_top_ratio: Optional[float] = None
    chat_bottom_ratio: Optional[float] = None
    chat_side_margin_ratio: Optional[float] = None
    area_threshold: Optional[int] = None
    time_threshold: Optional[float] = None
    accumulation_ratio: Optional[float] = None
    dynamic_frames: Optional[int] = None
    dynamic_cooldown: Optional[float] = None
    dynamic_block_size: Optional[int] = None
    cursor_mask_size: Optional[int] = None
    text_cursor_max_area: Optional[int] = None
    dirty_rect_merge_gap: Optional[int] = None
    dirty_rect_min_size: Optional[int] = None
    dirty_rect_merge_window_ms: Optional[int] = None
    config_poll_interval: Optional[int] = None
    windows: Optional[list[str]] = None


class WindowsUpdate(BaseModel):
    windows: list[str]


app = FastAPI(title="Visible Capture Receiver", version="0.2.0")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitize_session_id(value: str) -> str:
    cleaned = "".join(ch for ch in value if ch.isalnum() or ch in ("-", "_"))
    return cleaned[:128] or "session"


def _decode_image(payload: str) -> bytes:
    try:
        return base64.b64decode(payload, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="invalid image payload") from exc


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _load_config() -> dict[str, Any]:
    if CONFIG_PATH.exists():
        try:
            saved = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            merged = dict(DEFAULT_REMOTE_CONFIG)
            merged.update(saved)
            return merged
        except Exception:
            pass
    return dict(DEFAULT_REMOTE_CONFIG)


def _save_config(config: dict[str, Any]) -> None:
    CONFIG_PATH.write_text(json.dumps(config, ensure_ascii=True, indent=2), encoding="utf-8")


@app.on_event("startup")
def startup() -> None:
    STORAGE_ROOT.mkdir(parents=True, exist_ok=True)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "timestamp_utc": _utc_now()}


@app.get("/api/config")
def get_config() -> dict[str, Any]:
    return _load_config()


@app.put("/api/config")
def update_config(update: ConfigUpdate) -> dict[str, Any]:
    config = _load_config()
    changes = {}
    for key, value in update.model_dump(exclude_none=True).items():
        if key in config:
            config[key] = value
            changes[key] = value
    _save_config(config)
    return {"status": "updated", "changes": changes, "config": config}


@app.get("/api/windows")
def get_windows() -> dict[str, Any]:
    config = _load_config()
    return {"windows": config.get("windows", [])}


@app.put("/api/windows")
def set_windows(update: WindowsUpdate) -> dict[str, Any]:
    config = _load_config()
    config["windows"] = update.windows
    _save_config(config)
    return {"status": "updated", "windows": update.windows}


@app.post("/api/frame")
def upload_frame(frame: FrameUpload) -> dict[str, Any]:
    is_position_update = frame.event == "window_moved" or not frame.image_jpeg_base64

    session_id = _sanitize_session_id(frame.session_id)
    session_dir = STORAGE_ROOT / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    if is_position_update:
        meta_path = session_dir / f"{frame.sequence:08d}_position.json"
        _write_json(
            meta_path,
            {
                "session_id": session_id,
                "sequence": frame.sequence,
                "timestamp_utc": frame.timestamp_utc,
                "received_at_utc": _utc_now(),
                "window_title": frame.window_title,
                "window_rect": frame.window_rect,
                "screen_size": frame.screen_size,
                "event": "window_moved",
            },
        )
        return {
            "status": "position_updated",
            "session_id": session_id,
            "sequence": frame.sequence,
            "meta_path": os.fspath(meta_path),
        }

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
            "window_rect": frame.window_rect,
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
