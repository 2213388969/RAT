from __future__ import annotations

import base64
import hashlib
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
from pydantic import BaseModel, Field


STORAGE_ROOT = Path("data")
CONFIG_PATH = Path("config.json")

DEFAULT_REMOTE_CONFIG = {
    "poll_interval": 0.5,
    "webp_quality": 60,
    "anchor_top_ratio": 0.70,
    "anchor_bottom_ratio": 0.90,
    "search_top_ratio": 0.30,
    "search_bottom_ratio": 0.90,
    "match_confidence_threshold": 0.70,
    "scroll_threshold_ratio": 0.4,
    "no_scroll_capture_timeout": 5.0,
    "config_poll_interval": 30,
    "windows": [],
}

# Reconstruction constants
RECON_MATCH_THRESHOLD = 0.75
RECON_MIN_OVERLAP_RATIO = 0.10
RECON_MAX_OVERLAP_RATIO = 0.90


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class SessionCreate(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    software: str = Field(min_length=1, max_length=64)
    chat_name: str = Field(min_length=1, max_length=256)
    hwnd: int
    created_at: str


class FrameUpload(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    capture_id: str = Field(min_length=1, max_length=128)
    timestamp: float
    scroll_since_last: int = Field(ge=0)
    window_title: str = Field(min_length=1, max_length=512)
    software: str = Field(min_length=1, max_length=64)
    chat_name: str = Field(min_length=1, max_length=256)
    chat_rect: Optional[dict[str, int]] = None
    image_webp_base64: str = Field(min_length=0)


class ConfigUpdate(BaseModel):
    poll_interval: Optional[float] = None
    webp_quality: Optional[int] = None
    anchor_top_ratio: Optional[float] = None
    anchor_bottom_ratio: Optional[float] = None
    search_top_ratio: Optional[float] = None
    search_bottom_ratio: Optional[float] = None
    match_confidence_threshold: Optional[float] = None
    scroll_threshold_ratio: Optional[float] = None
    no_scroll_capture_timeout: Optional[float] = None
    config_poll_interval: Optional[int] = None
    windows: Optional[list[str]] = None


class TitleStripUpload(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    image_webp_base64: str = Field(min_length=0)


class WindowsUpdate(BaseModel):
    windows: list[str]


app = FastAPI(title="Chat Window Capture Archiver", version="1.0.0")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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


def _load_session_frames(session_id: str) -> list[tuple[Path, dict[str, Any]]]:
    """Load all frames for a session, sorted by sequence. Returns [(webp_path, meta_dict)]."""
    session_dir = STORAGE_ROOT / session_id
    if not session_dir.exists():
        return []
    frames = []
    for meta_path in sorted(session_dir.glob("*.json")):
        if meta_path.name == "session.json":
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        stem = meta_path.stem
        webp_path = session_dir / f"{stem}.webp"
        if webp_path.exists():
            frames.append((webp_path, meta))
    return frames


# ---------------------------------------------------------------------------
# 8. Server Reconstruction (frame stitching)
# ---------------------------------------------------------------------------
def _find_overlap_rows(prev_img: Image.Image, curr_img: Image.Image) -> Optional[int]:
    """Find how many rows at the bottom of prev_img overlap with the top of curr_img.

    Uses OpenCV matchTemplate: take bottom portion of prev as template,
    search in the top portion of curr.

    Returns the overlap height in pixels, or None if no reliable match found.
    """
    prev_arr = np.array(prev_img.convert("RGB"))
    curr_arr = np.array(curr_img.convert("RGB"))

    ph, pw = prev_arr.shape[:2]
    ch, cw = curr_arr.shape[:2]

    # Width must match (same chat region)
    if pw != cw:
        return None

    # Search for overlap in range [10%, 90%] of the shorter image height
    min_overlap = max(10, int(min(ph, ch) * RECON_MIN_OVERLAP_RATIO))
    max_overlap = int(min(ph, ch) * RECON_MAX_OVERLAP_RATIO)

    # Take bottom of prev as template, search in top of curr
    # Try from largest overlap to smallest, return first good match
    for overlap in range(max_overlap, min_overlap - 1, -1):
        template = prev_arr[ph - overlap : ph]
        search_region = curr_arr[0 : overlap]

        # Quick pixel-level comparison before expensive template matching
        diff = np.mean(np.abs(template.astype(float) - search_region.astype(float)))
        if diff < 10.0:
            return overlap

    # Fallback: use matchTemplate for more robust matching
    # Take bottom 30% of prev as template
    template_h = max(30, int(ph * 0.30))
    template = prev_arr[ph - template_h : ph]

    # Search in top 60% of curr
    search_h = max(template_h + 1, int(ch * 0.60))
    search_region = curr_arr[0:search_h]

    if search_region.shape[0] < template.shape[0] or search_region.shape[1] < template.shape[1]:
        return None

    result = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val < RECON_MATCH_THRESHOLD:
        return None

    # best_y is where the template was found in search_region
    best_y = max_loc[1]
    # overlap = template is at prev[ph-template_h:ph], found at curr[best_y:best_y+template_h]
    # The actual overlap = ph - (ph - template_h) mapped to curr means:
    # prev bottom starts at row (ph - template_h), curr top starts at row 0
    # If template is found at best_y in curr, then:
    # prev[ph-template_h:] == curr[best_y:best_y+template_h]
    # So overlap = ph - (ph - template_h) + best_y = template_h + best_y
    # Wait, let me think again...
    # prev bottom region: rows [ph-template_h, ph)
    # This matches curr rows [best_y, best_y+template_h)
    # So the overlap starts at: prev row (ph-template_h) == curr row best_y
    # Overlap height = ph - (ph - template_h) = template_h (the template itself)
    # But the full overlap extends further: prev rows below template_h also overlap
    # Actually: prev row X corresponds to curr row (X - ph + template_h + best_y)
    # For prev row (ph-1) to map to curr: curr_row = (ph-1) - ph + template_h + best_y = template_h + best_y - 1
    # For prev row (ph-template_h) to map to curr: curr_row = best_y
    # So overlap in prev = [ph-template_h, ph) = template_h rows
    # But there could be more overlap above the template match
    # Full overlap: prev rows that appear in curr starting from curr row 0
    # prev row (ph-template_h) == curr row best_y
    # So prev row (ph-template_h-best_y) == curr row 0
    # Full overlap from prev row (ph-template_h-best_y) to prev row (ph-1)
    # Full overlap height = template_h + best_y
    overlap = template_h + best_y
    if overlap < min_overlap or overlap > max_overlap:
        return None
    return overlap


def _stitch_frames(frames: list[tuple[Path, dict[str, Any]]]) -> Optional[Image.Image]:
    """Stitch a sequence of frames into a single tall image.

    For each pair (frame_n, frame_n+1):
    1. Find overlap rows using template matching
    2. Stack frame_n+1 below frame_n, skipping the overlapping top portion
    """
    if not frames:
        return None

    images = []
    for webp_path, _ in frames:
        img = Image.open(webp_path).convert("RGB")
        images.append(img)

    if len(images) == 1:
        return images[0]

    # Start with the first image
    result = images[0]
    overlaps = []

    for i in range(1, len(images)):
        overlap = _find_overlap_rows(result, images[i])
        if overlap is not None and overlap > 0:
            # Crop: take images[i] from row overlap onwards
            new_part = images[i].crop((0, overlap, images[i].width, images[i].height))
            # Stitch vertically
            new_height = result.height + new_part.height
            stitched = Image.new("RGB", (result.width, new_height))
            stitched.paste(result, (0, 0))
            stitched.paste(new_part, (0, result.height))
            result = stitched
            overlaps.append(overlap)
        else:
            # No overlap found, just append (with a separator gap)
            gap = 5
            new_height = result.height + gap + images[i].height
            stitched = Image.new("RGB", (result.width, new_height), (128, 128, 128))
            stitched.paste(result, (0, 0))
            stitched.paste(images[i], (0, result.height + gap))
            result = stitched
            overlaps.append(0)

    return result


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
@app.on_event("startup")
def startup() -> None:
    STORAGE_ROOT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "timestamp_utc": _utc_now()}


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------
@app.post("/api/session")
def create_session(session: SessionCreate) -> dict[str, Any]:
    session_id = _sanitize_session_id(session.session_id)
    session_dir = STORAGE_ROOT / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    meta_path = session_dir / "session.json"
    meta = {
        "session_id": session_id,
        "software": session.software,
        "chat_name": session.chat_name,
        "hwnd": session.hwnd,
        "created_at": session.created_at,
        "registered_at_utc": _utc_now(),
    }
    _write_json(meta_path, meta)

    return {"status": "registered", "session_id": session_id, "meta_path": os.fspath(meta_path)}


@app.get("/api/session/{session_id}")
def get_session(session_id: str) -> dict[str, Any]:
    session_id = _sanitize_session_id(session_id)
    meta_path = STORAGE_ROOT / session_id / "session.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="session not found")
    return json.loads(meta_path.read_text(encoding="utf-8"))


@app.post("/api/session/{session_id}/title")
def upload_title_strip(session_id: str, data: TitleStripUpload) -> dict[str, Any]:
    """Upload the 60px title strip image for a session."""
    session_id = _sanitize_session_id(session_id)
    session_dir = STORAGE_ROOT / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="session not found")

    image_bytes = _decode_image(data.image_webp_base64)
    title_path = session_dir / "title.webp"
    title_path.write_bytes(image_bytes)

    return {"status": "stored", "session_id": session_id, "title_path": os.fspath(title_path)}


@app.get("/api/sessions")
def list_sessions() -> dict[str, Any]:
    sessions = []
    if STORAGE_ROOT.exists():
        for d in sorted(STORAGE_ROOT.iterdir()):
            meta = d / "session.json"
            if meta.exists():
                sessions.append(json.loads(meta.read_text(encoding="utf-8")))
    return {"sessions": sessions, "count": len(sessions)}


# ---------------------------------------------------------------------------
# Frame upload
# ---------------------------------------------------------------------------
@app.post("/api/frame")
def upload_frame(frame: FrameUpload) -> dict[str, Any]:
    session_id = _sanitize_session_id(frame.session_id)
    session_dir = STORAGE_ROOT / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    # Determine sequence from existing frames
    existing = sorted(session_dir.glob("*.webp"))
    sequence = len(existing)

    image_bytes = _decode_image(frame.image_webp_base64)
    sha256 = hashlib.sha256(image_bytes).hexdigest()
    stem = f"{sequence:08d}_{sha256[:12]}"

    image_path = session_dir / f"{stem}.webp"
    meta_path = session_dir / f"{stem}.json"

    if image_path.exists():
        raise HTTPException(status_code=409, detail="duplicate frame upload")

    image_path.write_bytes(image_bytes)
    _write_json(
        meta_path,
        {
            "session_id": session_id,
            "capture_id": frame.capture_id,
            "sequence": sequence,
            "timestamp": frame.timestamp,
            "timestamp_utc": _utc_now(),
            "scroll_since_last": frame.scroll_since_last,
            "window_title": frame.window_title,
            "software": frame.software,
            "chat_name": frame.chat_name,
            "chat_rect": frame.chat_rect,
            "image_sha256": sha256,
            "image_size_bytes": len(image_bytes),
        },
    )

    return {
        "status": "stored",
        "session_id": session_id,
        "capture_id": frame.capture_id,
        "sequence": sequence,
        "image_path": os.fspath(image_path),
        "meta_path": os.fspath(meta_path),
    }


@app.get("/api/session/{session_id}/frames")
def list_frames(session_id: str) -> dict[str, Any]:
    session_id = _sanitize_session_id(session_id)
    session_dir = STORAGE_ROOT / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="session not found")

    frames = []
    for meta in sorted(session_dir.glob("*.json")):
        if meta.name == "session.json":
            continue
        frames.append(json.loads(meta.read_text(encoding="utf-8")))
    return {"session_id": session_id, "frames": frames, "count": len(frames)}


# ---------------------------------------------------------------------------
# 8. Server Reconstruction API
# ---------------------------------------------------------------------------
@app.get("/api/session/{session_id}/reconstruct")
def reconstruct_session(session_id: str) -> dict[str, Any]:
    """Reconstruct (stitch) all frames for a session and return metadata.

    The stitched image is saved to the session directory.
    """
    session_id = _sanitize_session_id(session_id)
    session_dir = STORAGE_ROOT / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="session not found")

    frames = _load_session_frames(session_id)
    if not frames:
        raise HTTPException(status_code=404, detail="no frames found for session")

    stitched = _stitch_frames(frames)
    if stitched is None:
        raise HTTPException(status_code=500, detail="stitching failed")

    # Save stitched image
    out_path = session_dir / "reconstructed.webp"
    stitched.save(out_path, format="WEBP", quality=90)

    return {
        "status": "reconstructed",
        "session_id": session_id,
        "frame_count": len(frames),
        "image_size": {"width": stitched.width, "height": stitched.height},
        "image_path": os.fspath(out_path),
    }


@app.get("/api/session/{session_id}/reconstruct/image")
def reconstruct_session_image(session_id: str) -> StreamingResponse:
    """Reconstruct and stream the stitched image directly."""
    session_id = _sanitize_session_id(session_id)
    session_dir = STORAGE_ROOT / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="session not found")

    frames = _load_session_frames(session_id)
    if not frames:
        raise HTTPException(status_code=404, detail="no frames found for session")

    stitched = _stitch_frames(frames)
    if stitched is None:
        raise HTTPException(status_code=500, detail="stitching failed")

    buf = io.BytesIO()
    stitched.save(buf, format="WEBP", quality=90)
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/webp")


@app.get("/api/session/{session_id}/reconstruct/overlap")
def reconstruct_overlap_info(session_id: str) -> dict[str, Any]:
    """Compute overlap information between consecutive frames without full stitching.

    Returns per-pair overlap data for debugging and verification.
    """
    session_id = _sanitize_session_id(session_id)
    session_dir = STORAGE_ROOT / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="session not found")

    frames = _load_session_frames(session_id)
    if not frames:
        raise HTTPException(status_code=404, detail="no frames found for session")

    overlaps = []
    images = [Image.open(p).convert("RGB") for p, _ in frames]

    for i in range(len(images) - 1):
        overlap = _find_overlap_rows(images[i], images[i + 1])
        overlaps.append({
            "frame_a": frames[i][1].get("sequence", i),
            "frame_b": frames[i + 1][1].get("sequence", i + 1),
            "overlap_rows": overlap,
            "frame_a_height": images[i].height,
            "frame_b_height": images[i + 1].height,
        })

    return {
        "session_id": session_id,
        "frame_count": len(frames),
        "overlaps": overlaps,
    }


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
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
