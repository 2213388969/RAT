from __future__ import annotations

import base64
import ctypes
import hashlib
import io
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import pygetwindow as gw
import requests
from mss import mss
from PIL import Image, ImageChops, ImageFilter, ImageOps, ImageStat

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("capture")

# ---------------------------------------------------------------------------
# Default configuration (can be overridden by remote config)
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "server_url": "http://127.0.0.1:8000",
    "poll_interval": 0.35,
    "jpeg_quality": 60,
    "diff_threshold": 8.0,
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

# Scroll detection constants
SCROLL_ANALYSIS_WIDTH = 256
SCROLL_MIN_SHIFT = 12
SCROLL_MAX_SHIFT = 240
SCROLL_OVERLAP_MIN_RATIO = 0.55
SCROLL_MATCH_THRESHOLD = 0.92
SCROLL_IMPROVEMENT_THRESHOLD = 0.005
SCROLL_REFINEMENT_WINDOW = 10
SCROLL_TIE_EPSILON = 0.002

# Overlap trimming constants
OVERLAP_SEARCH_WIDTH = 192
OVERLAP_MIN_HEIGHT = 48
OVERLAP_MAX_MEAN_DIFF = 5.0
OVERLAP_ACTIVE_PIXEL_THRESHOLD = 245
OVERLAP_ACTIVE_COLUMN_RATIO = 0.06
OVERLAP_MIN_WIDTH = 120

RESAMPLE_BILINEAR = getattr(Image, "Resampling", Image).BILINEAR

# ---------------------------------------------------------------------------
# Windows API structures and helpers
# ---------------------------------------------------------------------------
if hasattr(ctypes, "windll"):
    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32
    PW_RENDERFULLCONTENT = 0x00000002
else:
    user32 = None
    gdi32 = None
    PW_RENDERFULLCONTENT = 0


class RECT(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", ctypes.c_uint32),
        ("biWidth", ctypes.c_long),
        ("biHeight", ctypes.c_long),
        ("biPlanes", ctypes.c_ushort),
        ("biBitCount", ctypes.c_ushort),
        ("biCompression", ctypes.c_uint32),
        ("biSizeImage", ctypes.c_uint32),
        ("biXPelsPerMeter", ctypes.c_long),
        ("biYPelsPerMeter", ctypes.c_long),
        ("biClrUsed", ctypes.c_uint32),
        ("biClrImportant", ctypes.c_uint32),
    ]


class BITMAPINFO(ctypes.Structure):
    _fields_ = [
        ("bmiHeader", BITMAPINFOHEADER),
        ("bmiColors", ctypes.c_uint32 * 3),
    ]


class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


class CURSORINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("hCursor", ctypes.c_void_p),
        ("ptScreenPos", POINT),
    ]


class WINDOWPLACEMENT(ctypes.Structure):
    _fields_ = [
        ("length", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("showCmd", ctypes.c_uint32),
        ("ptMinPosition", POINT),
        ("ptMaxPosition", POINT),
        ("rcNormalPosition", RECT),
    ]

SW_SHOWMINIMIZED = 2


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def list_windows() -> list[str]:
    seen = []
    for title in gw.getAllTitles():
        cleaned = title.strip()
        if cleaned and cleaned not in seen:
            seen.append(cleaned)
    return seen


@dataclass
class WindowTarget:
    hwnd: int
    title: str
    left: int
    top: int
    width: int
    height: int


def resolve_window(title: str) -> Optional[WindowTarget]:
    matches = gw.getWindowsWithTitle(title)
    for win in matches:
        if not win.title.strip():
            continue
        hwnd = int(win._hWnd)

        # First try GetWindowRect for the actual current position/size
        # (correct for maximized/fullscreen windows)
        if user32 is not None:
            rect = RECT()
            if user32.GetWindowRect(hwnd, ctypes.byref(rect)):
                width = rect.right - rect.left
                height = rect.bottom - rect.top
                # GetWindowRect returns huge negative coords when minimized
                if width > 0 and height > 0 and rect.left > -10000:
                    return WindowTarget(
                        hwnd=hwnd,
                        title=win.title,
                        left=max(0, rect.left),
                        top=max(0, rect.top),
                        width=width,
                        height=height,
                    )

            # Fallback: GetWindowPlacement (works when minimized)
            wp = WINDOWPLACEMENT()
            wp.length = ctypes.sizeof(WINDOWPLACEMENT)
            if user32.GetWindowPlacement(hwnd, ctypes.byref(wp)):
                r = wp.rcNormalPosition
                width = r.right - r.left
                height = r.bottom - r.top
                if width > 0 and height > 0:
                    return WindowTarget(
                        hwnd=hwnd,
                        title=win.title,
                        left=max(0, r.left),
                        top=max(0, r.top),
                        width=width,
                        height=height,
                    )

        # Fallback to pygetwindow rect
        if win.width <= 0 or win.height <= 0:
            continue
        return WindowTarget(
            hwnd=hwnd,
            title=win.title,
            left=max(0, win.left),
            top=max(0, win.top),
            width=win.width,
            height=win.height,
        )
    return None


def capture_window(sct: mss, target: WindowTarget) -> Image.Image:
    bbox = {
        "left": target.left,
        "top": target.top,
        "width": target.width,
        "height": target.height,
    }
    shot = sct.grab(bbox)
    return Image.frombytes("RGB", shot.size, shot.rgb)


def capture_window_via_printwindow(target: WindowTarget) -> Optional[Image.Image]:
    if user32 is None or gdi32 is None:
        return None

    hwnd = target.hwnd

    # Get DPI scaling factor for this window
    dpi = user32.GetDpiForWindow(hwnd) if hasattr(user32, 'GetDpiForWindow') else 96
    dpi_scale = dpi / 96.0

    # Use target's stored dimensions (from GetWindowPlacement, in logical pixels)
    # and scale to physical pixels for the bitmap
    width = int(target.width * dpi_scale)
    height = int(target.height * dpi_scale)

    if width <= 0 or height <= 0:
        return None

    hwnd_dc = user32.GetDC(hwnd)
    if not hwnd_dc:
        return None

    mem_dc = gdi32.CreateCompatibleDC(hwnd_dc)
    bitmap = gdi32.CreateCompatibleBitmap(hwnd_dc, width, height)
    old_bitmap = gdi32.SelectObject(mem_dc, bitmap)

    try:
        flags = PW_RENDERFULLCONTENT
        result = user32.PrintWindow(target.hwnd, mem_dc, flags)
        if result != 1:
            return None

        bmi = BITMAPINFO()
        bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.bmiHeader.biWidth = width
        bmi.bmiHeader.biHeight = -height
        bmi.bmiHeader.biPlanes = 1
        bmi.bmiHeader.biBitCount = 32
        bmi.bmiHeader.biCompression = 0

        buffer_len = width * height * 4
        pixel_buffer = ctypes.create_string_buffer(buffer_len)
        rows = gdi32.GetDIBits(
            mem_dc, bitmap, 0, height, pixel_buffer, ctypes.byref(bmi), 0,
        )
        if rows != height:
            return None

        return Image.frombuffer("RGB", (width, height), pixel_buffer, "raw", "BGRX", 0, 1).copy()
    finally:
        gdi32.SelectObject(mem_dc, old_bitmap)
        gdi32.DeleteObject(bitmap)
        gdi32.DeleteDC(mem_dc)
        user32.ReleaseDC(target.hwnd, hwnd_dc)


def _is_black_image(image: Image.Image, threshold: float = 5.0) -> bool:
    """Check if an image is mostly black (average brightness below threshold)."""
    stat = ImageStat.Stat(image.convert("L"))
    return stat.mean[0] < threshold


def capture_window_prefer_window_api(
    sct: mss, target: WindowTarget,
) -> tuple[Image.Image, str]:
    image = capture_window_via_printwindow(target)
    if image is not None and not _is_black_image(image):
        return image, "window"
    return capture_window(sct, target), "screen"


# ---------------------------------------------------------------------------
# Chat region detection (layout analysis)
# ---------------------------------------------------------------------------
@dataclass
class ChatRegion:
    """Detected chat region within a window (header + messages, excluding sidebar and input box)."""
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


def detect_chat_region(image: Image.Image) -> ChatRegion:
    """Detect the chat region (header + messages) within a WeChat-style window.

    Uses layout analysis:
    1. Find vertical divider between contact list and chat area
    2. Find horizontal divider between messages and input box
    3. Chat region = right panel minus input box

    The algorithm is adaptive — it searches the full width/height rather
    than relying on fixed ratio ranges, so it works even when the sidebar
    is unusually wide or the input box is unusually tall.
    """
    import numpy as np

    w, h = image.size
    arr = np.array(image.convert("RGB"))
    gray = np.mean(arr, axis=2)

    # --- 1. Find vertical divider (left panel | right panel) ---
    # Strategy: find the first column where average brightness reaches
    # "white" level and stays there. The chat area has a white background
    # while the sidebar is darker gray. Search the full width.
    col_avg = np.mean(gray, axis=0)

    # Smooth column averages to reduce noise from text/icons
    kernel = max(5, w // 40)
    if kernel % 2 == 0:
        kernel += 1
    smooth_col = np.convolve(col_avg, np.ones(kernel) / kernel, mode='same')

    # Find the first column where brightness exceeds the white threshold
    # and is sustained for at least 20 columns. Skip the first 3% (window border).
    white_threshold = 235
    min_col = max(5, int(w * 0.03))

    divider_col = 0
    for c in range(min_col, w - 20):
        if smooth_col[c] >= white_threshold:
            if np.mean(smooth_col[c:c + 20]) >= white_threshold - 3:
                divider_col = c
                break

    if divider_col == 0:
        # Fallback: find the column with the biggest brightness jump
        # where right side is brighter (sidebar→chat transition)
        best_col = min_col
        best_diff = 0
        for c in range(min_col, w - 20):
            left_b = np.mean(smooth_col[max(0, c - 20):c])
            right_b = np.mean(smooth_col[c:c + 20])
            diff = right_b - left_b
            if diff > best_diff:
                best_diff = diff
                best_col = c
        divider_col = best_col

    # --- 2. Find horizontal divider (messages | input box) ---
    # Strategy: scan top-to-bottom and find the FIRST separator line where
    # the area above is bright (white message area). This is the top of the
    # input box. We pick the first match rather than the strongest, because
    # when the input box is very tall, the bottom border might have a bigger
    # gap but we want the top border.
    #
    # Key constraint: the input box is at most 270px tall, so the separator
    # must be within 270px from the bottom of the window.
    MAX_INPUT_BOX_HEIGHT = 270

    right_panel = gray[:, divider_col:]
    row_avg = np.mean(right_panel, axis=1)

    # Search range: input box top must be at least 10% from top (skip header)
    # and at most 270px from bottom (input box height limit)
    scan_top = max(10, int(h * 0.10))
    scan_bottom = h - 5

    # Collect all candidate separator rows (darker than above AND below, above is bright)
    candidates = []
    for r in range(scan_top, scan_bottom):
        above = np.mean(row_avg[max(0, r - 5):r])
        below = np.mean(row_avg[r + 1:r + 6])
        curr = row_avg[r]
        if curr < above - 3 and curr < below - 3 and above > 230:
            gap = min(above - curr, below - curr)
            if gap > 3:
                candidates.append((r, gap))

    input_top_row = 0
    if candidates:
        # Pick the FIRST candidate where the remaining height to bottom
        # is <= MAX_INPUT_BOX_HEIGHT (i.e., this separator is the input box top)
        for r, gap in candidates:
            remaining = h - r
            if gap > 5 and remaining <= MAX_INPUT_BOX_HEIGHT + 20:
                input_top_row = r
                break
        # If no candidate within input box height limit, take the first one
        if input_top_row == 0 and candidates:
            input_top_row = candidates[0][0]

    if input_top_row == 0:
        # Fallback: find the first significant brightness drop from bright area
        # within the input box height constraint
        best_drop = 0
        best_r = 0
        for r in range(max(scan_top, h - MAX_INPUT_BOX_HEIGHT - 50), h - 5):
            above = np.mean(row_avg[max(0, r - 10):r])
            below = np.mean(row_avg[r:r + 10])
            drop = above - below
            if drop > best_drop and above > 230:
                best_drop = drop
                best_r = r
        if best_drop > 5:
            input_top_row = best_r
        else:
            input_top_row = h - MAX_INPUT_BOX_HEIGHT

    # --- 3. Find chat header bottom (header | messages) ---
    # Strategy: scan from top-down in the right panel. Look for the first
    # row where brightness becomes consistently high (white message bg).
    header_bottom = 0
    scan_header_top = max(2, int(h * 0.01))  # skip window border
    scan_header_bottom = min(input_top_row - 10, int(h * 0.30))

    for r in range(scan_header_top, scan_header_bottom):
        above = np.mean(row_avg[max(0, r - 3):r])
        below = np.mean(row_avg[r:r + 3])
        # Header bottom: brightness jumps up (entering white message area)
        if below > above + 5 and below > 235:
            header_bottom = r
            break

    if header_bottom == 0:
        header_bottom = max(2, int(h * 0.04))

    return ChatRegion(
        left=int(divider_col),
        top=int(header_bottom),
        right=int(w),
        bottom=int(input_top_row),
    )


def get_cursor_pos() -> Optional[tuple[int, int]]:
    if user32 is None:
        return None
    ci = CURSORINFO()
    ci.cbSize = ctypes.sizeof(CURSORINFO)
    if not user32.GetCursorInfo(ctypes.byref(ci)):
        return None
    if ci.flags == 0:
        return None
    return ci.ptScreenPos.x, ci.ptScreenPos.y


# ---------------------------------------------------------------------------
# Image diff and scroll detection (preserved from original)
# ---------------------------------------------------------------------------
def diff_bbox(
    previous: Image.Image, current: Image.Image, threshold: float = 8.0,
) -> Optional[tuple[int, int, int, int]]:
    diff = ImageChops.difference(previous, current)
    bbox = diff.getbbox()
    if not bbox:
        return None
    stat = ImageStat.Stat(diff.crop(bbox).convert("L"))
    if stat.mean[0] < threshold:
        return None
    return bbox


def chat_content_bbox(
    image: Image.Image,
    top_ratio: float = 0.10,
    bottom_ratio: float = 0.22,
    side_margin_ratio: float = 0.04,
) -> tuple[int, int, int, int]:
    side_margin = max(8, int(image.width * side_margin_ratio))
    top = max(0, int(image.height * top_ratio))
    bottom = min(image.height, image.height - int(image.height * bottom_ratio))
    left = min(side_margin, max(0, image.width - 1))
    right = max(left + 1, image.width - side_margin)
    if bottom <= top:
        top = 0
        bottom = image.height
    return left, top, right, bottom


def _similarity_score(previous: Image.Image, current: Image.Image) -> float:
    diff = ImageChops.difference(previous, current)
    stat = ImageStat.Stat(diff)
    return max(0.0, 1.0 - (stat.mean[0] / 255.0))


def _prepare_scroll_analysis_roi(
    image: Image.Image, top_ratio: float, bottom_ratio: float, side_margin_ratio: float,
) -> tuple[Image.Image, float]:
    left, top, right, bottom = chat_content_bbox(image, top_ratio, bottom_ratio, side_margin_ratio)
    roi = image.crop((left, top, right, bottom)).convert("L")
    if roi.width <= 0 or roi.height <= 0:
        return roi, 1.0
    if roi.width <= SCROLL_ANALYSIS_WIDTH:
        return roi, 1.0
    scale = SCROLL_ANALYSIS_WIDTH / roi.width
    scaled_height = max(32, int(roi.height * scale))
    resized = roi.resize((SCROLL_ANALYSIS_WIDTH, scaled_height), RESAMPLE_BILINEAR)
    return resized, scale


def _scroll_signature(image: Image.Image) -> list[float]:
    edge = image.filter(ImageFilter.FIND_EDGES)
    ink = ImageOps.invert(image)
    edge_column = list(edge.resize((1, image.height), RESAMPLE_BILINEAR).getdata())
    ink_column = list(ink.resize((1, image.height), RESAMPLE_BILINEAR).getdata())
    return [(0.8 * e) + (0.2 * i) for e, i in zip(edge_column, ink_column)]


def _signature_similarity(prev: list[float], curr: list[float], shift: int) -> float:
    if shift > 0:
        overlap = len(prev) - shift
        if overlap <= 0:
            return 0.0
        p, c = prev[shift:], curr[:overlap]
    elif shift < 0:
        overlap = len(prev) + shift
        if overlap <= 0:
            return 0.0
        p, c = prev[:overlap], curr[-shift:]
    else:
        p, c = prev, curr
    mean_abs_diff = sum(abs(a - b) for a, b in zip(p, c)) / len(p)
    return max(0.0, 1.0 - (mean_abs_diff / 255.0))


def _best_scroll_shift_from_signature(
    prev_sig: list[float], curr_sig: list[float], shifts: list[int],
) -> tuple[int, float]:
    best_shift, best_score = 0, 0.0
    for shift in shifts:
        if shift == 0:
            continue
        score = _signature_similarity(prev_sig, curr_sig, shift)
        if score > best_score + SCROLL_TIE_EPSILON:
            best_score = score
            best_shift = shift
        elif abs(score - best_score) <= SCROLL_TIE_EPSILON and best_shift != 0:
            if abs(shift) < abs(best_shift):
                best_score = score
                best_shift = shift
    return best_shift, best_score


def _overlap_similarity(prev: Image.Image, curr: Image.Image, shift: int) -> float:
    w, h = prev.size
    if shift > 0:
        oh = h - shift
        if oh <= 0:
            return 0.0
        return _similarity_score(prev.crop((0, shift, w, h)), curr.crop((0, 0, w, oh)))
    elif shift < 0:
        oh = h + shift
        if oh <= 0:
            return 0.0
        return _similarity_score(prev.crop((0, 0, w, oh)), curr.crop((0, -shift, w, h)))
    return _similarity_score(prev, curr)


def _best_scroll_shift(
    prev: Image.Image, curr: Image.Image, shifts: range,
) -> tuple[int, float]:
    best_shift, best_score = 0, 0.0
    for shift in shifts:
        if shift == 0:
            continue
        score = _overlap_similarity(prev, curr, shift)
        if score > best_score + SCROLL_TIE_EPSILON:
            best_score = score
            best_shift = shift
        elif abs(score - best_score) <= SCROLL_TIE_EPSILON and best_shift != 0:
            if abs(shift) < abs(best_shift):
                best_score = score
                best_shift = shift
    return best_shift, best_score


def _scroll_shift_candidates(min_shift: int, max_shift: int) -> list[int]:
    return [*range(-max_shift, -min_shift + 1), *range(min_shift, max_shift + 1)]


def detect_scroll_shift(
    previous: Image.Image,
    current: Image.Image,
    top_ratio: float = 0.10,
    bottom_ratio: float = 0.22,
    side_margin_ratio: float = 0.04,
) -> Optional[int]:
    prev_small, scale = _prepare_scroll_analysis_roi(previous, top_ratio, bottom_ratio, side_margin_ratio)
    curr_small, _ = _prepare_scroll_analysis_roi(current, top_ratio, bottom_ratio, side_margin_ratio)
    if prev_small.size != curr_small.size:
        return None
    w, h = prev_small.size
    if w <= 0 or h <= 0:
        return None
    min_shift = max(2, int(SCROLL_MIN_SHIFT * scale))
    max_shift = min(int(SCROLL_MAX_SHIFT * scale), int(h * (1.0 - SCROLL_OVERLAP_MIN_RATIO)))
    if max_shift < min_shift:
        return None

    prev_sig = _scroll_signature(prev_small)
    curr_sig = _scroll_signature(curr_small)
    stationary_score = _signature_similarity(prev_sig, curr_sig, 0)
    best_shift_small, best_score = _best_scroll_shift_from_signature(
        prev_sig, curr_sig, _scroll_shift_candidates(min_shift, max_shift),
    )
    if best_shift_small == 0 or best_score < SCROLL_MATCH_THRESHOLD:
        return None
    if (best_score - stationary_score) < SCROLL_IMPROVEMENT_THRESHOLD:
        return None

    left, top, right, bottom = chat_content_bbox(previous, top_ratio, bottom_ratio, side_margin_ratio)
    prev_roi = previous.crop((left, top, right, bottom)).convert("L")
    curr_roi = current.crop((left, top, right, bottom)).convert("L")
    coarse_shift = int(round(best_shift_small / scale))
    refine_min = max(-SCROLL_MAX_SHIFT, coarse_shift - SCROLL_REFINEMENT_WINDOW)
    refine_max = min(SCROLL_MAX_SHIFT, coarse_shift + SCROLL_REFINEMENT_WINDOW)
    if refine_max < refine_min:
        return None

    refined_shift, refined_score = _best_scroll_shift(prev_roi, curr_roi, range(refine_min, refine_max + 1))
    if refined_shift == 0 or refined_score < SCROLL_MATCH_THRESHOLD:
        return None
    if (refined_score - _similarity_score(prev_roi, curr_roi)) < SCROLL_IMPROVEMENT_THRESHOLD:
        return None
    if abs(refined_shift) < SCROLL_MIN_SHIFT:
        return None
    return refined_shift


def detect_incremental_chat_bbox(
    previous: Image.Image,
    current: Image.Image,
    diff_threshold: float = 8.0,
    top_ratio: float = 0.10,
    bottom_ratio: float = 0.22,
    side_margin_ratio: float = 0.04,
) -> Optional[tuple[int, int, int, int]]:
    scroll_shift = detect_scroll_shift(previous, current, top_ratio, bottom_ratio, side_margin_ratio)
    if scroll_shift is None:
        return diff_bbox(previous, current, diff_threshold)
    left, top, right, bottom = chat_content_bbox(current, top_ratio, bottom_ratio, side_margin_ratio)
    if scroll_shift > 0:
        band_top = max(top, bottom - scroll_shift)
        if band_top >= bottom:
            return None
        return (left, band_top, right, bottom)
    band_bottom = min(bottom, top - scroll_shift)
    if band_bottom <= top:
        return None
    return (left, top, right, band_bottom)


# ---------------------------------------------------------------------------
# Overlap trimming (preserved from original)
# ---------------------------------------------------------------------------
def _active_column_bounds(images: list[Image.Image]) -> tuple[int, int]:
    width = min(img.width for img in images)
    height = min(img.height for img in images)
    threshold = max(8, int(height * OVERLAP_ACTIVE_COLUMN_RATIO))
    combined = [0] * width
    for img in images:
        gray = img.convert("L")
        pixels = gray.load()
        for x in range(width):
            count = sum(1 for y in range(height) if pixels[x, y] < OVERLAP_ACTIVE_PIXEL_THRESHOLD)
            combined[x] = max(combined[x], count)
    active = [i for i, c in enumerate(combined) if c >= threshold]
    if not active:
        return 0, width
    return min(active), max(active) + 1


def trim_patch_overlap(
    prev_patch: Image.Image, prev_bbox: tuple[int, int, int, int],
    curr_patch: Image.Image, curr_bbox: tuple[int, int, int, int],
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    pl, pt, pr, pb = prev_bbox
    cl, ct, cr, cb = curr_bbox
    ol = max(pl, cl)
    or_ = min(pr, cr)
    ow = or_ - ol
    if ow < OVERLAP_MIN_WIDTH:
        return curr_patch, curr_bbox

    prev_crop = prev_patch.crop((ol - pl, 0, or_ - pl, prev_patch.height))
    curr_crop = curr_patch.crop((ol - cl, 0, or_ - cl, curr_patch.height))
    if prev_crop.width <= 0 or curr_crop.width <= 0:
        return curr_patch, curr_bbox

    al, ar = _active_column_bounds([prev_crop, curr_crop])
    prev_crop = prev_crop.crop((al, 0, ar, prev_crop.height))
    curr_crop = curr_crop.crop((al, 0, ar, curr_crop.height))
    if prev_crop.width < OVERLAP_MIN_WIDTH or curr_crop.width < OVERLAP_MIN_WIDTH:
        return curr_patch, curr_bbox

    scale = 1.0
    if prev_crop.width > OVERLAP_SEARCH_WIDTH:
        scale = OVERLAP_SEARCH_WIDTH / prev_crop.width
        prev_crop = prev_crop.resize(
            (OVERLAP_SEARCH_WIDTH, max(24, int(prev_crop.height * scale))), RESAMPLE_BILINEAR,
        )
        curr_crop = curr_crop.resize(
            (OVERLAP_SEARCH_WIDTH, max(24, int(curr_crop.height * scale))), RESAMPLE_BILINEAR,
        )

    prev_gray = prev_crop.convert("L")
    curr_gray = curr_crop.convert("L")
    min_overlap = max(8, int(OVERLAP_MIN_HEIGHT * scale))
    max_overlap = min(prev_gray.height, curr_gray.height)
    best_overlap = 0

    for overlap in range(max_overlap, min_overlap - 1, -1):
        prev_part = prev_gray.crop((0, prev_gray.height - overlap, prev_gray.width, prev_gray.height))
        curr_part = curr_gray.crop((0, 0, curr_gray.width, overlap))
        mean_diff = ImageStat.Stat(ImageChops.difference(prev_part, curr_part)).mean[0]
        if mean_diff <= OVERLAP_MAX_MEAN_DIFF:
            best_overlap = overlap
            break

    if best_overlap == 0:
        return curr_patch, curr_bbox
    trim_top = int(round(best_overlap / scale))
    if trim_top <= 0 or trim_top >= curr_patch.height:
        return curr_patch, curr_bbox

    trimmed_patch = curr_patch.crop((0, trim_top, curr_patch.width, curr_patch.height))
    trimmed_bbox = (cl, ct + trim_top, cr, cb)
    return trimmed_patch, trimmed_bbox


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------
def encode_jpeg(image: Image.Image, quality: int = 60) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def block_hash(image: Image.Image, x: int, y: int, block_size: int) -> str:
    block = image.crop((x, y, min(x + block_size, image.width), min(y + block_size, image.height)))
    return hashlib.md5(block.tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# Dynamic content filter (3-frame hash state machine)
# ---------------------------------------------------------------------------
@dataclass
class DynamicBlock:
    x: int
    y: int
    hashes: list[str] = field(default_factory=list)
    is_dynamic: bool = False
    first_frame_sent: bool = False
    last_change_time: float = 0.0


class DynamicFilter:
    def __init__(self, block_size: int = 64, dynamic_frames: int = 3, cooldown: float = 0.8):
        self.block_size = block_size
        self.dynamic_frames = dynamic_frames
        self.cooldown = cooldown
        self.blocks: dict[tuple[int, int], DynamicBlock] = {}

    def update_and_filter(
        self, image: Image.Image, dirty_bbox: tuple[int, int, int, int],
    ) -> Optional[tuple[int, int, int, int]]:
        left, top, right, bottom = dirty_bbox
        bs = self.block_size
        now = time.monotonic()

        dynamic_rects: list[tuple[int, int, int, int]] = []
        stable_rects: list[tuple[int, int, int, int]] = []

        bx_start = (left // bs) * bs
        by_start = (top // bs) * bs

        for bx in range(bx_start, right, bs):
            for by in range(by_start, bottom, bs):
                key = (bx, by)
                h = block_hash(image, bx, by, bs)

                if key not in self.blocks:
                    self.blocks[key] = DynamicBlock(x=bx, y=by)
                block = self.blocks[key]

                if block.is_dynamic:
                    if now - block.last_change_time > self.cooldown:
                        block.is_dynamic = False
                        block.hashes = [h]
                    else:
                        block.hashes.append(h)
                        if len(block.hashes) > self.dynamic_frames:
                            block.hashes = block.hashes[-self.dynamic_frames:]
                        block.last_change_time = now
                        dynamic_rects.append((bx, by, bx + bs, by + bs))
                        continue

                block.hashes.append(h)
                if len(block.hashes) > self.dynamic_frames:
                    block.hashes = block.hashes[-self.dynamic_frames:]

                if len(block.hashes) >= self.dynamic_frames and len(set(block.hashes)) >= self.dynamic_frames:
                    block.is_dynamic = True
                    block.last_change_time = now
                    if not block.first_frame_sent:
                        block.first_frame_sent = True
                        stable_rects.append((bx, by, bx + bs, by + bs))
                    else:
                        dynamic_rects.append((bx, by, bx + bs, by + bs))
                else:
                    block.last_change_time = now
                    stable_rects.append((bx, by, bx + bs, by + bs))

        if not stable_rects:
            return None

        merged = _merge_rects(stable_rects)
        if not merged:
            return None

        ml = min(r[0] for r in merged)
        mt = min(r[1] for r in merged)
        mr = max(r[2] for r in merged)
        mb = max(r[3] for r in merged)
        ml = max(ml, left)
        mt = max(mt, top)
        mr = min(mr, right)
        mb = min(mb, bottom)
        if mr <= ml or mb <= mt:
            return None
        return (ml, mt, mr, mb)


def _merge_rects(rects: list[tuple[int, int, int, int]], gap: int = 16) -> list[tuple[int, int, int, int]]:
    if not rects:
        return []
    merged = list(rects)
    changed = True
    while changed:
        changed = False
        result = []
        used = [False] * len(merged)
        for i in range(len(merged)):
            if used[i]:
                continue
            r1 = merged[i]
            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                r2 = merged[j]
                if (r1[0] - gap <= r2[2] and r2[0] - gap <= r1[2] and
                        r1[1] - gap <= r2[3] and r2[1] - gap <= r1[3]):
                    r1 = (min(r1[0], r2[0]), min(r1[1], r2[1]),
                          max(r1[2], r2[2]), max(r1[3], r2[3]))
                    used[j] = True
                    changed = True
            result.append(r1)
        merged = result
    return merged


# ---------------------------------------------------------------------------
# Dirty rect buffer (merge nearby rects within time window)
# ---------------------------------------------------------------------------
@dataclass
class PendingRect:
    bbox: tuple[int, int, int, int]
    time: float


class DirtyRectBuffer:
    def __init__(self, merge_gap: int = 16, merge_window_ms: int = 120, min_size: int = 32):
        self.merge_gap = merge_gap
        self.merge_window_ms = merge_window_ms
        self.min_size = min_size
        self.pending: list[PendingRect] = []

    def add(self, bbox: tuple[int, int, int, int]) -> None:
        self.pending.append(PendingRect(bbox=bbox, time=time.monotonic()))

    def flush_ready(self) -> list[tuple[int, int, int, int]]:
        if not self.pending:
            return []
        now = time.monotonic()
        ready = [p for p in self.pending if (now - p.time) * 1000 >= self.merge_window_ms]
        if not ready:
            return []
        self.pending = [p for p in self.pending if (now - p.time) * 1000 < self.merge_window_ms]
        merged = _merge_rects([p.bbox for p in ready], self.merge_gap)
        return [r for r in merged if (r[2] - r[0]) >= self.min_size and (r[3] - r[1]) >= self.min_size]

    def force_flush(self) -> list[tuple[int, int, int, int]]:
        if not self.pending:
            return []
        result = list(self.pending)
        self.pending.clear()
        merged = _merge_rects([p.bbox for p in result], self.merge_gap)
        return [r for r in merged if (r[2] - r[0]) >= self.min_size and (r[3] - r[1]) >= self.min_size]


# ---------------------------------------------------------------------------
# Per-window capture state
# ---------------------------------------------------------------------------
@dataclass
class WindowCaptureState:
    window_title: str
    target: Optional[WindowTarget] = None
    previous_frame: Optional[Image.Image] = None
    last_uploaded_patch: Optional[Image.Image] = None
    last_uploaded_bbox: Optional[tuple[int, int, int, int]] = None
    last_window_rect: Optional[tuple[int, int, int, int]] = None
    sequence: int = 0
    pending_height: int = 0
    last_upload_time: float = 0.0
    dynamic_filter: DynamicFilter = field(default_factory=DynamicFilter)
    dirty_buffer: DirtyRectBuffer = field(default_factory=DirtyRectBuffer)
    capture_mode: str = "unknown"
    chat_region: Optional[ChatRegion] = None


# ---------------------------------------------------------------------------
# Multi-window capture manager
# ---------------------------------------------------------------------------
class CaptureManager:
    def __init__(self, config: dict) -> None:
        self.config = dict(DEFAULT_CONFIG)
        self.config.update(config)
        self.session_id = str(uuid.uuid4())
        self.states: dict[str, WindowCaptureState] = {}
        self.stop_event = threading.Event()
        self.lock = threading.Lock()

    def _cfg(self, key: str):
        return self.config.get(key, DEFAULT_CONFIG.get(key))

    def _server_url(self, path: str) -> str:
        base = self._cfg("server_url").rstrip("/")
        return f"{base}{path}"

    def _update_windows(self) -> None:
        wanted = self._cfg("windows")
        if not wanted:
            return
        with self.lock:
            for title in wanted:
                if title not in self.states:
                    self.states[title] = WindowCaptureState(window_title=title)
            for title in list(self.states.keys()):
                if title not in wanted:
                    del self.states[title]

    def _resolve_target(self, state: WindowCaptureState) -> Optional[WindowTarget]:
        target = resolve_window(state.window_title)
        state.target = target
        return target

    def _check_window_moved(self, state: WindowCaptureState) -> bool:
        if state.target is None:
            return False
        rect = (state.target.left, state.target.top, state.target.width, state.target.height)
        if state.last_window_rect is None:
            state.last_window_rect = rect
            return False
        moved = rect != state.last_window_rect
        state.last_window_rect = rect
        return moved

    def _apply_cursor_mask(
        self, image: Image.Image, target: WindowTarget, bbox: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int]:
        cursor_pos = get_cursor_pos()
        if cursor_pos is None:
            return bbox
        cx, cy = cursor_pos
        mask_size = self._cfg("cursor_mask_size")
        cursor_left = cx - mask_size // 2 - target.left
        cursor_top = cy - mask_size // 2 - target.top
        cursor_right = cursor_left + mask_size
        cursor_bottom = cursor_top + mask_size

        bl, bt, br, bb = bbox
        if cursor_right <= bl or cursor_left >= br or cursor_bottom <= bt or cursor_top >= bb:
            return bbox

        cl = max(cursor_left, bl)
        ct = max(cursor_top, bt)
        cr = min(cursor_right, br)
        cb = min(cursor_bottom, bb)
        area = (cr - cl) * (cb - ct)
        text_cursor_max = self._cfg("text_cursor_max_area")
        if area <= text_cursor_max:
            return bbox

        if cl - bl >= self._cfg("dirty_rect_min_size") and bb - bt >= self._cfg("dirty_rect_min_size"):
            return (bl, bt, cl, bb)
        return bbox

    def _should_upload(
        self, state: WindowCaptureState, patch_height: int, chat_roi_height: int,
    ) -> bool:
        state.pending_height += patch_height
        accum_ratio = self._cfg("accumulation_ratio")
        area_threshold = self._cfg("area_threshold")
        time_threshold = self._cfg("time_threshold")

        if state.pending_height >= chat_roi_height * accum_ratio:
            return True
        if state.pending_height * patch_height >= area_threshold:
            return True
        now = time.monotonic()
        if (now - state.last_upload_time) >= time_threshold and state.pending_height > 0:
            return True
        return False

    def _upload_patch(
        self, state: WindowCaptureState, target: WindowTarget,
        patch: Image.Image, bbox: tuple[int, int, int, int],
    ) -> None:
        left, top, right, bottom = bbox
        payload = {
            "session_id": self.session_id,
            "sequence": state.sequence,
            "timestamp_utc": utc_now(),
            "window_title": target.title,
            "window_rect": {
                "left": target.left, "top": target.top,
                "width": target.width, "height": target.height,
            },
            "screen_size": {"width": target.width, "height": target.height},
            "changed_region": {
                "left": left, "top": top,
                "width": right - left, "height": bottom - top,
            },
            "image_jpeg_base64": encode_jpeg(patch, self._cfg("jpeg_quality")),
        }
        try:
            resp = requests.post(
                self._server_url("/api/frame"), json=payload, timeout=10,
            )
            resp.raise_for_status()
            state.sequence += 1
            state.last_upload_time = time.monotonic()
            state.pending_height = 0
            log.info("Uploaded #%d from '%s' (%dx%d)", state.sequence - 1, target.title, right - left, bottom - top)
        except Exception as exc:
            log.warning("Upload failed for '%s': %s", target.title, exc)

    def _upload_position_update(self, state: WindowCaptureState, target: WindowTarget) -> None:
        payload = {
            "session_id": self.session_id,
            "sequence": state.sequence,
            "timestamp_utc": utc_now(),
            "window_title": target.title,
            "window_rect": {
                "left": target.left, "top": target.top,
                "width": target.width, "height": target.height,
            },
            "screen_size": {"width": target.width, "height": target.height},
            "changed_region": {"left": 0, "top": 0, "width": 0, "height": 0},
            "image_jpeg_base64": "",
            "event": "window_moved",
        }
        try:
            requests.post(self._server_url("/api/frame"), json=payload, timeout=10)
            state.sequence += 1
            log.debug("Position update for '%s'", target.title)
        except Exception:
            pass

    def _process_window(self, sct: mss, state: WindowCaptureState) -> None:
        target = self._resolve_target(state)
        if target is None:
            log.debug("Window '%s' not found or minimized", state.window_title)
            return

        window_moved = self._check_window_moved(state)
        current_frame, capture_mode = capture_window_prefer_window_api(sct, target)
        state.capture_mode = capture_mode

        # Detect chat region on first frame or when window moved
        if state.chat_region is None or window_moved:
            state.chat_region = detect_chat_region(current_frame)
            log.info("Chat region for '%s': left=%d top=%d right=%d bottom=%d (%dx%d)",
                     target.title, state.chat_region.left, state.chat_region.top,
                     state.chat_region.right, state.chat_region.bottom,
                     state.chat_region.width, state.chat_region.height)

        # Crop to chat region for incremental detection
        cr = state.chat_region
        chat_frame = current_frame.crop((cr.left, cr.top, cr.right, cr.bottom))

        if state.previous_frame is None:
            state.previous_frame = chat_frame
            state.last_upload_time = time.monotonic()
            self._upload_patch(state, target, chat_frame, (cr.left, cr.top, cr.right, cr.bottom))
            state.last_uploaded_patch = chat_frame.copy()
            state.last_uploaded_bbox = (cr.left, cr.top, cr.right, cr.bottom)
            return

        changed = detect_incremental_chat_bbox(
            state.previous_frame, chat_frame,
            diff_threshold=self._cfg("diff_threshold"),
            top_ratio=0.0,  # Already cropped to chat region, no need to skip top
            bottom_ratio=0.0,  # Already cropped, no need to skip bottom
            side_margin_ratio=0.0,  # Already cropped, no need to skip sides
        )
        state.previous_frame = chat_frame

        if window_moved and changed is None:
            self._upload_position_update(state, target)
            return

        if changed is None:
            return

        # Map changed coordinates back to full window
        cl, ct, cr2, cb = changed
        changed = (cl + cr.left, ct + cr.top, cr2 + cr.left, cb + cr.top)

        # Clamp changed region strictly within chat region
        left, top, right, bottom = changed
        left = max(left, cr.left)
        top = max(top, cr.top)
        right = min(right, cr.right)
        bottom = min(bottom, cr.bottom)
        changed = (left, top, right, bottom)

        changed = self._apply_cursor_mask(current_frame, target, changed)
        if changed is None:
            return

        left, top, right, bottom = changed
        # Clamp again after cursor mask (mask may have expanded the rect)
        left = max(left, cr.left)
        top = max(top, cr.top)
        right = min(right, cr.right)
        bottom = min(bottom, cr.bottom)
        if (right - left) <= 0 or (bottom - top) <= 0:
            return

        filtered = state.dynamic_filter.update_and_filter(current_frame, changed)
        if filtered is None:
            return

        # Clamp filtered result to chat region as well
        fl, ft, fr, fb = filtered
        fl = max(fl, cr.left)
        ft = max(ft, cr.top)
        fr = min(fr, cr.right)
        fb = min(fb, cr.bottom)
        filtered = (fl, ft, fr, fb)
        if fr <= fl or fb <= ft:
            return

        state.dirty_buffer.add(filtered)
        ready_rects = state.dirty_buffer.flush_ready()

        for rect in ready_rects:
            rl, rt, rr, rb = rect
            # Clamp each rect to chat region
            rl = max(rl, cr.left)
            rt = max(rt, cr.top)
            rr = min(rr, cr.right)
            rb = min(rb, cr.bottom)
            if rr <= rl or rb <= rt:
                continue

            patch = current_frame.crop((rl, rt, rr, rb))
            if state.last_uploaded_patch is not None and state.last_uploaded_bbox is not None:
                patch, rect_adjusted = trim_patch_overlap(
                    state.last_uploaded_patch, state.last_uploaded_bbox, patch, (rl, rt, rr, rb),
                )
                if patch.width <= 0 or patch.height <= 0:
                    continue
                rl, rt, rr, rb = rect_adjusted

            chat_roi_height = state.chat_region.height if state.chat_region else current_frame.height
            if self._should_upload(state, patch.height, chat_roi_height):
                self._upload_patch(state, target, patch, (rl, rt, rr, rb))
                state.last_uploaded_patch = patch.copy()
                state.last_uploaded_bbox = (rl, rt, rr, rb)

    def _capture_loop(self) -> None:
        with mss() as sct:
            while not self.stop_event.is_set():
                self._update_windows()
                with self.lock:
                    states = list(self.states.values())
                for state in states:
                    try:
                        self._process_window(sct, state)
                    except Exception as exc:
                        import traceback
                        log.warning("Error processing '%s': %s\n%s", state.window_title, exc, traceback.format_exc())
                time.sleep(self._cfg("poll_interval"))

        with self.lock:
            for state in self.states.values():
                remaining = state.dirty_buffer.force_flush()
                if remaining and state.target and state.previous_frame:
                    for rect in remaining:
                        rl, rt, rr, rb = rect
                        patch = state.previous_frame.crop((rl, rt, rr, rb))
                        self._upload_patch(state, state.target, patch, rect)

    def _config_poll_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                resp = requests.get(self._server_url("/api/config"), timeout=10)
                if resp.status_code == 200:
                    remote = resp.json()
                    with self.lock:
                        remote_windows = remote.pop("windows", None)
                        self.config.update(remote)
                        if remote_windows is not None:
                            local_windows = self.config.get("windows", [])
                            merged = list(dict.fromkeys(local_windows + remote_windows))
                            self.config["windows"] = merged
                    log.info("Config updated from server")
            except Exception:
                log.debug("Config poll failed, using current config")
            self.stop_event.wait(self._cfg("config_poll_interval"))

    def start(self) -> None:
        self._update_windows()
        capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        config_thread = threading.Thread(target=self._config_poll_loop, daemon=True)
        capture_thread.start()
        config_thread.start()
        log.info("CaptureManager started, session=%s", self.session_id)
        log.info("Monitoring windows: %s", list(self.states.keys()))

    def stop(self) -> None:
        self.stop_event.set()
        log.info("CaptureManager stopping")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def show_preview(window_title: str) -> None:
    """Capture a window and show the detected chat region with annotations."""
    import tkinter as tk
    from PIL import ImageTk, ImageDraw

    target = resolve_window(window_title)
    if target is None:
        print(f"Window '{window_title}' not found")
        return

    from mss import mss
    with mss() as sct:
        img, mode = capture_window_prefer_window_api(sct, target)

    region = detect_chat_region(img)
    print(f"Window: {img.size[0]}x{img.size[1]} (mode={mode})")
    print(f"Chat region: left={region.left} top={region.top} right={region.right} bottom={region.bottom}")
    print(f"Chat area: {region.width}x{region.height}")

    # Annotate image
    annotated = img.copy()
    draw = ImageDraw.Draw(annotated)
    draw.line([(region.left, 0), (region.left, img.height)], fill='red', width=3)
    draw.line([(region.left, region.bottom), (img.width, region.bottom)], fill='blue', width=3)
    draw.rectangle([region.left, region.top, region.right, region.bottom], outline='yellow', width=3)
    draw.text((region.left // 2 - 30, img.height // 2), "Contacts", fill='red')
    draw.text(((region.left + img.width) // 2 - 30, region.bottom + 10), "Input Box", fill='blue')
    draw.text(((region.left + region.right) // 2 - 40, (region.top + region.bottom) // 2), "Chat Area", fill='yellow')

    # Scale to fit screen
    max_w, max_h = 1200, 800
    scale = min(max_w / annotated.width, max_h / annotated.height, 1.0)
    display = annotated.resize((int(annotated.width * scale), int(annotated.height * scale)), Image.LANCZOS)

    root = tk.Tk()
    root.title(f"Chat Region Preview - {window_title}")
    root.configure(bg='black')

    photo = ImageTk.PhotoImage(display)
    label = tk.Label(root, image=photo, bg='black')
    label.pack(padx=10, pady=10)

    info = tk.Label(
        root,
        text=f"Window: {img.size[0]}x{img.size[1]}  |  Chat: {region.width}x{region.height}  |  "
             f"Mode: {mode}  |  Red=Sidebar  Blue=InputBox  Yellow=ChatArea",
        fg='white', bg='black', font=('Consolas', 11),
    )
    info.pack(pady=(0, 10))

    root.mainloop()


def show_chooser() -> None:
    """Show a GUI to pick a window and preview its chat region detection."""
    import tkinter as tk
    from tkinter import ttk
    from PIL import ImageTk, ImageDraw

    root = tk.Tk()
    root.title("Chat Region Detector")
    root.configure(bg='#1e1e1e')
    root.geometry("1280x900")

    # --- Top bar: window selector ---
    top_frame = tk.Frame(root, bg='#2d2d2d', pady=8, padx=10)
    top_frame.pack(fill=tk.X)

    tk.Label(top_frame, text="Select Window:", fg='white', bg='#2d2d2d',
             font=('Segoe UI', 11)).pack(side=tk.LEFT)

    windows = list_windows()
    combo = ttk.Combobox(top_frame, values=windows, width=40, state='readonly')
    combo.pack(side=tk.LEFT, padx=8)
    if windows:
        combo.current(0)

    # --- Image display area ---
    img_frame = tk.Frame(root, bg='black')
    img_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    img_label = tk.Label(img_frame, bg='black')
    img_label.pack(fill=tk.BOTH, expand=True)

    # --- Info bar ---
    info_var = tk.StringVar(value="Select a window and click 'Detect'")
    info_label = tk.Label(root, textvariable=info_var, fg='#cccccc', bg='#1e1e1e',
                          font=('Consolas', 10), anchor='w', padx=10, pady=5)
    info_label.pack(fill=tk.X)

    # --- State for keeping photo reference ---
    state = {"photo": None}

    def do_detect():
        title = combo.get()
        if not title:
            return

        # Refresh window list
        windows_new = list_windows()
        combo['values'] = windows_new
        if title not in windows_new:
            info_var.set(f"Window '{title}' not found")
            return

        target = resolve_window(title)
        if target is None:
            info_var.set(f"Cannot resolve window '{title}'")
            return

        from mss import mss
        with mss() as sct:
            img, mode = capture_window_prefer_window_api(sct, target)

        region = detect_chat_region(img)

        # Annotate
        annotated = img.copy()
        draw = ImageDraw.Draw(annotated)
        draw.line([(region.left, 0), (region.left, img.height)], fill='red', width=3)
        draw.line([(region.left, region.bottom), (img.width, region.bottom)], fill='blue', width=3)
        draw.rectangle([region.left, region.top, region.right, region.bottom], outline='yellow', width=3)
        draw.text((region.left // 2 - 30, img.height // 2), "Contacts", fill='red')
        draw.text(((region.left + img.width) // 2 - 30, region.bottom + 10), "Input Box", fill='blue')
        draw.text(((region.left + region.right) // 2 - 40, (region.top + region.bottom) // 2), "Chat Area", fill='yellow')

        # Scale to fit display area
        img_frame.update_idletasks()
        root.update_idletasks()
        frame_w = img_frame.winfo_width()
        frame_h = img_frame.winfo_height()
        # Use screen size as fallback if frame not yet laid out
        if frame_w <= 1 or frame_h <= 1:
            frame_w = root.winfo_screenwidth() - 40
            frame_h = root.winfo_screenheight() - 160
        max_w = max(frame_w - 20, 400)
        max_h = max(frame_h - 20, 300)
        scale = min(max_w / annotated.width, max_h / annotated.height, 1.0)
        display = annotated.resize((int(annotated.width * scale), int(annotated.height * scale)), Image.LANCZOS)

        state["photo"] = ImageTk.PhotoImage(display)
        img_label.configure(image=state["photo"])

        info_var.set(
            f"Window: {img.size[0]}x{img.size[1]}  |  Mode: {mode}  |  "
            f"Chat: {region.width}x{region.height} (left={region.left} top={region.top} "
            f"right={region.right} bottom={region.bottom})  |  "
            f"Red=Sidebar  Blue=InputBox  Yellow=ChatArea"
        )

    def do_refresh():
        windows_new = list_windows()
        combo['values'] = windows_new
        info_var.set(f"Refreshed: {len(windows_new)} windows found")

    btn_detect = tk.Button(top_frame, text="Detect", command=do_detect,
                           bg='#0078d4', fg='white', font=('Segoe UI', 10, 'bold'),
                           padx=15, pady=2, relief=tk.FLAT)
    btn_detect.pack(side=tk.LEFT, padx=5)

    btn_refresh = tk.Button(top_frame, text="Refresh List", command=do_refresh,
                            bg='#444', fg='white', font=('Segoe UI', 10),
                            padx=10, pady=2, relief=tk.FLAT)
    btn_refresh.pack(side=tk.LEFT, padx=5)

    root.mainloop()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Chat Window Incremental Capture Client")
    parser.add_argument("--server", default="http://127.0.0.1:8000", help="Server URL")
    parser.add_argument("--windows", nargs="+", default=[], help="Window titles to monitor")
    parser.add_argument("--poll", type=float, default=0.35, help="Poll interval in seconds")
    parser.add_argument("--quality", type=int, default=60, help="JPEG quality")
    parser.add_argument("--preview", action="store_true", help="Show chat region preview and exit")
    parser.add_argument("--gui", action="store_true", help="Show interactive window chooser GUI")
    args = parser.parse_args()

    if args.gui:
        show_chooser()
        return

    if args.preview:
        titles = args.windows or ["微信"]
        show_preview(titles[0])
        return

    config = {
        "server_url": args.server,
        "poll_interval": args.poll,
        "jpeg_quality": args.quality,
    }
    if args.windows:
        config["windows"] = args.windows

    manager = CaptureManager(config)
    manager.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        manager.stop()


if __name__ == "__main__":
    main()
