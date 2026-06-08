from __future__ import annotations

import base64
import ctypes
import hashlib
import io
import logging
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import cv2
import numpy as np
import pygetwindow as gw
import requests
from mss import mss
from PIL import Image, ImageChops, ImageDraw, ImageStat

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("capture")

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "server_url": "http://127.0.0.1:8000",
    "poll_interval": 0.5,
    "webp_quality": 60,
    # Scroll estimator
    "anchor_top_ratio": 0.70,
    "anchor_bottom_ratio": 0.90,
    "search_top_ratio": 0.30,
    "search_bottom_ratio": 0.90,
    "match_confidence_threshold": 0.70,
    # Capture trigger
    "scroll_threshold_ratio": 0.4,
    "no_scroll_capture_timeout": 5.0,
    # Config poll
    "config_poll_interval": 30,
    # Windows to monitor
    "windows": [],
}

# ---------------------------------------------------------------------------
# DPI awareness + Windows API structures and helpers
# ---------------------------------------------------------------------------
if hasattr(ctypes, "windll"):
    # Make this process DPI-aware so GetWindowRect returns physical pixels.
    # Without this, coordinates are virtualized and PrintWindow produces
    # images with black borders on the right/bottom.
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

    # Use separate WinDLL instances to avoid interfering with pygetwindow
    # and other libraries that also call user32/gdi32.
    _user32 = ctypes.WinDLL("user32")
    _gdi32 = ctypes.WinDLL("gdi32")

    user32 = ctypes.windll.user32  # shared instance for pygetwindow compatibility
    gdi32 = _gdi32

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

# Set proper argtypes/restype for 64-bit Windows compatibility.
# Must be done AFTER structure classes are defined.
# Uses _user32/_gdi32 (private instances) to avoid breaking pygetwindow.
if user32 is not None:
    _user32.GetDC.argtypes = [ctypes.c_void_p]
    _user32.GetDC.restype = ctypes.c_void_p
    _user32.ReleaseDC.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _user32.ReleaseDC.restype = ctypes.c_int
    _user32.GetWindowDC.argtypes = [ctypes.c_void_p]
    _user32.GetWindowDC.restype = ctypes.c_void_p
    _user32.PrintWindow.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]
    _user32.PrintWindow.restype = ctypes.c_int
    _user32.GetWindowRect.argtypes = [ctypes.c_void_p, ctypes.POINTER(RECT)]
    _user32.GetWindowRect.restype = ctypes.c_int
    _user32.GetWindowPlacement.argtypes = [ctypes.c_void_p, ctypes.POINTER(WINDOWPLACEMENT)]
    _user32.GetWindowPlacement.restype = ctypes.c_int
    _user32.IsIconic.argtypes = [ctypes.c_void_p]
    _user32.IsIconic.restype = ctypes.c_int

    _gdi32.CreateCompatibleDC.argtypes = [ctypes.c_void_p]
    _gdi32.CreateCompatibleDC.restype = ctypes.c_void_p
    _gdi32.CreateCompatibleBitmap.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    _gdi32.CreateCompatibleBitmap.restype = ctypes.c_void_p
    _gdi32.SelectObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _gdi32.SelectObject.restype = ctypes.c_void_p
    _gdi32.DeleteObject.argtypes = [ctypes.c_void_p]
    _gdi32.DeleteObject.restype = ctypes.c_int
    _gdi32.DeleteDC.argtypes = [ctypes.c_void_p]
    _gdi32.DeleteDC.restype = ctypes.c_int
    _gdi32.GetDIBits.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint,
        ctypes.c_void_p, ctypes.POINTER(BITMAPINFO), ctypes.c_uint,
    ]
    _gdi32.GetDIBits.restype = ctypes.c_int
    _gdi32.BitBlt.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_uint,
    ]
    _gdi32.BitBlt.restype = ctypes.c_int


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


# ---------------------------------------------------------------------------
# 1. Window Capture (PrintWindow + mss fallback)
# ---------------------------------------------------------------------------
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

        if user32 is not None:
            rect = RECT()
            if _user32.GetWindowRect(hwnd, ctypes.byref(rect)):
                width = rect.right - rect.left
                height = rect.bottom - rect.top
                if width > 0 and height > 0 and rect.left > -10000:
                    return WindowTarget(
                        hwnd=hwnd,
                        title=win.title,
                        left=max(0, rect.left),
                        top=max(0, rect.top),
                        width=width,
                        height=height,
                    )

            wp = WINDOWPLACEMENT()
            wp.length = ctypes.sizeof(WINDOWPLACEMENT)
            if _user32.GetWindowPlacement(hwnd, ctypes.byref(wp)):
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


def _printwindow_to_image(mem_dc: int, bitmap: int, width: int, height: int) -> Optional[Image.Image]:
    """Extract image data from a PrintWindow bitmap."""
    bmi = BITMAPINFO()
    bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    bmi.bmiHeader.biWidth = width
    bmi.bmiHeader.biHeight = -height
    bmi.bmiHeader.biPlanes = 1
    bmi.bmiHeader.biBitCount = 32
    bmi.bmiHeader.biCompression = 0

    buffer_len = width * height * 4
    pixel_buffer = ctypes.create_string_buffer(buffer_len)
    rows = _gdi32.GetDIBits(
        mem_dc, bitmap, 0, height, pixel_buffer, ctypes.byref(bmi), 0,
    )
    if rows != height:
        return None

    return Image.frombuffer("RGB", (width, height), pixel_buffer, "raw", "BGRX", 0, 1).copy()


def capture_window_via_printwindow(target: WindowTarget) -> Optional[Image.Image]:
    """Capture window using PrintWindow API. Supports background/occluded windows."""
    if user32 is None or gdi32 is None:
        return None

    hwnd = target.hwnd
    width = target.width
    height = target.height

    if width <= 0 or height <= 0:
        return None

    for dc_func, pw_flags_list in [
        (_user32.GetWindowDC, [PW_RENDERFULLCONTENT, 3, 0]),
        (_user32.GetDC, [PW_RENDERFULLCONTENT, 3, 0]),
    ]:
        hwnd_dc = dc_func(hwnd)
        if not hwnd_dc:
            continue

        mem_dc = _gdi32.CreateCompatibleDC(hwnd_dc)
        bitmap = _gdi32.CreateCompatibleBitmap(hwnd_dc, width, height)
        if not bitmap:
            _gdi32.DeleteDC(mem_dc)
            _user32.ReleaseDC(hwnd, hwnd_dc)
            continue
        old_bitmap = _gdi32.SelectObject(mem_dc, bitmap)

        try:
            for flags in pw_flags_list:
                result = _user32.PrintWindow(hwnd, mem_dc, flags)
                if result == 1:
                    image = _printwindow_to_image(mem_dc, bitmap, width, height)
                    if image is not None and not _is_black_image(image):
                        return image

            # Try BitBlt from window DC as last resort
            result = _gdi32.BitBlt(mem_dc, 0, 0, width, height, hwnd_dc, 0, 0, 0x00CC0020)
            if result:
                image = _printwindow_to_image(mem_dc, bitmap, width, height)
                if image is not None and not _is_black_image(image):
                    return image
        finally:
            _gdi32.SelectObject(mem_dc, old_bitmap)
            _gdi32.DeleteObject(bitmap)
            _gdi32.DeleteDC(mem_dc)
            _user32.ReleaseDC(hwnd, hwnd_dc)

    return None


def _is_black_image(image: Image.Image, threshold: float = 5.0) -> bool:
    stat = ImageStat.Stat(image.convert("L"))
    return stat.mean[0] < threshold


def capture_window_auto(sct: mss, target: WindowTarget) -> tuple[Image.Image, str]:
    """Capture window, preferring PrintWindow for background/occluded support."""
    image = capture_window_via_printwindow(target)
    if image is not None:
        if not _is_black_image(image):
            return image, "printwindow"
        else:
            log.debug("PrintWindow returned black image for '%s', falling back to screen capture", target.title)
    else:
        log.debug("PrintWindow failed for '%s', falling back to screen capture", target.title)
    return capture_window(sct, target), "screen"


# ---------------------------------------------------------------------------
# 2. Session mechanism
# ---------------------------------------------------------------------------
@dataclass
class Session:
    session_id: str
    software: str   # "wechat" or "telegram"
    chat_name: str
    hwnd: int
    created_at: str


def identify_software(title: str) -> tuple[str, str]:
    """Identify software type and chat name from window title.

    Returns (software, chat_name).
    """
    # Telegram Desktop: "Chat Name - Telegram"
    tg_match = re.match(r"(.+?)\s*[-\u2013\u2014]\s*Telegram$", title, re.IGNORECASE)
    if tg_match:
        return "telegram", tg_match.group(1).strip()
    # WeChat: window title is the chat name
    return "wechat", title


# ---------------------------------------------------------------------------
# 3. Chat Region Locator
# ---------------------------------------------------------------------------
@dataclass
class ChatRegion:
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


def detect_chat_region(image: Image.Image, software: str = "wechat") -> ChatRegion:
    """Detect the chat region within a window.

    For WeChat: finds sidebar divider, header bottom, and input box top.
    For Telegram: similar layout analysis with adjusted parameters.

    Strategy:
    1. Sidebar divider: find the sharp brightness transition from dark sidebar
       to bright chat area. WeChat sidebar is gray (~230), chat is white (~250).
    2. Input box top: find the FIRST dark separator line from the bottom where
       the area above is bright (white message area). WeChat input box has a
       visible border line and a darker background.
    3. Header bottom: find where the title bar transitions to the white message area.
    """
    w, h = image.size
    arr = np.array(image.convert("RGB"))
    gray = np.mean(arr, axis=2)

    # --- 1. Find vertical divider (sidebar | chat area) ---
    # The sidebar is darker gray, the chat area is brighter white.
    # Look for a sustained brightness increase in column averages.
    col_avg = np.mean(gray, axis=0)
    kernel = max(5, w // 40)
    if kernel % 2 == 0:
        kernel += 1
    smooth_col = np.convolve(col_avg, np.ones(kernel) / kernel, mode="same")

    # Method A: find first column where brightness exceeds threshold and stays high
    white_threshold = 240 if software == "wechat" else 230
    min_col = max(5, int(w * 0.03))
    divider_col = 0

    for c in range(min_col, w - 20):
        if smooth_col[c] >= white_threshold:
            if np.mean(smooth_col[c : c + 20]) >= white_threshold - 3:
                divider_col = c
                break

    # Method B: find the biggest brightness jump (fallback)
    if divider_col == 0:
        best_col = min_col
        best_diff = 0
        for c in range(min_col, w - 20):
            left_b = np.mean(smooth_col[max(0, c - 20) : c])
            right_b = np.mean(smooth_col[c : c + 20])
            diff = right_b - left_b
            if diff > best_diff:
                best_diff = diff
                best_col = c
        divider_col = best_col

    # For Telegram with hidden sidebar, divider may be at 0
    if software == "telegram" and divider_col < int(w * 0.05):
        divider_col = 0

    # --- 2. Find horizontal divider (messages | input box) ---
    # Strategy: WeChat has a thin uniform separator line between the message
    # area and the input box. This line is:
    #   - Dark (low brightness)
    #   - Uniform across its width (low standard deviation)
    #   - Spans nearly the full width of the chat panel
    # Message content, by contrast, has high std dev (text, bubbles, avatars).
    #
    # We identify candidate rows by:
    #   1. Low row-wise standard deviation (uniform line, not message content)
    #   2. Low brightness (dark line)
    #   3. Above the line is bright (white message area)
    #   4. Within reasonable distance from the bottom (input box height)

    MAX_INPUT_BOX_HEIGHT = 370 if software == "wechat" else 250
    MIN_INPUT_BOX_HEIGHT = 70 if software == "wechat" else 40

    right_panel = gray[:, divider_col:]
    row_avg = np.mean(right_panel, axis=1)
    row_std = np.std(right_panel, axis=1)

    # Smooth row averages to reduce noise
    row_kernel = max(3, h // 200)
    if row_kernel % 2 == 0:
        row_kernel += 1
    smooth_row = np.convolve(row_avg, np.ones(row_kernel) / row_kernel, mode="same")

    input_top_row = 0
    scan_bottom = h - 5
    scan_top = max(10, int(h * 0.05))

    # Method A: Find separator line using low std dev + low brightness
    # Scan bottom-up, find the first row that looks like a uniform dark line
    # with bright content above it.
    candidates = []
    for r in range(scan_bottom, scan_top, -1):
        remaining = h - r
        if remaining > MAX_INPUT_BOX_HEIGHT or remaining < MIN_INPUT_BOX_HEIGHT:
            continue

        # A separator line has: low std dev (uniform) AND low brightness
        if row_std[r] < 15 and row_avg[r] < 230:
            # Check that above is bright (message area)
            above_window = min(20, r - scan_top)
            if above_window < 3:
                continue
            above_avg = np.mean(smooth_row[max(0, r - above_window) : r])
            if above_avg > 225:
                # Verify: region BELOW the line should be uniform (input box)
                # while region ABOVE should be content-rich (messages)
                below_check_end = min(h, r + remaining)
                below_std = np.mean(row_std[r:below_check_end])
                above_check_start = max(0, r - min(40, r))
                above_std = np.mean(row_std[above_check_start:r])
                # Input box has low std dev; message area has high std dev
                if below_std < 30 and above_std > below_std:
                    score = above_avg - row_avg[r] - row_std[r]
                    # Bonus: prefer candidates where above std is much higher than below
                    score += (above_std - below_std) * 0.5
                    candidates.append((r, score))

    if candidates:
        # Pick the candidate with the highest score
        candidates.sort(key=lambda x: (-x[1], -x[0]))
        input_top_row = candidates[0][0]

    # Method B: Find the biggest brightness drop from bottom-up
    if input_top_row == 0:
        best_drop = 0
        best_r = 0
        window_size = 15
        for r in range(scan_bottom - window_size, scan_top + window_size, -1):
            above_avg = np.mean(smooth_row[max(0, r - window_size) : r])
            below_avg = np.mean(smooth_row[r : min(h, r + window_size)])
            drop = above_avg - below_avg
            remaining = h - r
            if drop > best_drop and above_avg > 220 and MIN_INPUT_BOX_HEIGHT <= remaining <= MAX_INPUT_BOX_HEIGHT:
                best_drop = drop
                best_r = r
        if best_drop > 2:
            input_top_row = best_r

    # Method C: Last resort fallback
    if input_top_row == 0:
        input_top_row = max(int(h * 0.60), h - MAX_INPUT_BOX_HEIGHT)

    # --- 3. Find chat header bottom ---
    # The header is a slightly darker bar at the top of the chat panel.
    # Below it, the message area is bright white.
    header_bottom = 0
    scan_header_top = max(2, int(h * 0.01))
    scan_header_bottom = min(input_top_row - 10, int(h * 0.30))

    for r in range(scan_header_top, scan_header_bottom):
        above = np.mean(smooth_row[max(0, r - 5) : r])
        below = np.mean(smooth_row[r : min(h, r + 5)])
        # Header bottom: brightness jumps up (entering white message area)
        if below > above + 4 and below > 235:
            header_bottom = r
            break

    if header_bottom == 0:
        header_bottom = max(2, int(h * 0.05))

    return ChatRegion(
        left=int(divider_col),
        top=int(header_bottom),
        right=int(w),
        bottom=int(input_top_row),
    )


# ---------------------------------------------------------------------------
# 4. Scroll Estimator (bottom anchor template matching)
# ---------------------------------------------------------------------------
class ScrollEstimator:
    """Estimate vertical scroll delta using bottom-anchor template matching.

    Algorithm:
    1. From frame_prev, extract bottom 20% as anchor (0.70H ~ 0.90H)
    2. In frame_curr, search for anchor in range 0.30H ~ 0.90H
    3. Use OpenCV matchTemplate (TM_CCOEFF_NORMED)
    4. scroll_delta = anchor_old_y - best_y
       Positive = content scrolled up (new messages appeared at bottom)
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.anchor: Optional[np.ndarray] = None
        self.anchor_y_start: int = 0

    # Width of the right-aligned anchor strip (pixels).
    # Using a fixed width avoids anchor mismatch when chat region left boundary changes.
    ANCHOR_STRIP_WIDTH = 256

    def set_anchor(self, chat_image: Image.Image) -> None:
        """Extract anchor region from the bottom-right portion of the chat image."""
        arr = np.array(chat_image.convert("RGB"))
        h, w = arr.shape[:2]
        y_start = int(h * self.config.get("anchor_top_ratio", 0.70))
        y_end = int(h * self.config.get("anchor_bottom_ratio", 0.90))
        x_start = max(0, w - self.ANCHOR_STRIP_WIDTH)
        self.anchor = arr[y_start:y_end, x_start:w].copy()
        self.anchor_y_start = y_start

    def estimate_scroll(self, chat_image: Image.Image) -> Optional[int]:
        """Estimate scroll delta between anchor and current frame.

        Returns scroll_delta in pixels, or None if anchor not found.
        """
        if self.anchor is None or self.anchor.size == 0:
            return None

        arr = np.array(chat_image.convert("RGB"))
        h, w = arr.shape[:2]
        ah, aw = self.anchor.shape[:2]

        # Only search in the right-aligned strip
        x_start = max(0, w - self.ANCHOR_STRIP_WIDTH)
        search_top = int(h * self.config.get("search_top_ratio", 0.30))
        search_bottom = int(h * self.config.get("search_bottom_ratio", 0.90))

        search_region = arr[search_top:search_bottom, x_start:w]

        if search_region.shape[0] < ah or search_region.shape[1] < aw:
            return None

        result = cv2.matchTemplate(search_region, self.anchor, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        threshold = self.config.get("match_confidence_threshold", 0.70)
        if max_val < threshold:
            return None

        best_y = max_loc[1] + search_top
        scroll_delta = self.anchor_y_start - best_y
        return scroll_delta

    def reset(self) -> None:
        self.anchor = None
        self.anchor_y_start = 0


class MotionRegionDetector:
    """Detect chat region by observing which area of the window changes.

    Algorithm:
    1. Capture frames over time
    2. When a significant change is detected, compute the diff
    3. Accumulate change masks across multiple events
    4. Find the bounding box of accumulated changes in the right portion
    5. That bounding box = chat region (scrollable area)

    Why this works:
    - When new messages arrive, the chat area scrolls → large pixel change
    - The input box doesn't scroll → no change → naturally excluded
    - The contact list is on the left → we prefer the right side
    - Chat switches cause even larger changes → also detected correctly
    """

    def __init__(self, right_bias: float = 0.25, diff_threshold: int = 25,
                 min_change_ratio: float = 0.003, min_events: int = 1) -> None:
        self.right_bias = right_bias  # Skip left 25% of window (contact list)
        self.diff_threshold = diff_threshold
        self.min_change_ratio = min_change_ratio
        self.min_events = min_events
        self.prev_frame: Optional[np.ndarray] = None
        self.accumulated_mask: Optional[np.ndarray] = None
        self.event_count: int = 0
        self.chat_region: Optional[ChatRegion] = None
        self.sidebar_boundary: int = 0  # detected sidebar/chat divider x position

    def reset(self) -> None:
        """Reset all accumulated state for fresh detection."""
        self.prev_frame = None
        self.accumulated_mask = None
        self.event_count = 0
        self.chat_region = None

    def detect_sidebar_from_image(self, image: Image.Image) -> int:
        """Detect the vertical divider line between sidebar and chat area
        using edge detection on a single image. Returns the x position
        of the divider, or 0 if not found."""
        arr = np.array(image.convert("RGB"))
        h, w = arr.shape[:2]
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

        # Vertical edge detection
        edges = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        edge_mag = np.abs(edges)

        mask = (edge_mag > 30).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        col_sums = np.sum(mask, axis=0)
        max_col = np.max(col_sums)

        min_sidebar_width = int(w * 0.08)
        max_sidebar_width = int(w * 0.85)

        left_boundary = 0
        if max_col > 0:
            peak_threshold = max_col * 0.9
            max_peak_width = 5
            center_x = w // 2

            # Collect all candidate peaks
            candidates = []
            x = min_sidebar_width
            while x < max_sidebar_width:
                if col_sums[x] >= peak_threshold:
                    peak_start = x
                    peak_end = x
                    while peak_end < max_sidebar_width and col_sums[peak_end] >= peak_threshold:
                        peak_end += 1
                    peak_width = peak_end - peak_start
                    if peak_width <= max_peak_width:
                        peak_center = peak_start + peak_width // 2
                        candidates.append(peak_center)
                    x = peak_end
                else:
                    x += 1

            # Pick the candidate closest to window center
            if candidates:
                left_boundary = min(candidates, key=lambda cx: abs(cx - center_x))

        self.sidebar_boundary = left_boundary
        return left_boundary

    def process_frame(self, image: Image.Image) -> Optional[ChatRegion]:
        """Process a frame. Returns ChatRegion once detected, or None."""
        arr = np.array(image.convert("RGB"))
        h, w = arr.shape[:2]

        if self.prev_frame is None:
            self.prev_frame = arr
            return None

        # Window resized: reset accumulated data
        if self.prev_frame.shape[:2] != (h, w):
            self.prev_frame = arr
            self.accumulated_mask = None
            self.event_count = 0
            return None

        # Compute diff
        diff = np.abs(arr.astype(np.int16) - self.prev_frame.astype(np.int16))
        gray_diff = np.mean(diff, axis=2)

        # Threshold: significant pixel change
        mask = (gray_diff > self.diff_threshold).astype(np.uint8)

        # Morphological close to fill small gaps (text characters, thin lines)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Use pre-detected sidebar boundary (from detect_sidebar_from_image)
        left_boundary = self.sidebar_boundary

        # Filter connected components:
        # - Remove tiny noise (cursor blink, < 200px area) everywhere
        # - Remove short components (< 50px height) ONLY in left/sidebar area
        # - Keep short components in right/chat area (new messages can be short)
        min_component_area = 200   # pixels - cursor is ~1x20=20px
        min_component_height = 50  # pixels - sidebar preview is ~30px
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, num_labels):  # skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            cx = stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] / 2
            if area < min_component_area:
                mask[labels == i] = 0
            elif height < min_component_height and cx < left_boundary:
                # Short component in sidebar area - filter it
                mask[labels == i] = 0

        # Now mask out the left/sidebar area
        mask[:, :left_boundary] = 0
        self.sidebar_boundary = left_boundary  # store for external use

        # Check if enough pixels changed after filtering
        change_ratio = np.sum(mask) / mask.size
        if change_ratio < self.min_change_ratio:
            self.prev_frame = arr
            return None

        # Accumulate change mask
        if self.accumulated_mask is None:
            self.accumulated_mask = mask
        else:
            self.accumulated_mask = cv2.bitwise_or(self.accumulated_mask, mask)

        self.event_count += 1
        self.prev_frame = arr

        if self.event_count < self.min_events:
            return None

        # Find bounding box of accumulated changes
        coords = cv2.findNonZero(self.accumulated_mask)
        if coords is None:
            return None

        x, y, bw, bh = cv2.boundingRect(coords)

        # Chat area spans from the leftmost change to the window right edge.
        # Use the detected left boundary (not a fixed ratio) so we don't
        # cut off content that's left of the 25% mark.
        new_region = ChatRegion(
            left=int(x),                   # actual detected left boundary
            top=int(y),
            right=int(w),                   # window right edge
            bottom=int(y + bh),
        )

        # Region only grows, never shrinks
        if self.chat_region is not None:
            new_region = ChatRegion(
                left=min(self.chat_region.left, new_region.left),
                top=min(self.chat_region.top, new_region.top),
                right=max(self.chat_region.right, new_region.right),
                bottom=max(self.chat_region.bottom, new_region.bottom),
            )

        self.chat_region = new_region
        return self.chat_region


# ---------------------------------------------------------------------------
# 5. Accumulated Scroll + Threshold Trigger + Full Chat Capture
# ---------------------------------------------------------------------------
@dataclass
class SessionState:
    """Per-session capture state."""
    session: Session
    target: Optional[WindowTarget] = None
    chat_region: Optional[ChatRegion] = None
    previous_chat_frame: Optional[Image.Image] = None
    scroll_estimator: ScrollEstimator = field(default=None)
    motion_detector: MotionRegionDetector = field(default=None)
    accumulated_scroll: int = 0
    capture_count: int = 0
    last_capture_time: float = 0.0
    last_change_time: float = 0.0
    last_redetect_time: float = 0.0
    capture_mode: str = "unknown"
    last_window_rect: Optional[tuple[int, int, int, int]] = None
    first_frame: bool = True
    calibrating: bool = True  # Start in calibration mode by default
    anchor_lost_time: Optional[float] = None  # When anchor was first lost
    prev_frame_size: Optional[tuple[int, int]] = None  # (width, height) to detect resize
    paused: bool = False  # True when window is minimized or not renderable


def encode_webp(image: Image.Image, quality: int = 60) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="WEBP", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _frames_differ(prev: Image.Image, curr: Image.Image, threshold: float = 3.0) -> bool:
    # Right-align frames so left boundary changes don't cause false positives
    w = min(prev.width, curr.width)
    h = min(prev.height, curr.height)
    prev_r = prev.crop((prev.width - w, 0, prev.width, h))
    curr_r = curr.crop((curr.width - w, 0, curr.width, h))
    diff = ImageChops.difference(prev_r, curr_r)
    stat = ImageStat.Stat(diff.convert("L"))
    return stat.mean[0] > threshold


# ---------------------------------------------------------------------------
# Capture Manager (multi-window, multi-session)
# ---------------------------------------------------------------------------
class CaptureManager:
    def __init__(self, config: dict) -> None:
        self.config = dict(DEFAULT_CONFIG)
        self.config.update(config)
        self.states: dict[str, SessionState] = {}
        self.stop_event = threading.Event()
        self.lock = threading.Lock()

    def _cfg(self, key: str, default=None):
        if default is not None:
            return self.config.get(key, DEFAULT_CONFIG.get(key, default))
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
                if title in self.states:
                    continue
                target = resolve_window(title)
                if target is None:
                    log.debug("Window '%s' not found yet", title)
                    continue
                software, chat_name = identify_software(target.title)
                session = Session(
                    session_id=f"{software}_{uuid.uuid4().hex[:8]}",
                    software=software,
                    chat_name=chat_name,
                    hwnd=target.hwnd,
                    created_at=utc_now(),
                )
                state = SessionState(
                    session=session,
                    target=target,
                    scroll_estimator=ScrollEstimator(self.config),
                    motion_detector=MotionRegionDetector(),
                    calibrating=True,
                )
                self.states[title] = state
                self._register_session(state)
                log.info(
                    "New session: %s software=%s chat=%s",
                    session.session_id, session.software, session.chat_name,
                )
            for title in list(self.states.keys()):
                if title not in wanted:
                    del self.states[title]

    def _register_session(self, state: SessionState) -> None:
        payload = {
            "session_id": state.session.session_id,
            "software": state.session.software,
            "chat_name": state.session.chat_name,
            "hwnd": state.session.hwnd,
            "created_at": state.session.created_at,
        }
        try:
            resp = requests.post(
                self._server_url("/api/session"), json=payload, timeout=10,
            )
            resp.raise_for_status()
            log.info(
                "Session registered: %s (%s/%s)",
                state.session.session_id,
                state.session.software,
                state.session.chat_name,
            )
        except Exception as exc:
            log.warning("Session registration failed: %s", exc)

    def _check_window_moved(self, state: SessionState) -> bool:
        if state.target is None:
            return False
        rect = (state.target.left, state.target.top, state.target.width, state.target.height)
        if state.last_window_rect is None:
            state.last_window_rect = rect
            return False
        moved = rect != state.last_window_rect
        state.last_window_rect = rect
        return moved

    def _should_capture(self, state: SessionState) -> bool:
        if state.chat_region is None:
            return False
        threshold = int(state.chat_region.height * self._cfg("scroll_threshold_ratio"))
        if state.accumulated_scroll >= threshold:
            return True
        # # Time-based fallback: content changed but scroll below threshold for too long
        # timeout = self._cfg("no_scroll_capture_timeout")
        # now = time.monotonic()
        # if (now - state.last_change_time) >= timeout and state.accumulated_scroll > 0:
        #     return True
        return False

    def _upload_capture(
        self, state: SessionState, chat_image: Image.Image, scroll_since_last: int,
    ) -> None:
        capture_id = str(uuid.uuid4())
        payload = {
            "session_id": state.session.session_id,
            "capture_id": capture_id,
            "timestamp": time.time(),
            "scroll_since_last": scroll_since_last,
            "window_title": state.target.title if state.target else "",
            "software": state.session.software,
            "chat_name": state.session.chat_name,
            "chat_rect": {
                "x": state.chat_region.left,
                "y": state.chat_region.top,
                "width": state.chat_region.width,
                "height": state.chat_region.height,
            }
            if state.chat_region
            else None,
            "image_webp_base64": encode_webp(chat_image, self._cfg("webp_quality")),
        }
        try:
            resp = requests.post(
                self._server_url("/api/frame"), json=payload, timeout=10,
            )
            resp.raise_for_status()
            state.capture_count += 1
            state.last_capture_time = time.monotonic()
            log.info(
                "Captured #%d '%s' (scroll=%d, size=%dx%d)",
                state.capture_count,
                state.session.chat_name,
                scroll_since_last,
                chat_image.width,
                chat_image.height,
            )
        except Exception as exc:
            log.warning("Upload failed for '%s': %s", state.session.chat_name, exc)

    def _upload_title_strip(self, state: SessionState, title_strip: Image.Image) -> None:
        """Upload the 60px title strip from the top of the chat region."""
        payload = {
            "session_id": state.session.session_id,
            "software": state.session.software,
            "timestamp": time.time(),
            "image_webp_base64": encode_webp(title_strip, self._cfg("webp_quality")),
        }
        try:
            resp = requests.post(
                self._server_url("/api/session/{}/title".format(state.session.session_id)),
                json=payload,
                timeout=10,
            )
            resp.raise_for_status()
            log.info(
                "Title strip uploaded for '%s' (%dx%d)",
                state.session.chat_name,
                title_strip.width,
                title_strip.height,
            )
        except Exception as exc:
            log.warning("Title strip upload failed for '%s': %s", state.session.chat_name, exc)

    def _save_debug_image(self, debug_dir: str, chat_name: str, frame: Image.Image, region: ChatRegion) -> None:
        """Save a diagnostic image with chat region overlay."""
        try:
            from pathlib import Path
            out_dir = Path(debug_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            annotated = frame.copy()
            from PIL import ImageDraw
            draw = ImageDraw.Draw(annotated)
            # Draw chat region rectangle (green)
            draw.rectangle([region.left, region.top, region.right, region.bottom], outline="lime", width=3)
            # Draw horizontal line at input_top (red)
            draw.line([(region.left, region.bottom), (region.right, region.bottom)], fill="red", width=2)
            # Draw horizontal line at header_bottom (blue)
            draw.line([(region.left, region.top), (region.right, region.top)], fill="blue", width=2)
            # Add text
            threshold = int(region.height * self._cfg("scroll_threshold_ratio"))
            draw.text((region.left + 5, region.top + 5),
                      f"Chat: {region.width}x{region.height}  Threshold: {threshold}px",
                      fill="red")
            safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in chat_name)
            out_path = out_dir / f"debug_{safe_name}.png"
            annotated.save(out_path, format="PNG")
            log.info("Debug image saved: %s", out_path)
        except Exception as exc:
            log.warning("Failed to save debug image: %s", exc)

    # Map window title keywords to process names for app detection
    APP_PROCESS_MAP = {
        "微信": ["WeChat.exe", "WeChatApp.exe"],
        "QQ": ["QQ.exe"],
        "Telegram": ["telegram.exe"],
    }

    def _is_app_running(self, window_title: str) -> bool:
        """Check if the application process is still running (window may be closed to tray)."""
        try:
            import psutil
        except ImportError:
            # Without psutil, assume app is running (don't pause)
            return True

        # Find matching process names from the map
        process_names = None
        for keyword, procs in self.APP_PROCESS_MAP.items():
            if keyword in window_title:
                process_names = procs
                break

        if process_names is None:
            # Unknown app, assume still running
            return True

        for proc in psutil.process_iter(["name"]):
            try:
                if proc.info["name"] in process_names:
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return False

    def _process_window(self, sct: mss, state: SessionState) -> None:
        # Re-resolve window (it may have moved or changed)
        target = resolve_window(state.session.chat_name)
        if target is None:
            for title, s in self.states.items():
                if s is state:
                    target = resolve_window(title)
                    break
        if target is None:
            # Window not found: could be closed-to-tray or app exited
            if not state.paused:
                # Check if the app process is still running
                app_running = self._is_app_running(state.session.chat_name)
                if app_running:
                    log.info("Window '%s' closed to tray, pausing capture",
                             state.session.chat_name)
                else:
                    log.info("App '%s' exited, pausing capture",
                             state.session.chat_name)
                state.paused = True
            return

        state.target = target
        window_moved = self._check_window_moved(state)

        # Check if window is minimized — skip capture to save resources
        if _user32.IsIconic(target.hwnd):
            if not state.paused:
                log.info("Window '%s' is minimized, pausing capture",
                         state.session.chat_name)
                state.paused = True
            return
        elif state.paused:
            log.info("Window '%s' restored, resuming capture",
                     state.session.chat_name)
            state.paused = False

        # Capture window
        current_frame, capture_mode = capture_window_auto(sct, target)
        state.capture_mode = capture_mode

        # Check if captured frame is all black (window not renderable)
        if _is_black_image(current_frame):
            if not state.paused:
                log.info("Window '%s' not renderable (black frame), pausing capture",
                         state.session.chat_name)
                state.paused = True
            return
        elif state.paused:
            log.info("Window '%s' renderable again, resuming capture",
                     state.session.chat_name)
            state.paused = False

        # Detect window resize: if frame size changed, re-calibrate
        frame_w, frame_h = current_frame.size
        if (state.prev_frame_size is not None
                and state.prev_frame_size != (frame_w, frame_h)):
            log.info("Window resized for '%s': %dx%d -> %dx%d, re-calibrating",
                     state.session.chat_name,
                     state.prev_frame_size[0], state.prev_frame_size[1],
                     frame_w, frame_h)
            state.accumulated_scroll = 0
            state.anchor_lost_time = None
            state.scroll_estimator.anchor = None
            state.previous_chat_frame = None
            state.first_frame = True
            state.motion_detector = MotionRegionDetector()
            state.chat_region = None
            state.calibrated = False
            state.calibrating = True
        state.prev_frame_size = (frame_w, frame_h)

        # --- Calibration mode: detect chat region via motion observation ---
        if state.calibrating:
            if state.motion_detector is None:
                state.motion_detector = MotionRegionDetector()

            # Detect sidebar boundary if not yet detected
            if state.motion_detector.sidebar_boundary == 0:
                boundary = state.motion_detector.detect_sidebar_from_image(current_frame)
                if boundary > 0:
                    log.info("Sidebar boundary for '%s': x=%d", state.session.chat_name, boundary)
                else:
                    log.info("No sidebar boundary detected for '%s'", state.session.chat_name)

            region = state.motion_detector.process_frame(current_frame)

            if region is not None:
                # Fix width to span the full right portion (sidebar_boundary -> window right)
                sb = state.motion_detector.sidebar_boundary
                w = np.array(current_frame.convert("RGB")).shape[1]
                h = np.array(current_frame.convert("RGB")).shape[0]
                region = ChatRegion(
                    left=sb,
                    top=max(0, region.top - 5),
                    right=w,
                    bottom=min(h, region.bottom + 5),
                )
                # Only expand, never shrink
                if state.chat_region is not None:
                    region = ChatRegion(
                        left=min(state.chat_region.left, region.left),
                        top=min(state.chat_region.top, region.top),
                        right=max(state.chat_region.right, region.right),
                        bottom=max(state.chat_region.bottom, region.bottom),
                    )
                state.chat_region = region
                state.calibrating = False
                state.last_redetect_time = time.monotonic()
                log.info(
                    "Calibration complete for '%s': left=%d top=%d right=%d bottom=%d (%dx%d) "
                    "| events=%d",
                    state.session.chat_name,
                    region.left, region.top, region.right, region.bottom,
                    region.width, region.height,
                    state.motion_detector.event_count,
                )
                debug_dir = self._cfg("debug_dir")
                if debug_dir:
                    self._save_debug_image(debug_dir, state.session.chat_name, current_frame, region)
            else:
                if state.motion_detector.event_count == 0:
                    log.info("Calibrating '%s': waiting for content changes...",
                             state.session.chat_name)
                else:
                    log.debug("Calibrating '%s': %d change events observed, need %d more",
                              state.session.chat_name,
                              state.motion_detector.event_count,
                              state.motion_detector.min_events - state.motion_detector.event_count)
            return

        # --- Normal mode: use detected chat region ---

        # Periodically re-detect sidebar boundary (every 60s)
        # to handle user resizing. Chat region width always = window width - sidebar.
        if (state.motion_detector is not None
                and time.monotonic() - state.last_redetect_time >= 60):
            old_boundary = state.motion_detector.sidebar_boundary
            new_boundary = state.motion_detector.detect_sidebar_from_image(current_frame)
            if new_boundary > 0 and new_boundary != old_boundary:
                log.info("Sidebar boundary changed for '%s': %d -> %d",
                         state.session.chat_name, old_boundary, new_boundary)
                state.motion_detector.sidebar_boundary = new_boundary
                # Immediately update chat region width
                if state.chat_region is not None:
                    w = np.array(current_frame.convert("RGB")).shape[1]
                    state.chat_region = ChatRegion(
                        left=new_boundary,
                        top=state.chat_region.top,
                        right=w,
                        bottom=state.chat_region.bottom,
                    )
                    log.info("Chat region width updated for '%s': left=%d right=%d",
                             state.session.chat_name, new_boundary, w)
                    # Reset anchor since frame width changed
                    state.scroll_estimator.anchor = None

            # Re-detect chat region height via motion (only expand)
            new_region = state.motion_detector.process_frame(current_frame)
            if new_region is not None and state.chat_region is not None:
                sb = state.motion_detector.sidebar_boundary
                w = np.array(current_frame.convert("RGB")).shape[1]
                expanded = ChatRegion(
                    left=sb,
                    top=min(state.chat_region.top, new_region.top),
                    right=w,
                    bottom=max(state.chat_region.bottom, new_region.bottom),
                )
                if (expanded.top != state.chat_region.top or
                    expanded.bottom != state.chat_region.bottom):
                    log.info(
                        "Chat region height expanded for '%s': (%d,%d,%d,%d) -> (%d,%d,%d,%d)",
                        state.session.chat_name,
                        state.chat_region.left, state.chat_region.top,
                        state.chat_region.right, state.chat_region.bottom,
                        expanded.left, expanded.top, expanded.right, expanded.bottom,
                    )
                    state.chat_region = expanded
            state.last_redetect_time = time.monotonic()

        cr = state.chat_region
        if cr is None:
            return

        # Crop to chat region
        chat_frame = current_frame.crop((cr.left, cr.top, cr.right, cr.bottom))

        # First frame: capture title strip, then always capture and set anchor
        if state.first_frame or state.previous_chat_frame is None:
            # Capture a 60px tall strip just above the chat region top (likely the title bar)
            # Crop left sidebar if detected
            title_left = state.motion_detector.sidebar_boundary if state.motion_detector else 0
            title_height = min(60, cr.top)
            if title_height > 0:
                title_strip = current_frame.crop((title_left, cr.top - title_height, current_frame.width, cr.top))
                self._upload_title_strip(state, title_strip)

            state.previous_chat_frame = chat_frame
            state.scroll_estimator.set_anchor(chat_frame)
            state.accumulated_scroll = 0
            state.last_capture_time = time.monotonic()
            state.last_change_time = time.monotonic()
            state.first_frame = False
            self._upload_capture(state, chat_frame, 0)
            return

        # Estimate scroll delta
        scroll_delta = state.scroll_estimator.estimate_scroll(chat_frame)

        if scroll_delta is not None and abs(scroll_delta) >= 2:
            state.accumulated_scroll += scroll_delta
            state.last_change_time = time.monotonic()
            state.anchor_lost_time = None  # Anchor is working, clear lost timer
            log.debug(
                "Scroll delta=%d accumulated=%d (threshold=%d) '%s'",
                scroll_delta,
                state.accumulated_scroll,
                int(cr.height * self._cfg("scroll_threshold_ratio")),
                state.session.chat_name,
            )
        elif _frames_differ(state.previous_chat_frame, chat_frame):
            # Anchor lost but content changed significantly (e.g. chat switch)
            # Wait 1 second of continuous anchor loss before resetting
            now = time.monotonic()
            if state.anchor_lost_time is None:
                state.anchor_lost_time = now
                log.info("Anchor lost for '%s', waiting 1s before reset...",
                         state.session.chat_name)
            elif now - state.anchor_lost_time >= 1.0:
                state.last_change_time = now
                log.info(
                    "Anchor lost for 1s, re-detecting chat region for '%s'",
                    state.session.chat_name,
                )
                state.accumulated_scroll = 0
                state.anchor_lost_time = None
                # Don't set anchor yet - wait for new chat region to be detected first
                state.scroll_estimator.anchor = None
                state.previous_chat_frame = None
                state.first_frame = True
                # Reset motion detector and chat region for fresh detection
                state.motion_detector = MotionRegionDetector()
                state.chat_region = None
                state.calibrated = False
                state.calibrating = True
            return

        # Update anchor from current frame for next comparison
        state.scroll_estimator.set_anchor(chat_frame)
        state.previous_chat_frame = chat_frame

        # Check if we should capture
        if self._should_capture(state):
            self._upload_capture(state, chat_frame, state.accumulated_scroll)
            state.accumulated_scroll = 0
            state.scroll_estimator.set_anchor(chat_frame)

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
                        log.warning(
                            "Error processing '%s': %s\n%s",
                            state.session.chat_name,
                            exc,
                            traceback.format_exc(),
                        )
                time.sleep(self._cfg("poll_interval"))

    def _config_poll_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                resp = requests.get(self._server_url("/api/config"), timeout=10)
                if resp.status_code == 200:
                    remote = resp.json()
                    with self.lock:
                        remote_windows = remote.pop("windows", None)
                        self.config.update(remote)
                        if remote_windows is not None and not self.config.get("windows_override"):
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
        log.info("CaptureManager started")
        log.info("Monitoring windows: %s", list(self.states.keys()))

    def stop(self) -> None:
        self.stop_event.set()
        log.info("CaptureManager stopping")


# ---------------------------------------------------------------------------
# Preview / GUI utilities
# ---------------------------------------------------------------------------
def show_preview(window_title: str) -> None:
    """Capture a window and show the detected chat region with annotations."""
    import tkinter as tk
    from PIL import ImageTk

    target = resolve_window(window_title)
    if target is None:
        print(f"Window '{window_title}' not found")
        return

    software, chat_name = identify_software(target.title)
    with mss() as sct:
        img, mode = capture_window_auto(sct, target)

    region = detect_chat_region(img, software)
    print(f"Window: {img.size[0]}x{img.size[1]} (mode={mode}, software={software})")
    print(f"Chat region: left={region.left} top={region.top} right={region.right} bottom={region.bottom}")
    print(f"Chat area: {region.width}x{region.height}")

    annotated = img.copy()
    draw = ImageDraw.Draw(annotated)
    draw.line([(region.left, 0), (region.left, img.height)], fill="red", width=3)
    draw.line([(region.left, region.bottom), (img.width, region.bottom)], fill="blue", width=3)
    draw.rectangle(
        [region.left, region.top, region.right, region.bottom], outline="yellow", width=3,
    )
    draw.text((region.left // 2 - 30, img.height // 2), "Contacts", fill="red")
    draw.text(((region.left + img.width) // 2 - 30, region.bottom + 10), "Input Box", fill="blue")
    draw.text(
        ((region.left + region.right) // 2 - 40, (region.top + region.bottom) // 2),
        "Chat Area",
        fill="yellow",
    )

    max_w, max_h = 1200, 800
    scale = min(max_w / annotated.width, max_h / annotated.height, 1.0)
    display = annotated.resize(
        (int(annotated.width * scale), int(annotated.height * scale)), Image.LANCZOS,
    )

    root = tk.Tk()
    root.title(f"Chat Region Preview - {window_title} ({software})")
    root.configure(bg="black")

    photo = ImageTk.PhotoImage(display)
    label = tk.Label(root, image=photo, bg="black")
    label.pack(padx=10, pady=10)

    info = tk.Label(
        root,
        text=f"Window: {img.size[0]}x{img.size[1]}  |  Chat: {region.width}x{region.height}  |  "
        f"Mode: {mode}  |  Software: {software}  |  Red=Sidebar  Blue=InputBox  Yellow=ChatArea",
        fg="white",
        bg="black",
        font=("Consolas", 11),
    )
    info.pack(pady=(0, 10))

    root.mainloop()


def show_chooser() -> None:
    """Show a GUI to pick a window and preview its chat region detection."""
    import tkinter as tk
    from tkinter import ttk
    from PIL import ImageTk

    root = tk.Tk()
    root.title("Chat Region Detector")
    root.configure(bg="#1e1e1e")
    root.geometry("1280x900")

    top_frame = tk.Frame(root, bg="#2d2d2d", pady=8, padx=10)
    top_frame.pack(fill=tk.X)

    tk.Label(
        top_frame, text="Select Window:", fg="white", bg="#2d2d2d", font=("Segoe UI", 11),
    ).pack(side=tk.LEFT)

    windows = list_windows()
    combo = ttk.Combobox(top_frame, values=windows, width=40, state="readonly")
    combo.pack(side=tk.LEFT, padx=8)
    if windows:
        combo.current(0)

    img_frame = tk.Frame(root, bg="black")
    img_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    img_label = tk.Label(img_frame, bg="black")
    img_label.pack(fill=tk.BOTH, expand=True)

    info_var = tk.StringVar(value="Select a window and click 'Detect'")
    info_label = tk.Label(
        root, textvariable=info_var, fg="#cccccc", bg="#1e1e1e",
        font=("Consolas", 10), anchor="w", padx=10, pady=5,
    )
    info_label.pack(fill=tk.X)

    gui_state = {"photo": None}

    def do_detect():
        title = combo.get()
        if not title:
            return

        target = resolve_window(title)
        if target is None:
            info_var.set(f"Cannot resolve window '{title}'")
            return

        software, chat_name = identify_software(target.title)
        with mss() as sct:
            img, mode = capture_window_auto(sct, target)

        region = detect_chat_region(img, software)

        annotated = img.copy()
        draw = ImageDraw.Draw(annotated)
        draw.line([(region.left, 0), (region.left, img.height)], fill="red", width=3)
        draw.line([(region.left, region.bottom), (img.width, region.bottom)], fill="blue", width=3)
        draw.rectangle(
            [region.left, region.top, region.right, region.bottom], outline="yellow", width=3,
        )
        draw.text((region.left // 2 - 30, img.height // 2), "Contacts", fill="red")
        draw.text(((region.left + img.width) // 2 - 30, region.bottom + 10), "Input Box", fill="blue")
        draw.text(
            ((region.left + region.right) // 2 - 40, (region.top + region.bottom) // 2),
            "Chat Area",
            fill="yellow",
        )

        img_frame.update_idletasks()
        root.update_idletasks()
        frame_w = img_frame.winfo_width()
        frame_h = img_frame.winfo_height()
        if frame_w <= 1 or frame_h <= 1:
            frame_w = root.winfo_screenwidth() - 40
            frame_h = root.winfo_screenheight() - 160
        max_w = max(frame_w - 20, 400)
        max_h = max(frame_h - 20, 300)
        scale = min(max_w / annotated.width, max_h / annotated.height, 1.0)
        display = annotated.resize(
            (int(annotated.width * scale), int(annotated.height * scale)), Image.LANCZOS,
        )

        gui_state["photo"] = ImageTk.PhotoImage(display)
        img_label.configure(image=gui_state["photo"])

        info_var.set(
            f"Window: {img.size[0]}x{img.size[1]}  |  Mode: {mode}  |  "
            f"Software: {software}  |  Chat: {region.width}x{region.height} "
            f"(left={region.left} top={region.top} right={region.right} bottom={region.bottom})  |  "
            f"Red=Sidebar  Blue=InputBox  Yellow=ChatArea"
        )

    def do_refresh():
        windows_new = list_windows()
        combo["values"] = windows_new
        info_var.set(f"Refreshed: {len(windows_new)} windows found")

    btn_detect = tk.Button(
        top_frame, text="Detect", command=do_detect,
        bg="#0078d4", fg="white", font=("Segoe UI", 10, "bold"),
        padx=15, pady=2, relief=tk.FLAT,
    )
    btn_detect.pack(side=tk.LEFT, padx=5)

    btn_refresh = tk.Button(
        top_frame, text="Refresh List", command=do_refresh,
        bg="#444", fg="white", font=("Segoe UI", 10),
        padx=10, pady=2, relief=tk.FLAT,
    )
    btn_refresh.pack(side=tk.LEFT, padx=5)

    root.mainloop()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Chat Window Low-Frequency Capture Archiver")
    parser.add_argument("--server", default="http://127.0.0.1:8000", help="Server URL")
    parser.add_argument("--windows", nargs="+", default=[], help="Window titles to monitor")
    parser.add_argument("--poll", type=float, default=0.5, help="Poll interval in seconds")
    parser.add_argument("--quality", type=int, default=60, help="WebP quality")
    parser.add_argument("--scroll-ratio", type=float, default=0.4, help="Scroll threshold as ratio of chat height")
    parser.add_argument("--preview", action="store_true", help="Show chat region preview and exit")
    parser.add_argument("--gui", action="store_true", help="Show interactive window chooser GUI")
    parser.add_argument("--debug-dir", default="", help="Save diagnostic images to this directory")
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
        "webp_quality": args.quality,
        "scroll_threshold_ratio": args.scroll_ratio,
        "debug_dir": args.debug_dir,
    }
    if args.windows:
        config["windows"] = args.windows
        config["windows_override"] = True  # CLI override: don't merge with server

    manager = CaptureManager(config)
    manager.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        manager.stop()


if __name__ == "__main__":
    main()
