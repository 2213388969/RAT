from __future__ import annotations

import base64
import ctypes
import io
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import pygetwindow as gw
import requests
import tkinter as tk
from mss import mss
from PIL import Image, ImageChops, ImageFilter, ImageOps, ImageStat
from tkinter import messagebox, ttk


SERVER_URL = "http://127.0.0.1:8000/api/frame"
POLL_INTERVAL_SECONDS = 0.35
JPEG_QUALITY = 60
DIFF_THRESHOLD = 8.0
CHAT_CONTENT_TOP_RATIO = 0.10
CHAT_CONTENT_BOTTOM_RATIO = 0.22
CHAT_CONTENT_SIDE_MARGIN_RATIO = 0.04
SCROLL_ANALYSIS_WIDTH = 256
SCROLL_MIN_SHIFT = 12
SCROLL_MAX_SHIFT = 240
SCROLL_OVERLAP_MIN_RATIO = 0.55
SCROLL_MATCH_THRESHOLD = 0.92
SCROLL_IMPROVEMENT_THRESHOLD = 0.005
SCROLL_REFINEMENT_WINDOW = 10
SCROLL_TIE_EPSILON = 0.002
OVERLAP_SEARCH_WIDTH = 192
OVERLAP_MIN_HEIGHT = 48
OVERLAP_MAX_MEAN_DIFF = 5.0
OVERLAP_ACTIVE_PIXEL_THRESHOLD = 245
OVERLAP_ACTIVE_COLUMN_RATIO = 0.06
OVERLAP_MIN_WIDTH = 120


@dataclass
class WindowTarget:
    hwnd: int
    title: str
    left: int
    top: int
    width: int
    height: int


if hasattr(ctypes, "windll"):
    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32
    PW_RENDERFULLCONTENT = 0x00000002
else:  # pragma: no cover - non-Windows guard
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


RESAMPLE_BILINEAR = getattr(Image, "Resampling", Image).BILINEAR


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def list_windows() -> list[str]:
    seen = []
    for title in gw.getAllTitles():
        cleaned = title.strip()
        if cleaned and cleaned not in seen:
            seen.append(cleaned)
    return seen


def resolve_window(title: str) -> Optional[WindowTarget]:
    matches = gw.getWindowsWithTitle(title)
    for win in matches:
        if not win.title.strip():
            continue
        if win.width <= 0 or win.height <= 0:
            continue
        if win.isMinimized:
            continue
        return WindowTarget(
            hwnd=int(win._hWnd),
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

    rect = RECT()
    if not user32.GetClientRect(target.hwnd, ctypes.byref(rect)):
        return None

    width = rect.right - rect.left
    height = rect.bottom - rect.top
    if width <= 0 or height <= 0:
        return None

    hwnd_dc = user32.GetDC(target.hwnd)
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
            mem_dc,
            bitmap,
            0,
            height,
            pixel_buffer,
            ctypes.byref(bmi),
            0,
        )
        if rows != height:
            return None

        return Image.frombuffer("RGB", (width, height), pixel_buffer, "raw", "BGRX", 0, 1).copy()
    finally:
        gdi32.SelectObject(mem_dc, old_bitmap)
        gdi32.DeleteObject(bitmap)
        gdi32.DeleteDC(mem_dc)
        user32.ReleaseDC(target.hwnd, hwnd_dc)


def capture_window_prefer_window_api(sct: mss, target: WindowTarget) -> tuple[Image.Image, str]:
    image = capture_window_via_printwindow(target)
    if image is not None:
        return image, "window"
    return capture_window(sct, target), "screen"


def diff_bbox(previous: Image.Image, current: Image.Image) -> Optional[tuple[int, int, int, int]]:
    diff = ImageChops.difference(previous, current)
    bbox = diff.getbbox()
    if not bbox:
        return None

    # Avoid uploading noise from cursor shimmer or tiny rendering changes.
    stat = ImageStat.Stat(diff.crop(bbox).convert("L"))
    if stat.mean[0] < DIFF_THRESHOLD:
        return None
    return bbox


def chat_content_bbox(image: Image.Image) -> tuple[int, int, int, int]:
    side_margin = max(8, int(image.width * CHAT_CONTENT_SIDE_MARGIN_RATIO))
    top = max(0, int(image.height * CHAT_CONTENT_TOP_RATIO))
    bottom = min(image.height, image.height - int(image.height * CHAT_CONTENT_BOTTOM_RATIO))
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


def _prepare_scroll_analysis_roi(image: Image.Image) -> tuple[Image.Image, float]:
    left, top, right, bottom = chat_content_bbox(image)
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
    return [(0.8 * edge_value) + (0.2 * ink_value) for edge_value, ink_value in zip(edge_column, ink_column)]


def _signature_similarity(previous: list[float], current: list[float], shift: int) -> float:
    if shift > 0:
        overlap = len(previous) - shift
        if overlap <= 0:
            return 0.0
        previous_overlap = previous[shift:]
        current_overlap = current[:overlap]
    elif shift < 0:
        overlap = len(previous) + shift
        if overlap <= 0:
            return 0.0
        previous_overlap = previous[:overlap]
        current_overlap = current[-shift:]
    else:
        previous_overlap = previous
        current_overlap = current

    mean_abs_diff = sum(
        abs(previous_value - current_value)
        for previous_value, current_value in zip(previous_overlap, current_overlap)
    ) / len(previous_overlap)
    return max(0.0, 1.0 - (mean_abs_diff / 255.0))


def _best_scroll_shift_from_signature(
    previous_signature: list[float],
    current_signature: list[float],
    shifts: list[int],
) -> tuple[int, float]:
    best_shift = 0
    best_score = 0.0
    for shift in shifts:
        if shift == 0:
            continue
        score = _signature_similarity(previous_signature, current_signature, shift)
        if score > best_score + SCROLL_TIE_EPSILON:
            best_score = score
            best_shift = shift
            continue
        if abs(score - best_score) <= SCROLL_TIE_EPSILON and best_shift != 0:
            if abs(shift) < abs(best_shift):
                best_score = score
                best_shift = shift
    return best_shift, best_score


def _overlap_similarity(previous: Image.Image, current: Image.Image, shift: int) -> float:
    width, height = previous.size
    if shift > 0:
        overlap_height = height - shift
        if overlap_height <= 0:
            return 0.0
        previous_overlap = previous.crop((0, shift, width, height))
        current_overlap = current.crop((0, 0, width, overlap_height))
    elif shift < 0:
        overlap_height = height + shift
        if overlap_height <= 0:
            return 0.0
        previous_overlap = previous.crop((0, 0, width, overlap_height))
        current_overlap = current.crop((0, -shift, width, height))
    else:
        previous_overlap = previous
        current_overlap = current

    return _similarity_score(previous_overlap, current_overlap)


def _best_scroll_shift(previous: Image.Image, current: Image.Image, shifts: range) -> tuple[int, float]:
    best_shift = 0
    best_score = 0.0
    for shift in shifts:
        if shift == 0:
            continue
        score = _overlap_similarity(previous, current, shift)
        if score > best_score + SCROLL_TIE_EPSILON:
            best_score = score
            best_shift = shift
            continue
        if abs(score - best_score) <= SCROLL_TIE_EPSILON and best_shift != 0:
            if abs(shift) < abs(best_shift):
                best_score = score
                best_shift = shift
    return best_shift, best_score


def _scroll_shift_candidates(min_shift: int, max_shift: int) -> list[int]:
    negative = range(-max_shift, -min_shift + 1)
    positive = range(min_shift, max_shift + 1)
    return [*negative, *positive]


def detect_scroll_shift(previous: Image.Image, current: Image.Image) -> Optional[int]:
    previous_small, scale = _prepare_scroll_analysis_roi(previous)
    current_small, _ = _prepare_scroll_analysis_roi(current)
    if previous_small.size != current_small.size:
        return None

    width, height = previous_small.size
    if width <= 0 or height <= 0:
        return None

    min_shift = max(2, int(SCROLL_MIN_SHIFT * scale))
    max_shift = min(int(SCROLL_MAX_SHIFT * scale), int(height * (1.0 - SCROLL_OVERLAP_MIN_RATIO)))
    if max_shift < min_shift:
        return None

    previous_signature = _scroll_signature(previous_small)
    current_signature = _scroll_signature(current_small)
    stationary_score = _signature_similarity(previous_signature, current_signature, 0)
    best_shift_small, best_score = _best_scroll_shift_from_signature(
        previous_signature,
        current_signature,
        _scroll_shift_candidates(min_shift, max_shift),
    )

    if best_shift_small == 0:
        return None
    if best_score < SCROLL_MATCH_THRESHOLD:
        return None
    if (best_score - stationary_score) < SCROLL_IMPROVEMENT_THRESHOLD:
        return None

    left, top, right, bottom = chat_content_bbox(previous)
    previous_roi = previous.crop((left, top, right, bottom)).convert("L")
    current_roi = current.crop((left, top, right, bottom)).convert("L")
    coarse_shift = int(round(best_shift_small / scale))
    refine_min = max(-SCROLL_MAX_SHIFT, coarse_shift - SCROLL_REFINEMENT_WINDOW)
    refine_max = min(SCROLL_MAX_SHIFT, coarse_shift + SCROLL_REFINEMENT_WINDOW)
    if refine_max < refine_min:
        return None

    refined_shift, refined_score = _best_scroll_shift(
        previous_roi,
        current_roi,
        range(refine_min, refine_max + 1),
    )
    if refined_shift == 0:
        return None
    if refined_score < SCROLL_MATCH_THRESHOLD:
        return None
    if (refined_score - _similarity_score(previous_roi, current_roi)) < SCROLL_IMPROVEMENT_THRESHOLD:
        return None

    if abs(refined_shift) < SCROLL_MIN_SHIFT:
        return None
    return refined_shift


def detect_incremental_chat_bbox(
    previous: Image.Image,
    current: Image.Image,
) -> Optional[tuple[int, int, int, int]]:
    scroll_shift = detect_scroll_shift(previous, current)
    if scroll_shift is None:
        return diff_bbox(previous, current)

    left, top, right, bottom = chat_content_bbox(current)
    if scroll_shift > 0:
        band_top = max(top, bottom - scroll_shift)
        if band_top >= bottom:
            return None
        return (left, band_top, right, bottom)

    band_bottom = min(bottom, top - scroll_shift)
    if band_bottom <= top:
        return None
    return (left, top, right, band_bottom)


def encode_jpeg(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _active_column_bounds(images: list[Image.Image]) -> tuple[int, int]:
    width = min(image.width for image in images)
    height = min(image.height for image in images)
    threshold = max(8, int(height * OVERLAP_ACTIVE_COLUMN_RATIO))
    combined_counts = [0] * width

    for image in images:
        gray = image.convert("L")
        pixels = gray.load()
        for x in range(width):
            count = 0
            for y in range(height):
                if pixels[x, y] < OVERLAP_ACTIVE_PIXEL_THRESHOLD:
                    count += 1
            combined_counts[x] = max(combined_counts[x], count)

    active = [index for index, count in enumerate(combined_counts) if count >= threshold]
    if not active:
        return 0, width
    return min(active), max(active) + 1


def trim_patch_overlap(
    previous_patch: Image.Image,
    previous_bbox: tuple[int, int, int, int],
    current_patch: Image.Image,
    current_bbox: tuple[int, int, int, int],
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    prev_left, prev_top, prev_right, prev_bottom = previous_bbox
    curr_left, curr_top, curr_right, curr_bottom = current_bbox
    overlap_left = max(prev_left, curr_left)
    overlap_right = min(prev_right, curr_right)
    overlap_width = overlap_right - overlap_left
    if overlap_width < OVERLAP_MIN_WIDTH:
        return current_patch, current_bbox

    prev_crop = previous_patch.crop((overlap_left - prev_left, 0, overlap_right - prev_left, previous_patch.height))
    curr_crop = current_patch.crop((overlap_left - curr_left, 0, overlap_right - curr_left, current_patch.height))
    if prev_crop.width <= 0 or curr_crop.width <= 0:
        return current_patch, current_bbox

    active_left, active_right = _active_column_bounds([prev_crop, curr_crop])
    prev_crop = prev_crop.crop((active_left, 0, active_right, prev_crop.height))
    curr_crop = curr_crop.crop((active_left, 0, active_right, curr_crop.height))
    if prev_crop.width < OVERLAP_MIN_WIDTH or curr_crop.width < OVERLAP_MIN_WIDTH:
        return current_patch, current_bbox

    scale = 1.0
    if prev_crop.width > OVERLAP_SEARCH_WIDTH:
        scale = OVERLAP_SEARCH_WIDTH / prev_crop.width
        resized_height_prev = max(24, int(prev_crop.height * scale))
        resized_height_curr = max(24, int(curr_crop.height * scale))
        prev_crop = prev_crop.resize((OVERLAP_SEARCH_WIDTH, resized_height_prev), RESAMPLE_BILINEAR)
        curr_crop = curr_crop.resize((OVERLAP_SEARCH_WIDTH, resized_height_curr), RESAMPLE_BILINEAR)

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
        return current_patch, current_bbox

    trim_top = int(round(best_overlap / scale))
    if trim_top <= 0 or trim_top >= current_patch.height:
        return current_patch, current_bbox

    trimmed_patch = current_patch.crop((0, trim_top, current_patch.width, current_patch.height))
    trimmed_bbox = (curr_left, curr_top + trim_top, curr_right, curr_bottom)
    return trimmed_patch, trimmed_bbox


class CaptureApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Visible Window Capture Client")
        self.root.geometry("560x260")

        self.server_var = tk.StringVar(value=SERVER_URL)
        self.window_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Idle")
        self.capture_mode_var = tk.StringVar(value="Preferred capture path: window API")
        self.session_var = tk.StringVar(value=str(uuid.uuid4()))
        self.sequence = 0
        self.worker: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.last_uploaded_patch: Optional[Image.Image] = None
        self.last_uploaded_bbox: Optional[tuple[int, int, int, int]] = None

        self._build_ui()
        self.refresh_windows()

    def _build_ui(self) -> None:
        frame = ttk.Frame(self.root, padding=16)
        frame.pack(fill=tk.BOTH, expand=True)

        consent = (
            "This client only captures a window you explicitly choose. "
            "Capture is visible, runs only after you press Start, and can be stopped at any time."
        )
        ttk.Label(frame, text=consent, wraplength=500, justify=tk.LEFT).pack(anchor=tk.W, pady=(0, 14))

        ttk.Label(frame, text="Session ID").pack(anchor=tk.W)
        ttk.Entry(frame, textvariable=self.session_var).pack(fill=tk.X, pady=(0, 10))

        ttk.Label(frame, text="Server URL").pack(anchor=tk.W)
        ttk.Entry(frame, textvariable=self.server_var).pack(fill=tk.X, pady=(0, 10))

        header = ttk.Frame(frame)
        header.pack(fill=tk.X)
        ttk.Label(header, text="Window").pack(side=tk.LEFT)
        ttk.Button(header, text="Refresh", command=self.refresh_windows).pack(side=tk.RIGHT)

        self.window_combo = ttk.Combobox(frame, textvariable=self.window_var, state="readonly")
        self.window_combo.pack(fill=tk.X, pady=(0, 10))

        controls = ttk.Frame(frame)
        controls.pack(fill=tk.X, pady=(0, 10))
        self.start_button = ttk.Button(controls, text="Start Capture", command=self.start_capture)
        self.start_button.pack(side=tk.LEFT)
        self.stop_button = ttk.Button(controls, text="Stop", command=self.stop_capture, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(8, 0))

        ttk.Label(frame, textvariable=self.capture_mode_var).pack(anchor=tk.W, pady=(0, 4))
        ttk.Label(frame, textvariable=self.status_var, foreground="#0b5").pack(anchor=tk.W)

    def refresh_windows(self) -> None:
        titles = list_windows()
        self.window_combo["values"] = titles
        if titles and not self.window_var.get():
            self.window_var.set(titles[0])
        self.status_var.set(f"Loaded {len(titles)} visible window titles")

    def start_capture(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        if not self.window_var.get():
            messagebox.showerror("No window selected", "Choose a window before starting capture.")
            return

        self.sequence = 0
        self.stop_event.clear()
        self.last_uploaded_patch = None
        self.last_uploaded_bbox = None
        self.worker = threading.Thread(target=self._capture_loop, daemon=True)
        self.worker.start()
        self.start_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)
        self.status_var.set("Capturing selected window")

    def stop_capture(self) -> None:
        self.stop_event.set()
        self.start_button.configure(state=tk.NORMAL)
        self.stop_button.configure(state=tk.DISABLED)
        self.status_var.set("Stopping capture")

    def set_status(self, text: str) -> None:
        self.root.after(0, self.status_var.set, text)

    def set_capture_mode(self, text: str) -> None:
        self.root.after(0, self.capture_mode_var.set, text)

    def on_capture_finished(self) -> None:
        self.root.after(0, self.start_button.configure, {"state": tk.NORMAL})
        self.root.after(0, self.stop_button.configure, {"state": tk.DISABLED})

    def _capture_loop(self) -> None:
        previous_frame: Optional[Image.Image] = None
        with mss() as sct:
            while not self.stop_event.is_set():
                target = resolve_window(self.window_var.get())
                if not target:
                    self.set_status("Selected window unavailable or minimized")
                    time.sleep(POLL_INTERVAL_SECONDS)
                    continue

                current_frame, capture_mode = capture_window_prefer_window_api(sct, target)
                if capture_mode == "window":
                    self.set_capture_mode("Capture path: window API")
                else:
                    self.set_capture_mode("Capture path: screen fallback (may be occluded)")

                if previous_frame is None:
                    previous_frame = current_frame
                    self.set_status("Baseline captured, waiting for changes")
                    time.sleep(POLL_INTERVAL_SECONDS)
                    continue

                changed = detect_incremental_chat_bbox(previous_frame, current_frame)
                previous_frame = current_frame

                if not changed:
                    time.sleep(POLL_INTERVAL_SECONDS)
                    continue

                patch = current_frame.crop(changed)
                if self.last_uploaded_patch is not None and self.last_uploaded_bbox is not None:
                    patch, changed = trim_patch_overlap(
                        self.last_uploaded_patch,
                        self.last_uploaded_bbox,
                        patch,
                        changed,
                    )
                    if patch.height <= 0 or patch.width <= 0:
                        time.sleep(POLL_INTERVAL_SECONDS)
                        continue
                try:
                    self._upload_patch(target, patch, changed)
                    self.last_uploaded_patch = patch.copy()
                    self.last_uploaded_bbox = changed
                    self.set_status(
                        f"Uploaded delta frame #{self.sequence - 1} via {capture_mode} capture"
                    )
                except Exception as exc:
                    self.set_status(f"Upload failed: {exc}")

                time.sleep(POLL_INTERVAL_SECONDS)

        self.on_capture_finished()
        self.set_status("Capture stopped")

    def _upload_patch(
        self,
        target: WindowTarget,
        patch: Image.Image,
        changed_bbox: tuple[int, int, int, int],
    ) -> None:
        left, top, right, bottom = changed_bbox
        payload = {
            "session_id": self.session_var.get().strip() or "session",
            "sequence": self.sequence,
            "timestamp_utc": utc_now(),
            "window_title": target.title,
            "screen_size": {"width": target.width, "height": target.height},
            "changed_region": {
                "left": left,
                "top": top,
                "width": right - left,
                "height": bottom - top,
            },
            "image_jpeg_base64": encode_jpeg(patch),
        }
        response = requests.post(self.server_var.get().strip(), json=payload, timeout=10)
        response.raise_for_status()
        self.sequence += 1


def main() -> None:
    root = tk.Tk()
    app = CaptureApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.stop_capture(), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    main()
