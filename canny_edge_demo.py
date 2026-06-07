import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Canny edge detection and locate the chat top-right plus input top-left."
    )
    parser.add_argument(
        "--input",
        default="new_wechat.png",
        help="Input image path. Defaults to new_wechat.png in the current directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="canny_output",
        help="Output directory. Defaults to canny_output.",
    )
    parser.add_argument(
        "--blur-kernel",
        type=int,
        default=5,
        help="Gaussian blur kernel size. Defaults to 5.",
    )
    return parser.parse_args()


def ensure_odd(value):
    return value if value % 2 == 1 else value + 1


def load_image(image_path):
    image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")
    return image


def save_image(image_path, image):
    suffix = image_path.suffix.lower() or ".png"
    success, encoded = cv2.imencode(suffix, image)
    if not success:
        raise RuntimeError(f"Failed to save image: {image_path}")
    encoded.tofile(str(image_path))


def colorize_edges(base_image, edges, color=(0, 0, 255)):
    overlay = base_image.copy()
    edge_mask = edges > 0
    overlay[edge_mask] = color
    mixed = cv2.addWeighted(base_image, 0.78, overlay, 0.22, 0)
    mixed[edge_mask] = color
    return mixed


def make_preview_panel(images, labels):
    annotated = []
    font = cv2.FONT_HERSHEY_SIMPLEX

    for image, label in zip(images, labels):
        if image.ndim == 2:
            canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            canvas = image.copy()

        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 40), (245, 245, 245), -1)
        cv2.putText(canvas, label, (20, 27), font, 0.8, (20, 20, 20), 2, cv2.LINE_AA)
        annotated.append(canvas)

    return cv2.hconcat(annotated)


def edge_ratio(edges):
    return float(np.count_nonzero(edges)) / float(edges.size)


def strongest_diff_index(values, start, end):
    window = values[start:end]
    if window.size == 0:
        raise ValueError("Empty search window; cannot locate boundary.")
    return start + int(np.argmax(window))


def detect_chat_left_boundary(gray):
    height, width = gray.shape
    y0 = int(height * 0.14)
    y1 = int(height * 0.82)
    profile = gray[y0:y1, :].astype(np.float32).mean(axis=0)
    profile_diff = np.abs(np.diff(profile))

    search_start = int(width * 0.20)
    search_end = int(width * 0.60)
    window = profile_diff[search_start:search_end]
    max_score = float(window.max())
    threshold = max(12.0, max_score * 0.60)
    candidates = [
        search_start + idx
        for idx, score in enumerate(window)
        if float(score) >= threshold
    ]
    if candidates:
        return candidates[0]
    return strongest_diff_index(profile_diff, search_start, search_end)


def detect_chat_bounds(gray):
    height, width = gray.shape
    main_x_start = int(width * 0.22)
    main_x_end = width - 8

    right_panel = gray[:, main_x_start:main_x_end].astype(np.int16)
    row_mean = right_panel.mean(axis=1)
    row_diff = np.abs(np.diff(row_mean))

    body_panel = gray[120 : height - 16, :].astype(np.int16)
    col_mean = body_panel.mean(axis=0)
    col_diff = np.abs(np.diff(col_mean))

    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    col_edge_counts = edges[120:, :].sum(axis=0) // 255

    chat_left_x = detect_chat_left_boundary(gray)
    chat_right_x = strongest_diff_index(col_edge_counts, width - 40, width - 2)
    chat_top_y = strongest_diff_index(row_diff, 90, 220)

    fallback_input_top_y = strongest_diff_index(
        row_diff, int(height * 0.58), int(height * 0.74)
    )

    return {
        "chat_left_x": chat_left_x,
        "chat_right_x": chat_right_x,
        "chat_top_y": chat_top_y,
        "fallback_input_top_y": fallback_input_top_y,
    }


def detect_input_top_line(image, chat_left_x, chat_right_x):
    height, width = image.shape[:2]
    roi_x0 = max(int(width * 0.26), chat_left_x - 20)
    roi_x1 = width - 10
    roi_y0 = int(height * 0.60)
    roi_y1 = int(height * 0.96)

    roi = image[roi_y0:roi_y1, roi_x0:roi_x1]

    # Use color-derived lightness instead of plain grayscale so dark-theme borders keep contrast.
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    lightness = lab[:, :, 0]
    blurred = cv2.GaussianBlur(lightness, (5, 5), 0)
    edges = cv2.Canny(blurred, 15, 45)

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=60,
        minLineLength=int(roi.shape[1] * 0.35),
        maxLineGap=45,
    )

    expected_width = chat_right_x - chat_left_x
    min_coverage = int(expected_width * 0.72)
    candidates = []
    if lines is not None:
        for line in lines[:, 0]:
            x1, y1, x2, y2 = map(int, line)
            if abs(y1 - y2) > 3:
                continue

            global_y = roi_y0 + (y1 + y2) // 2
            if not (int(height * 0.62) <= global_y <= int(height * 0.95)):
                continue

            start_x = roi_x0 + min(x1, x2)
            end_x = roi_x0 + max(x1, x2)
            length = end_x - start_x
            if length < min_coverage:
                continue

            left_gap = abs(start_x - chat_left_x)
            right_gap = abs(end_x - chat_right_x)
            candidates.append((start_x, global_y, end_x, length, left_gap + right_gap))

    if not candidates:
        return None

    # Prefer a line that spans most of the chat width and aligns with the panel borders.
    candidates.sort(key=lambda item: (item[4], -item[3], item[1], item[0]))
    return candidates[0]


def detect_layout_points(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    layout = detect_chat_bounds(gray)

    input_line = detect_input_top_line(
        image, layout["chat_left_x"], layout["chat_right_x"]
    )
    if input_line is not None:
        input_left_x, input_top_y, input_right_x, _, _ = input_line
        input_line_found = True
    else:
        input_left_x = layout["chat_left_x"]
        input_right_x = layout["chat_right_x"]
        input_top_y = layout["fallback_input_top_y"]
        input_line_found = False

    layout.update(
        {
            "input_top_y": input_top_y,
            "input_left_x": input_left_x,
            "input_right_x": input_right_x,
            "input_top_left": (input_left_x, input_top_y),
            "chat_top_right": (layout["chat_right_x"], layout["chat_top_y"]),
            "input_line_found": input_line_found,
        }
    )
    return layout


def annotate_layout(image, layout):
    annotated = image.copy()

    input_point = layout["input_top_left"]
    chat_point = layout["chat_top_right"]
    chat_left_x = layout["chat_left_x"]
    chat_right_x = layout["chat_right_x"]
    chat_top_y = layout["chat_top_y"]
    input_top_y = layout["input_top_y"]
    input_right_x = layout["input_right_x"]

    line_color = (255, 140, 0)
    point_color_a = (0, 0, 255)
    point_color_b = (255, 0, 0)

    cv2.line(annotated, (chat_left_x, 0), (chat_left_x, image.shape[0] - 1), line_color, 2)
    cv2.line(annotated, (chat_right_x, 0), (chat_right_x, image.shape[0] - 1), line_color, 2)
    cv2.line(annotated, (chat_left_x, chat_top_y), (chat_right_x, chat_top_y), line_color, 2)
    cv2.line(annotated, (input_point[0], input_top_y), (input_right_x, input_top_y), line_color, 2)

    cv2.circle(annotated, input_point, 8, point_color_a, -1)
    cv2.circle(annotated, chat_point, 8, point_color_b, -1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        annotated,
        f"input_top_left {input_point}",
        (input_point[0] + 14, input_point[1] - 14),
        font,
        0.7,
        point_color_a,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        f"chat_top_right {chat_point}",
        (max(20, chat_point[0] - 360), chat_point[1] - 14),
        font,
        0.7,
        point_color_b,
        2,
        cv2.LINE_AA,
    )

    return annotated


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    blur_kernel = ensure_odd(max(1, args.blur_kernel))

    image = load_image(input_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

    presets = {
        "sensitive": (20, 60),
        "balanced": (50, 150),
    }

    generated = []
    for name, (low_threshold, high_threshold) in presets.items():
        edges = cv2.Canny(blurred, low_threshold, high_threshold)
        overlay = colorize_edges(image, edges)

        edge_path = output_dir / f"{input_path.stem}_{name}_edges.png"
        overlay_path = output_dir / f"{input_path.stem}_{name}_overlay.png"
        save_image(edge_path, edges)
        save_image(overlay_path, overlay)

        ratio = edge_ratio(edges)
        print(
            f"{name}: low={low_threshold} high={high_threshold} "
            f"edge_pixels={np.count_nonzero(edges)} ratio={ratio:.4%}"
        )

        generated.append(
            {
                "label": f"{name} ({low_threshold}, {high_threshold})",
                "edges": edges,
                "overlay": overlay,
            }
        )

    layout = detect_layout_points(image)
    annotated = annotate_layout(image, layout)
    annotated_path = output_dir / f"{input_path.stem}_layout_points.png"
    save_image(annotated_path, annotated)

    print(f"input_top_left={layout['input_top_left']}")
    print(f"chat_top_right={layout['chat_top_right']}")
    print(
        f"boundaries: left_x={layout['chat_left_x']} right_x={layout['chat_right_x']} "
        f"chat_top_y={layout['chat_top_y']} input_top_y={layout['input_top_y']}"
    )
    print(
        f"input_line_found={layout['input_line_found']} "
        f"fallback_input_top_y={layout['fallback_input_top_y']}"
    )

    panel = make_preview_panel(
        [
            image,
            generated[0]["overlay"],
            generated[1]["overlay"],
            annotated,
        ],
        [
            "original",
            generated[0]["label"] + " overlay",
            generated[1]["label"] + " overlay",
            "detected points",
        ],
    )
    panel_path = output_dir / f"{input_path.stem}_canny_panel.png"
    save_image(panel_path, panel)
    print(f"preview_panel={panel_path}")


if __name__ == "__main__":
    main()
