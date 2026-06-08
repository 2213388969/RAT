"""Standalone script to visualize sidebar/chat boundary detection on a screenshot."""

import argparse
import cv2
import numpy as np
from PIL import Image


def detect_sidebar_boundary(image_path: str):
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)
    h, w = arr.shape[:2]

    # Need two frames to compute diff - use the image itself and a slightly shifted version
    # For a single image, we simulate by detecting the vertical gap in the image itself
    # Convert to grayscale and look for the divider line
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    # Method: look for vertical edges that span most of the image height
    # Edge detection along vertical direction
    edges = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    edge_mag = np.abs(edges)

    # Sum edges per column
    col_sums = np.sum(edge_mag, axis=0)

    # Find the divider: a column with strong vertical edges spanning much of the height
    # This is the separator line between sidebar and chat area
    min_sidebar_width = int(w * 0.08)
    max_sidebar_width = int(w * 0.85)  # sidebar can be up to 85% of window

    # Also try the motion-based approach (from client.py)
    # Simulate by using edge magnitude as "change mask"
    mask = (edge_mag > 30).astype(np.uint8)

    # Morphological close
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    col_sums_mask = np.sum(mask, axis=0)
    max_col = np.max(col_sums_mask)
    gap_threshold = max_col * 0.05
    min_gap_width = 3
    # Find the divider line: a narrow, tall peak in column sums.
    # The divider is a vertical line spanning most of the window height,
    # so it appears as a very thin but very high spike in the column chart.
    min_sidebar_width = int(w * 0.08)
    max_sidebar_width = int(w * 0.85)  # sidebar can be up to 85% of window

    left_boundary = 0
    if max_col > 0:
        # A divider peak should be: very high (>= 90% of max) and very narrow (<= 5px wide)
        peak_threshold = max_col * 0.9
        max_peak_width = 5
        center_x = w // 2

        # Collect all candidate peaks
        candidates = []
        x = min_sidebar_width
        while x < max_sidebar_width:
            if col_sums_mask[x] >= peak_threshold:
                peak_start = x
                peak_end = x
                while peak_end < max_sidebar_width and col_sums_mask[peak_end] >= peak_threshold:
                    peak_end += 1
                peak_width = peak_end - peak_start
                if peak_width <= max_peak_width:
                    peak_center = peak_start + peak_width // 2
                    peak_height = col_sums_mask[peak_center]
                    print(f"Divider candidate at x={peak_center}: width={peak_width}px, height={peak_height:.0f} (max={max_col:.0f}, ratio={peak_height/max_col:.2f})")
                    candidates.append(peak_center)
                x = peak_end
            else:
                x += 1

        # Pick the candidate closest to window center
        if candidates:
            left_boundary = min(candidates, key=lambda cx: abs(cx - center_x))
            print(f"Selected divider at x={left_boundary} (closest to center x={center_x})")

    # Draw result
    result = arr.copy()
    if left_boundary > 0:
        cv2.line(result, (left_boundary, 0), (left_boundary, h), (0, 255, 0), 2)
        cv2.putText(result, f"Chat boundary: x={left_boundary}", (left_boundary + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(result, f"Sidebar: 0-{left_boundary}px", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(result, f"Chat: {left_boundary}-{w}px", (left_boundary + 10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(result, "No sidebar boundary detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Also draw column sum chart at the bottom
    chart_h = 100
    chart = np.zeros((chart_h, w, 3), dtype=np.uint8)
    if max_col > 0:
        normalized = (col_sums_mask / max_col * (chart_h - 10)).astype(int)
        for x in range(w):
            cv2.line(chart, (x, chart_h), (x, chart_h - normalized[x]), (255, 200, 0), 1)
        if left_boundary > 0:
            cv2.line(chart, (left_boundary, 0), (left_boundary, chart_h), (0, 255, 0), 1)
        # Draw gap threshold line
        thresh_y = chart_h - int(gap_threshold / max_col * (chart_h - 10))
        cv2.line(chart, (0, thresh_y), (w, thresh_y), (0, 0, 255), 1)

    # Stack image and chart
    result_with_chart = np.vstack([result, chart])

    output_path = image_path.rsplit(".", 1)[0] + "_boundary.png"
    cv2.imwrite(output_path, cv2.cvtColor(result_with_chart, cv2.COLOR_RGB2BGR))
    print(f"Result saved to: {output_path}")
    print(f"Image size: {w}x{h}")
    print(f"Sidebar boundary: x={left_boundary}" if left_boundary > 0 else "No sidebar detected")
    print(f"Sidebar width: {left_boundary}px ({left_boundary/w*100:.1f}%)" if left_boundary > 0 else "")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize sidebar/chat boundary detection")
    parser.add_argument("--input", default="image.png", help="Input screenshot")
    args = parser.parse_args()
    detect_sidebar_boundary(args.input)
