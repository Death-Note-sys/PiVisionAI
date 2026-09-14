"""
Interactively select a tight crop box on a static image and print the JSON
snippet to paste into references_manifest.json. Requires a GUI-capable
OpenCV build (opencv-python, not opencv-python-headless) — if this errors
with something like "function not implemented" or a Qt/GTK backend error,
that's why; run `pip install opencv-python` to fix.

Usage:
    python benchmarks/ai_identify/select_roi.py good/quarter_heads.jpg

Drag a box with the mouse, press ENTER or SPACE to confirm, ESC to cancel.
"""
import sys
import os
import json
import cv2


def main():
    if len(sys.argv) < 2:
        print("Usage: python select_roi.py <path-to-image>")
        sys.exit(1)

    path = sys.argv[1]
    img = cv2.imread(path)
    if img is None:
        print(f"Could not read image: {path}")
        sys.exit(1)

    print("Drag a tight box around the distinguishing feature.")
    print("Press ENTER or SPACE when done, ESC to cancel.")

    x, y, w, h = cv2.selectROI("Select ROI - " + os.path.basename(path), img, showCrosshair=True)
    cv2.destroyAllWindows()

    if w == 0 or h == 0:
        print("No region selected.")
        sys.exit(1)

    filename = os.path.basename(path)
    snippet = {filename: {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}}
    print("\nAdd this under \"good\" or \"bad\" in references_manifest.json:\n")
    print(json.dumps(snippet, indent=2))


if __name__ == "__main__":
    main()
