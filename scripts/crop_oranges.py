# scripts/crop_oranges.py
#
# Create "orange-only" crops from the original dataset using simple color-based detection.

import os
from pathlib import Path
import cv2
import numpy as np

# ============================================================
# Config
# ============================================================

# Root of your original dataset
# D:\GAN\Data\orange_dataset\
DATA_ROOT = r"D:\GAN\Data\orange_dataset"

# Output root for cropped oranges
# D:\GAN\Data\orange_crops\
OUT_ROOT = r"D:\GAN\Data\orange_crops"

# Target size for crops (matches your GAN)
OUTPUT_SIZE = 64  # 64x64

# Whether to show a few debug windows (optional, set to False for speed)
SHOW_DEBUG = False

# HSV ranges for orange color (may need slight tuning per dataset)
# Hue in OpenCV HSV is [0,179]; orange ~ [5, 25]
LOWER_ORANGE = np.array([5, 80, 80])
UPPER_ORANGE = np.array([25, 255, 255])


# ============================================================
# Helpers
# ============================================================

def find_orange_bbox(img_bgr):
    """
    Find bounding box around the largest orange-ish blob in the image.
    Returns (x, y, w, h) or None if nothing found.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_ORANGE, UPPER_ORANGE)

    # Optional: clean up noise with morphology
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Largest contour by area
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)

    # Ignore tiny blobs
    if w * h < 100:  # area threshold; adjust if needed
        return None

    return x, y, w, h, mask


def crop_and_resize(img_bgr, bbox):
    h, w = img_bgr.shape[:2]

    if bbox is not None:
        x, y, bw, bh, _ = bbox

        # Expand bbox a bit so we keep context around fruit
        pad_x = int(0.2 * bw)
        pad_y = int(0.2 * bh)

        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(w, x + bw + pad_x)
        y1 = min(h, y + bh + pad_y)

        crop = img_bgr[y0:y1, x0:x1]
    else:
        # Fallback: center crop
        side = min(h, w)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        crop = img_bgr[y0:y0+side, x0:x0+side]

    crop_resized = cv2.resize(crop, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_AREA)
    return crop_resized


def process_class(class_name):
    in_dir = Path(DATA_ROOT) / class_name
    out_dir = Path(OUT_ROOT) / class_name
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        image_paths.extend(in_dir.glob(ext))

    print(f"\nClass '{class_name}': found {len(image_paths)} images.")

    if not image_paths:
        return

    for idx, img_path in enumerate(sorted(image_paths), start=1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [WARN] Could not read {img_path}, skipping.")
            continue

        bbox = find_orange_bbox(img)
        crop = crop_and_resize(img, bbox)

        out_path = out_dir / f"{class_name}_{idx:04d}.jpg"
        cv2.imwrite(str(out_path), crop)

        if idx % 50 == 0:
            print(f"  Processed {idx}/{len(image_paths)} images...")

        if SHOW_DEBUG and idx <= 3:
            # Show a couple of debug windows
            debug = img.copy()
            if bbox is not None:
                x, y, w, h, mask = bbox
                cv2.rectangle(debug, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.imshow("mask", mask)
            cv2.imshow("original", img)
            cv2.imshow("crop", crop)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    print(f"  Done. Cropped images saved to: {out_dir}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("========================================")
    print("Cropping orange dataset to orange-only images")
    print("Input root: ", DATA_ROOT)
    print("Output root:", OUT_ROOT)
    print("Target size:", OUTPUT_SIZE)
    print("========================================")

    # These should match your folder names under Data\orange_dataset\
    classes = ["ripe_oranges", "unripe_orange"]

    for cls in classes:
        process_class(cls)

    print("\nAll classes processed. Cropped dataset is ready.")
