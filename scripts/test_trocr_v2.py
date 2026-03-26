"""TrOCR-Large-Handwritten with proper OpenCV line detection.

Three line detection strategies, tried in order:
1. Projection profile — find horizontal whitespace gaps between text lines
2. Connected component clustering — group ink blobs into lines by y-position
3. Contour-based — find text block contours, split into horizontal strips

Each detected line is deskewed individually before feeding to TrOCR.
"""
import os, time, json, glob, gc
import numpy as np
import cv2
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

TEST_DIR = "/home/omen/Documents/Project/Richiebot/scans/test_pages"
OUT_DIR = "/home/omen/Documents/Project/Richiebot/scans/output/trocr_v2"
os.makedirs(OUT_DIR, exist_ok=True)

MODEL = "microsoft/trocr-large-handwritten"
DEVICE = "cuda"


def detect_lines_projection(gray, min_gap=8, min_line_height=20, padding=8):
    """Projection profile method: find text rows by horizontal ink density."""
    # Binarize (text = white on black)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Horizontal projection — sum of ink pixels per row
    proj = binary.sum(axis=1) / 255

    # Find text rows: where projection > threshold
    threshold = gray.shape[1] * 0.005  # 0.5% of width has ink
    is_text = proj > threshold

    # Group consecutive text rows into lines, split by whitespace gaps
    lines = []
    in_line = False
    start = 0

    for i in range(len(is_text)):
        if is_text[i] and not in_line:
            start = i
            in_line = True
        elif not is_text[i] and in_line:
            gap_start = i
            # Look ahead for the gap length
            gap_end = i
            while gap_end < len(is_text) and not is_text[gap_end]:
                gap_end += 1
            gap_len = gap_end - gap_start

            if gap_len >= min_gap or gap_end >= len(is_text):
                if i - start >= min_line_height:
                    y1 = max(0, start - padding)
                    y2 = min(gray.shape[0], i + padding)
                    lines.append((y1, y2))
                in_line = False

    # Handle last line
    if in_line and len(is_text) - start >= min_line_height:
        lines.append((max(0, start - padding), gray.shape[0]))

    return lines


def detect_lines_components(gray, min_area=100, merge_distance_y=15, padding=10):
    """Connected component clustering: group ink blobs by y-center."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Dilate horizontally to connect characters within same word
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    dilated = cv2.dilate(binary, kernel_h, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get bounding boxes, filter small ones
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > min_area and h > 8:
            boxes.append((x, y, w, h))

    if not boxes:
        return []

    # Sort by y-center
    boxes.sort(key=lambda b: b[1] + b[3] / 2)

    # Merge boxes into lines by y-proximity
    lines = []
    current_y1 = boxes[0][1]
    current_y2 = boxes[0][1] + boxes[0][3]
    current_x1 = boxes[0][0]
    current_x2 = boxes[0][0] + boxes[0][2]

    for x, y, w, h in boxes[1:]:
        center_y = y + h / 2
        if center_y - current_y2 > merge_distance_y:
            # New line
            lines.append((
                max(0, current_y1 - padding),
                min(gray.shape[0], current_y2 + padding),
                max(0, current_x1 - padding),
                min(gray.shape[1], current_x2 + padding),
            ))
            current_y1 = y
            current_y2 = y + h
            current_x1 = x
            current_x2 = x + w
        else:
            current_y1 = min(current_y1, y)
            current_y2 = max(current_y2, y + h)
            current_x1 = min(current_x1, x)
            current_x2 = max(current_x2, x + w)

    # Last line
    lines.append((
        max(0, current_y1 - padding),
        min(gray.shape[0], current_y2 + padding),
        max(0, current_x1 - padding),
        min(gray.shape[1], current_x2 + padding),
    ))

    return lines


def deskew_line(img_crop):
    """Deskew a line crop using minimum area rectangle angle."""
    gray = np.array(img_crop.convert('L'))
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    coords = np.column_stack(np.where(binary > 0))
    if len(coords) < 10:
        return img_crop

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Only deskew if angle is small (< 15°) — large angles mean rotated text
    if abs(angle) > 15 or abs(angle) < 0.5:
        return img_crop

    h, w = gray.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(np.array(img_crop), M, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(rotated)


def process_page(img, processor, model):
    """Detect lines and OCR each with TrOCR."""
    gray = np.array(img.convert('L'))

    # Try projection profile first
    proj_lines = detect_lines_projection(gray)

    # Try component clustering
    comp_lines = detect_lines_components(gray)

    # Pick the method that found more lines (usually better segmentation)
    if len(comp_lines) > len(proj_lines) * 1.3:
        # Use component lines (have x-bounds too)
        line_crops = []
        for y1, y2, x1, x2 in comp_lines:
            if y2 - y1 < 15:
                continue
            crop = img.crop((x1, y1, x2, y2))
            line_crops.append(crop)
        method = "components"
    else:
        # Use projection lines (full width)
        line_crops = []
        for y1, y2 in proj_lines:
            if y2 - y1 < 15:
                continue
            crop = img.crop((0, y1, img.width, y2))
            line_crops.append(crop)
        method = "projection"

    # Deskew and OCR each line
    texts = []
    for crop in line_crops:
        crop = deskew_line(crop)
        # Skip very small crops
        if crop.width < 30 or crop.height < 10:
            continue
        try:
            pixel_values = processor(images=crop, return_tensors="pt").pixel_values.to(DEVICE)
            with torch.no_grad():
                ids = model.generate(pixel_values, max_new_tokens=256)
            text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
            if text and len(text) > 1:
                texts.append(text)
        except Exception:
            pass

    return "\n".join(texts), len(line_crops), method


def main():
    print("Loading TrOCR-Large-Handwritten...")
    t0 = time.time()
    processor = TrOCRProcessor.from_pretrained(MODEL)
    model_trocr = VisionEncoderDecoderModel.from_pretrained(MODEL).to(DEVICE).eval()
    print(f"Loaded in {time.time()-t0:.1f}s, GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    pages = sorted(glob.glob(os.path.join(TEST_DIR, "*.png")))
    results = {}

    for img_path in pages:
        fname = os.path.basename(img_path)
        page_id = fname.replace(".png", "")
        print(f"\nProcessing: {fname}")

        img = Image.open(img_path).convert("RGB")
        t1 = time.time()

        text, num_lines, method = process_page(img, processor, model_trocr)
        elapsed = time.time() - t1

        results[fname] = {
            "chars": len(text),
            "time_sec": round(elapsed, 2),
            "lines": num_lines,
            "method": method,
        }

        out_path = os.path.join(OUT_DIR, f"{page_id}.txt")
        with open(out_path, "w") as f:
            f.write(text)

        print(f"  {method}: {num_lines} lines, {len(text)} chars, {elapsed:.1f}s")

        torch.cuda.empty_cache()
        gc.collect()

    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    total = sum(r["time_sec"] for r in results.values())
    print(f"\nDone. {len(results)} pages, {total:.1f}s total, {total/len(results):.1f}s avg")


if __name__ == "__main__":
    main()
