"""Test TrOCR-Large-Handwritten on test pages.

TrOCR is a line-level model — it needs pre-segmented text lines.
We use a simple approach: split the image into horizontal strips
and feed each strip to TrOCR. For production, use a proper line
detector (Surya, CRAFT, etc).
"""
import os, time, json, glob, gc
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

TEST_DIR = "/home/omen/Documents/Project/Richiebot/scans/test_pages"
OUT_DIR = "/home/omen/Documents/Project/Richiebot/scans/output/trocr_handwritten"
os.makedirs(OUT_DIR, exist_ok=True)

MODEL = "microsoft/trocr-large-handwritten"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading TrOCR-Large-Handwritten...")
t0 = time.time()
processor = TrOCRProcessor.from_pretrained(MODEL)
model = VisionEncoderDecoderModel.from_pretrained(MODEL).to(DEVICE).eval()
print(f"Loaded in {time.time()-t0:.1f}s, GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB")


def segment_lines(img, min_height=30, max_lines=60):
    """Simple horizontal strip segmentation.
    Splits image into strips based on whitespace gaps.
    """
    import numpy as np
    gray = np.array(img.convert('L'))
    # Row-wise mean intensity — white rows are ~255, text rows are darker
    row_means = gray.mean(axis=1)
    threshold = 240  # below this = likely has text

    # Find runs of text rows
    is_text = row_means < threshold
    lines = []
    in_line = False
    start = 0

    for i, t in enumerate(is_text):
        if t and not in_line:
            start = i
            in_line = True
        elif not t and in_line:
            if i - start >= min_height:
                # Add some padding
                y1 = max(0, start - 5)
                y2 = min(gray.shape[0], i + 5)
                lines.append((y1, y2))
            in_line = False

    if in_line and len(is_text) - start >= min_height:
        lines.append((max(0, start - 5), gray.shape[0]))

    # If no lines detected, just split into strips
    if not lines:
        h = gray.shape[0]
        strip_h = max(min_height * 2, h // max_lines)
        lines = [(i, min(i + strip_h, h)) for i in range(0, h, strip_h)]

    return lines[:max_lines]


def ocr_line(img_crop):
    """Run TrOCR on a single line image."""
    # TrOCR expects 384x384 — processor handles resizing
    pixel_values = processor(images=img_crop, return_tensors="pt").pixel_values.to(DEVICE)
    with torch.no_grad():
        generated_ids = model.generate(pixel_values, max_new_tokens=128)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text


pages = sorted(glob.glob(os.path.join(TEST_DIR, "*.png")))
results = {}

for img_path in pages:
    fname = os.path.basename(img_path)
    page_id = fname.replace(".png", "")
    print(f"\nProcessing: {fname}")

    img = Image.open(img_path).convert("RGB")
    t1 = time.time()

    # Segment into lines
    line_regions = segment_lines(img)
    print(f"  Detected {len(line_regions)} line regions")

    # OCR each line
    texts = []
    for y1, y2 in line_regions:
        crop = img.crop((0, y1, img.width, y2))
        # Skip very thin strips
        if crop.height < 15:
            continue
        try:
            text = ocr_line(crop)
            if text.strip():
                texts.append(text)
        except Exception as e:
            pass

    full_text = "\n".join(texts)
    elapsed = time.time() - t1

    results[fname] = {"chars": len(full_text), "time_sec": round(elapsed, 2), "lines": len(texts)}
    out_path = os.path.join(OUT_DIR, f"{page_id}.txt")
    with open(out_path, "w") as f:
        f.write(full_text)

    print(f"  {len(texts)} lines OCR'd, {len(full_text)} chars, {elapsed:.1f}s")

    torch.cuda.empty_cache()
    gc.collect()

# Save summary
with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
    json.dump(results, f, indent=2)

total = sum(r["time_sec"] for r in results.values())
print(f"\nDone. {len(results)} pages, {total:.1f}s total, {total/len(results):.1f}s avg")
