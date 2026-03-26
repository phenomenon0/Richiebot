"""Test Surya OCR on test pages."""
import os, time, json, glob
from PIL import Image

TEST_DIR = "/home/omen/Documents/Project/Richiebot/scans/test_pages"
OUT_DIR = "/home/omen/Documents/Project/Richiebot/scans/output/surya"
os.makedirs(OUT_DIR, exist_ok=True)

from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor

print("Loading Surya models...")
t0 = time.time()
det = DetectionPredictor()
rec = RecognitionPredictor()
print(f"Loaded in {time.time()-t0:.1f}s")

pages = sorted(glob.glob(os.path.join(TEST_DIR, "*.png")))
results = {}

for img_path in pages:
    fname = os.path.basename(img_path)
    page_id = fname.replace(".png", "")
    print(f"\nProcessing: {fname}")

    img = Image.open(img_path).convert("RGB")
    t1 = time.time()

    # Detect text lines
    det_result = det([img])
    # Recognize text
    rec_result = rec([img], det_result)

    elapsed = time.time() - t1

    # Extract text from recognition results
    lines = []
    if rec_result and len(rec_result) > 0:
        for text_line in rec_result[0].text_lines:
            lines.append(text_line.text)

    text = "\n".join(lines)

    results[fname] = {"chars": len(text), "time_sec": round(elapsed, 2)}
    out_path = os.path.join(OUT_DIR, f"{page_id}.txt")
    with open(out_path, "w") as f:
        f.write(text)

    print(f"  {len(lines)} lines, {len(text)} chars, {elapsed:.1f}s")

# Save summary
with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
    json.dump(results, f, indent=2)

total = sum(r["time_sec"] for r in results.values())
print(f"\nDone. {len(results)} pages, {total:.1f}s total, {total/len(results):.1f}s avg")
