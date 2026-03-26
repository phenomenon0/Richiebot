"""Test DeepSeek-OCR via Ollama on test pages."""
import os, time, json, base64, glob
import requests

MODEL = "deepseek-ocr"
TEST_DIR = "/home/omen/Documents/Project/Richiebot/scans/test_pages"
OUT_DIR = "/home/omen/Documents/Project/Richiebot/scans/output/deepseek_ocr2"
OLLAMA_URL = "http://localhost:11434/api/generate"
os.makedirs(OUT_DIR, exist_ok=True)

pages = sorted(glob.glob(os.path.join(TEST_DIR, "*.png")))
results = {}

for img_path in pages:
    fname = os.path.basename(img_path)
    page_id = fname.replace(".png", "")

    with open(img_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    print(f"Processing: {fname}...", end=" ", flush=True)
    t0 = time.time()
    resp = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "prompt": "Perform OCR on this document image. Extract ALL text exactly as written, preserving structure. For handwritten text, transcribe your best reading even if messy. Output in markdown.",
        "images": [img_b64],
        "stream": False,
        "options": {"num_predict": 4096, "temperature": 0.1},
    }, timeout=600)
    elapsed = time.time() - t0
    text = resp.json().get("response", "")

    results[fname] = {"chars": len(text), "time_sec": round(elapsed, 2)}
    with open(os.path.join(OUT_DIR, f"{page_id}.txt"), "w") as f:
        f.write(text)
    print(f"{len(text)} chars, {elapsed:.1f}s")

with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
    json.dump(results, f, indent=2)

total = sum(r["time_sec"] for r in results.values())
print(f"\nDone. {len(results)} pages, {total:.1f}s total, {total/len(results):.1f}s avg")
