"""Test VLM OCR via Ollama on extracted test pages."""
import os, time, json, base64, requests

MODEL = "qwen2.5vl:7b"
TEST_DIR = "/home/omen/Documents/Project/Richiebot/scans/test_pages"
OUT_DIR = "/home/omen/Documents/Project/Richiebot/scans/output/qwen25vl_7b"
OLLAMA_URL = "http://localhost:11434/api/generate"

os.makedirs(OUT_DIR, exist_ok=True)

results = {}
test_files = sorted([f for f in os.listdir(TEST_DIR) if f.endswith(".png")])

for fname in test_files:
    img_path = os.path.join(TEST_DIR, fname)
    print(f"\n{'='*60}")
    print(f"Processing: {fname}")

    # Encode image as base64
    with open(img_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    t1 = time.time()
    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": MODEL,
            "prompt": "Perform OCR on this document image. Extract ALL text exactly as it appears, preserving structure, headers, bullet points, and table layouts. For handwritten text, do your best to transcribe even if messy. Output the text in markdown format.",
            "images": [img_b64],
            "stream": False,
            "options": {"num_predict": 4096, "temperature": 0.1},
        }, timeout=300)
        elapsed = time.time() - t1

        data = resp.json()
        result = data.get("response", "")

        print(f"  Time: {elapsed:.1f}s")
        print(f"  Output length: {len(result)} chars")
        print(f"  Preview: {result[:300]}...")

        results[fname] = {
            "text": result,
            "time_sec": round(elapsed, 2),
            "chars": len(result),
        }

        # Save individual result
        out_path = os.path.join(OUT_DIR, fname.replace(".png", ".txt"))
        with open(out_path, "w") as f:
            f.write(result)

    except Exception as e:
        elapsed = time.time() - t1
        print(f"  ERROR ({elapsed:.1f}s): {e}")
        results[fname] = {"error": str(e), "time_sec": round(elapsed, 2), "chars": 0}

# Save summary
with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
    json.dump(results, f, indent=2)

print(f"\n\n{'='*60}")
print("DONE.")
total_time = sum(r["time_sec"] for r in results.values())
print(f"Total inference time: {total_time:.1f}s for {len(results)} pages")
if results:
    print(f"Avg: {total_time/len(results):.1f}s/page")
