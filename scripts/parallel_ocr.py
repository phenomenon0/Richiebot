"""
Parallel OCR pipeline with rotation detection.
Uses ThreadPoolExecutor to send 2 concurrent requests to Ollama.
"""
import os, sys, time, json, base64, io
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from PIL import Image

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5vl:7b"
WORKERS = 2  # Match OLLAMA_NUM_PARALLEL


def ocr_image(img: Image.Image, prompt: str = None) -> dict:
    """Send image to Ollama VLM."""
    if prompt is None:
        prompt = (
            "Perform OCR on this document image. Extract ALL text exactly as written. "
            "For handwritten text, transcribe your best reading even if messy. "
            "Output plain text only, no commentary."
        )
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    t0 = time.time()
    resp = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "prompt": prompt,
        "images": [img_b64],
        "stream": False,
        "options": {"num_predict": 4096, "temperature": 0.1},
    }, timeout=300)
    elapsed = time.time() - t0
    text = resp.json().get("response", "")
    return {"text": text, "time_sec": round(elapsed, 2), "chars": len(text)}


def score_text(text: str) -> float:
    """Heuristic quality score."""
    if not text.strip():
        return 0.0
    words = text.split()
    n = len(words)
    if n == 0:
        return 0.0
    unique_ratio = len(set(w.lower() for w in words)) / n
    reasonable = sum(1 for w in words if 2 <= len(w) <= 15) / n
    has_digits = 10 if any(c.isdigit() for c in text) else 0
    punct = min(len(set(c for c in text if c in ".,;:!?()-/")), 5) * 3
    score = unique_ratio * 40 + reasonable * 20 + has_digits + punct
    if n < 10:
        score *= 0.5
    return max(0, min(100, score))


def try_rotation(img: Image.Image, angle: int) -> dict:
    """OCR at a specific rotation."""
    rotated = img.copy().rotate(angle, expand=True) if angle != 0 else img.copy()
    result = ocr_image(rotated)
    result["angle"] = angle
    result["quality_score"] = score_text(result["text"])
    return result


def best_rotation_parallel(img: Image.Image) -> dict:
    """Try 0° and 180° in parallel, then 90°/270° if needed."""
    # Phase 1: try 0° and 180° concurrently
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(try_rotation, img, a): a for a in [0, 180]}
        results = {}
        for f in as_completed(futures):
            r = f.result()
            results[r["angle"]] = r

    best = max(results.values(), key=lambda r: r["quality_score"])
    if best["quality_score"] >= 60:
        return best

    # Phase 2: try 90° and 270° concurrently
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(try_rotation, img, a): a for a in [90, 270]}
        for f in as_completed(futures):
            r = f.result()
            results[r["angle"]] = r

    return max(results.values(), key=lambda r: r["quality_score"])


def process_page(args):
    """Process a single page — used as thread target."""
    img_path, fname = args
    img = Image.open(img_path).convert("RGB")
    img.load()  # Force load before threading
    t0 = time.time()
    result = best_rotation_parallel(img)
    elapsed = time.time() - t0
    return {
        "filename": fname,
        "text": result["text"],
        "angle": result["angle"],
        "quality_score": result["quality_score"],
        "chars": result["chars"],
        "total_time": round(elapsed, 2),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Directory with PNG page images")
    parser.add_argument("output_dir", help="Output directory for results")
    parser.add_argument("--limit", type=int, default=0, help="Max pages to process (0=all)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    pages = sorted([f for f in os.listdir(args.input_dir) if f.endswith(".png")])
    if args.limit > 0:
        pages = pages[:args.limit]

    print(f"Processing {len(pages)} pages with {WORKERS} parallel workers")
    print(f"Model: {MODEL}")
    print()

    results = []
    total_t0 = time.time()

    for fname in pages:
        img_path = os.path.join(args.input_dir, fname)
        r = process_page((img_path, fname))
        results.append(r)

        # Save text
        out_path = os.path.join(args.output_dir, fname.replace(".png", ".txt"))
        with open(out_path, "w") as f:
            f.write(r["text"])

        print(f"  {fname}: angle={r['angle']}° score={r['quality_score']:.1f} chars={r['chars']} time={r['total_time']}s")

    total_elapsed = time.time() - total_t0

    # Save summary
    summary = {
        "total_pages": len(results),
        "total_time_sec": round(total_elapsed, 2),
        "avg_time_per_page": round(total_elapsed / len(results), 2) if results else 0,
        "pages_per_minute": round(len(results) / (total_elapsed / 60), 2) if total_elapsed > 0 else 0,
        "results": results,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"DONE: {len(results)} pages in {total_elapsed:.1f}s")
    print(f"Avg: {summary['avg_time_per_page']}s/page, {summary['pages_per_minute']} pages/min")
    scores = [r["quality_score"] for r in results]
    rotated = [r for r in results if r["angle"] != 0]
    print(f"Avg quality score: {sum(scores)/len(scores):.1f}")
    print(f"Pages auto-rotated: {len(rotated)}/{len(results)}")


if __name__ == "__main__":
    main()
