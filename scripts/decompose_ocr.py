"""
Phase 4 Prototype: Page Decomposition + Multi-Rotation OCR Pipeline

For the hardest pages (composited, overlapping, multi-orientation):
1. Try all 4 rotations, pick best
2. Detect text regions via connected components
3. Estimate per-region orientation
4. Mask/crop each region, deskew independently
5. OCR each region separately
6. Reassemble with spatial metadata
"""
import os, sys, time, json, base64, math
import requests
import numpy as np
from PIL import Image
import cv2

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5vl:7b"
TEST_DIR = "/home/omen/Documents/Project/Richiebot/scans/test_pages"
OUT_DIR = "/home/omen/Documents/Project/Richiebot/scans/output/decomposed"
os.makedirs(OUT_DIR, exist_ok=True)


def ocr_image(img: Image.Image, prompt: str = None) -> dict:
    """Send image to Ollama VLM, return text + timing."""
    if prompt is None:
        prompt = (
            "Perform OCR on this document image. Extract ALL text exactly as written. "
            "For handwritten text, transcribe your best reading even if messy. "
            "Output plain text only, no commentary."
        )

    # Convert PIL to PNG bytes -> base64
    import io
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


def score_ocr_quality(text: str) -> float:
    """Heuristic quality score for OCR output.
    Higher = more likely to be real text vs hallucination/noise.
    """
    if not text.strip():
        return 0.0

    score = 0.0
    words = text.split()
    num_words = len(words)

    if num_words == 0:
        return 0.0

    # Reward: more unique words (penalize repetition like "Planned vs Actual" x80)
    unique_ratio = len(set(w.lower() for w in words)) / num_words
    score += unique_ratio * 40  # max 40

    # Reward: reasonable word lengths (2-15 chars)
    reasonable = sum(1 for w in words if 2 <= len(w) <= 15)
    score += (reasonable / num_words) * 20  # max 20

    # Reward: has digits (dates, numbers = real content)
    has_digits = any(c.isdigit() for c in text)
    score += 10 if has_digits else 0

    # Reward: has punctuation variety
    punct_types = set(c for c in text if c in ".,;:!?()-/")
    score += min(len(punct_types) * 3, 15)  # max 15

    # Penalty: very short output for a full page
    if num_words < 10:
        score *= 0.5

    # Penalty: placeholder text
    placeholders = sum(1 for w in words if w.lower() in ("[handwritten", "text]", "[handwritten]"))
    score -= placeholders * 5

    return max(0, min(100, score))


def try_all_rotations(img: Image.Image) -> dict:
    """Try OCR at 0°, 90°, 180°, 270° — return best result."""
    results = {}
    for angle in [0, 180, 90, 270]:
        print(f"    Trying {angle}°...", end=" ", flush=True)
        rotated = img.rotate(angle, expand=True) if angle != 0 else img
        result = ocr_image(rotated)
        result["angle"] = angle
        result["quality_score"] = score_ocr_quality(result["text"])
        results[angle] = result
        print(f"{result['chars']} chars, score={result['quality_score']:.1f}, {result['time_sec']}s")

        # Early exit if we get a high-quality result
        if result["quality_score"] > 70:
            print(f"    -> High quality at {angle}°, skipping remaining rotations")
            break

    # Pick best by quality score
    best_angle = max(results, key=lambda a: results[a]["quality_score"])
    best = results[best_angle]
    best["all_rotations"] = {a: {"score": r["quality_score"], "chars": r["chars"]} for a, r in results.items()}
    return best


def detect_text_regions(img_path: str) -> list:
    """Detect text regions using OpenCV morphological operations.
    Returns list of (x, y, w, h, estimated_angle) tuples.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return []

    # Binarize
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Dilate to connect nearby text into regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 20))
    dilated = cv2.dilate(binary, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    h_img, w_img = img.shape
    min_area = (h_img * w_img) * 0.005  # ignore tiny regions (<0.5% of page)

    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (w, h), angle = rect

        # Normalize angle to text orientation
        if w < h:
            angle = angle + 90

        area = w * h
        if area < min_area:
            continue

        # Get bounding box
        x, y, bw, bh = cv2.boundingRect(cnt)
        regions.append({
            "bbox": (x, y, bw, bh),
            "angle": round(angle, 1),
            "area": round(area),
            "center": (round(cx), round(cy)),
        })

    # Sort by area (largest first)
    regions.sort(key=lambda r: r["area"], reverse=True)
    return regions


def extract_and_ocr_regions(img_path: str, regions: list) -> list:
    """Crop each detected region, deskew, and OCR independently."""
    img = Image.open(img_path)
    results = []

    for i, region in enumerate(regions[:6]):  # max 6 regions
        x, y, w, h = region["bbox"]
        # Add padding
        pad = 20
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img.width, x + w + pad)
        y2 = min(img.height, y + h + pad)

        crop = img.crop((x1, y1, x2, y2))

        # Skip tiny crops
        if crop.width < 100 or crop.height < 50:
            continue

        print(f"  Region {i+1}: ({x},{y}) {w}x{h}, angle={region['angle']}°")

        # Try the detected angle and 180° flip
        best_result = None
        for try_angle in [0, 180]:
            rotated = crop.rotate(try_angle, expand=True) if try_angle != 0 else crop
            result = ocr_image(rotated)
            result["quality_score"] = score_ocr_quality(result["text"])
            if best_result is None or result["quality_score"] > best_result["quality_score"]:
                best_result = result
                best_result["used_rotation"] = try_angle

        best_result["region"] = region
        best_result["region_index"] = i
        results.append(best_result)
        print(f"    -> {best_result['chars']} chars, score={best_result['quality_score']:.1f}, rot={best_result['used_rotation']}°")

    return results


def process_page(img_path: str, fname: str) -> dict:
    """Full decomposition pipeline for one page."""
    print(f"\n{'='*70}")
    print(f"PROCESSING: {fname}")

    img = Image.open(img_path)
    page_result = {
        "filename": fname,
        "size": img.size,
        "passes": [],
    }

    # PASS 1: Full-page multi-rotation OCR
    print("\n  PASS 1: Full-page multi-rotation OCR")
    full_page = try_all_rotations(img)
    page_result["passes"].append({
        "type": "full_page_rotation",
        "best_angle": full_page["angle"],
        "quality_score": full_page["quality_score"],
        "text": full_page["text"],
        "time_sec": full_page["time_sec"],
        "all_rotations": full_page.get("all_rotations", {}),
    })

    # If full-page score is good enough, skip decomposition
    if full_page["quality_score"] >= 60:
        print(f"\n  Full-page quality={full_page['quality_score']:.1f} — good enough, skipping decomposition")
        page_result["final_text"] = full_page["text"]
        page_result["strategy"] = "full_page"
        page_result["best_angle"] = full_page["angle"]
        return page_result

    # PASS 2: Region decomposition
    print(f"\n  PASS 2: Region decomposition (full-page score={full_page['quality_score']:.1f} too low)")
    regions = detect_text_regions(img_path)
    print(f"  Detected {len(regions)} text regions")

    if not regions:
        page_result["final_text"] = full_page["text"]
        page_result["strategy"] = "full_page_fallback"
        return page_result

    region_results = extract_and_ocr_regions(img_path, regions)
    page_result["passes"].append({
        "type": "region_decomposition",
        "num_regions": len(regions),
        "regions_ocrd": len(region_results),
        "results": [{
            "region_index": r["region_index"],
            "bbox": r["region"]["bbox"],
            "rotation": r["used_rotation"],
            "quality_score": r["quality_score"],
            "text": r["text"],
            "chars": r["chars"],
        } for r in region_results],
    })

    # Combine: use best full-page if it beats sum of regions
    region_texts = [r["text"] for r in region_results if r["quality_score"] > 20]
    combined_region_text = "\n\n---\n\n".join(region_texts)
    avg_region_score = np.mean([r["quality_score"] for r in region_results]) if region_results else 0

    if avg_region_score > full_page["quality_score"]:
        page_result["final_text"] = combined_region_text
        page_result["strategy"] = "decomposed"
        print(f"\n  Using decomposed regions (avg score={avg_region_score:.1f} > full page={full_page['quality_score']:.1f})")
    else:
        page_result["final_text"] = full_page["text"]
        page_result["strategy"] = "full_page_best"
        print(f"\n  Full page still better (score={full_page['quality_score']:.1f} >= regions={avg_region_score:.1f})")

    return page_result


def main():
    # Test on the hardest pages
    hard_pages = [
        "R_-_Written_Notes_p1.png",   # composited, overlapping — hardest
        "R_-_Written_Notes_p3.png",   # messy cursive
        "R_-_Written_Notes_p4.png",   # cursive product notes
        "R_-_Org_Charts_p2.png",      # handwritten org chart
    ]

    all_results = {}
    total_t0 = time.time()

    for fname in hard_pages:
        img_path = os.path.join(TEST_DIR, fname)
        if not os.path.exists(img_path):
            print(f"SKIP: {fname} not found")
            continue

        result = process_page(img_path, fname)
        all_results[fname] = result

        # Save individual text output
        out_txt = os.path.join(OUT_DIR, fname.replace(".png", ".txt"))
        with open(out_txt, "w") as f:
            f.write(result["final_text"])

        # Save full result JSON
        out_json = os.path.join(OUT_DIR, fname.replace(".png", ".json"))
        with open(out_json, "w") as f:
            json.dump(result, f, indent=2, default=str)

    total_elapsed = time.time() - total_t0

    # Summary
    print(f"\n\n{'='*70}")
    print("DECOMPOSITION PIPELINE SUMMARY")
    print(f"{'='*70}")
    for fname, r in all_results.items():
        strategy = r["strategy"]
        angle = r.get("best_angle", "N/A")
        score = r["passes"][0]["quality_score"]
        chars = len(r["final_text"])
        print(f"  {fname}")
        print(f"    Strategy: {strategy}, Best angle: {angle}°, Score: {score:.1f}, Output: {chars} chars")
    print(f"\nTotal time: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
