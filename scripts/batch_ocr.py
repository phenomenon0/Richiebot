"""Batch OCR runner with model routing.

Reads classified pages from SQLite, routes each to the appropriate model:
- typed    → Marker (via marker-pdf library)
- handwritten → Qwen2.5-VL via Ollama
- hardest  → Chandra OCR 2 via HF + rotation detection
- diagram  → Qwen2.5-VL via Ollama (description prompt)
- blank    → skip

Processes in batches with checkpoint/resume via SQLite status field.
"""
import os, sys, time, json, base64, io, sqlite3, gc
import requests
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from init_db import DB_PATH

OLLAMA_URL = "http://localhost:11434/api/generate"


class OllamaOCR:
    """Qwen2.5-VL via Ollama for handwritten + diagram pages."""

    def __init__(self, model="qwen2.5vl:7b"):
        self.model = model
        self.name = model

    def ocr(self, image_path, prompt=None):
        if prompt is None:
            prompt = ("Perform OCR on this document image. Extract ALL text exactly as written. "
                      "For handwritten text, transcribe your best reading even if messy. "
                      "Output in markdown format.")
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()

        t0 = time.time()
        resp = requests.post(OLLAMA_URL, json={
            "model": self.model,
            "prompt": prompt,
            "images": [img_b64],
            "stream": False,
            "options": {"num_predict": 4096, "temperature": 0.1},
        }, timeout=600)
        elapsed = time.time() - t0
        text = resp.json().get("response", "")
        return text, elapsed


class ChandraOCR:
    """Chandra OCR 2 via HuggingFace for hardest pages + rotation."""

    def __init__(self):
        self.manager = None
        self.name = "chandra_ocr2"

    def _load(self):
        if self.manager is not None:
            return
        from chandra.model import InferenceManager
        print("  Loading Chandra OCR 2...")
        t0 = time.time()
        self.manager = InferenceManager(method='hf')
        print(f"  Chandra loaded in {time.time()-t0:.1f}s")

    def ocr(self, image_path, rotation=0):
        self._load()
        from chandra.model.schema import BatchInputItem
        img = Image.open(image_path).convert("RGB")
        if rotation:
            img = img.rotate(rotation, expand=True)

        t0 = time.time()
        batch = [BatchInputItem(image=img, prompt_type='ocr_layout')]
        result = self.manager.generate(batch)[0]
        elapsed = time.time() - t0
        text = result.markdown if hasattr(result, 'markdown') else str(result)
        return text, elapsed

    def unload(self):
        if self.manager is not None:
            del self.manager
            self.manager = None
            torch.cuda.empty_cache()
            gc.collect()


class MarkerOCR:
    """Marker for typed pages — runs marker on the source PDF."""

    def __init__(self):
        self.name = "marker"
        self._cache = {}  # pdf_path -> {page_num: text}

    def ocr(self, image_path, pdf_path=None, page_num=None):
        """For typed pages, we use Marker's PDF-level output.
        Since Marker processes whole PDFs, we run it once and cache."""
        # Marker is already run — check if output exists
        # For batch pipeline, fall back to Qwen for individual pages
        # (Marker doesn't have a clean per-page API)
        # Use a simple fallback: read the image and do basic extraction
        t0 = time.time()

        # Try pdftext (Marker's text extractor) for typed pages
        try:
            from pdftext.extraction import plain_text_output
            if pdf_path and os.path.exists(pdf_path):
                if pdf_path not in self._cache:
                    pages_text = plain_text_output(pdf_path)
                    self._cache[pdf_path] = pages_text
                pages = self._cache[pdf_path]
                if page_num and page_num <= len(pages):
                    text = pages[page_num - 1]
                    return text, time.time() - t0
        except Exception:
            pass

        # Fallback: use pdftext directly
        return "", time.time() - t0


def score_quality(text):
    """Quick quality score heuristic."""
    if not text.strip():
        return 0.0
    words = text.split()
    n = len(words)
    if n == 0:
        return 0.0
    unique = len(set(w.lower() for w in words)) / n
    reasonable = sum(1 for w in words if 2 <= len(w) <= 15) / n
    has_digits = 10 if any(c.isdigit() for c in text) else 0
    punct = min(len(set(c for c in text if c in ".,;:!?()-/")), 5) * 3
    score = unique * 40 + reasonable * 20 + has_digits + punct
    if n < 10:
        score *= 0.5
    return max(0, min(100, score))


def process_batch(conn, limit=0, model_filter=None):
    """Process unprocessed pages from SQLite."""
    c = conn.cursor()

    # Get pages to process
    query = "SELECT id, pdf_name, page_num, image_path, class, rotation_hint FROM pages WHERE status = 'classified'"
    if model_filter:
        class_map = {"marker": "typed", "qwen": "handwritten", "chandra": "hardest", "diagram": "diagram"}
        target_class = class_map.get(model_filter)
        if target_class:
            query += f" AND class = '{target_class}'"
    query += " ORDER BY pdf_name, page_num"
    if limit > 0:
        query += f" LIMIT {limit}"

    pages = c.execute(query).fetchall()
    if not pages:
        print("No pages to process.")
        return

    print(f"Processing {len(pages)} pages...")

    # Count by class
    classes = {}
    for _, _, _, _, cls, _ in pages:
        classes[cls] = classes.get(cls, 0) + 1
    for cls, count in sorted(classes.items()):
        print(f"  {cls}: {count}")

    # Sort pages by class so we process all of one model type before switching.
    # Order: blank (skip) → typed (Marker, CPU) → handwritten+diagram (Qwen, GPU) → hardest (Chandra, GPU)
    # This avoids GPU memory conflicts between Chandra and Ollama.
    class_order = {"blank": 0, "typed": 1, "handwritten": 2, "diagram": 3, "hardest": 4}
    pages.sort(key=lambda p: (class_order.get(p[4], 5), p[1], p[2]))

    # Initialize models lazily
    qwen = OllamaOCR("minicpm-v")
    chandra = ChandraOCR()
    marker = MarkerOCR()

    processed = 0
    errors = 0
    t_start = time.time()
    current_model_group = None

    for page_id, pdf_name, page_num, image_path, page_class, rotation_hint in pages:
        # Unload Chandra before Qwen pages and vice versa
        if page_class in ("handwritten", "diagram") and current_model_group == "chandra":
            print("\n  Unloading Chandra, switching to Qwen...")
            chandra.unload()
        current_model_group = "chandra" if page_class == "hardest" else "qwen"
        # Mark as processing
        c.execute("UPDATE pages SET status = 'processing' WHERE id = ?", (page_id,))
        conn.commit()

        try:
            if page_class == "blank":
                text, elapsed = "", 0
                model_used = "skip"
            elif page_class == "typed":
                # Try Marker's pdftext, fall back to Qwen
                pdf_path = _find_pdf(pdf_name)
                text, elapsed = marker.ocr(image_path, pdf_path, page_num)
                model_used = "marker"
                if not text.strip():
                    # Fallback to Qwen for pages where pdftext fails
                    text, elapsed = qwen.ocr(image_path)
                    model_used = qwen.name
            elif page_class == "handwritten":
                text, elapsed = qwen.ocr(image_path)
                model_used = qwen.name
            elif page_class == "hardest":
                # Use rotation hint from classifier
                text, elapsed = chandra.ocr(image_path, rotation=rotation_hint)
                model_used = "chandra_ocr2"
            elif page_class == "diagram":
                text, elapsed = qwen.ocr(image_path,
                    prompt="Describe this document image in detail. Extract all text, describe any diagrams, "
                           "charts, or tables. Preserve structure in markdown format.")
                model_used = qwen.name
            else:
                text, elapsed = qwen.ocr(image_path)
                model_used = qwen.name

            quality = score_quality(text)
            chars = len(text)

            c.execute("""UPDATE pages SET
                         status = 'done', model_used = ?, text = ?, chars = ?,
                         time_sec = ?, quality_score = ?, processed_at = CURRENT_TIMESTAMP
                         WHERE id = ?""",
                      (model_used, text, chars, round(elapsed, 2), round(quality, 1), page_id))
            processed += 1

        except Exception as e:
            c.execute("UPDATE pages SET status = 'error', model_used = ? WHERE id = ?",
                      (str(e)[:200], page_id))
            errors += 1
            print(f"  ERROR on {pdf_name} p{page_num}: {e}")

        conn.commit()

        # Progress
        total_elapsed = time.time() - t_start
        rate = processed / total_elapsed if total_elapsed > 0 else 0
        remaining = (len(pages) - processed - errors) / rate if rate > 0 else 0
        print(f"  [{processed+errors}/{len(pages)}] {pdf_name} p{page_num}: "
              f"{page_class} → {model_used if 'model_used' in dir() else '?'} "
              f"({chars if 'chars' in dir() else 0} chars, {elapsed if 'elapsed' in dir() else 0:.1f}s) "
              f"ETA: {remaining/60:.0f}min")

    # Cleanup
    chandra.unload()

    total_elapsed = time.time() - t_start
    print(f"\nDone: {processed} processed, {errors} errors in {total_elapsed:.1f}s")


def _find_pdf(pdf_name):
    """Find the source PDF file by name."""
    scans_dir = os.path.join(os.path.dirname(__file__), "..", "scans")
    # Check common locations
    for subdir in ["", "nas/Brian_Dema", "nas/Richard_Dema", "nas/Richard_+_Brian"]:
        path = os.path.join(scans_dir, subdir, pdf_name)
        if os.path.exists(path):
            return path
    return None


def print_status(conn):
    """Print current processing status."""
    c = conn.cursor()
    print("\n--- Pipeline Status ---")
    for row in c.execute("SELECT status, COUNT(*) FROM pages GROUP BY status ORDER BY status"):
        print(f"  {row[0]:15s} {row[1]:5d}")
    print()
    for row in c.execute("SELECT class, COUNT(*), SUM(CASE WHEN status='done' THEN 1 ELSE 0 END) FROM pages GROUP BY class"):
        print(f"  {row[0]:15s} {row[1]:5d} total, {row[2] or 0:5d} done")
    total_time = c.execute("SELECT SUM(time_sec) FROM pages WHERE status='done'").fetchone()[0]
    if total_time:
        print(f"\n  Total processing time: {total_time:.0f}s ({total_time/3600:.1f}h)")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Max pages to process (0=all)")
    parser.add_argument("--filter", choices=["marker", "qwen", "chandra", "diagram"], help="Only process this model's pages")
    parser.add_argument("--status", action="store_true", help="Print status and exit")
    args = parser.parse_args()

    conn = sqlite3.connect(DB_PATH)

    if args.status:
        print_status(conn)
        conn.close()
        return

    process_batch(conn, limit=args.limit, model_filter=args.filter)
    print_status(conn)
    conn.close()


if __name__ == "__main__":
    main()
