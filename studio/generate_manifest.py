#!/usr/bin/env python3
"""Generate manifest.json for the OCR QA Studio.

Scans test_pages/ for images and output/ for model results,
producing a single JSON manifest the frontend consumes.

Usage: python3 generate_manifest.py
"""
import os, json, re, glob
from datetime import datetime, timezone

SCANS_DIR = os.path.join(os.path.dirname(__file__), "..", "scans")
TEST_PAGES_DIR = os.path.join(SCANS_DIR, "test_pages")
OUTPUT_DIR = os.path.join(SCANS_DIR, "output")

# --- Page metadata ---

PDF_NAME_MAP = {
    "B_-_Notes1": "B - Notes(1).pdf",
    "B_-_The_Keys_to_Demand_Generation": "B - The Keys to Demand Generation.pdf",
    "R_-_Org_Charts": "R - Org Charts.pdf",
    "R_-_Presenataions": "R - Presenataions.pdf",
    "R_-_Theory_of_the_Business_-_Underlying_Philosophy": "R - Theory of the Business - Underlying Philosophy.pdf",
    "R_-_Written_Notes": "R - Written Notes.pdf",
    "RandB_-_Low_Hanging_Fruit": "R&B - Low Hanging Fruit.pdf",
}

DIFFICULTY_MAP = {
    "B_-_The_Keys_to_Demand_Generation_p1": ("Easy", "Clean typed"),
    "B_-_Notes1_p1": ("Very Hard", "Handwritten notes"),
    "B_-_Notes1_p2": ("Very Hard", "Handwritten notes"),
    "R_-_Org_Charts_p1": ("Medium", "Printed org chart"),
    "R_-_Org_Charts_p2": ("Hard", "Handwritten org chart"),
    "R_-_Presenataions_p9": ("Hard", "Marketing collateral"),
    "R_-_Theory_of_the_Business_-_Underlying_Philosophy_p1": ("Medium", "Annotated book"),
    "R_-_Theory_of_the_Business_-_Underlying_Philosophy_p3": ("Medium", "Annotated book"),
    "R_-_Written_Notes_p1": ("Very Hard", "Composited cursive"),
    "R_-_Written_Notes_p3": ("Hard", "Dense cursive"),
    "R_-_Written_Notes_p4": ("Hard", "Dense cursive"),
    "RandB_-_Low_Hanging_Fruit_p1": ("Medium", "Mixed typed"),
    "RandB_-_Low_Hanging_Fruit_p5": ("Hard", "Table + handwritten"),
}

# --- Model metadata ---

MODEL_META = {
    "chandra_ocr2": {
        "name": "Chandra OCR 2",
        "params": "4B",
        "vram": "15.3GB",
        "type": "vlm",
        "description": "SOTA handwriting (90.8%), March 2026",
    },
    "qwen25vl_7b": {
        "name": "Qwen2.5-VL 7B",
        "params": "7B",
        "vram": "22.8GB",
        "type": "vlm",
        "description": "Best tables (A-), September 2024",
    },
    "minicpm_v": {
        "name": "MiniCPM-V",
        "params": "4B",
        "vram": "5.5GB",
        "type": "vlm",
        "description": "Fast general-purpose, 7.3s/page",
    },
    "decomposed": {
        "name": "Decomposed Pipeline",
        "params": None,
        "vram": None,
        "type": "pipeline",
        "description": "Multi-rotation + region decomposition",
    },
    "marker": {
        "name": "Marker (Surya)",
        "params": None,
        "vram": "4GB",
        "type": "traditional",
        "description": "Traditional OCR baseline, 3.2s/page",
    },
    "glm_ocr": {
        "name": "GLM-OCR",
        "params": "0.9B",
        "vram": "2GB",
        "type": "vlm",
        "description": "Broken on Ollama — empty outputs",
    },
    "trocr_handwritten": {
        "name": "TrOCR-Large HW v1",
        "params": "660M",
        "vram": "2.4GB",
        "type": "specialized",
        "description": "Naive line segmentation (v1)",
    },
    "trocr_v2": {
        "name": "TrOCR-Large HW v2",
        "params": "660M",
        "vram": "2.4GB",
        "type": "specialized",
        "description": "Projection + component line detection, deskew",
    },
    "deepseek_ocr2": {
        "name": "DeepSeek-OCR-2",
        "params": "3B",
        "vram": "6.7GB",
        "type": "vlm",
        "description": "Broken on Ollama — echoes prompt back",
    },
}

# Models to skip (empty/broken beyond repair)
SKIP_MODELS = {"got_ocr", "paddleocr_vl", "parallel_test", "surya", "pipeline"}


def parse_page_id(filename):
    """Extract page ID from filename like 'R_-_Written_Notes_p3.png'."""
    return filename.replace(".png", "")


def parse_page_meta(page_id):
    """Get source PDF, page number, difficulty from page ID."""
    match = re.match(r"(.+)_p(\d+)$", page_id)
    if not match:
        return None, None, "Unknown", "Unknown"
    base, page_num = match.group(1), int(match.group(2))
    pdf_name = PDF_NAME_MAP.get(base, f"{base}.pdf")
    difficulty, category = DIFFICULTY_MAP.get(page_id, ("Unknown", "Unknown"))
    return pdf_name, page_num, difficulty, category


def load_summary(model_dir):
    """Load summary.json from a model output directory."""
    path = os.path.join(model_dir, "summary.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    # Normalize: could be a dict keyed by filename, or a list, or nested
    if isinstance(data, dict):
        return data
    return {}


def find_marker_output(page_id, pdf_name, page_num):
    """Find Marker's .md output for a page. Marker outputs whole PDFs."""
    if not pdf_name:
        return None
    # Marker dir name = PDF name without .pdf
    marker_dir_name = pdf_name.replace(".pdf", "")
    md_path = os.path.join(OUTPUT_DIR, "marker", marker_dir_name, f"{marker_dir_name}.md")
    if os.path.exists(md_path):
        return {
            "textPath": f"../scans/output/marker/{marker_dir_name}/{marker_dir_name}.md",
            "chars": os.path.getsize(md_path),
            "timeSec": None,
            "format": "md",
            "wholeDocument": True,
            "pageNum": page_num,
        }
    return None


def find_decomposed_output(page_id):
    """Find decomposed pipeline output with JSON sidecar."""
    txt_path = os.path.join(OUTPUT_DIR, "decomposed", f"{page_id}.txt")
    json_path = os.path.join(OUTPUT_DIR, "decomposed", f"{page_id}.json")
    if not os.path.exists(txt_path):
        return None

    result = {
        "textPath": f"../scans/output/decomposed/{page_id}.txt",
        "chars": None,
        "timeSec": None,
        "format": "txt",
    }

    # Read text for char count
    with open(txt_path) as f:
        result["chars"] = len(f.read())

    # Read JSON sidecar for metadata
    if os.path.exists(json_path):
        with open(json_path) as f:
            meta = json.load(f)
        result["rotation"] = meta.get("best_angle")
        result["qualityScore"] = meta.get("passes", [{}])[0].get("quality_score")
        result["strategy"] = meta.get("strategy")
        if meta.get("passes") and len(meta["passes"]) > 0:
            result["timeSec"] = meta["passes"][0].get("time_sec")

    return result


def find_standard_output(model_id, page_id, summary_data):
    """Find standard .txt output for a model/page pair."""
    txt_path = os.path.join(OUTPUT_DIR, model_id, f"{page_id}.txt")
    if not os.path.exists(txt_path):
        return None

    result = {
        "textPath": f"../scans/output/{model_id}/{page_id}.txt",
        "chars": None,
        "timeSec": None,
        "format": "txt",
    }

    # Try to get metadata from summary.json
    # Summary keys could be filename (with .png) or page_id
    for key in [f"{page_id}.png", page_id]:
        if key in summary_data:
            entry = summary_data[key]
            if isinstance(entry, dict):
                result["chars"] = entry.get("chars")
                result["timeSec"] = entry.get("time_sec") or entry.get("time") or entry.get("timeSec")
                if entry.get("error"):
                    result["error"] = entry["error"]
            break

    # Fallback: compute chars from file
    if result["chars"] is None:
        with open(txt_path) as f:
            content = f.read()
            result["chars"] = len(content)
            # Flag effectively empty outputs
            if result["chars"] < 50:
                result["empty"] = True

    return result


def generate_from_sqlite():
    """Generate manifest from SQLite database (batch pipeline results)."""
    import sqlite3
    db_path = os.path.join(os.path.dirname(__file__), "..", "data", "richiebot.db")
    if not os.path.exists(db_path):
        return None

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    # Check if we have processed pages
    count = c.execute("SELECT COUNT(*) FROM pages WHERE status = 'done'").fetchone()[0]
    if count == 0:
        conn.close()
        return None

    models_seen = set()
    pages = []
    for row in c.execute("""SELECT pdf_name, page_num, image_path, class, class_confidence,
                                   rotation_hint, model_used, text, chars, time_sec, quality_score, status
                            FROM pages ORDER BY pdf_name, page_num"""):
        page_id = f"{safe_name(row['pdf_name'])}_p{row['page_num']}"
        difficulty_map = {"typed": "Easy", "handwritten": "Medium", "hardest": "Very Hard",
                          "diagram": "Medium", "blank": "Skip"}
        pages.append({
            "id": page_id,
            "filename": f"p{row['page_num']}.png",
            "imagePath": row['image_path'] if row['image_path'] else "",
            "sourcePdf": row['pdf_name'],
            "pageNum": row['page_num'],
            "difficulty": difficulty_map.get(row['class'], "Unknown"),
            "category": row['class'] or "unknown",
            "outputs": {},
        })
        if row['status'] == 'done' and row['text']:
            model_id = (row['model_used'] or 'unknown').replace('.', '_').replace(':', '_')
            out_dir = os.path.join(os.path.dirname(__file__), "..", "scans", "output", f"batch_{model_id}")
            os.makedirs(out_dir, exist_ok=True)
            txt_path = os.path.join(out_dir, f"{page_id}.txt")
            with open(txt_path, "w") as f:
                f.write(row['text'])
            pages[-1]["outputs"][f"batch_{model_id}"] = {
                "textPath": f"../scans/output/batch_{model_id}/{page_id}.txt",
                "chars": row['chars'],
                "timeSec": row['time_sec'],
                "format": "txt",
                "qualityScore": row['quality_score'],
                "rotation": row['rotation_hint'] if row['rotation_hint'] else None,
                "pageClass": row['class'],
            }
            models_seen.add(model_id)

    conn.close()

    MODEL_DISPLAY = {
        "minicpm-v": ("MiniCPM-V (batch)", "4B", "5.5GB", "Handwritten pages via Ollama"),
        "chandra_ocr2": ("Chandra OCR 2 (batch)", "4B", "15.3GB", "Hardest pages + rotation"),
        "qwen2_5vl_7b": ("Qwen2.5-VL (batch)", "7B", "22.8GB", "Fallback / typed pages"),
        "marker": ("Marker (batch)", None, "4GB", "Typed pages via pdftext"),
    }
    models = []
    for mid in sorted(models_seen):
        display = MODEL_DISPLAY.get(mid, (mid, None, None, ""))
        models.append({
            "id": f"batch_{mid}",
            "name": display[0],
            "params": display[1],
            "vram": display[2],
            "type": "batch",
            "description": display[3],
        })

    return {
        "generated": datetime.now(timezone.utc).isoformat(),
        "models": models,
        "pages": pages,
        "stats": {
            "totalPages": len(pages),
            "totalModels": len(models),
            "coverageMatrix": {
                m["id"]: sum(1 for p in pages if m["id"] in p["outputs"])
                for m in models
            },
        },
    }


def safe_name(pdf_name):
    """Convert PDF name to safe ID."""
    return pdf_name.replace(".pdf", "").replace(" ", "_").replace("&", "and").replace("(", "").replace(")", "")


def main():
    # Try SQLite first (batch pipeline results)
    sqlite_manifest = generate_from_sqlite()

    # Discover test pages
    pages = []
    for png in sorted(glob.glob(os.path.join(TEST_PAGES_DIR, "*.png"))):
        filename = os.path.basename(png)
        page_id = parse_page_id(filename)
        pdf_name, page_num, difficulty, category = parse_page_meta(page_id)
        file_size = os.path.getsize(png)

        pages.append({
            "id": page_id,
            "filename": filename,
            "imagePath": f"../scans/test_pages/{filename}",
            "sourcePdf": pdf_name,
            "pageNum": page_num,
            "difficulty": difficulty,
            "category": category,
            "fileSizeBytes": file_size,
            "outputs": {},
        })

    # Discover models and their outputs
    models = []
    for model_dir_name in sorted(os.listdir(OUTPUT_DIR)):
        model_dir = os.path.join(OUTPUT_DIR, model_dir_name)
        if not os.path.isdir(model_dir) or model_dir_name in SKIP_MODELS:
            continue

        meta = MODEL_META.get(model_dir_name, {
            "name": model_dir_name,
            "params": None,
            "vram": None,
            "type": "unknown",
            "description": "",
        })
        models.append({"id": model_dir_name, **meta})

        # Load summary data for this model
        summary_data = load_summary(model_dir)

        # Match outputs to pages
        for page in pages:
            page_id = page["id"]

            if model_dir_name == "marker":
                output = find_marker_output(page_id, page["sourcePdf"], page["pageNum"])
            elif model_dir_name == "decomposed":
                output = find_decomposed_output(page_id)
            else:
                output = find_standard_output(model_dir_name, page_id, summary_data)

            if output:
                page["outputs"][model_dir_name] = output

    # Merge SQLite pipeline results if available
    if sqlite_manifest:
        # Add batch models
        for m in sqlite_manifest["models"]:
            models.append(m)
        # Add pipeline pages that aren't in test_pages (the full 369)
        existing_ids = {p["id"] for p in pages}
        for sp in sqlite_manifest["pages"]:
            if sp["id"] in existing_ids:
                # Merge pipeline output into existing test page
                for p in pages:
                    if p["id"] == sp["id"]:
                        p["outputs"].update(sp["outputs"])
                        break
            else:
                # New page from batch pipeline — fix image path for studio
                # Image path from SQLite is absolute, make relative
                abs_path = sp["imagePath"]
                if abs_path and os.path.isabs(abs_path):
                    rel = os.path.relpath(abs_path, os.path.join(os.path.dirname(__file__), ".."))
                    sp["imagePath"] = f"../{rel}"
                pages.append(sp)
        total_merged = sum(sqlite_manifest['stats']['coverageMatrix'].values())
        print(f"  Merged {total_merged} batch results from SQLite ({len(sqlite_manifest['models'])} models)")

    # Build manifest
    manifest = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "models": models,
        "pages": pages,
        "stats": {
            "totalPages": len(pages),
            "totalModels": len(models),
            "coverageMatrix": {
                m["id"]: sum(1 for p in pages if m["id"] in p["outputs"])
                for m in models
            },
        },
    }

    # Write manifest
    out_path = os.path.join(os.path.dirname(__file__), "manifest.json")
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Generated {out_path}")
    print(f"  {len(pages)} pages, {len(models)} models")
    for m in models:
        coverage = manifest["stats"]["coverageMatrix"][m["id"]]
        print(f"  {m['id']}: {coverage}/{len(pages)} pages")


if __name__ == "__main__":
    main()
