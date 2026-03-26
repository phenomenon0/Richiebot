"""Convert PDFs to page images and classify each page.

Processes all PDFs in scans/ (or a specified directory), converts
to 300 DPI PNGs, runs the page classifier, and stores metadata in SQLite.
"""
import os, sys, glob, time, sqlite3
from pdf2image import convert_from_path
sys.path.insert(0, os.path.dirname(__file__))
from classify_page import classify_page
from init_db import DB_PATH, init_db

SCANS_DIR = os.path.join(os.path.dirname(__file__), "..", "scans")
PAGES_DIR = os.path.join(SCANS_DIR, "pages")
DPI = 300


def safe_dirname(pdf_name):
    """Convert PDF filename to safe directory name."""
    return pdf_name.replace(".pdf", "").replace(" ", "_").replace("&", "and").replace("(", "").replace(")", "")


def process_pdf(pdf_path, conn):
    """Convert one PDF to pages, classify each, store in DB."""
    pdf_name = os.path.basename(pdf_path)
    folder = os.path.basename(os.path.dirname(pdf_path))
    safe_name = safe_dirname(pdf_name)
    out_dir = os.path.join(PAGES_DIR, safe_name)
    os.makedirs(out_dir, exist_ok=True)

    c = conn.cursor()

    # Check if already in DB
    c.execute("SELECT id, num_pages, status FROM pdfs WHERE local_path = ?", (pdf_path,))
    row = c.fetchone()
    if row and row[2] == 'done':
        print(f"  SKIP (already done): {pdf_name}")
        return row[1] or 0

    # Register PDF
    size = os.path.getsize(pdf_path)
    c.execute("""INSERT OR REPLACE INTO pdfs (nas_path, local_path, filename, folder, size_bytes, status, downloaded_at)
                 VALUES (?, ?, ?, ?, ?, 'processing', CURRENT_TIMESTAMP)""",
              (pdf_path, pdf_path, pdf_name, folder, size))
    conn.commit()
    pdf_id = c.lastrowid or c.execute("SELECT id FROM pdfs WHERE local_path = ?", (pdf_path,)).fetchone()[0]

    # Convert to images
    print(f"  Converting {pdf_name} at {DPI} DPI...")
    t0 = time.time()
    try:
        images = convert_from_path(pdf_path, dpi=DPI)
    except Exception as e:
        print(f"  ERROR converting {pdf_name}: {e}")
        c.execute("UPDATE pdfs SET status = 'error' WHERE id = ?", (pdf_id,))
        conn.commit()
        return 0

    num_pages = len(images)
    elapsed = time.time() - t0
    print(f"  {num_pages} pages in {elapsed:.1f}s")

    # Save each page and classify
    for i, img in enumerate(images):
        page_num = i + 1
        img_path = os.path.join(out_dir, f"p{page_num}.png")

        # Check if page already processed
        c.execute("SELECT status FROM pages WHERE pdf_name = ? AND page_num = ?", (pdf_name, page_num))
        existing = c.fetchone()
        if existing and existing[0] == 'done':
            continue

        # Save image
        img.save(img_path, "PNG")

        # Classify
        result = classify_page(img_path)

        c.execute("""INSERT OR REPLACE INTO pages
                     (pdf_id, pdf_name, page_num, image_path, class, class_confidence, rotation_hint, status)
                     VALUES (?, ?, ?, ?, ?, ?, ?, 'classified')""",
                  (pdf_id, pdf_name, page_num, img_path,
                   result.page_class, result.confidence, result.rotation_hint))

        if page_num % 20 == 0:
            conn.commit()
            print(f"    {page_num}/{num_pages} pages classified...")

    # Update PDF record
    c.execute("UPDATE pdfs SET num_pages = ?, status = 'done' WHERE id = ?", (num_pages, pdf_id))
    conn.commit()
    print(f"  Done: {num_pages} pages classified")
    return num_pages


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=os.path.join(SCANS_DIR), help="Directory containing PDFs")
    parser.add_argument("--pdf", help="Process a single PDF")
    args = parser.parse_args()

    init_db()
    conn = sqlite3.connect(DB_PATH)

    if args.pdf:
        pdfs = [args.pdf]
    else:
        pdfs = sorted(glob.glob(os.path.join(args.dir, "*.pdf")))
        # Also check subdirectories
        pdfs += sorted(glob.glob(os.path.join(args.dir, "*", "*.pdf")))

    print(f"Found {len(pdfs)} PDFs to process")
    total_pages = 0
    t0 = time.time()

    for pdf_path in pdfs:
        print(f"\n{'='*60}")
        n = process_pdf(pdf_path, conn)
        total_pages += n

    elapsed = time.time() - t0
    conn.close()

    # Print classification summary
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    print(f"\n{'='*60}")
    print(f"SUMMARY: {total_pages} pages from {len(pdfs)} PDFs in {elapsed:.1f}s")
    for row in c.execute("SELECT class, COUNT(*) FROM pages GROUP BY class ORDER BY COUNT(*) DESC"):
        print(f"  {row[0]:15s} {row[1]:5d} pages")
    conn.close()


if __name__ == "__main__":
    main()
