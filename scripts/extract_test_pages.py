"""Extract specific test pages from PDFs for VLM evaluation."""
import os
from pdf2image import convert_from_path

SCANS = "/home/omen/Documents/Project/Richiebot/scans"
OUT = "/home/omen/Documents/Project/Richiebot/scans/test_pages"
os.makedirs(OUT, exist_ok=True)

# Pick representative pages from each difficulty tier
test_pages = {
    # Easy: clean typed
    "B - The Keys to Demand Generation.pdf": [1],
    # Medium: org chart (printed)
    "R - Org Charts.pdf": [1, 2],  # printed chart + handwritten chart
    # Hard: table with numbers
    "R - Presenataions.pdf": [9],  # YNN results spreadsheet
    # Very hard: messy cursive
    "R - Written Notes.pdf": [1, 3, 4],  # worst handwriting pages
    # Hard: cursive + mixed
    "R&B - Low Hanging Fruit.pdf": [1, 5],
    # Medium: book with annotations
    "R - Theory of the Business - Underlying Philosophy.pdf": [1, 3],
}

for pdf_name, pages in test_pages.items():
    pdf_path = os.path.join(SCANS, pdf_name)
    if not os.path.exists(pdf_path):
        print(f"SKIP: {pdf_name} not found")
        continue
    for page_num in pages:
        print(f"Extracting {pdf_name} page {page_num}...")
        images = convert_from_path(pdf_path, first_page=page_num, last_page=page_num, dpi=300)
        if images:
            safe_name = pdf_name.replace(".pdf", "").replace(" ", "_").replace("&", "and")
            out_path = os.path.join(OUT, f"{safe_name}_p{page_num}.png")
            images[0].save(out_path, "PNG")
            print(f"  -> {out_path} ({images[0].size})")

print(f"\nDone. {len(os.listdir(OUT))} test pages in {OUT}")
