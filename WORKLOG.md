# Richiebot — Work Log

## 2026-03-25

Connected to Richie's Synology NAS via Tailscale. Downloaded 7 PDFs (369 pages) from the Scans folder. Did a visual pass — **66% of the corpus is messy cursive handwriting.** Traditional OCR is dead on arrival.

### The Corpus

| PDF | Pages | Type |
|-----|-------|------|
| B - Notes(1) | 237 | Handwritten notes, legal pad, some upside-down |
| R - Org Charts | 47 | Printed + handwritten org charts |
| R&B - Low Hanging Fruit | 39 | Mixed typed frameworks + handwritten strategy |
| R - Theory of Business | 18 | Drucker book pages with highlighter + margin notes |
| R - Presenataions | 16 | Marketing collateral, webinar screenshots, YNN spreadsheet |
| R - Written Notes | 7 | Dense cursive. SaaS plans, product notes, composited sticky notes |
| B - Keys to Demand Gen | 5 | Clean typed memo. JP Morgan demand gen strategy |

### 8 Models Tested

| Model | Speed | Handwriting | Tables | Verdict |
|-------|-------|-------------|--------|---------|
| **Chandra OCR 2** (4B) | 67s/pg | **A-** | A | Best quality. Handwriting king. |
| **Qwen2.5-VL** (7B) | 18.5s/pg | B+ | **A-** | Best tables. Too large for batch (22.8GB). |
| **MiniCPM-V** (4B) | 7.3s/pg | B | C+ | Best speed/quality. Production workhorse. |
| TrOCR-Large | 1.7s/pg | D | C | Line-level only. Needs proper segmentation. |
| Marker (Surya) | 3.2s/pg | F | D | Typed text only. |
| GLM-OCR (0.9B) | — | — | — | Broken on Ollama. |
| DeepSeek-OCR-2 | — | — | — | Broken on Ollama. |
| Surya | — | — | — | Broken (transformers conflict). |

### Rotation Breakthrough

All models fail on upside-down pages. A 180° flip turned the hardest page from **F to C+** — from hallucinated garbage to real content ("SMBS OFFERINGS", "Training", "Analytics", "Lead & Targeting").

The "impossible" pages aren't unreadable — they're composited. Upside-down sticky notes over lined paper. It's a decomposition problem, not a recognition problem.

Built a multi-rotation pipeline: try 0°, 180°, 90°, 270° — pick the best by quality score. Early-exit when score > 70.

### Batch Pipeline

Built a classify → route → process pipeline:
- **Classifier**: OpenCV image statistics (ink density, component size variance, line regularity). No ML model needed.
- **Router**: typed → Marker, handwritten → MiniCPM-V, hardest → Chandra + rotation, diagrams → MiniCPM-V.
- **SQLite tracking**: checkpoint/resume, quality scores, model attribution per page.

Processed all 369 pages in **1.4 hours**, zero errors:
- 215 pages via MiniCPM-V
- 140 pages via Chandra OCR 2
- 10 pages via Qwen2.5-VL
- 4 blank pages skipped

### Quality Results

| Grade | Pages | % |
|-------|-------|---|
| A (80-100) — excellent | 7 | 2% |
| B (60-80) — usable | 237 | 64% |
| C (40-60) — partial | 104 | 28% |
| D (20-40) — fragments | 16 | 4% |
| F (<20) — unusable | 1 | 0.3% |

**~66% of pages scored good or better.** Weighted raw text accuracy: ~45-55%. But most of the corpus is terse jot-notes ("SPAC - omni-bond - regulatory risk") — keyword-dense fragments that embed well even at lower accuracy.

### NAS Discovery

Re-scanned the NAS — found way more content than the test set:

| Folder | Size |
|--------|------|
| Richard Dema | 852 MB (65 PDFs) |
| Brian Dema | 85 MB (7 PDFs) |
| Richard + Brian | 83 MB (5 PDFs) |

**Full corpus: ~1 GB, ~5,000 pages.** Estimated processing time with routed pipeline: ~36 hours.

### QA Studio

Built a web-based evaluation tool: dark theme, page browser with thumbnails, image viewer with zoom/pan/rotate, model output comparison with tabs, metadata chips, keyboard navigation. Deployed to `femiadeniran.com/experiments/richiebot/`.

### What's Next

- [ ] Download full NAS corpus and run the batch pipeline (~36 hours)
- [ ] Embed all pages into vector store + cluster by topic
- [ ] Build RAG application — query Richie's knowledge base
- [ ] Context engineering, not fine-tuning — the corpus is reference material, not training data
