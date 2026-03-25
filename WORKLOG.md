# Richiebot Work Log

## 2026-03-25 — Initial OCR Pipeline Research & Prototyping

### What we did
- Connected to Richie-NAS (Synology) via Tailscale at `100.106.45.12:5000`
- Downloaded 7 PDFs (41MB, 369 pages) from `/Richie-Bot/Scans/Femi Test/`
- Visual pass of all PDFs to categorize difficulty tiers
- Deep research on state-of-the-art VLMs for document OCR on RTX 3090
- Tested 4 OCR models head-to-head on 11 representative test pages
- Built rotation-aware decomposition pipeline prototype

### Document corpus breakdown
| Type | Pages | % |
|------|-------|---|
| Messy handwriting (B-Notes, Written Notes) | ~244 | 66% |
| Mixed typed + handwriting (Low Hanging Fruit) | ~39 | 11% |
| Org charts / diagrams | ~47 | 13% |
| Book pages + annotations (Theory of Business) | ~18 | 5% |
| Marketing collateral + tables (Presenataions) | ~16 | 4% |
| Clean typed (Keys to Demand Gen) | ~5 | 1% |

### Models tested

| Model | Type | VRAM | Speed (avg/page) | Handwriting | Tables | Clean text |
|-------|------|------|-------------------|-------------|--------|-----------|
| **Marker** (Surya) | Traditional OCR | ~4GB | 3.2s | F | D | A |
| **MiniCPM-V** (4B) | VLM via Ollama | 5.5GB | 7.3s | B | C+ | A |
| **Qwen2.5-VL-7B** | VLM via Ollama | 22.8GB | 18.5s | B+ | A- | A |
| **Chandra OCR 2** (4B) | VLM (chandra-ocr) | ~8-10GB | testing... | 90.8% (bench) | 89.9% (bench) | — |
| **GLM-OCR** (0.9B) | VLM via Ollama | ~2GB | broken (Ollama packaging issue) | 87% (bench) | — | — |

### Key findings

1. **66% of the corpus is messy handwriting** — VLMs are essential, not optional
2. **Qwen2.5-VL-7B wins overall** — best tables (A-), good handwriting (B+), but slowest (18.5s/page)
3. **MiniCPM-V is the speed/quality sweet spot** — 2.5x faster, nearly as good on handwriting
4. **Rotation detection is a game changer** — flipping upside-down pages from F to C+ instantly
5. **Marker is only useful for clean typed text** — fails completely on handwriting
6. **Can't parallelize on 3090** with 7B models (22.8GB VRAM leaves no room)
7. **Chandra OCR 2 (March 2026)** scores 85.9 on olmOCR-bench, 90.8% on handwriting — testing in progress

### Decomposition pipeline concept
The hardest pages aren't unreadable — they're composited (overlapping sticky notes, rotated text layers). The pipeline:
1. Auto-detect best rotation per page (0°/90°/180°/270°)
2. Score OCR quality heuristically (unique words, punctuation, digits)
3. If score too low, detect text regions via OpenCV
4. Crop/deskew each region independently
5. OCR each with best rotation
6. Reassemble with spatial metadata

### Commits
- `cef1dcd` — .gitignore
- `aa32868` — test page extraction script
- `a160bae` — VLM OCR test script (Ollama)
- `281edc0` — decomposition + multi-rotation pipeline
- `ebd7d4a` — parallel rotation-aware pipeline

### Next steps
- [ ] Finish Chandra OCR 2 evaluation
- [ ] Fix GLM-OCR Ollama integration or run via HuggingFace
- [ ] Full 369-page run with best model + rotation detection
- [ ] Build SQLite tracking DB for batch processing
- [ ] Test decomposition Pass 2 (region-level OCR) on composited pages
- [ ] Scale to full NAS scan collection
