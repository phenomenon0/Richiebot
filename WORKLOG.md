# Richiebot Work Log

## 2026-03-25 — OCR Pipeline: Research, Model Evaluation & Prototype

### Context

Richie (Brian Dema) shared access to his Synology NAS via Tailscale. The NAS contains scanned business documents — marketing strategy memos, org charts, handwritten meeting notes, sales tracking spreadsheets, and historical materials from multiple companies (JP Morgan commercial banking, ComplyNet/GHS compliance, and Richie's own consulting frameworks). The goal: build a local OCR pipeline that can ingest all of this into searchable, structured text — ultimately feeding into a knowledge system for Richiebot.

### Phase 0: NAS Connection & Data Acquisition

Connected to **Richie-NAS** (Synology) via Tailscale at `100.106.45.12:5000`. Login as `Femi_Admin` with 2FA (authenticator OTP required each session). Used the Synology FileStation API to browse and download files programmatically.

**NAS structure:**
```
/Richie-Bot/
  ├── Brian/
  ├── Christian/
  ├── Femi/         (empty)
  ├── Han/
  ├── Olga/
  ├── Phanuel/
  ├── Scans/
  │   └── Femi Test/   ← 7 PDFs, 41MB total
  └── Video/
```

Downloaded all 7 PDFs to `/home/omen/Documents/Project/Richiebot/scans/`.

### Phase 1: Visual Assessment & Corpus Analysis

Did a visual pass of every PDF, reading sample pages to categorize content and difficulty. This turned out to be critical — the corpus is not what you'd expect.

**The 7 PDFs (369 pages total):**

| PDF | Pages | Content | Difficulty |
|-----|-------|---------|-----------|
| **B - Notes(1).pdf** | 237 | Handwritten meeting notes, business plans, product strategy. Yellow legal pad, cursive, some pages upside-down. Multiple dates spanning 2011-2018. | Very Hard |
| **R - Org Charts.pdf** | 47 | Functional org charts for "The Firm" — some printed (Jan 2017), some handwritten with annotations. Marketing/Sales/Product/Finance structure. | Medium-Hard |
| **R&B - Low Hanging Fruit.pdf** | 39 | Mixed: typed business framework docs (History Files, Marketing Platform, Lead Systems Engine) plus handwritten notes on products, outreach, SaaS strategy. | Mixed |
| **R - Theory of the Business.pdf** | 18 | Scanned Drucker book pages ("Theory of the Business") with yellow highlighter and handwritten margin notes. | Medium |
| **R - Presenataions.pdf** | 16 | Marketing collateral: ComplyNet postcards, GHS compliance webinar screenshots, OSHA training ads, YNN sales results spreadsheet. | Hard |
| **R - Written Notes.pdf** | 7 | Dense cursive handwriting. SaaS plan notes, product characteristics, business model sketches. Some pages composited with overlapping sticky notes. | Very Hard |
| **B - Keys to Demand Generation.pdf** | 5 | Clean typed memo by Brian Dema (2/1/22) on JP Morgan commercial banking demand gen strategy. | Easy |

**The surprise: 66% of the corpus is messy handwriting.** This completely changed the pipeline strategy — traditional OCR is useless for the majority of the data.

| Document Type | Pages | % of corpus |
|---------------|-------|-------------|
| Messy cursive handwriting | ~244 | 66% |
| Mixed typed + handwriting | ~39 | 11% |
| Org charts / diagrams | ~47 | 13% |
| Book pages + annotations | ~18 | 5% |
| Marketing collateral + tables | ~16 | 4% |
| Clean typed text | ~5 | 1% |

### Phase 2: Research — State of the Art VLMs for Document OCR (2024-2026)

Ran three parallel deep-research agents covering:
1. VLM models that fit in 24GB VRAM (RTX 3090)
2. Document processing pipeline architecture
3. Handwriting OCR and table extraction specifically

**Key models identified (fit on 3090):**

| Model | Params | VRAM (FP16) | OmniDocBench | olmOCR-bench | Handwriting | Released |
|-------|--------|-------------|--------------|--------------|-------------|---------|
| PaddleOCR-VL-0.9B | 0.9B | ~2GB | 92.56 | — | Good | 2025 |
| PaddleOCR-VL-7B | 7B | ~16GB | 92.86 (#1) | — | Good | 2025 |
| Qwen2.5-VL-7B | 7B | ~16GB | High | 65.5 | 3.8% CER (IAM) | Sept 2024 |
| MiniCPM-V-4.5 | ~8B | ~18GB | OCRBench leader | — | Good | 2025 |
| GOT-OCR-2.0 | 0.6B | ~2GB | — | — | Moderate | 2024 |
| OlmOCR-2-7B | 7.7B | ~17GB | — | 82.4 | Good | 2025 |
| **Chandra OCR 2** | 4B | ~10GB | — | **85.9** (SOTA) | **90.8%** | **Mar 2026** |
| **GLM-OCR** | 0.9B | ~2GB | **94.62** (SOTA) | — | **87%** | **Jan 2026** |

**Traditional OCR for comparison:**

| Tool | Speed (3090) | Best for |
|------|-------------|----------|
| Surya | 300-900 pg/min | Layout analysis, reading order |
| Marker (uses Surya) | ~70-120 pg/min | PDF-to-markdown, clean docs |
| MinerU | ~285 pg/min | Tables, complex layouts |

**Handwriting accuracy reality check:**

| Quality | VLM accuracy | Traditional OCR |
|---------|-------------|-----------------|
| Neat cursive | 85-92% | 75-85% |
| Messy but readable | 70-85% | 50-70% |
| Very messy (our corpus) | 40-65% | 20-40% |

**Decision: VLMs are the only viable path for 66% of this corpus.** Traditional OCR would produce unusable output on the handwriting. The question became: which VLM, and how to handle the hardest pages?

### Phase 3: Model Evaluation — Head-to-Head Testing

Extracted 11 representative test pages across all difficulty tiers using `scripts/extract_test_pages.py`:
- 1 clean typed page (Demand Gen p1)
- 2 org charts (printed p1, handwritten p2)
- 1 table/spreadsheet (YNN results p9)
- 3 messy cursive (Written Notes p1, p3, p4)
- 2 mixed content (Low Hanging Fruit p1, p5)
- 2 annotated book pages (Theory of Business p1, p3)

#### Model 1: Marker (baseline — traditional OCR)

**369 pages in 20 minutes** (0.31 pg/sec). Output: markdown per PDF + extracted images.

- **Clean typed (Demand Gen):** A — near-perfect markdown, headers, bullets preserved
- **Typed business docs (Low Hanging Fruit):** B+ — good structure, diagrams saved as images
- **Org charts:** B- — title extracted but spatial layout lost, tables become pipe-delimited noise
- **Marketing collateral:** C — partial, massive hallucination on scanned screenshots
- **Handwriting:** F — total failure. Garbled nonsense, repetitive hallucinations, unusable

**Verdict:** Marker is only useful for the 1% of clean typed pages.

#### Model 2: MiniCPM-V (4B, via Ollama)

**11 pages in 80 seconds** (7.3s/page avg). Uses 5.5GB VRAM.

| Page type | Grade | Example output |
|-----------|-------|---------------|
| Clean typed | A | Near-identical to Marker |
| Org chart (printed) | A- | Got structure, departments, hierarchy |
| Org chart (handwritten) | B | Got departments, some misreads |
| Table (YNN results) | C+ | Partial structure, some numbers, rep names |
| Handwriting (plan notes p3) | B | "SMS PLAN NOTES 8/4/17" — got date, action items |
| Handwriting (products p4) | C+ | Got ~65% of content with misreads |
| Handwriting (composited p1) | F+ | Gave up — returned `[Handwritten Text]` placeholders |

#### Model 3: Qwen2.5-VL-7B (via Ollama)

**11 pages in 203 seconds** (18.5s/page avg). Uses 22.8GB VRAM (nearly all of 3090).

| Page type | Grade | vs MiniCPM-V |
|-----------|-------|-------------|
| Clean typed | A | Tie |
| Org chart (printed) | A- | Tie |
| Org chart (handwritten) | B | Tie |
| Table (YNN results) | **A-** | **Much better** — full markdown table, all rep names (Ruffolo R, Frey G, Price N, Imhoff J, Westerberg K, Valliyil L), weekly totals, summary stats |
| Handwriting (plan notes p3) | **B+** | Slightly better — got "SaaS" correct vs MiniCPM's "SMS" |
| Handwriting (products p4) | C+ | Tie |
| Handwriting (composited p1) | F | Worse — hallucinated "Planned vs Actual" x80 |

**Qwen2.5-VL wins on tables** (full structured markdown) and has slightly better contextual handwriting recognition. But 2.5x slower and maxes out VRAM (no room for parallelization).

#### Model 4: Chandra OCR 2 (4B, March 2026, via chandra-ocr pip)

**11 pages in 741 seconds** (67.4s/page avg). Uses 15.3GB VRAM. No flash-attention optimization installed (would be faster with it).

| Page type | Grade | vs Qwen2.5-VL |
|-----------|-------|--------------|
| Clean typed | A | Tie |
| Org chart (printed) | A | Slightly better layout |
| Table (YNN results) | A (benchmark: 89.9%) | Comparable |
| Handwriting (plan notes p3) | **A-** | **Best** — got "Dalesforce.com", "ORACLE", "12 steps TO CREATE an APP", "signal.co" |
| Handwriting (products p4) | **B+** | **Best** — preserved strikethrough annotations, numbered deliverables list |
| Handwriting (composited p1) | D+ | Started well ("SMS OPERATIONS", channels list) then hallucinated `~~now~~` x500 |

**Chandra is the handwriting king** — extracted more detail from cursive than any other model, including crossed-out text and annotations. But 3.6x slower than Qwen without flash-attention, and still fails on composited/overlapping pages.

#### Model 5: GLM-OCR (0.9B, January 2026, via Ollama)

**Broken.** The Ollama packaging doesn't properly handle image inputs — returns empty `<table>` tags for every page. Would need to run via HuggingFace transformers directly (requires transformers 5.0+, incompatible with Marker/Surya). Parked for now.

#### Final comparison table

| Model | Params | VRAM | Speed/pg | Handwriting | Tables | Composited pages |
|-------|--------|------|----------|-------------|--------|-----------------|
| Marker | — | 4GB | 3.2s | F | D | F |
| MiniCPM-V | 4B | 5.5GB | 7.3s | B | C+ | F+ |
| Qwen2.5-VL | 7B | 22.8GB | 18.5s | B+ | **A-** | F |
| **Chandra OCR 2** | 4B | 15.3GB | 67.4s | **A-** | A | D+ |

### Phase 4: The Rotation Breakthrough

While testing, noticed all models completely fail on pages that are upside-down or rotated. Tested a simple 180° flip on the hardest page (Written Notes p1 — composited, overlapping, multiple orientations):

| Approach | Qwen2.5-VL output | Grade |
|----------|-------------------|-------|
| Original (upside-down) | "Planned vs Actual" x80 — hallucinated garbage | F |
| **Flipped 180°** | "SMBS OFFERINGS", "Training", "Data & Platforms", "Analytics", "Lead & Targeting", dates (11/3/17, 3/2/18) | **C+** |

**A simple rotation turned an F into a C+.** This led to the core insight:

> The "impossible" pages aren't unreadable — they're composited. Upside-down sticky notes overlapping lined paper, rotated scrawls layered on typed text. This is a **masking/decomposition problem**, not a recognition problem.

### Phase 5: Decomposition Pipeline Prototype

Built `scripts/decompose_ocr.py` — a multi-pass pipeline:

**Pass 1: Multi-rotation OCR**
1. Try page at 0° — OCR and score quality
2. If score < threshold, try 180° — OCR and score
3. If still low, try 90° and 270°
4. Pick rotation with highest quality score
5. Early-exit if any rotation scores > 70 (saves time)

**Quality scoring heuristic:**
- Unique word ratio (penalizes repetitive hallucinations like "Planned vs Actual" x80)
- Reasonable word lengths (2-15 chars)
- Has digits (dates/numbers = real content)
- Punctuation variety
- Penalties for placeholder text and very short output

**Pass 2: Region decomposition** (for pages that still score low)
1. Binarize image, dilate to connect text into regions (OpenCV morphology)
2. Find contours, get bounding boxes + rotation angles
3. Filter by minimum area (>0.5% of page)
4. Crop each region with padding
5. Try 0° and 180° on each crop independently
6. OCR each with best rotation
7. Reassemble all region texts with spatial metadata

**Test results on hardest pages:**

| Page | Auto-detected angle | Quality score | Time | Strategy |
|------|-------------------|--------------|------|----------|
| Written Notes p1 (composited) | **180°** | 73.7 | 58s | Full-page rotation |
| Written Notes p3 (cursive) | 0° | 74.2 | 7.5s | Good on first try |
| Written Notes p4 (cursive) | 0° | 78.9 | 7.9s | Good on first try |
| Org Charts p2 (handwritten) | **90°** | 72.6 | 63s | Tried 0°, 180°, 90° |

The pipeline correctly identified that p1 was upside-down and p2 was rotated 90°. Pages already in correct orientation were processed in one pass (~8s).

### Parallelization Assessment

Tested `OLLAMA_NUM_PARALLEL=2` with Qwen2.5-VL-7B. **Failed** — the model uses 22.8GB of 24GB VRAM, leaving no room for a second concurrent KV cache. Two simultaneous requests caused a timeout (likely OOM thrash).

**Options explored:**
- `NUM_PARALLEL=2` with 7B model: OOM, doesn't work
- Smaller model (MiniCPM-V at 5.5GB): could fit 2-3 parallel, but worse quality
- CPU prep + GPU overlap: marginal gain (~1.2x), GPU is the bottleneck
- Parallel rotation probing: works within a single page (try 0° and 180° concurrently)

**Conclusion:** Sequential processing is the reality on a single 3090 with 7B+ models. The 369-page run will take ~2 hours with Qwen2.5-VL, ~7 hours with Chandra. Rotation detection adds ~50% overhead only for pages that need it.

### Architecture Decision: Classify → Route → Process → Score

Based on all findings, the production pipeline design:

```
PDF Input
  │
  ├─ Step 1: PDF → Images (300 DPI, PNG)
  ├─ Step 2: Auto-rotate (multi-rotation OCR with quality scoring)
  ├─ Step 3: Classify page type (typed/table/handwriting/diagram/noise)
  ├─ Step 4: Route to best model
  │    ├─ Clean text → Marker (fast, 98%+ accuracy)
  │    ├─ Tables → Qwen2.5-VL-7B (best structure preservation)
  │    ├─ Handwriting → Chandra OCR 2 (best cursive, annotations)
  │    ├─ Diagrams → VLM natural language description
  │    └─ Composited → Decomposition pipeline (region detection + per-region OCR)
  ├─ Step 5: Confidence scoring & quality flags
  │    ├─ >90% → auto-accept
  │    ├─ 70-90% → flag for review
  │    └─ <70% → route to decomposition or human
  └─ Step 6: Output (Markdown + JSON metadata sidecar)
```

**Nothing gets discarded.** Every page gets:
- Best-effort transcription (even if low confidence)
- Difficulty flag and decomposition metadata
- Original image preserved for future re-processing

### Phase 6: Additional Model Testing (Round 2)

Tested 3 more models to complete the evaluation suite.

#### Model 6: TrOCR-Large-Handwritten (Microsoft, 660M params)

**13 pages in 18.6s** (1.4s/page avg). 2.44GB VRAM. Blazing fast.

TrOCR is a **line-level** model — it needs pre-segmented text lines. We used simple horizontal strip detection (row-wise intensity thresholding). This crude segmentation limits accuracy significantly.

| Page type | Result | Notes |
|-----------|--------|-------|
| Clean typed (Demand Gen) | C+ | Got content but with "1961" artifacts, missing structure |
| Handwriting (Written Notes p3) | D | "Williams P.MANNOTES", "1961 W.S.T Duke" — mostly garbled |
| Handwriting (Written Notes p4) | D | "INSE characteristics of products", "MULTI Phang" — fragments |
| Composited (Written Notes p1) | F | Only 10 chars — segmentation found 1 line |
| Tables (YNN results) | C | Got some numbers but lost structure |

**Verdict:** TrOCR is hampered by our naive line segmentation. With proper Surya/CRAFT line detection it would likely score B+ on handwriting. As-is, it's the fastest model (1.4s/page) but lowest quality on our test set.

#### Model 7: DeepSeek-OCR-2 (3B, via Ollama)

**Broken.** Same issue as GLM-OCR — the Ollama vision pipeline doesn't properly pass images to these specialized OCR models. Output was either empty (0 chars) or echoed the prompt back. 6.7GB model downloaded for nothing.

#### Model 8: Surya OCR (0.17.1, already installed)

**Broken.** Surya 0.17.1 is incompatible with transformers 5.3.0 (which we upgraded to for Chandra OCR 2). Crashes with `AttributeError: 'SuryaDecoderConfig' object has no attribute 'pad_token_id'`. Would need a separate venv with transformers 4.56 to run.

#### Updated Final Comparison (8 models tested)

| Model | Params | VRAM | Speed/pg | Handwriting | Tables | Status |
|-------|--------|------|----------|-------------|--------|--------|
| **Marker** | — | 4GB | 3.2s | F | D | Working |
| **MiniCPM-V** | 4B | 5.5GB | 7.3s | B | C+ | Working |
| **Qwen2.5-VL** | 7B | 22.8GB | 18.5s | B+ | **A-** | Working |
| **Chandra OCR 2** | 4B | 15.3GB | 67.4s | **A-** | A | Working |
| **TrOCR-Large HW** | 660M | 2.4GB | 1.4s | D (bad segmentation) | C | Working (needs line det.) |
| GLM-OCR | 0.9B | 2GB | — | — | — | Broken (Ollama) |
| DeepSeek-OCR-2 | 3B | 6.7GB | — | — | — | Broken (Ollama) |
| Surya | — | — | — | — | — | Broken (transformers conflict) |

**Working models: 5.** Chandra, Qwen2.5-VL, MiniCPM-V, TrOCR, Marker.
**Broken: 3.** GLM-OCR, DeepSeek-OCR-2, Surya (all fixable with dedicated venvs or HF direct).

### QA Evaluation Studio

Built `studio/index.html` — a dark-themed single-file web app for comparing model outputs side-by-side against source images. Features: page browser with thumbnails + difficulty badges, image viewer with zoom/pan/rotate, model switcher tabs, markdown rendering, metadata chips, keyboard navigation. Served via `python3 -m http.server 8080`.

### NAS Scan Update

Re-checked Richie-NAS and discovered massive new content:

| Folder | Files | Size |
|--------|-------|------|
| Richard Dema | 65 PDFs + 2 subdirs | 852 MB |
| Brian Dema | 7 PDFs | ~85 MB |
| Richard + Brian | 5 PDFs | ~83 MB |
| Femi Test | 7 PDFs (what we have) | 41 MB |
| Test Image Files | 2 TIFs | 10 MB |

**Total corpus: ~1 GB, 77+ PDFs** — far larger than our initial 7 PDFs.

### Commits

| Hash | Description |
|------|-------------|
| `cef1dcd` | .gitignore for PDFs, caches, outputs |
| `aa32868` | Test page extraction script |
| `a160bae` | VLM OCR test script (Ollama) |
| `281edc0` | Decomposition + multi-rotation pipeline |
| `ebd7d4a` | Parallel rotation-aware pipeline |
| `9d04725` | Expanded worklog |
| `94d5197` | OCR QA Evaluation Studio |
| `7170aa6` | Rotate button + markdown renderer |
| `ed081e5` | Fixed rotation to rotate in place |
| `c98fc66` | Generated manifest.json |

### Open Issues

- **Ollama vision pipeline** — GLM-OCR and DeepSeek-OCR-2 both broken. These models need to be run via HuggingFace transformers directly, which requires transformers 5.0+ (conflicts with Surya/Marker).
- **Surya** — needs transformers 4.56, incompatible with Chandra (needs 5.0+). Solution: separate venv or use Marker's bundled Surya.
- **TrOCR** — needs proper line detection (Surya or CRAFT) to reach its potential. Current naive segmentation wastes the model.
- **Chandra speed** — 67.4s/page without flash-linear-attention.
- **1 GB NAS corpus** — 77+ PDFs discovered, need to plan batch download and processing strategy.

### Next Steps

- [ ] Full 369-page run with Qwen2.5-VL + rotation detection
- [ ] Download and sample from the 852MB Richard Dema collection
- [ ] Set up separate venv for Surya/GLM-OCR (transformers 4.56)
- [ ] Add proper line detection for TrOCR (CRAFT or Surya-based)
- [ ] Build SQLite tracking DB for batch processing
- [ ] Embed + cluster pipeline for semantic grouping of noisy OCR output
- [ ] Scale to full NAS corpus
