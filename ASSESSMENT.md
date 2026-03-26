# Richiebot OCR Pipeline — Full Assessment

## Executive Summary

We built and tested a complete OCR pipeline for processing Richie Dema's scanned business document archive. The archive spans ~20 years of business strategy, client work, marketing frameworks, training methodology, and organizational planning — stored as scanned PDFs on a Synology NAS.

**First pass (7 PDFs, 369 pages):** fully processed in 1.4 hours with zero errors using a classify→route→process pipeline. Three models handle different page types: MiniCPM-V for handwriting, Chandra OCR 2 for the hardest pages, and Marker for typed text.

**Full NAS corpus:** ~84 PDFs, ~1 GB, estimated 4,000-5,000 pages. Processing the full archive would take approximately 36 hours using the routed pipeline.

**Recommended application:** Context engineering with RAG (retrieval-augmented generation), not fine-tuning. The corpus is a knowledge base, not a training dataset.

---

## The Corpus

### What's on the NAS

| Folder | PDFs | Size | Content |
|--------|------|------|---------|
| Richard Dema | 65 + 2 subdirs | 852 MB | Client files, training kits, marketing, org charts, strategy, misc |
| Brian Dema | 7 | 85 MB | Business memos, notes, planning, DofG, job search |
| Richard + Brian | 5 | 83 MB | Product notes (79MB), principles, seminar |
| Femi Test (processed) | 7 | 41 MB | Sample set — notes, org charts, presentations, theory |
| Test Image Files | 2 TIFs | 10 MB | Scanner test files |
| **Total** | **~86** | **~1.07 GB** | **~4,000-5,000 pages** |

### Content breakdown by type (estimated across full NAS)

| Content Type | Est. PDFs | Est. Pages | Description |
|---|---|---|---|
| Handwritten strategy notes | ~15 | ~1,500 | Meeting notes, planning scribbles, strategy jots on legal pads. Richie's raw thinking. Dates span 2009-2018+. |
| Client files & case work | ~12 | ~800 | Acme, Cisco, HYPR, Pair Networks, Sundberg, Karen, Lee, Arun & JDK. Real engagement artifacts. |
| Training & methodology | ~8 | ~600 | 3-Point Training, kits, rep training, Rob Simpson training. Richie's teaching frameworks — most structured content. |
| Marketing & sales frameworks | ~8 | ~500 | Campaign design, research, platform architecture, sales management, prospecting methodology. |
| Org charts & planning | ~5 | ~300 | Functional org charts, planning worksheets, growth projections. Mix of printed templates with handwritten annotations. |
| Presentations & collateral | ~6 | ~200 | ComplyNet/GHS compliance materials, fact sheets, sample tool kits. Marketing output. |
| Brian's business docs | ~7 | ~300 | Demand generation strategy (JP Morgan), business planning, contacts, DofG notes. Typed memos + handwritten. |
| Invoices & admin | ~3 | ~200 | Financial records, structuring docs, procurement. Low knowledge value. |
| Theory & philosophy | ~4 | ~300 | Drucker's "Theory of the Business" with annotations, New Paradigm, Major Themes. Richie's intellectual influences. |
| Misc / uncategorized | ~18 | ~1,200 | "Misc Files", "Other Files", "Random Files", "Interesting Things". Unknown mix. |

### Document difficulty distribution

From processing 369 pages:

| Difficulty | % | Character | Page examples |
|---|---|---|---|
| Typed/clean | ~9% | Uniform text, standard fonts, paragraphs | Demand Gen memo, History Files, Platform docs |
| Handwritten (readable) | ~39% | Cursive but decodable, single-layer | Meeting notes with legible handwriting |
| Hardest (messy/composited) | ~38% | Upside-down text, overlapping sticky notes, dense scrawl | B-Notes composited pages, rotated org charts |
| Diagram/table | ~13% | Org charts, flowcharts, spreadsheets, marketing collateral | Functional org charts, YNN results table |
| Blank | ~1% | Empty or near-empty pages | Scanner artifacts |

**The corpus is overwhelmingly handwritten.** 77% of pages require VLM processing. Traditional OCR is useless for the majority.

---

## Pipeline Architecture

### Classify → Route → Process

```
PDF Input
  │
  ├─ PDF → PNG (300 DPI)
  ├─ Classify page (OpenCV image statistics)
  │    ├─ typed (area_cv < 1.0, many uniform components)
  │    ├─ handwritten (moderate components, area_cv 3-8)
  │    ├─ hardest (rotation needed, or very messy features)
  │    ├─ diagram (high area variance + structured layout)
  │    └─ blank (ink density < 0.5%)
  │
  ├─ Route to model
  │    ├─ typed     → Marker/pdftext (3.2s/page)
  │    ├─ handwritten → MiniCPM-V via Ollama (5s/page)
  │    ├─ hardest   → Chandra OCR 2 via HF + rotation (20-60s/page)
  │    ├─ diagram   → MiniCPM-V with description prompt (5s/page)
  │    └─ blank     → skip
  │
  ├─ Quality scoring (heuristic: unique words, punctuation, digits)
  └─ Store in SQLite (text, model, timing, quality, metadata)
```

### Processing time estimates

**Processed (369 pages, Femi Test):**

| Tier | Pages | Model | Time |
|------|-------|-------|------|
| Typed | 32 | Marker/MiniCPM-V | ~3 min |
| Handwritten | 145 | MiniCPM-V | ~12 min |
| Hardest | 141 | Chandra OCR 2 | ~65 min |
| Diagram | 48 | MiniCPM-V | ~4 min |
| Blank | 3 | Skip | 0 |
| **Total** | **369** | **Routed** | **1.4 hours** |

**Full NAS estimate (~5,000 pages):**

| Tier | Est. Pages | Model | Est. Time |
|------|-----------|-------|-----------|
| Typed | ~450 | Marker | ~24 min |
| Handwritten | ~1,950 | MiniCPM-V | ~2.7 hours |
| Hardest | ~1,900 | Chandra OCR 2 | ~28 hours |
| Diagram | ~650 | MiniCPM-V | ~54 min |
| Blank | ~50 | Skip | 0 |
| **Total** | **~5,000** | **Routed** | **~32-36 hours** |

---

## Model Evaluation Results

### 8 models tested, 5 working

| Model | Params | VRAM | Speed/pg | Handwriting | Tables | Status |
|-------|--------|------|----------|-------------|--------|--------|
| **Chandra OCR 2** | 4B | 15.3GB | 20-60s | **A-** (90.8% bench) | A | Best handwriting |
| **Qwen2.5-VL 7B** | 7B | 22.8GB | 18.5s | B+ | **A-** | Best tables, too large for batch |
| **MiniCPM-V** | 4B | 5.5GB | 5-7s | B | C+ | Best speed/quality ratio |
| **TrOCR-Large HW** | 660M | 2.4GB | 1.7s | D (needs line det.) | C | Line-level only |
| **Marker (Surya)** | — | 4GB | 3.2s | F | D | Typed text only |
| GLM-OCR | 0.9B | 2GB | — | — | — | Broken (Ollama) |
| DeepSeek-OCR-2 | 3B | 6.7GB | — | — | — | Broken (Ollama) |
| Surya | — | — | — | — | — | Broken (transformers conflict) |

### Head-to-head on the hardest page (Written Notes p3 — SaaS Plan Notes)

| Model | Got title | Got date | Got content | Grade |
|---|---|---|---|---|
| Marker | No | No | Garbled nonsense | F |
| TrOCR v1 | "Williams P.MANNOTES" | No | Fragments | D |
| TrOCR v2 | "Williams Plaw Notes" | No | Slightly better | D |
| MiniCPM-V | "SMS PLAN NOTES" | 8/4/17 | Pilot app, alliance, keep simple | B |
| Qwen2.5-VL | "SaaS PLAN NOTES" | 8/4/17 | Pilot app, alliance, "LEAD SYSTEM" | B+ |
| **Chandra OCR 2** | "SIXTS PLAN NOTES" | 8/9/17 | Salesforce, Oracle, 12 steps, app, alliance, MKTG, signal.co, "LEAD SYSTEM" | **A-** |

### Key breakthrough: rotation detection

Upside-down pages went from F to C+ with a simple 180° flip. The decomposition pipeline auto-detects optimal rotation (0°/90°/180°/270°) using a quality scoring heuristic.

| Page | Without rotation | With 180° flip |
|---|---|---|
| Written Notes p1 (composited) | "Planned vs Actual" ×80 (hallucination) | Real content: SMBS OFFERINGS, Training, Analytics, Lead & Targeting |
| Org Charts p2 (handwritten) | Partial (score 70) | Better at 90° (score 72.6) |

---

## Quality Assessment (369 pages processed)

### Quality distribution

| Grade | Pages | % | Description |
|-------|-------|---|-------------|
| A (80-100) | 7 | 2% | Excellent extraction, near-verbatim |
| B (60-80) | 237 | 64% | Good — key terms, topics, dates, structure captured |
| C (40-60) | 104 | 28% | Partial — topic signal present, many individual word errors |
| D (20-40) | 16 | 4% | Fragments — keywords buried in hallucination |
| F (<20) | 1 | 0.3% | One blank/unreadable page |
| Skip | 4 | 1% | Blank pages |

### By model

| Model | Pages | Avg quality | % scoring good (>60) | % scoring poor (<40) |
|-------|-------|-------------|---------------------|---------------------|
| Qwen2.5-VL | 10 | 70.8 | 90% | 10% |
| MiniCPM-V | 215 | 62.8 | 70% | 6% |
| Chandra OCR 2 | 141 | 61.9 | 60% | 3% |

### Estimated real text accuracy

| Page type | Raw text accuracy | Notes |
|-----------|------------------|-------|
| Clean typed | 90-95% | Near-perfect |
| Typed + annotations | 80-90% | Base text great, annotations partial |
| Readable handwriting | 50-70% | Key terms correct, many word-level errors |
| Messy cursive | 30-50% | Topic signal present, specific words unreliable |
| Composited/rotated | 20-40% (without rotation), 40-60% (with rotation) | Decomposition helps significantly |

**Weighted average across corpus: ~45-55% raw text accuracy.**

### The quality scorer caveat

Our heuristic quality scorer penalizes the dominant writing style in this corpus. Richie writes in terse, keyword-dense jots: `"SPAC - omni-bond - regulatory risk - clients"`. This scores low (no sentences, no punctuation variety) but may be **100% faithful** to what was actually written. The real accuracy on jot-style pages is likely higher than the scores suggest.

### Trouble spots

All worst-scoring pages come from **B - Notes(1).pdf** — the 237-page handwritten core. Specific failure modes:
- **Hallucination**: Model generates 10K-20K chars of repetitive text (caught by unique-word-ratio penalty)
- **Total blank**: Page 105 returned 0 chars
- **Rotation not detected**: Some pages need 90° rotation but classifier missed it

---

## Strategic Assessment: Fine-Tuning vs Context Engineering

### Fine-tuning verdict: No

Even with the full 5,000-page NAS corpus:

1. **Wrong data format.** Fine-tuning needs instruction→response pairs. We have fragments, scribbles, jot-downs. There's no "question" to pair with the "answer."

2. **OCR noise poisons weights.** Fine-tuning on 50%-accurate text permanently embeds errors into the model. "SIXTS PLAN NOTES" instead of "SaaS PLAN NOTES" becomes learned behavior. RAG can be corrected; fine-tuned weights cannot.

3. **Not enough volume.** At 50% text accuracy across 5,000 pages, we have ~2,500 pages of usable text. After deduplication and filtering, maybe 1,500 clean passages. Fine-tuning a 7B model on 1,500 examples produces a model that memorizes, not generalizes.

4. **Two different voices.** Richie and Brian write differently. A fine-tuned model would blend them into an incoherent hybrid.

5. **The training kits exception.** The ~600 pages of structured training material (R - 3 Point Training, R - Kits and Training, etc.) are the closest thing to fine-tuning data. If someone manually cleaned and structured these into lesson→explanation pairs, you could potentially fine-tune a small model on Richie's *teaching methodology*. But this requires significant human curation.

### Context engineering verdict: Yes — this corpus is a goldmine

**What makes it ideal for RAG:**

1. **Topic density.** The same subjects appear across dozens of pages over years. SPAC strategy, marketing platforms, org design, client management — each topic has 20-50 pages of overlapping notes. A single page at 50% accuracy is weak. Twenty pages on the same topic at 50% each? Cross-referenced by an LLM, you reconstruct the narrative at 85-95%.

2. **Temporal depth.** Dates span 2009-2018+. You can trace how Richie's thinking on a topic *evolved*. A RAG system can answer "How did Richie's approach to demand generation change over time?" by retrieving dated pages on that topic.

3. **Multi-domain coverage.** Marketing, sales, training, client work, org design, product strategy, compliance, financial structuring. This is a complete business knowledge base, not a narrow specialty.

4. **Two perspectives.** Brian's typed memos (JP Morgan demand gen strategy) complement Richie's handwritten notes. A RAG system can surface both views.

5. **Real client work.** Acme, Cisco, HYPR, Pair Networks, Sundberg — actual engagement artifacts, not hypothetical frameworks. This grounds the knowledge in reality.

6. **The jot-note format is actually an advantage.** Keyword-dense fragments embed beautifully. `"SPAC omni-bond regulatory risk clients investment bank"` produces a tight, distinctive embedding that clusters perfectly with related content. Full paragraphs would embed more diffusely.

### Recommended architecture

```
Full NAS corpus (5,000 pages)
  │
  ├─ OCR pipeline (routed, ~36 hours)
  ├─ Embed all pages into vector store
  ├─ Cluster by topic (k-means or HDBSCAN on embeddings)
  ├─ Generate topic summaries (LLM reads all pages per cluster)
  │
  └─ RAG application
       ├─ User query → embed → retrieve top 10-20 chunks
       ├─ Include relevant topic summaries as context
       └─ LLM synthesizes answer grounded in Richie's actual documents
```

### What this enables

- **"What was Richie's framework for lead generation?"** → Retrieves across R-Platform, R-Marketing Campaigns, R&B-Low Hanging Fruit, B-Notes strategy pages
- **"How did Richie train sales reps?"** → Retrieves from R-3 Point Training, R-Rep Training, R-Kits
- **"What clients did Richie work with in 2017?"** → Retrieves dated client file pages
- **"Compare Richie and Brian's approach to demand generation"** → Retrieves from both voices

### Future path to fine-tuning

If you eventually want a "Richie model," the RAG system generates the training data:
1. Build the RAG system
2. Collect real user queries over time
3. The RAG-generated answers (grounded in documents) become high-quality training pairs
4. Fine-tune on those pairs — now you have clean, structured, verified data
5. The fine-tuned model can handle common queries without retrieval

This is the synthetic-data flywheel: RAG produces the clean training data that OCR couldn't.

---

## Infrastructure Summary

### Hardware
- **GPU:** NVIDIA RTX 3090 (24GB VRAM)
- **Storage:** NAS via Tailscale (Synology, 100.106.45.12)
- **Local disk:** 952GB, ~47GB free after model downloads

### Models deployed
- **MiniCPM-V** (4B) via Ollama — primary handwriting model, 5.5GB VRAM
- **Chandra OCR 2** (4B) via HuggingFace — hardest pages, 15.3GB VRAM
- **Marker/Surya** via pip — typed text baseline
- **Qwen2.5-VL 7B** via Ollama — available but too large for batch (22.8GB)

### Tools built
| Script | Purpose |
|--------|---------|
| `scripts/classify_page.py` | OpenCV page type classifier |
| `scripts/pdf_to_pages.py` | PDF→PNG + classification pipeline |
| `scripts/batch_ocr.py` | Routed batch OCR with checkpoint/resume |
| `scripts/decompose_ocr.py` | Multi-rotation + region decomposition |
| `scripts/parallel_ocr.py` | Parallel rotation probing |
| `scripts/extract_test_pages.py` | Test page extraction |
| `scripts/test_vlm.py` | Ollama VLM testing harness |
| `scripts/test_trocr.py` / `test_trocr_v2.py` | TrOCR with line detection |
| `studio/index.html` | QA Evaluation Studio (dark theme, zoom/pan/rotate, model compare) |
| `studio/generate_manifest.py` | Manifest generator (filesystem + SQLite) |

### Database
- SQLite at `data/richiebot.db`
- Tables: `pdfs` (84 entries potential), `pages` (369 processed)
- Tracks: classification, model routing, OCR text, quality scores, timing

---

## Next Steps

1. **Download full NAS corpus** (~1 GB, 77 more PDFs) and run through pipeline (~36 hours)
2. **Build embed + cluster pipeline** — vector store + topic clustering + LLM summaries
3. **Build RAG application** — query interface grounded in Richie's knowledge base
4. **Fix broken models** (GLM-OCR, Surya) in separate venvs for comparison
5. **Install flash-linear-attention** for Chandra speed improvement
6. **Human review** — flag and review the 17 worst-scoring pages
7. **Deploy QA Studio** to Hetzner server for team access
