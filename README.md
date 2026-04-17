# Wedding Photo Curator

An automated AI pipeline that culls and curates thousands of wedding photos down to a final album — running entirely on a local GPU, no cloud APIs, no per-photo cost.

Built in Python on a Windows laptop (RTX 3050, 4 GB VRAM) to process a real 21,000-photo Indian wedding shoot.

---

## Pipeline Overview

```
21,000 raw photos
      │
      ▼
Phase 1 — Filter          blur / exposure / duplicate removal
      │
      ▼
Phase 1b — Burst          keep best 3 from each rapid-fire sequence
      │
      ▼
Phase 2 — Enrich          face detection · shot type · CLIP event tags · moment clustering
      │
      ▼
Phase 3 — Score           aesthetic · emotion · composition · memorability
      │
      ▼
Phase 4 — Select          MMR diversity selection → top 800
      │
      ▼
Phase 5 — Review          keyboard-driven UI: approve / reject / tag people
      │
      ▼
Phase 6 — Export          copy finals to organised output folders
```

Run the full pipeline in one command:
```bash
python run_pipeline.py
python run_pipeline.py --from 3        # resume from phase 3
python run_pipeline.py --only 5        # run only the review UI
python run_pipeline.py --dry-run       # print what would run, do nothing
```

---

## Scoring Formula

Every photo gets a `final_score` in [0, 1]:

```
final_score = aesthetic×0.25 + emotion×0.35 + memorability×0.15
            + rule_of_thirds×0.10 + prominence×0.15
```

Where `aesthetic_score` is itself a blend:

```
aesthetic_score = BRISQUE×0.20 + LAION×0.40 + MUSIQ×0.40
```

All weights live in `config.py` — tune them without re-running models using `--reweight`.

---

## Tech Stack

| Component | Model / Library |
|-----------|----------------|
| Aesthetic quality | [LAION aesthetic predictor](https://github.com/LAION-AI/aesthetic-predictor) (ViT-B/32 linear) |
| Perceptual quality | [MUSIQ](https://github.com/chaofengc/IQA-PyTorch) via pyiqa |
| Traditional quality | BRISQUE (OpenCV) |
| Emotion scoring | [LLaVA 7B](https://ollama.com/library/llava) via Ollama (local) |
| Face detection | DeepFace + RetinaFace |
| Diversity selection | CLIP ViT-B/32 embeddings + MMR (λ=0.7) |
| Duplicate detection | Perceptual hashing (ImageHash) — folder-sharded O(N²/buckets) |
| RAW support | rawpy |
| GPU | CUDA 12.4, PyTorch 2.6 |

LLaVA runs via [Ollama](https://ollama.com/) locally — no API keys, no cost per photo.

---

## Results (CANDID subset — 1,476 photos)

| Phase | Input | Output | Time |
|-------|-------|--------|------|
| Filter | 1,755 | 1,476 (208 dupes removed, 64 soft-blur kept) | ~25 min |
| Enrich | 1,476 | enriched JSON (100% face detection, avg 2.5 faces/photo) | 3h 17m |
| Score | 1,477 | scored JSON (range 0.22–0.78, mean 0.59) | 4h 35m |
| Select | 1,477 | top 800 via MMR (34% solo / 34% group / 32% couple) | 17s |
| Review | 800 | 80 approved (10% acceptance rate) | manual |

Phase 3 speed: ~11s/photo on RTX 3050 (LAION + MUSIQ + LLaVA in sequence).

---

## Setup

### Requirements

- Python 3.11+
- CUDA-capable GPU recommended (CPU fallback works, slower)
- [Ollama](https://ollama.com/) for LLaVA emotion scoring

### Install

```bash
git clone <repo-url>
cd wedding-photo-curator

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS

pip install -r requirements.txt

# Extra packages (not in requirements.txt)
pip install aesthetic-predictor open-clip-torch
pip install pyiqa
pip install transformers  # for CLIP in phase 2

# LLaVA via Ollama
ollama pull llava:7b-v1.6-mistral-q4_K_M
```

### Configure

Edit `config.py`:

```python
INPUT_FOLDER = "E:\\Your Wedding Folder"   # path to your photos on disk
```

All other thresholds, weights, and paths are documented inline in `config.py`.

### Run

```bash
# Optional: generate a scan report first (fast duplicate pre-filter)
python phase1_filter.py --scan-report

# Full pipeline
python run_pipeline.py

# Or phase by phase
python phase1_filter.py
python phase2_enrich.py
python phase1b_burst.py
python phase3_score.py
python phase4_select.py
python phase5_review.py
```

To enroll a person for tagging in the review UI:
```bash
python enroll_face.py --name yourname
```

---

## Project Structure

```
phase1_filter.py      blur / exposure / duplicate filter
phase1b_burst.py      burst sequence limiter (keep best 3)
phase2_enrich.py      face detection, CLIP tagging, moment clustering
phase3_score.py       aesthetic + emotion + composition scoring
phase4_select.py      MMR diversity selection
phase5_review.py      keyboard-driven review UI
phase6_export.py      final export to output folders
run_pipeline.py       orchestrates all phases
enroll_face.py        enroll a person's face for review tagging
config.py             all tunable parameters
utils.py              shared utilities (logging, JSON, EXIF, file copy)
```

### Output structure

```
output/
├── album_top800/          top 800 by score + diversity
├── album_final200/        manually approved finals
└── person_highlights/
    └── {name}/            approved photos tagged with that person
```

---

## Key Design Decisions

- **No originals ever touched** — every output is a copy; source files are read-only
- **Crash-safe checkpoints** — all JSON writes are atomic (write to `.tmp`, then `os.replace`)
- **Candid photos protected** — soft-blur tier preserved with 1.10× bonus instead of hard rejection
- **LLaVA timeout** — 90s wall-clock timeout via `ThreadPoolExecutor` (TCP keepalive blocked `requests` timeout)
- **CUDA OOM prevention** — all images resized to ≤800px before GPU model inference
- **Diversity via MMR** — prevents the top-800 from being all similar shots of the same moment

---

## Author

Built by **Gautam** using [Claude Code](https://claude.ai/code).