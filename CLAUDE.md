# Wedding Photo Curator — Claude Code Rules

## Current Status
- Last completed: Pre-21k code review fixes ✅ (7 issues — 2026-04-17)
- utils.py ✅ — atomic JSON writes (tmp+os.replace); all checkpoints now crash-safe
- phase1_filter.py ✅ — error logging WARNING (was DEBUG); phash dedup sharded by folder; two-tier phash threshold (cross-folder <=4, same-folder <=8); scan_report.json atomic write via save_json; warning on 0-path scan_report match
- phase2_enrich.py ✅ — DeepFace 60s timeout; face_bboxes stored; +--resume flag with checkpoint every 500 photos; fixed save_json arg order at step5 checkpoint; +step6_clip_embeddings() ViT-B/32, --skip-clip
- phase3_score.py ✅ — LAION/MUSIQ/BRISQUE aesthetic blend; composition+burst compare; +ollama stop before LLaVA; +consecutive LLaVA failure detection (10 in a row = abort+checkpoint)
- phase4_select.py ✅ — +MMR selection (lambda=0.7, CLIP embeddings, moment cap), --no-mmr flag, mmr_score field
- config.py ✅ — all weights, thresholds, and checkpoint intervals
- phase1b_burst.py ✅ — burst limiter phase (between phase2 and phase3); BURST_MAX_KEEP=3
- run_pipeline.py ✅ — 7-phase pipeline runner (--from/--only/--skip-review/--dry-run)
- phase5_review.py ✅ — CANDID review complete: 800 reviewed, 80 approved (10%), 1 person tagged, 0 errors
- README.md ✅ — added for public repo (2026-04-17)
- Next: full 21k pipeline run (see Phase Order below).

## Non-Negotiable Rules
- NEVER delete, move, or modify original photo files
- ALWAYS show a plan and wait for approval before writing any code
- ALWAYS run one phase at a time, confirm it works before moving to the next
- ALWAYS write to logs/ with timestamps so every action is traceable
- ALWAYS handle errors gracefully — if one photo fails, log it and continue, never crash the pipeline

## Self-Maintenance Rules
- After completing ANY phase, script, or significant change, Claude Code MUST automatically update context.md and CLAUDE.md without being asked
- context.md updates required after every completion:
  - Last Updated timestamp
  - Current Status
  - Phase table (flip ⏳ to ✅, fill in output file)
  - Key Numbers (fill in real counts where available)
  - Known Issues (add any flags encountered)
- CLAUDE.md Current Status block must be updated after every phase
- These updates are NON-NEGOTIABLE — treat them as part of the definition of "done" for any task
- Never consider a phase complete until context.md and CLAUDE.md are updated

## Code Style
- Every script must have a main() function and if __name__ == "__main__" block
- Every function must have a docstring
- Use tqdm for all progress bars
- Use config.py for every configurable value — no hardcoded numbers anywhere else
- Print a clear summary at the end of every phase (processed / skipped / failed counts)

## Project Context
- Windows 11, Python 3.11.6, CUDA 12.4, RTX 3050 Laptop GPU (4GB VRAM)
- Virtual environment is always active
- Photos live on an external hard drive, path set in config.py
- Pipeline has 6 phases — always build and test one at a time
- All outputs are copies — originals on hard drive are never touched

## Phase Order
0. Download LAION model: save sa_0_4_vit_l_14_linear.pth as aesthetic_model.pth in project root
1. phase1_filter.py --scan-report  (generates scan_report.json for pre-filtering)
2. phase1_filter.py                (full filter run; reads scan_report.json automatically)
3. phase2_enrich.py                (EXIF moments + capture_time + subfolder fallback + faces + CLIP embeddings)
   - use --skip-clip if transformers not installed; CLIP embeddings needed for MMR in phase4
4. phase1b_burst.py                (burst limiter — runs between phase2 and phase3)
5. phase3_score.py                 (LAION/BRISQUE + composition + LLaVA + burst compare)
   - use --skip-nima to skip LAION and use BRISQUE (same as before)
   - use --skip-composition if cv2 saliency unavailable
6. phase4_select.py                (MMR-balanced top-800 selection; --no-mmr for greedy fallback)
7. phase5_review.py                (keyboard-driven review UI)
8. phase6_export.py                (copy finals to output folders)

## Output Structure
output/
├── album_top800/
├── album_final200/
└── person_highlights/{name}/

## Memory
- Never assume a previous conversation — always read CLAUDE.md and config.py first
- If unsure about any setting, check config.py before asking