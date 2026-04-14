# Wedding Photo Curator — Claude Code Rules

## Current Status
- Last completed: Pre-21k improvements ✅ (all 3 built and verified, 2026-04-14)
- phase1_filter.py ✅ — + --scan-report (Pass A filename dedup + Pass B phash dedup + step1_ingest pre-filter)
- phase2_enrich.py ✅ — + capture_time field + MOMENT_SUBFOLDER_FALLBACK subfolder moment assignment
- phase1b_burst.py ✅ — new burst limiter phase (between phase2 and phase3); BURST_MAX_KEEP=3
- run_pipeline.py ✅ — 7-phase pipeline runner (--from/--only/--skip-review/--dry-run)
- phase5_review.py ✅ — tkinter review UI: StartupDialog + ReviewApp, 3-col grid, background loads, filter panel, atomic progress save, export to album_approved/ + person_highlights/
- phase4_select.py ✅ — selected_photos_candid.json: 800 in top800, rank 1 score=0.825
- phase3_score.py ✅ — scored_photos_candid.json: 1,476 scored, mean 0.6464, median 0.6750
- phase2_enrich.py — needs re-run on full 21k dataset (capture_time + subfolder fallback are new)
- Next: Run full 21k pipeline starting with phase1_filter.py --scan-report

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
1. phase1_filter.py --scan-report  (run first — generates scan_report.json for pre-filtering)
2. phase1_filter.py                (full filter run; reads scan_report.json automatically)
3. phase2_enrich.py                (EXIF moments + capture_time + subfolder fallback + faces)
4. phase1b_burst.py                (burst limiter — runs between phase2 and phase3)
5. phase3_score.py                 (NIMA + LLaVA scoring)
6. phase4_select.py                (moment-balanced top-800 selection)
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