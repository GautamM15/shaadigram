## Last Updated
2026-04-14

## Current Status
Pre-21k improvements — COMPLETE (all 3 improvements built and syntax-verified)
  Improvement 1: phase1_filter.py --scan-report flag (Pass A filename dedup + Pass B phash dedup)
  Improvement 2: phase2_enrich.py subfolder moment fallback + capture_time field per record
  Improvement 3: phase1b_burst.py burst limiter (new phase, runs between phase2 and phase3)
  run_pipeline.py — full 7-phase pipeline runner with --from/--only/--skip-review/--dry-run
Phase 5 — COMPLETE, phase5_review.py written (1,313 lines), import verified, syntax OK
Phase 4 — COMPLETE, 800/800 selected in 0.08s, solo 39% / group 32% / couple 29%, no diversity warnings
Phase 3 — COMPLETE, full run done on 1,476 photos, 5 LLaVA fallbacks, 4h 1m
Phase 2 — COMPLETE, full run done on 1,476 photos, 0 errors, 3h 17m (needs re-run for capture_time + subfolder fallback)
Phase 1 — COMPLETE and tested on CANDID subfolder (1,755 photos)
enroll_face.py — COMPLETE (GUI + threaded DeepFace + --list flag)

## Phase Completion
| Phase | File | Status | Output |
|-------|------|--------|--------|
| 1 | phase1_filter.py | ✅ Complete | surviving_photos_candid.json |
| 2 | phase2_enrich.py | ✅ Complete, full run done | enriched_photos_candid.json |
| 3 | phase3_score.py | ✅ Complete, full run done | scored_photos_candid.json |
| 4 | phase4_select.py | ✅ Complete | selected_photos_candid.json |
| 5 | phase5_review.py | ✅ Complete | review_progress.json + output/album_approved/ |
| 6 | phase6_export.py | ⏳ Not started | - |
| - | enroll_face.py | ✅ Complete | {name}_face.pkl |

## Key Numbers (update after each phase)
- Test folder scanned (CANDID subfolder): 1,755
- Surviving after phase 1 (candid test): 1,476
  - Sharp: 1,412 | Soft blur: 64 | Duplicates removed: 208
- Phase 1 speed: 0.87s/photo → full 21,581 run estimate: ~5.2 hours
- Full dataset scanned: 21,581 total (full run not yet complete)
- Phase 2 full run (1,476 photos): 0 errors, 3h 17m, ~8.0s/photo
  - Faces detected: 1,476/1,476 (100%) | Avg 2.5 faces/photo
  - Shot types: solo_portrait(549) couple(433) group(494) candid(64)
  - Group photos (3+ faces): 494 | Smile scores: off | Person matching: disabled
  - Moments: 8 identified, 1,197 photos with EXIF timestamps; 279 rejected records have none
  - Errors: 0
- Enrolled persons: none (both gautam_face.pkl and siddharth_face.pkl deleted — enrollments were contaminated)
- Phase 3 full run (1,476 photos): 5 LLaVA fallbacks, 14,505s (~4h 1m), ~9.0s/photo
  - BRISQUE: 1,476/1,476 | NIMA: 0 (skip-nima flag used) | LLaVA fallbacks: 5
  - Score range: 0.2437–0.8250 | Mean: 0.6464 | Median: 0.6750
  - Candid bonus applied: 64 | Closed-eyes penalty: 0 | Soft-blur penalty: 64
  - Top photo: TMS00200.jpg at 0.8250
  - Output: scored_photos_candid.json (1,755 records, 1,476 scored)
- Photos scored: 1,476
- Phase 4 selection (0.08s):
  - Top 800 selected: 800/800 (rank 1: TMS00200.jpg score=0.825, rank 800: TMS02443.jpg score=0.656)
  - Moment balancing: all photos moment_label=unknown (no EXIF in CANDID set — no cap applied)
  - Shot type distribution in top 800: solo_portrait 314 (39.2%), group 252 (31.5%), couple 234 (29.2%)
  - No diversity warnings (no type > 60%)
  - Output: selected_photos_candid.json (1,755 records, in_top800 + selection_rank on every record)
  - Gautam identification removed from phase 4 — done manually in phase 5 review UI
- Final album selection: 800 (top 800 selected)

## Decisions Made
- Candid photos protected via soft_blur tier (not hard rejected)
- CANDID_BONUS_MULTIPLIER = 1.10 applied in phase 3
- No photos ever deleted — all outputs are copies
- Keyboard-driven review UI (J/K navigate, Space approve, R reject)
- Output folders: album_top800, album_final200, person_highlights/{name}
- Social ready export removed from scope
- Gautam identification is manual in Phase 5 review UI — not automated in phase 4

## Pre-21k Improvements (2026-04-14)
- phase1_filter.py: --scan-report flag (Pass A filename dedup + Pass B phash, ThreadPool, draft mode)
- phase1_filter.py: step1_ingest reads scan_report.json on startup and pre-filters known dupes
- phase2_enrich.py: capture_time field added per surviving record (ISO string from EXIF)
- phase2_enrich.py: MOMENT_SUBFOLDER_FALLBACK=True — no-EXIF photos get moment_label from parent directory, moment_id starting at 10001
- phase1b_burst.py: new phase between phase2 and phase3; BURST_MAX_KEEP=3, BURST_SEQUENCE_GAP=5; EXIF 2-min windows, fallback to filename-seq proximity
- run_pipeline.py: full 7-phase runner with --from/--only/--skip-review/--dry-run
- config.py: SCAN_REPORT_TXT, SCAN_REPORT_JSON, SCAN_REPORT_WORKERS, MOMENT_SUBFOLDER_FALLBACK, BURST_MAX_KEEP, BURST_SEQUENCE_GAP, PHASE1B_INPUT added

## Known Issues / Flags
- Phase 2: facial_expression_model_weights.h5 manually downloaded and verified valid (5.7 MB HDF5). Emotion off by default (--include-emotion flag to enable).
- Phase 2: opencv detector rejected — detects 0 faces on wedding photos. retinaface is the correct backend.
- Phase 2 speed: ~8.0s/photo actual (3h 17m for 1,476 photos)
- Phase 2: Both enrollments deleted (contaminated — gautam scored 0.60-0.63 on siddharth's face). Gautam added to MANUAL_PERSON_REVIEW in config.py; will be tagged in Phase 5 review UI. Re-enroll siddharth with clean photos before full run.
- Phase 2: Gautam-only guard rail added to step3_4_5_enrich() — if only gautam is enrolled, face matching is disabled with a clear log message.
- Phase 3: Ollama requests timeout=120 did not fire on hung connections (TCP kept alive during GPU compute). Fixed with concurrent.futures thread-level 90s timeout. First run killed after 9h with multi-hour hangs; fixed run completed cleanly in 4h.
- Phase 3: LLaVA 'emotion' key missing on ~5-8% of responses (model returns text instead of JSON) — retry with simplified prompt recovers most; 5 fallbacks to 0.5 across 1,476 photos.
- Phase 3: --resume flag added; BRISQUE checkpoint saved after step1; LLaVA checkpoint saved every 10 photos. Resume with: python phase3_score.py --skip-nima --resume
- Phase 3: NIMA weights not yet downloaded (--skip-nima used, BRISQUE fallback). Download mobilenet_weights.h5 and remove --skip-nima for true neural aesthetic scoring on full run.
- Phase 4: All 1,476 CANDID photos have moment_id=None (phase 2 moment assignment produced no non-None moment_ids — EXIF timestamps may not be present in CANDID subfolder, or phase 2 moment clustering did not write back to records). Moment balancing degrades gracefully to top-N-by-score. Needs investigation before full 21,581-photo run.

## Next Session Quickstart
1. Read CLAUDE.md
2. Read context.md
3. Read config.py
4. Ready for full 21k run. Recommended order:
   a. python phase1_filter.py --scan-report  (generates scan_report.json — fast, <60s)
   b. python phase1_filter.py                (full run; reads scan_report.json auto)
   c. python phase2_enrich.py --input surviving_photos.json --output enriched_photos.json
   d. python phase1b_burst.py --input enriched_photos.json --test  (verify burst groups first)
   e. python phase1b_burst.py --input enriched_photos.json
   f. python phase3_score.py --skip-nima --input enriched_photos.json
   g. python phase4_select.py
   h. python phase5_review.py
