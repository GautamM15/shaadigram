## Last Updated
2026-04-17 (pre-21k code review fixes — 7 issues resolved)

## Current Status
Pre-21k code review fixes — COMPLETE (2026-04-17)
  phase2_enrich.py: +--resume flag with checkpoint every 500 photos; fixed save_json arg order at step5 checkpoint
  phase1_filter.py: error logging DEBUG→WARNING; two-tier phash threshold (cross-folder <=4, same-folder <=8); scan_report.json atomic write; warning on 0-path scan_report match
  phase3_score.py: +ollama stop before LLaVA; +consecutive failure detection (10 in a row = abort+checkpoint)
Eng-review fixes — COMPLETE (2026-04-15)
  utils.py: atomic JSON writes (tmp+os.replace) — all checkpoints now crash-safe
  phase1_filter.py: error logging in process_single_photo; O(N²) phash dedup sharded by folder (~900x reduction at 21k)
  phase2_enrich.py: DeepFace.represent() wrapped with 60s timeout; face_bboxes stored per record for phase3 composition
  phase3_score.py: manual LAION mode 800px resize (CUDA OOM fix); step_composition reads stored bboxes (no Haar re-detect); _coalesce() fixes `or` bug on zero scores; LLaVA payload resized to 1024px (100x smaller)
MUSIQ aesthetic blend upgrade — COMPLETE (2026-04-15)
  phase3_score.py: +MUSIQ scoring (pyiqa, 0-100→0-1), +blend_aesthetic_scores() (BRISQUE×0.20+LAION×0.40+MUSIQ×0.40), +--reweight flag, +--skip-musiq flag, aesthetic_score field replaces laion_score in formula
  config.py: +BRISQUE_BLEND_WEIGHT=0.20, +LAION_BLEND_WEIGHT=0.40, +MUSIQ_BLEND_WEIGHT=0.40
  download_models.py: +verify_musiq() step 4
Phase 2/3/4 upgrades — COMPLETE (CLIP, LAION, composition scoring, MMR — syntax-verified)
  phase2_enrich.py: +step6_clip_embeddings() (ViT-B/32, batch=32, clip_embeddings.npz, clip_event_tags, --skip-clip)
  phase3_score.py: +LAION aesthetic (ViT-L/14+MLP aesthetic_model.pth, fallback BRISQUE), +step_composition() (rot_score/prominence_score/distraction_penalty), +step_burst_compare() (LLaVA grid, burst_rank, emotion adjustment), updated formula LAION×0.25+emotion×0.35+memorab×0.15+rot×0.10+prominence×0.15=1.00, --skip-composition flag
  phase4_select.py: +_mmr_select() MMR lambda=0.7, +_load_clip_embeddings(), moment cap enforced, --no-mmr flag, mmr_score field
  config.py: EMOTION_WEIGHT 0.40→0.35, +14 new constants (LAION_WEIGHT, ROT_WEIGHT, PROMINENCE_WEIGHT, DISTRACTION_PENALTY, BURST_RANK_BONUS/PENALTY, CLIP_MODEL/EMBEDDINGS_FILE/BATCH_SIZE/EVENT_PROMPTS, MMR_LAMBDA, PHASE4_CLIP_EMBEDDINGS)
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
| 3 | phase3_score.py | ✅ Complete, re-run done (LAION) | scored_photos_candid.json |
| 4 | phase4_select.py | ✅ Complete | selected_photos_candid.json |
| 5 | phase5_review.py | ✅ Complete | review_progress.json + output/album_approved/ (80 approved) |
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
- Phase 3 full run (1,477 photos): 74 LLaVA fallbacks, 16,500s (~4h 35m), ~11.2s/photo
  - LAION: 1,477/1,477 | BRISQUE fallbacks: 0 | LLaVA fallbacks: 74
  - Score range: 0.2220–0.7821 | Mean: 0.5897 | Median: 0.6114
  - Candid bonus applied: 64 | Closed-eyes penalty: 0 | Soft-blur penalty: 64 | Distraction penalty: 0
  - Top photo: FMP00678.jpg at 0.7821
  - Output: scored_photos_candid.json (1,521 records, 1,477 scored, 44 rejected)
- Photos scored: 1,477
- Phase 4 selection (11s):
  - Mode: MMR (CLIP diversity, lambda=0.7, 100% embedding coverage)
  - Top 800 selected: 800/800 (rank 1: FMP00678.jpg score=0.7821, rank 800: TMS09896.jpg score=0.6077)
  - Moment balancing: all photos moment_label=unknown (no EXIF in CANDID set — no cap applied)
  - Shot type distribution in top 800: solo_portrait 274 (34.2%), group 273 (34.1%), couple 253 (31.6%)
  - No diversity warnings (no type > 60%)
  - Score distribution: >=0.80: 0 | 0.70-0.79: 109 (13.6%) | 0.60-0.69: 642 (80.2%) | 0.50-0.59: 49 (6.1%)
  - Mean: 0.6577 | Median: 0.6556
  - Output: selected_photos_candid.json (1,521 records, in_top800 + selection_rank on every record)
  - Gautam identification removed from phase 4 — done manually in phase 5 review UI
- Final album selection: 800 (top 800 selected)
- Phase 5 CANDID review (2026-04-17):
  - 800 photos reviewed, 80 approved (10% acceptance rate)
  - 1 person tagged (G=gautam)
  - 0 errors
  - Output: output/album_approved/

## Decisions Made
- Candid photos protected via soft_blur tier (not hard rejected)
- CANDID_BONUS_MULTIPLIER = 1.10 applied in phase 3
- No photos ever deleted — all outputs are copies
- Keyboard-driven review UI (J/K navigate, Space approve, R reject)
- Output folders: album_top800, album_final200, person_highlights/{name}
- Social ready export removed from scope
- Gautam identification is manual in Phase 5 review UI — not automated in phase 4

## Pre-21k Improvements (2026-04-14)
- phase1_filter.py: --scan-report flag (phash-only dedup; filename dedup Pass A removed 2026-04-17 — cameras reuse filenames across folders, so same filename ≠ same photo)
- phase1_filter.py: step1_ingest reads scan_report.json on startup and pre-filters known phash dupes
- phase2_enrich.py: capture_time field added per surviving record (ISO string from EXIF)
- phase2_enrich.py: MOMENT_SUBFOLDER_FALLBACK=True — no-EXIF photos get moment_label from parent directory, moment_id starting at 10001
- phase1b_burst.py: new phase between phase2 and phase3; BURST_MAX_KEEP=3, BURST_SEQUENCE_GAP=5; EXIF 2-min windows, fallback to filename-seq proximity
- run_pipeline.py: full 7-phase runner with --from/--only/--skip-review/--dry-run
- config.py: SCAN_REPORT_TXT, SCAN_REPORT_JSON, SCAN_REPORT_WORKERS, MOMENT_SUBFOLDER_FALLBACK, BURST_MAX_KEEP, BURST_SEQUENCE_GAP, PHASE1B_INPUT added

## Phase 2/3/4 Upgrades (2026-04-14)
- phase2_enrich.py: step6_clip_embeddings() added as new step 6; old step6_output renamed step7_output; clip_event_tags field added to _NULL_ENRICHMENT; --skip-clip flag; clip_embeddings.npz written to project root
- phase3_score.py: LAION aesthetic replaces BRISQUE/NIMA in step1 (priority: LAION→NIMA→BRISQUE); step_composition() added (step 2b: rot/prominence/distraction); step_burst_compare() added (step 2c: LLaVA grid ranking, emotion_score adjustment, burst_rank field); compute_final_score() updated to new formula; --skip-composition flag added; aesthetic_model.pth must be downloaded manually
- phase4_select.py: _load_clip_embeddings() + _mmr_select() added; step2_moment_balanced_selection() extended with embeddings= param; --no-mmr flag added; mmr_score field in output
- config.py: EMOTION_WEIGHT changed from 0.40 to 0.35 (new formula requires 0.35); LIGHTING_WEIGHT commented out (replaced by ROT_WEIGHT + PROMINENCE_WEIGHT)

## MUSIQ Blend Upgrade (2026-04-15)
- phase3_score.py: step1_nima() restructured into 3 sub-passes: A=LAION+BRISQUE, B=MUSIQ, C=blend
- New fields per record: brisque_score, musiq_score, aesthetic_score (brisque×0.20 + laion×0.40 + musiq×0.40)
- compute_final_score() now uses aesthetic_score (falls back to laion_score → nima_score for old JSONs)
- --reweight flag: re-blends + re-scores in <10s without re-running models; edit blend weights in config.py then run
- --skip-musiq flag: aesthetic blend uses BRISQUE+LAION only (weights redistributed)
- download_models.py: verify_musiq() added as step 4
- pyiqa must be installed: pip install pyiqa

## Known Issues / Flags
- Phase 1 scan report: FIXED (2026-04-17). Two-tier phash threshold — cross-folder <=4 (strict), same-folder <=8 (normal). Prevents false positives from different cameras while catching burst dupes.
- Phase 1 scan report: FIXED (2026-04-17). scan_report.json now written via save_json (atomic). Warning logged if 0 paths match (stale report detection).
- Phase 1: FIXED (2026-04-17). process_single_photo errors now logged at WARNING (was DEBUG — silent data loss).
- Phase 2: FIXED (2026-04-17). --resume flag added with checkpoint every 500 photos. Crash at hour 20 loses ~1h max instead of entire run. Fixed save_json arg order at step5 checkpoint.
- Phase 3 LAION: FIXED (2026-04-15). Uses aesthetic-predictor pip package (predict_aesthetic, open-clip-torch, vit_b_32 nn.Linear(512,1)). No weights file needed.
- Phase 3: FIXED (2026-04-17). ollama stop runs before LLaVA scoring to free stale VRAM. Consecutive failure detection: 10 in a row = abort + checkpoint save.
- Phase 3 composition: cv2.saliency.StaticSaliencySpectralResidual_create() requires opencv-contrib-python (not plain opencv-python). If unavailable, distraction_penalty returns 1.0 (no penalty). Use --skip-composition if module not found.
- Phase 4 MMR: requires clip_embeddings.npz generated by phase2 step6. Will fall back to greedy selection if file missing. Use --no-mmr to force greedy.
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
4. LAION model: already handled — aesthetic-predictor pip package installed, no weights file needed
5. Ready for full 21k run. Recommended order:
   a. python phase1_filter.py --scan-report  (generates scan_report.json — fast, <60s)
   b. python phase1_filter.py                (full run; reads scan_report.json auto)
   c. python phase2_enrich.py --input surviving_photos.json --output enriched_photos.json
      (CLIP step6 runs automatically; use --skip-clip if transformers not installed)
   d. python phase1b_burst.py --input enriched_photos.json --test  (verify burst groups first)
   e. python phase1b_burst.py --input enriched_photos.json
   f. python phase3_score.py --input enriched_photos.json
      (LAION runs via aesthetic-predictor pip package automatically; no extra flags needed)
      (add --skip-composition if cv2 saliency module unavailable)
   g. python phase4_select.py  (MMR if clip_embeddings.npz found; --no-mmr for greedy)
   h. python phase5_review.py
