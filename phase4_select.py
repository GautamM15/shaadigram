"""
phase4_select.py — Phase 4: Moment-balanced top-800 selection.

Reads scored_photos_candid.json (phase 3 output), applies a moment-balanced
greedy selection to pick the best TOP_N_ALBUM (800) photos, checks shot-type
diversity in the selection, and writes selected_photos_candid.json with two
new fields per record:

    in_top800        bool  — whether the photo is in the top-800 selection
    selection_rank   int   — 1-based rank within top 800 (null if not selected)

Usage:
    python phase4_select.py
    python phase4_select.py --input scored_photos_candid.json --output selected_photos_candid.json
"""

import argparse
import os
import time
from collections import defaultdict

import numpy as np

import config
from utils import load_json, save_json, setup_logging

# Statuses that are eligible for selection
_SURVIVING = {"surviving", "soft_blur_surviving"}


# ── Step 1 — Load & validate ──────────────────────────────────────────────────

def step1_load_validate(input_path: str, logger) -> tuple:
    """Load scored records and filter to eligible (surviving + valid score).

    Args:
        input_path: Path to scored_photos_candid.json.
        logger:     Logger instance.

    Returns:
        Tuple (all_records, eligible) where all_records is the full 1,755-record
        list and eligible is the subset with status in _SURVIVING and a non-null
        final_score.

    Raises:
        ValueError: If the input file is missing or empty.
    """
    all_records = load_json(input_path)
    if not all_records:
        raise ValueError(f"Could not load {input_path} — file missing or empty")

    eligible = [
        r for r in all_records
        if r.get("status") in _SURVIVING and r.get("final_score") is not None
    ]
    ineligible = len(all_records) - len(eligible)

    logger.info(
        "Step 1 -- loaded %d records  eligible=%d  ineligible=%d",
        len(all_records), len(eligible), ineligible,
    )
    print(
        f"\nStep 1: loaded {len(all_records)} records  "
        f"eligible={len(eligible)}  ineligible={ineligible}"
    )
    return all_records, eligible


# ── CLIP embedding helpers ────────────────────────────────────────────────────

def _load_clip_embeddings(emb_path: str, records: list, logger) -> dict:
    """Load CLIP embeddings from a .npz file and build a path→embedding dict.

    The .npz file must have two arrays:
        paths      — shape (N,) string array of absolute image paths
        embeddings — shape (N, 512) float32 unit-normalised embeddings

    Args:
        emb_path: Path to clip_embeddings.npz.
        records:  All eligible records (used to validate coverage).
        logger:   Logger instance.

    Returns:
        Dict mapping image path (str) to unit-normalised float32 numpy array,
        or None if the file is missing, corrupt, or covers fewer than 50% of
        eligible records (indicating a stale/partial embedding run).
    """
    if not os.path.exists(emb_path):
        logger.info("CLIP embeddings file not found (%s) -- MMR disabled", emb_path)
        return None

    try:
        data   = np.load(emb_path, allow_pickle=True)
        paths  = data["paths"]
        embs   = data["embeddings"].astype(np.float32)
        if len(paths) != len(embs):
            raise ValueError("paths/embeddings array length mismatch")
    except Exception as exc:
        logger.warning("CLIP embeddings load failed (%s) -- MMR disabled", exc)
        return None

    path_to_emb = {str(p): embs[i] for i, p in enumerate(paths)}

    # Coverage check
    eligible_paths  = {r["path"] for r in records}
    covered         = sum(1 for p in eligible_paths if p in path_to_emb)
    coverage_pct    = covered / max(len(eligible_paths), 1) * 100

    if coverage_pct < 50:
        logger.warning(
            "CLIP embeddings cover only %.0f%% of eligible photos -- MMR disabled",
            coverage_pct,
        )
        return None

    logger.info(
        "CLIP embeddings loaded: %d vectors  coverage=%.0f%%  file=%s",
        len(path_to_emb), coverage_pct, emb_path,
    )
    print(f"\nCLIP embeddings: {len(path_to_emb)} vectors  coverage={coverage_pct:.0f}%")
    return path_to_emb


def _mmr_select(candidates: list, embeddings: dict, n: int,
                 lambda_: float, is_capped_fn, register_fn, logger) -> list:
    """Select n photos using Maximal Marginal Relevance (MMR).

    MMR balances quality (final_score) against visual diversity (CLIP cosine
    similarity to already-selected photos):

        mmr_score = lambda_ × final_score
                  − (1 − lambda_) × max_cosine_sim_to_selected

    For the first pick max_cosine_sim_to_selected = 0 so the highest-scoring
    photo is always selected first.

    Caps are enforced via is_capped_fn; register_fn is called immediately on
    each chosen photo so caps are updated before the next iteration.

    Args:
        candidates:  List of eligible record dicts with final_score and path.
        embeddings:  Dict of path → unit-normalised float32 (512,) embedding.
        n:           Target number of photos to select.
        lambda_:     MMR trade-off (0 = pure diversity, 1 = pure score).
        is_capped_fn: Callable(record) → bool — True means "skip (capped)".
        register_fn:  Callable(record) → None — called right after each pick.
        logger:       Logger instance.

    Returns:
        List of selected record dicts (up to n), in selection order.
    """
    remaining  = list(candidates)
    selected   = []
    sel_embs   = []   # list of (512,) arrays for dot-product batch

    for _ in range(n):
        if not remaining:
            break

        best_score = -1e9
        best_idx   = -1

        # Build (K, 512) matrix from selected embeddings for batch cosine sim
        if sel_embs:
            sel_matrix = np.stack(sel_embs, axis=0)

        for i, rec in enumerate(remaining):
            if is_capped_fn(rec):
                continue

            fs = rec.get("final_score", 0.0) or 0.0

            if not sel_embs:
                mmr = lambda_ * fs
            else:
                emb = embeddings.get(rec["path"])
                if emb is None:
                    max_sim = 0.0
                else:
                    sims    = sel_matrix @ emb   # (K,)
                    max_sim = float(sims.max())

                mmr = lambda_ * fs - (1.0 - lambda_) * max_sim

            if mmr > best_score:
                best_score = mmr
                best_idx   = i

        if best_idx == -1:
            break   # all remaining are capped

        chosen = remaining.pop(best_idx)
        chosen["mmr_score"] = round(best_score, 6)
        selected.append(chosen)
        register_fn(chosen)   # update caps immediately for next iteration

        emb = embeddings.get(chosen["path"])
        if emb is not None:
            sel_embs.append(emb)

    logger.info("MMR selection: selected %d / %d target", len(selected), n)
    return selected


# ── Step 2 — Moment-balanced top-800 selection ────────────────────────────────

def step2_moment_balanced_selection(eligible: list, logger,
                                     embeddings: dict = None) -> list:
    """Select up to TOP_N_ALBUM photos using moment-balanced selection.

    When CLIP embeddings are available (embeddings dict passed), uses Maximal
    Marginal Relevance (MMR) with lambda=MMR_LAMBDA for diversity-aware
    selection.  Falls back to greedy score-ordered selection when embeddings
    are None.

    Moment cap (MAX_PHOTOS_PER_MOMENT) is enforced in both modes.
    Photos with moment_label=="unknown" (no EXIF) are never capped.

    Sets on each eligible record:
        in_top800      (bool)
        selection_rank (int or None)
        mmr_score      (float or None — only set in MMR mode)

    Args:
        eligible:   Eligible records (status in _SURVIVING, final_score set).
        logger:     Logger instance.
        embeddings: Optional dict of path → float32 (512,) CLIP embedding.

    Returns:
        eligible list with in_top800, selection_rank (and optionally mmr_score)
        populated.
    """
    # Initialise all eligible with defaults
    for record in eligible:
        record["in_top800"]      = False
        record["selection_rank"] = None
        record.setdefault("mmr_score", None)

    moment_counts: dict    = defaultdict(int)
    shot_type_counts: dict = defaultdict(int)
    _shot_cap = (
        int(config.MAX_SHOT_TYPE_FRACTION * config.TOP_N_ALBUM)
        if config.MAX_SHOT_TYPE_FRACTION > 0 else 0
    )

    def _is_capped(rec: dict) -> bool:
        """Return True if this record would exceed its moment or shot-type cap."""
        lbl = rec.get("moment_label") or "unknown"
        mid = rec.get("moment_id")
        if lbl != "unknown" and moment_counts[mid] >= config.MAX_PHOTOS_PER_MOMENT:
            return True
        if _shot_cap > 0:
            shot = rec.get("primary_shot_type") or "unknown"
            if shot != "unknown" and shot_type_counts[shot] >= _shot_cap:
                return True
        return False

    def _register(rec: dict) -> None:
        """Increment moment and shot-type counters for a just-selected record."""
        lbl = rec.get("moment_label") or "unknown"
        mid = rec.get("moment_id")
        if lbl != "unknown":
            moment_counts[mid] += 1
        shot = rec.get("primary_shot_type") or "unknown"
        if shot != "unknown":
            shot_type_counts[shot] += 1

    use_mmr = embeddings is not None
    mode    = "MMR" if use_mmr else "greedy"
    logger.info(
        "Step 2 -- selection mode=%s  target=%d  shot_cap=%s",
        mode, config.TOP_N_ALBUM,
        f"{_shot_cap}/type ({config.MAX_SHOT_TYPE_FRACTION:.0%})" if _shot_cap else "disabled",
    )
    print(f"\nStep 2: selection  mode={mode}  target={config.TOP_N_ALBUM}  "
          f"shot_cap={f'{_shot_cap}/type' if _shot_cap else 'disabled'}")

    if use_mmr:
        selected_records = _mmr_select(
            candidates  = sorted(eligible, key=lambda r: r.get("final_score", 0.0), reverse=True),
            embeddings  = embeddings,
            n           = config.TOP_N_ALBUM,
            lambda_     = config.MMR_LAMBDA,
            is_capped_fn= _is_capped,
            register_fn = _register,
            logger      = logger,
        )
        for rank, rec in enumerate(selected_records, 1):
            rec["in_top800"]      = True
            rec["selection_rank"] = rank
        selected = len(selected_records)
        skipped_cap = 0
    else:
        # Greedy score-ordered selection (original behaviour)
        sorted_eligible = sorted(
            eligible, key=lambda r: r.get("final_score", 0.0), reverse=True
        )
        selected    = 0
        skipped_cap = 0

        for record in sorted_eligible:
            if selected >= config.TOP_N_ALBUM:
                break
            if _is_capped(record):
                skipped_cap += 1
                continue
            _register(record)
            selected += 1
            record["in_top800"]      = True
            record["selection_rank"] = selected

    if selected < config.TOP_N_ALBUM:
        logger.warning(
            "Only %d of %d target photos selected "
            "(insufficient eligible photos after moment balancing)",
            selected, config.TOP_N_ALBUM,
        )
        print(
            f"\nWARNING: Only {selected}/{config.TOP_N_ALBUM} photos selected "
            "— insufficient eligible photos after moment balancing"
        )

    if moment_counts:
        moment_summary = "  ".join(
            f"moment_{mid}={cnt}" for mid, cnt in sorted(moment_counts.items())
        )
    else:
        moment_summary = "all unknown (no EXIF — no cap applied)"

    logger.info(
        "Step 2 -- selection complete  mode=%s  selected=%d  skipped_cap=%d  moments=[%s]",
        mode, selected, skipped_cap, moment_summary,
    )
    print(
        f"\nStep 2: selected {selected}/{config.TOP_N_ALBUM}  "
        f"mode={mode}  skipped_cap={skipped_cap}  [{moment_summary}]"
    )
    return eligible


# ── Step 3 — Shot-type diversity check ───────────────────────────────────────

def step3_diversity_check(eligible: list, logger) -> None:
    """Log shot-type distribution in the top-800 and warn if any type > 60%.

    No data modifications — logging and printing only.

    Args:
        eligible: Eligible records with in_top800 populated.
        logger:   Logger instance.
    """
    top800 = [r for r in eligible if r.get("in_top800")]
    n = len(top800)
    if n == 0:
        logger.warning("Step 3 -- no photos in top 800, skipping diversity check")
        return

    shot_counts: dict[str, int] = defaultdict(int)
    for record in top800:
        shot_counts[record.get("primary_shot_type") or "unknown"] += 1

    w = 56
    lines = ["-" * w, " Step 3: Shot-type diversity in top 800", "-" * w]
    diversity_warnings = []

    for shot_type, count in sorted(shot_counts.items(), key=lambda x: -x[1]):
        pct  = count / n * 100
        flag = "  ← WARNING: >60%" if pct > 60 else ""
        lines.append(f"   {shot_type:<24} : {count:>5}  ({pct:>5.1f}%){flag}")
        if pct > 60:
            diversity_warnings.append((shot_type, pct))

    lines.append("-" * w)
    block = "\n".join(lines)
    print("\n" + block)
    logger.info("Step 3 -- diversity check\n%s", block)

    for shot_type, pct in diversity_warnings:
        msg = (
            f"WARNING: Album may lack variety — {shot_type} photos are "
            f"{pct:.0f}% of selection"
        )
        logger.warning(msg)
        print(msg)


# ── Step 4 — Output + summary ─────────────────────────────────────────────────

def step4_output(all_records: list, eligible: list,
                 output_path: str, logger) -> None:
    """Merge selection fields into all records, save JSON, and print summary.

    Eligible records already carry in_top800 and selection_rank.
    Ineligible (rejected) records receive False/None defaults before saving.

    Args:
        all_records:  Full 1,755-record list (eligible + rejected).
        eligible:     Eligible subset with selection fields populated.
        output_path:  Destination JSON path.
        logger:       Logger instance.
    """
    eligible_paths = {r["path"] for r in eligible}

    for record in all_records:
        if record["path"] not in eligible_paths:
            record["in_top800"]      = False
            record["selection_rank"] = None
            record.setdefault("mmr_score", None)

    save_json(output_path, all_records)
    logger.info("Saved %s  (%d records)", output_path, len(all_records))

    # ── Build summary ─────────────────────────────────────────────────────────
    top800 = [r for r in eligible if r.get("in_top800")]
    n_top  = len(top800)

    mmr_used = any(r.get("mmr_score") is not None for r in top800)

    # Moment distribution within top 800
    moment_dist: dict[str, int] = defaultdict(int)
    for record in top800:
        mid = record.get("moment_id")
        lbl = record.get("moment_label") or "unknown"
        key = "unknown (no EXIF)" if mid is None else f"moment_{mid} ({lbl})"
        moment_dist[key] += 1

    # Shot-type distribution within top 800
    shot_dist: dict[str, int] = defaultdict(int)
    for record in top800:
        shot_dist[record.get("primary_shot_type") or "unknown"] += 1

    w = 56
    lines = [
        "-" * w,
        " Phase 4 -- Selection Summary",
        "-" * w,
        f" Total records              : {len(all_records):>6}",
        f" Eligible (surviving)       : {len(eligible):>6}",
        f" Ineligible (rejected)      : {len(all_records) - len(eligible):>6}",
        f" Selection mode             : {'MMR (CLIP diversity)' if mmr_used else 'greedy (score-only)'}",
        "-" * w,
        " TOP 800 SELECTION",
        f" Target                     : {config.TOP_N_ALBUM:>6}",
        f" Selected                   : {n_top:>6}",
        "-" * w,
        " MOMENT DISTRIBUTION IN TOP 800",
    ]
    for key, cnt in sorted(moment_dist.items(), key=lambda x: -x[1]):
        pct = cnt / n_top * 100 if n_top else 0.0
        lines.append(f"   {key:<34} : {cnt:>5}  ({pct:.1f}%)")

    lines += ["-" * w, " SHOT TYPE DISTRIBUTION IN TOP 800"]
    for shot, cnt in sorted(shot_dist.items(), key=lambda x: -x[1]):
        pct  = cnt / n_top * 100 if n_top else 0.0
        flag = "  ← WARNING" if pct > 60 else ""
        lines.append(f"   {shot:<34} : {cnt:>5}  ({pct:.1f}%){flag}")

    lines += [
        "-" * w,
        f" Output: {os.path.basename(output_path)}  ({len(all_records)} records)",
        "-" * w,
    ]

    summary = "\n".join(lines)
    print("\n" + summary)
    logger.info("\n%s", summary)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    """Orchestrate Phase 4: load → (MMR) select → diversity check → output."""
    parser = argparse.ArgumentParser(
        description="Phase 4 -- Moment-balanced top-800 selection (MMR when CLIP available)."
    )
    parser.add_argument(
        "--input",
        default=config.PHASE4_INPUT,
        help="Scored JSON to read (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default=config.PHASE4_OUTPUT,
        help="Selected JSON to write (default: %(default)s)",
    )
    parser.add_argument(
        "--no-mmr",
        action="store_true",
        dest="no_mmr",
        help="Disable MMR and use greedy score-ordered selection even if CLIP embeddings exist",
    )
    args = parser.parse_args()

    logger = setup_logging("phase4_select")
    start  = time.time()
    logger.info(
        "=== Phase 4 started  input=%s  output=%s  no_mmr=%s ===",
        args.input, args.output, args.no_mmr,
    )

    all_records, eligible = step1_load_validate(args.input, logger)

    # Load CLIP embeddings for MMR selection (optional)
    embeddings = None
    if not args.no_mmr:
        embeddings = _load_clip_embeddings(config.PHASE4_CLIP_EMBEDDINGS, eligible, logger)

    eligible = step2_moment_balanced_selection(eligible, logger, embeddings=embeddings)
    step3_diversity_check(eligible, logger)
    step4_output(all_records, eligible, args.output, logger)

    elapsed = time.time() - start
    logger.info("=== Phase 4 complete in %.2fs ===", elapsed)
    print(f"\nPhase 4 complete in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
