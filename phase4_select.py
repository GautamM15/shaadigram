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


# ── Step 2 — Moment-balanced top-800 selection ────────────────────────────────

def step2_moment_balanced_selection(eligible: list, logger) -> list:
    """Select up to TOP_N_ALBUM photos using moment-balanced greedy selection.

    Algorithm:
        1. Sort eligible photos by final_score descending (stable).
        2. Walk the sorted list; for each photo:
           - If moment_label == "unknown" (no EXIF): always accept — no cap.
           - Else: accept only if moment_count[moment_id] < MAX_PHOTOS_PER_MOMENT.
        3. Stop when TOP_N_ALBUM selected or list exhausted.

    Sets on each eligible record:
        in_top800      (bool)
        selection_rank (int or None)

    Args:
        eligible: Eligible records (status in _SURVIVING, final_score set).
        logger:   Logger instance.

    Returns:
        eligible list with in_top800 and selection_rank populated.
    """
    # Initialise all eligible with defaults so every record has the fields set
    # regardless of whether the loop reaches them.
    for record in eligible:
        record["in_top800"]      = False
        record["selection_rank"] = None

    sorted_eligible = sorted(
        eligible, key=lambda r: r.get("final_score", 0.0), reverse=True
    )

    moment_counts: dict = defaultdict(int)  # moment_id → photos selected so far
    selected    = 0
    skipped_cap = 0

    for record in sorted_eligible:
        if selected >= config.TOP_N_ALBUM:
            break

        moment_label = record.get("moment_label") or "unknown"
        moment_id    = record.get("moment_id")

        if moment_label == "unknown":
            # No EXIF timestamp — competes freely, no moment cap applied
            pass
        else:
            if moment_counts[moment_id] >= config.MAX_PHOTOS_PER_MOMENT:
                skipped_cap += 1
                continue
            moment_counts[moment_id] += 1

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
        "Step 2 -- selection complete  selected=%d  skipped_cap=%d  moments=[%s]",
        selected, skipped_cap, moment_summary,
    )
    print(
        f"\nStep 2: selected {selected}/{config.TOP_N_ALBUM}  "
        f"skipped_cap={skipped_cap}  [{moment_summary}]"
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

    save_json(output_path, all_records)
    logger.info("Saved %s  (%d records)", output_path, len(all_records))

    # ── Build summary ─────────────────────────────────────────────────────────
    top800 = [r for r in eligible if r.get("in_top800")]
    n_top  = len(top800)

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
    """Orchestrate Phase 4: load → select → diversity check → output."""
    parser = argparse.ArgumentParser(
        description="Phase 4 -- Moment-balanced top-800 selection."
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
    args = parser.parse_args()

    logger = setup_logging("phase4_select")
    start  = time.time()
    logger.info(
        "=== Phase 4 started  input=%s  output=%s ===",
        args.input, args.output,
    )

    all_records, eligible = step1_load_validate(args.input, logger)
    eligible = step2_moment_balanced_selection(eligible, logger)
    step3_diversity_check(eligible, logger)
    step4_output(all_records, eligible, args.output, logger)

    elapsed = time.time() - start
    logger.info("=== Phase 4 complete in %.2fs ===", elapsed)
    print(f"\nPhase 4 complete in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
