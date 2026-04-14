"""
phase1b_burst.py — Phase 1b: Burst-group limiter.

Runs AFTER phase2_enrich.py and BEFORE phase3_score.py.  Reads the
enriched JSON, groups surviving photos into burst groups using per-photo
EXIF capture_time (2-minute windows within each moment) or filename
sequence proximity when no timestamps are available, keeps the best
BURST_MAX_KEEP photos per group, and marks the rest "rejected_burst".

Two new fields are written to every surviving record:
    burst_group_id  int or None — None for singletons
    burst_rank      int or None — 1 = kept, 2+ = rejected; None for singletons

Usage:
    python phase1b_burst.py
    python phase1b_burst.py --input enriched_photos_candid.json
    python phase1b_burst.py --input enriched_photos_candid.json --test
"""

import argparse
import os
import re
import time
from collections import defaultdict
from datetime import datetime, timedelta

from tqdm import tqdm

import config
from utils import load_json, save_json, setup_logging

_SURVIVING = {"surviving", "soft_blur_surviving"}
_BURST_WINDOW = timedelta(minutes=2)


# ── Step 1 — Load ─────────────────────────────────────────────────────────────

def step1_load(input_path: str, logger) -> tuple:
    """Load enriched JSON and initialise burst fields on surviving records.

    Args:
        input_path: Path to enriched_photos JSON.
        logger:     Logger instance.

    Returns:
        Tuple (all_records, surviving) where surviving is the subset with
        status in _SURVIVING.

    Raises:
        ValueError: If the file is missing or contains no records.
    """
    all_records = load_json(input_path)
    if not all_records:
        raise ValueError(f"Could not load {input_path} — file missing or empty")

    surviving = [r for r in all_records if r.get("status") in _SURVIVING]

    for r in surviving:
        r["burst_group_id"] = None
        r["burst_rank"]     = None

    logger.info(
        "Step 1 -- loaded %d records  surviving=%d",
        len(all_records), len(surviving),
    )
    print(f"\nStep 1: loaded {len(all_records)} records  surviving={len(surviving)}")
    return all_records, surviving


# ── Step 2 — Build burst groups ───────────────────────────────────────────────

def _seq_number(path: str) -> int:
    """Extract the leading sequence number from a filename, or -1 if none."""
    fname = os.path.basename(path)
    m = re.search(r'(\d+)', fname)
    return int(m.group(1)) if m else -1


def step2_build_burst_groups(surviving: list, logger) -> list:
    """Group surviving photos into burst groups.

    Strategy (applied per moment_id):
        Priority 1 — EXIF capture_time available:
            Sort photos by capture_time within the moment, then split on
            gaps > 2 minutes (_BURST_WINDOW).  Each continuous run is a
            burst group.

        Priority 2 — No capture_time (subfolder-fallback moments):
            Sort photos by filename sequence number within the moment, then
            split on sequence gaps > BURST_SEQUENCE_GAP.

    Groups of size 1 are singletons and are not assigned a burst_group_id.
    Photos with moment_id=None are treated individually (no burst grouping).

    Args:
        surviving: Surviving record dicts with burst fields initialised.
        logger:    Logger instance.

    Returns:
        List of burst groups, where each group is a list of record dicts.
        Singletons are not included.
    """
    # Group by moment_id
    by_moment: dict = defaultdict(list)
    no_moment  = []
    for r in surviving:
        mid = r.get("moment_id")
        if mid is None:
            no_moment.append(r)
        else:
            by_moment[mid].append(r)

    burst_groups = []

    for mid, members in by_moment.items():
        # Determine whether this moment has per-photo EXIF timestamps
        has_ts = any(r.get("capture_time") for r in members)

        if has_ts:
            # Sort by capture_time; photos without ts go to end
            def _ts_sort_key(r):
                ct = r.get("capture_time")
                if ct:
                    try:
                        return datetime.fromisoformat(ct)
                    except ValueError:
                        pass
                return datetime.max

            members_sorted = sorted(members, key=_ts_sort_key)
            # Split on 2-minute gaps
            cluster = [members_sorted[0]]
            for i in range(1, len(members_sorted)):
                prev_ct = members_sorted[i - 1].get("capture_time")
                curr_ct = members_sorted[i].get("capture_time")
                if prev_ct and curr_ct:
                    try:
                        gap = datetime.fromisoformat(curr_ct) - datetime.fromisoformat(prev_ct)
                        if gap > _BURST_WINDOW:
                            if len(cluster) > 1:
                                burst_groups.append(cluster)
                            cluster = [members_sorted[i]]
                            continue
                    except ValueError:
                        pass
                cluster.append(members_sorted[i])
            if len(cluster) > 1:
                burst_groups.append(cluster)

        else:
            # No per-photo timestamps — use filename sequence proximity
            members_sorted = sorted(members, key=lambda r: _seq_number(r["path"]))
            cluster = [members_sorted[0]]
            for i in range(1, len(members_sorted)):
                prev_seq = _seq_number(members_sorted[i - 1]["path"])
                curr_seq = _seq_number(members_sorted[i]["path"])
                gap = (curr_seq - prev_seq) if (prev_seq >= 0 and curr_seq >= 0) else 999
                if gap > config.BURST_SEQUENCE_GAP:
                    if len(cluster) > 1:
                        burst_groups.append(cluster)
                    cluster = [members_sorted[i]]
                else:
                    cluster.append(members_sorted[i])
            if len(cluster) > 1:
                burst_groups.append(cluster)

    total_in_bursts = sum(len(g) for g in burst_groups)
    logger.info(
        "Step 2 -- %d burst groups found  photos_in_bursts=%d  "
        "no_moment_singletons=%d",
        len(burst_groups), total_in_bursts, len(no_moment),
    )
    print(
        f"\nStep 2: {len(burst_groups)} burst groups  "
        f"{total_in_bursts} photos in bursts  "
        f"{len(no_moment)} no-moment singletons"
    )
    return burst_groups


# ── Step 3 — Select best within each burst ────────────────────────────────────

def _select_burst_keepers(group: list, max_keep: int) -> list:
    """Choose up to max_keep records from a burst group.

    Selection algorithm:
        1. Sort all candidates by blur_score descending.
        2. Pick greedily, applying a shot-type variety check:
           if the next candidate has the same primary_shot_type as all
           already-kept photos AND a different type exists later in the
           sorted list, defer it and pick the different type first.
        3. Fill any remaining keep slots with the next-highest blur_score.

    Args:
        group:    List of record dicts in one burst group.
        max_keep: Maximum number of records to keep (BURST_MAX_KEEP).

    Returns:
        List of records designated as keepers (at most max_keep items).
    """
    if len(group) <= max_keep:
        return list(group)

    # Sort by blur_score descending; tie-break by max_smile_score
    candidates = sorted(
        group,
        key=lambda r: (-(r.get("blur_score") or 0.0), -(r.get("max_smile_score") or 0.0)),
    )

    keepers    = []
    deferred   = []
    kept_types: set[str] = set()

    for r in candidates:
        if len(keepers) >= max_keep:
            break
        shot_type = r.get("primary_shot_type") or "unknown"

        # Check if all current keepers share the same type and this one matches
        if keepers and shot_type in kept_types and len(kept_types) == 1:
            # See if a different type exists later among remaining candidates
            remaining_idx = candidates.index(r)
            has_other_type = any(
                (c.get("primary_shot_type") or "unknown") != shot_type
                for c in candidates[remaining_idx + 1:]
            )
            if has_other_type:
                deferred.append(r)
                continue

        keepers.append(r)
        kept_types.add(shot_type)

    # Fill remaining slots from deferred (same-type) then rest of candidates
    fill_pool = deferred + [c for c in candidates if c not in keepers and c not in deferred]
    for r in fill_pool:
        if len(keepers) >= max_keep:
            break
        keepers.append(r)

    return keepers


def step3_apply_burst_limits(burst_groups: list, logger, test_mode: bool = False) -> tuple:
    """Apply BURST_MAX_KEEP limit to every burst group and assign IDs.

    Sets burst_group_id and burst_rank on every record in every group.
    Marks rejected burst records with status="rejected_burst".

    Args:
        burst_groups: List of burst groups from step2.
        logger:       Logger instance.
        test_mode:    If True, process only the first 5 groups.

    Returns:
        Tuple (total_kept, total_rejected).
    """
    groups_to_process = burst_groups[:5] if test_mode else burst_groups
    if test_mode:
        print(f"\n[TEST MODE] Processing first {len(groups_to_process)} burst groups only")

    total_kept     = 0
    total_rejected = 0
    largest_group  = 0

    for gid, group in enumerate(
        tqdm(groups_to_process, desc="Limiting bursts", unit="group"), start=1
    ):
        keepers_set = set(id(r) for r in _select_burst_keepers(group, config.BURST_MAX_KEEP))
        largest_group = max(largest_group, len(group))

        kept_rank    = 1
        reject_rank  = config.BURST_MAX_KEEP + 1

        for r in group:
            r["burst_group_id"] = gid
            if id(r) in keepers_set:
                r["burst_rank"] = kept_rank
                kept_rank += 1
                total_kept += 1
            else:
                r["burst_rank"]  = reject_rank
                r["status"]      = "rejected_burst"
                reject_rank += 1
                total_rejected += 1

    logger.info(
        "Step 3 -- burst limiting complete  kept=%d  rejected=%d  largest_group=%d",
        total_kept, total_rejected, largest_group,
    )
    return total_kept, total_rejected, largest_group


# ── Step 4 — Output ───────────────────────────────────────────────────────────

def step4_output(
    all_records: list,
    surviving: list,
    output_path: str,
    total_kept: int,
    total_rejected: int,
    largest_group: int,
    n_groups: int,
    logger,
) -> None:
    """Merge burst fields into all records, save JSON, and print summary.

    Non-surviving records that were never processed receive
    burst_group_id=None and burst_rank=None.

    Args:
        all_records:     Full record list (surviving + rejected).
        surviving:       Surviving subset with burst fields already set.
        output_path:     Path to write the updated JSON.
        total_kept:      Count of photos kept within burst groups.
        total_rejected:  Count of photos rejected as burst duplicates.
        largest_group:   Size of the largest burst group found.
        n_groups:        Total burst groups processed.
        logger:          Logger instance.
    """
    surviving_paths = {r["path"] for r in surviving}

    # Ensure non-surviving records have the burst fields
    for r in all_records:
        if r["path"] not in surviving_paths:
            r.setdefault("burst_group_id", None)
            r.setdefault("burst_rank",     None)

    save_json(output_path, all_records)
    logger.info("Saved %s (%d records)", output_path, len(all_records))

    total_surviving_after = sum(
        1 for r in all_records if r.get("status") in _SURVIVING
    )

    w = 50
    lines = [
        "-" * w,
        " Phase 1b -- Burst Limiter Summary",
        "-" * w,
        f" Total records              : {len(all_records):>6}",
        f" Burst groups found         : {n_groups:>6}",
        f" Largest burst group        : {largest_group:>6}",
        f" Photos rejected (burst)    : {total_rejected:>6}",
        f" Surviving after burst      : {total_surviving_after:>6}",
        "-" * w,
        f" Output: {os.path.basename(output_path)}  ({len(all_records)} records)",
        "-" * w,
    ]
    summary = "\n".join(lines)
    print("\n" + summary)
    logger.info("\n%s", summary)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    """Orchestrate Phase 1b: load -> build groups -> limit -> output."""
    parser = argparse.ArgumentParser(
        description="Phase 1b -- Burst-group limiter. "
                    "Runs after phase2_enrich.py, before phase3_score.py."
    )
    parser.add_argument(
        "--input",
        default=config.PHASE1B_INPUT,
        help="Enriched JSON to read and update in place (default: %(default)s)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Process first 5 burst groups only; print groups without saving",
    )
    args = parser.parse_args()

    logger = setup_logging("phase1b_burst")
    start  = time.time()
    logger.info(
        "=== Phase 1b started  input=%s  test=%s ===",
        args.input, args.test,
    )

    all_records, surviving = step1_load(args.input, logger)
    burst_groups            = step2_build_burst_groups(surviving, logger)

    if args.test:
        # In test mode: print group summaries and exit without saving
        print("\n[TEST MODE] First 5 burst groups:")
        for i, group in enumerate(burst_groups[:5], 1):
            keepers = _select_burst_keepers(group, config.BURST_MAX_KEEP)
            keeper_paths = {id(r) for r in keepers}
            print(f"\n  Group {i} — {len(group)} photos  "
                  f"(moment {group[0].get('moment_label', '?')}):")
            for r in sorted(group, key=lambda r: _seq_number(r["path"])):
                tag = "KEEP" if id(r) in keeper_paths else "skip"
                print(f"    [{tag}] {os.path.basename(r['path'])}  "
                      f"blur={r.get('blur_score', 0):.0f}  "
                      f"type={r.get('primary_shot_type', '?')}")
        elapsed = time.time() - start
        print(f"\nTest complete in {elapsed:.2f}s  (no files modified)")
        logger.info("=== Phase 1b test complete in %.2fs (no save) ===", elapsed)
        return

    total_kept, total_rejected, largest_group = step3_apply_burst_limits(
        burst_groups, logger, test_mode=False
    )
    step4_output(
        all_records, surviving, args.input,
        total_kept, total_rejected, largest_group,
        len(burst_groups), logger,
    )

    elapsed = time.time() - start
    logger.info("=== Phase 1b complete in %.2fs ===", elapsed)
    print(f"\nPhase 1b complete in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
