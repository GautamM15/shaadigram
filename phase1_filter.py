"""
phase1_filter.py — Phase 1: blur detection, exposure check, duplicate clustering.

Scans INPUT_FOLDER recursively for all supported image files, applies a
three-tier blur filter, checks exposure, clusters near-duplicates, and writes
surviving_photos.json to the project root.

Each image is loaded exactly once.  Early-exit logic skips unnecessary
computation for rejected photos (hard-rejected blur photos never reach
exposure or hashing; exposure-failed photos never get hashed).

Usage:
    python phase1_filter.py
    python phase1_filter.py --input "E:\\Other Wedding"
    python phase1_filter.py --test
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import time
from collections import defaultdict
from datetime import datetime

import cv2
import imagehash
import numpy as np
import rawpy
from PIL import Image
from tqdm import tqdm

import config
from utils import save_json, setup_logging

# ── Derived constants (computed once from config, never hardcoded) ─────────────
RAW_EXTENSIONS = {".nef", ".NEF", ".cr2", ".CR2", ".arw", ".ARW", ".dng", ".DNG"}
SUPPORTED_SET = set(config.SUPPORTED_EXTENSIONS)
BLUR_HARD_THRESHOLD = config.BLUR_THRESHOLD * config.BLUR_HARD_REJECT_MULTIPLIER


# ── Union-Find ────────────────────────────────────────────────────────────────

class UnionFind:
    """Union-Find with path compression and union by rank."""

    def __init__(self, n: int):
        """Initialise n disjoint sets.

        Args:
            n: Number of elements.
        """
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        """Return the root of the set containing x (with path compression).

        Args:
            x: Element index.

        Returns:
            Root index.
        """
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # two-step compression
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        """Merge the sets containing x and y.

        Args:
            x: First element index.
            y: Second element index.
        """
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1


# ── Per-photo helpers ─────────────────────────────────────────────────────────

def load_image(path: str) -> tuple:
    """Load an image and return a PIL Image without writing anything to disk.

    RAW files (.nef, .cr2, .arw, .dng) are decoded entirely in memory via
    rawpy using camera white-balance.  JPEG/PNG files are opened directly
    with PIL.  Errors are propagated to the caller so per-photo handling
    works correctly.

    Args:
        path: Absolute path to the image file.

    Returns:
        Tuple of (PIL.Image.Image, is_raw: bool).

    Raises:
        Exception: Any load or decode failure — caller must catch.
    """
    ext = os.path.splitext(path)[1]
    if ext in RAW_EXTENSIONS:
        with rawpy.imread(path) as raw:
            rgb = raw.postprocess(use_camera_wb=True)
        return Image.fromarray(rgb), True
    img = Image.open(path)
    img.load()  # force full decode so errors surface here, not later
    return img, False


def compute_blur_score(pil_image: Image.Image) -> float:
    """Measure sharpness using the variance of the Laplacian.

    Higher values indicate a sharper image.  The image is converted to
    grayscale before applying the Laplacian kernel.

    Args:
        pil_image: Any-mode PIL image.

    Returns:
        Laplacian variance as a float.
    """
    gray = np.array(pil_image.convert("L"))
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def get_blur_tier(blur_score: float) -> str:
    """Classify a blur score into one of three tiers.

    Tiers:
        "hard_reject" — score < BLUR_THRESHOLD * BLUR_HARD_REJECT_MULTIPLIER.
                        Genuinely unusable, pipeline rejects immediately.
        "soft_blur"   — score in [hard-reject threshold, BLUR_THRESHOLD).
                        Candid/motion shots that may be emotionally valuable;
                        kept for scoring phase to decide.
        "pass"        — score >= BLUR_THRESHOLD.  Acceptably sharp.

    Args:
        blur_score: Value from compute_blur_score().

    Returns:
        One of "hard_reject", "soft_blur", or "pass".
    """
    if blur_score < BLUR_HARD_THRESHOLD:
        return "hard_reject"
    if blur_score < config.BLUR_THRESHOLD:
        return "soft_blur"
    return "pass"


def compute_exposure_mean(pil_image: Image.Image) -> float:
    """Compute the mean grayscale pixel value to detect over/under-exposure.

    Args:
        pil_image: Any-mode PIL image.

    Returns:
        Mean pixel value in [0.0, 255.0].
    """
    gray = np.array(pil_image.convert("L"))
    return float(np.mean(gray))


def compute_phash(pil_image: Image.Image) -> str:
    """Compute the perceptual hash (pHash) of an image.

    Args:
        pil_image: Any-mode PIL image.

    Returns:
        Hex string representation of the perceptual hash.
    """
    return str(imagehash.phash(pil_image))


def process_single_photo(path: str) -> dict:
    """Load one image and compute all quality metrics in a single pass.

    Early-exit logic avoids unnecessary work:
        - hard_reject blur  → exposure and phash are skipped
        - exposure failure  → phash is skipped
        - Only photos that survive both checks get a phash computed

    A phash failure is treated as non-fatal: the photo retains its
    surviving status but phash is set to None, so it is excluded from
    duplicate clustering.

    Never raises — all exceptions produce status="error".

    Args:
        path: Absolute path to the image file.

    Returns:
        Complete record dict with keys: path, blur_score, blur_tier,
        exposure_mean, phash, cluster_id, is_cluster_best, status.
    """
    record = {
        "path": path,
        "blur_score": None,
        "blur_tier": None,
        "exposure_mean": None,
        "phash": None,
        "cluster_id": None,
        "is_cluster_best": True,
        "status": "error",
    }
    try:
        img, _ = load_image(path)

        # ── Blur ──────────────────────────────────────────────────────────────
        score = compute_blur_score(img)
        tier = get_blur_tier(score)
        record["blur_score"] = score
        record["blur_tier"] = tier

        if tier == "hard_reject":
            record["status"] = "rejected_blur"
            return record  # early exit — no further computation needed

        # ── Exposure ──────────────────────────────────────────────────────────
        mean = compute_exposure_mean(img)
        record["exposure_mean"] = mean

        too_dark = mean < config.EXPOSURE_LOW
        too_bright = mean > config.EXPOSURE_HIGH

        if too_dark or too_bright:
            if tier == "soft_blur":
                record["status"] = "rejected_blur_and_exposure"
            else:
                record["status"] = "rejected_dark" if too_dark else "rejected_bright"
            return record  # early exit — no phash needed for rejected photos

        # ── Perceptual hash (only for survivors) ──────────────────────────────
        try:
            record["phash"] = compute_phash(img)
        except Exception:
            pass  # non-fatal: photo survives but won't participate in dedup

        record["status"] = (
            "soft_blur_surviving" if tier == "soft_blur" else "surviving"
        )

    except Exception:
        pass  # record["status"] remains "error" from initialisation

    return record


# ── Duplicate clustering ──────────────────────────────────────────────────────

def cluster_duplicates(records: list) -> list:
    """Group near-duplicate photos and mark the best keeper in each cluster.

    Only considers records with status "surviving" or "soft_blur_surviving".
    Pairwise Hamming distances between perceptual hashes are computed; pairs
    within DUPLICATE_HASH_THRESHOLD are linked via Union-Find.

    Note: deduplication operates across ALL records passed in, regardless of
    which subfolder they came from.  Cross-folder near-duplicates (same photo
    exported to multiple event subfolders) are caught here via phash distance.
    For filename-based cross-folder deduplication before this pipeline runs,
    use phase1_filter.py --scan-report to generate scan_report.json first.

    Selection rule within a cluster:
        1. Prefer "pass" blur tier over "soft_blur".
        2. Break ties by highest blur_score.
    The winner retains its current status; all others become
    "rejected_duplicate".

    Singletons (no near-duplicate neighbours) receive cluster_id = None and
    is_cluster_best = True.

    Args:
        records: List of photo dicts.  Each must have phash, blur_score,
                 blur_tier, and status fields already set.

    Returns:
        The same list with cluster_id and is_cluster_best filled in on
        every record.
    """
    candidates = [
        i for i, r in enumerate(records)
        if r["status"] in ("surviving", "soft_blur_surviving")
        and r["phash"] is not None
    ]
    n = len(candidates)

    # Default: every record is its own singleton
    for r in records:
        r.setdefault("cluster_id", None)
        r.setdefault("is_cluster_best", True)

    if n == 0:
        return records

    hashes = [imagehash.hex_to_hash(records[i]["phash"]) for i in candidates]
    uf = UnionFind(n)

    for a in range(n):
        for b in range(a + 1, n):
            if hashes[a] - hashes[b] <= config.DUPLICATE_HASH_THRESHOLD:
                uf.union(a, b)

    # Group candidate positions by their Union-Find root
    groups = defaultdict(list)
    for idx, record_idx in enumerate(candidates):
        groups[uf.find(idx)].append(record_idx)

    cluster_id_counter = 0
    for _, members in groups.items():
        if len(members) == 1:
            # True singleton — leave defaults untouched
            continue

        cluster_id_counter += 1
        cid = cluster_id_counter

        def sort_key(rec_idx: int) -> tuple:
            """Sort so the best candidate comes first."""
            r = records[rec_idx]
            tier_priority = 0 if r["blur_tier"] == "pass" else 1
            return (tier_priority, -r["blur_score"])

        members_sorted = sorted(members, key=sort_key)
        best_idx = members_sorted[0]

        for mi in members:
            records[mi]["cluster_id"] = cid
            if mi == best_idx:
                records[mi]["is_cluster_best"] = True
            else:
                records[mi]["is_cluster_best"] = False
                records[mi]["status"] = "rejected_duplicate"

    return records


# ── Scan-report helpers ───────────────────────────────────────────────────────

def _open_header_only(path: str) -> tuple:
    """Read image dimensions from the file header without full pixel decode.

    Uses PIL's lazy-open behaviour: Image.open() reads only the header.
    Accessing img.size returns (width, height) without loading pixel data,
    making this ~10–50x faster than a full decode for large JPEGs.

    Args:
        path: Absolute path to the image file.

    Returns:
        Tuple (width, height, file_size_bytes).  Returns (0, 0, file_size)
        on any header-read error so one bad file never aborts the scan.
    """
    file_size = 0
    try:
        file_size = os.path.getsize(path)
    except OSError:
        pass
    try:
        with Image.open(path) as img:
            w, h = img.size
        return w, h, file_size
    except Exception:
        return 0, 0, file_size


def _compute_phash_fast(path: str) -> str | None:
    """Compute a perceptual hash using a low-resolution draft decode.

    For JPEG/PNG files, PIL's draft() hint instructs the JPEG decoder to
    produce a ~32×32 thumbnail instead of decoding the full image.  This
    is ~5–10x faster than a full decode and produces an identical phash to
    the full-resolution version for near-duplicate detection purposes.

    RAW files are skipped (returns None) — rawpy decodes are too slow for
    a scan pass and RAW files are unlikely to be cross-folder duplicates.

    Args:
        path: Absolute path to the image file.

    Returns:
        Hex phash string, or None if the file is RAW or cannot be read.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in {".nef", ".cr2", ".arw", ".dng"}:
        return None
    try:
        img = Image.open(path)
        img.draft("L", (32, 32))
        img.load()
        return str(imagehash.phash(img))
    except Exception:
        return None


def load_precomputed_phashes(logger) -> dict:
    """Load phash values already computed by phase 1 from any *_photos*.json files.

    Scans the project root for files matching *_photos*.json (e.g.
    surviving_photos.json, surviving_photos_candid.json) and builds a
    path → phash_string lookup.  These values are reused by sr_step3 to
    avoid re-decoding images that phase 1 already processed.

    Args:
        logger: Logger instance.

    Returns:
        Dict mapping absolute path strings to hex phash strings.
        Empty dict if no matching files are found or all fail to load.
    """
    import glob as _glob
    pattern = os.path.join(os.path.dirname(os.path.abspath(__file__)), "*_photos*.json")
    candidates = _glob.glob(pattern)

    phashes: dict[str, str] = {}
    loaded_files = []

    for fpath in candidates:
        try:
            with open(fpath, encoding="utf-8") as f:
                records = json.load(f)
            count_before = len(phashes)
            for r in records:
                p = r.get("path")
                h = r.get("phash")
                if p and h:
                    phashes[p] = h
            added = len(phashes) - count_before
            loaded_files.append(f"{os.path.basename(fpath)} ({added})")
        except Exception as exc:
            logger.warning("Could not load precomputed phashes from %s: %s", fpath, exc)

    if phashes:
        logger.info(
            "load_precomputed_phashes -- %d phashes from: %s",
            len(phashes), ", ".join(loaded_files),
        )
    else:
        logger.info("load_precomputed_phashes -- no cached phashes found")

    return phashes


def sr_step1_collect(scan_folder: str, logger) -> tuple:
    """Walk scan_folder and collect metadata for every supported image file.

    Uses header-only PIL reads for resolution — no full pixel decode.
    Returns both the file list and a per-folder count dict.

    Args:
        scan_folder: Root directory to scan recursively.
        logger:      Logger instance.

    Returns:
        Tuple (files, folder_counts) where files is a list of dicts:
            {path, filename_lower, folder_rel, file_size, width, height, pixels}
        and folder_counts maps relative folder path → file count.
    """
    logger.info("SR step 1 -- collecting files under %s", scan_folder)
    all_entries = []
    for dirpath, _, filenames in os.walk(scan_folder):
        for fname in filenames:
            if os.path.splitext(fname)[1] in SUPPORTED_SET:
                all_entries.append(os.path.join(dirpath, fname))

    files = []
    folder_counts: dict[str, int] = defaultdict(int)

    for path in tqdm(all_entries, desc="Scanning headers", unit="file"):
        w, h, fsz = _open_header_only(path)
        try:
            rel_dir = os.path.relpath(os.path.dirname(path), scan_folder)
        except ValueError:
            rel_dir = os.path.dirname(path)
        folder_counts[rel_dir] += 1
        files.append({
            "path":           path,
            "filename_lower": os.path.basename(path).lower(),
            "folder_rel":     rel_dir,
            "file_size":      fsz,
            "width":          w,
            "height":         h,
            "pixels":         w * h,
        })

    logger.info("SR step 1 complete -- %d files in %d folders", len(files), len(folder_counts))
    return files, dict(folder_counts)


def sr_step2_filename_dedup(files: list, logger) -> tuple:
    """Group files by filename (case-insensitive) and mark lower-res copies.

    Within each duplicate group the keeper is the file with the highest
    pixel count (width * height).  Ties are broken by file size (largest
    wins).  Files with unique names are all keepers.

    Args:
        files:  List of file dicts from sr_step1_collect.
        logger: Logger instance.

    Returns:
        Tuple (keepers, dupes, dup_groups):
            keepers   — list of file dicts that survive Pass A
            dupes     — list of file dicts marked as filename duplicates
            dup_groups — list of group detail dicts for the report
    """
    by_name: dict[str, list] = defaultdict(list)
    for f in files:
        by_name[f["filename_lower"]].append(f)

    keepers    = []
    dupes      = []
    dup_groups = []

    for fname_lower, group in by_name.items():
        if len(group) == 1:
            keepers.append(group[0])
            continue
        # Sort: highest pixels first, then largest file_size
        group_sorted = sorted(group, key=lambda f: (-f["pixels"], -f["file_size"]))
        keeper = group_sorted[0]
        keepers.append(keeper)
        entries = []
        for f in group_sorted:
            decision = "keep" if f is keeper else "skip"
            if f is not keeper:
                dupes.append(f)
            sz_mb = f["file_size"] / 1_048_576
            entries.append({
                "path":      f["path"],
                "folder":    f["folder_rel"],
                "pixels":    f["pixels"],
                "file_size": f["file_size"],
                "size_mb":   round(sz_mb, 1),
                "res":       f"{f['width']}x{f['height']}",
                "decision":  decision,
            })
        dup_groups.append({"filename": fname_lower, "entries": entries})

    logger.info(
        "SR step 2 -- %d filename-dup groups  keepers=%d  skipped=%d",
        len(dup_groups), len(keepers), len(dupes),
    )
    return keepers, dupes, dup_groups


def sr_step3_hash_dedup(keepers: list, logger) -> tuple:
    """Compute phash on Pass-A survivors and cluster near-duplicates.

    Reuses phash values already computed by phase 1 (loaded from any
    *_photos*.json files in the project root) to avoid re-decoding images.
    Only files not present in the cache are decoded via _compute_phash_fast().
    RAW files are excluded from hashing and pass through as unconditional
    keepers.

    Args:
        keepers: List of file dicts that survived Pass A.
        logger:  Logger instance.

    Returns:
        Tuple (final_keep_paths, hash_dupe_paths, hash_groups, cache_stats):
            final_keep_paths — set of paths to keep after both passes
            hash_dupe_paths  — set of paths to skip (hash duplicates)
            hash_groups      — list of group detail dicts for the report
            cache_stats      — dict with keys "reused" and "computed"
    """
    # Separate RAW files — they skip hashing entirely
    raw_exts = {".nef", ".cr2", ".arw", ".dng"}
    non_raw  = [f for f in keepers if os.path.splitext(f["path"])[1].lower() not in raw_exts]
    raw_pass = [f for f in keepers if os.path.splitext(f["path"])[1].lower() in raw_exts]

    logger.info(
        "SR step 3 -- %d non-RAW files to hash  (%d RAW pass-through)",
        len(non_raw), len(raw_pass),
    )

    # Load pre-computed phashes from phase 1 output files
    precomputed = load_precomputed_phashes(logger)

    # Split into cache-hit and cache-miss buckets
    path_to_hash: dict[str, str | None] = {}
    to_compute   = []

    for f in non_raw:
        cached = precomputed.get(f["path"])
        if cached:
            path_to_hash[f["path"]] = cached
        else:
            to_compute.append(f)

    n_reused   = len(non_raw) - len(to_compute)
    n_computed = len(to_compute)

    logger.info(
        "SR step 3 -- %d reused from phase1 cache  %d to compute fresh",
        n_reused, n_computed,
    )
    print(f"\n  Pass B: {n_reused} reused from phase1 cache  |  {n_computed} to compute fresh")

    # Compute fresh phashes for cache misses (thread pool + draft mode)
    if to_compute:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=config.SCAN_REPORT_WORKERS
        ) as executor:
            futures = {
                executor.submit(_compute_phash_fast, f["path"]): f["path"]
                for f in to_compute
            }
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Hashing (Pass B — fresh)",
                unit="file",
            ):
                path_to_hash[futures[future]] = future.result()

    # Union-Find over hash distances
    hashable = [f for f in non_raw if path_to_hash.get(f["path"]) is not None]
    n = len(hashable)
    hash_objs = [imagehash.hex_to_hash(path_to_hash[f["path"]]) for f in hashable]

    uf = UnionFind(n)
    for a in range(n):
        for b in range(a + 1, n):
            if hash_objs[a] - hash_objs[b] <= config.DUPLICATE_HASH_THRESHOLD:
                uf.union(a, b)

    groups: dict[int, list] = defaultdict(list)
    for i in range(n):
        groups[uf.find(i)].append(i)

    hash_dupe_paths: set[str] = set()
    hash_groups = []

    for _, members in groups.items():
        if len(members) == 1:
            continue
        member_files = [hashable[i] for i in members]
        member_files_sorted = sorted(member_files, key=lambda f: (-f["pixels"], -f["file_size"]))
        keeper_path = member_files_sorted[0]["path"]
        entries = []
        for f in member_files_sorted:
            is_keeper = f["path"] == keeper_path
            if not is_keeper:
                hash_dupe_paths.add(f["path"])
            entries.append({
                "path":      f["path"],
                "folder":    f["folder_rel"],
                "res":       f"{f['width']}x{f['height']}",
                "size_mb":   round(f["file_size"] / 1_048_576, 1),
                "decision":  "keep" if is_keeper else "skip",
            })
        hash_groups.append({"entries": entries})

    # Build final keep set: all raw + non-dup non-raw
    final_keep_paths: set[str] = set()
    for f in raw_pass:
        final_keep_paths.add(f["path"])
    for f in non_raw:
        if f["path"] not in hash_dupe_paths:
            final_keep_paths.add(f["path"])

    cache_stats = {"reused": n_reused, "computed": n_computed}
    logger.info(
        "SR step 3 -- %d hash-dup groups  skipped=%d  "
        "(reused=%d  computed=%d)",
        len(hash_groups), len(hash_dupe_paths), n_reused, n_computed,
    )
    return final_keep_paths, hash_dupe_paths, hash_groups, cache_stats


def sr_step4_write(
    scan_folder: str,
    all_files: list,
    folder_counts: dict,
    dup_groups: list,
    hash_groups: list,
    skip_paths: set,
    keep_paths: set,
    logger,
    cache_stats: dict | None = None,
) -> None:
    """Write scan_report.txt and scan_report.json to the project root.

    scan_report.txt — human-readable summary with folder breakdown and
    duplicate details.

    scan_report.json — machine-readable; consumed by phase1_filter.py
    step1_ingest to pre-filter known duplicates before processing.

    Args:
        scan_folder:  Root folder that was scanned.
        all_files:    All file dicts from sr_step1_collect.
        folder_counts: Relative folder → file count.
        dup_groups:   Filename-duplicate groups from sr_step2.
        hash_groups:  Hash-duplicate groups from sr_step3.
        skip_paths:   Set of paths to skip (all duplicates).
        keep_paths:   Set of paths to keep.
        logger:       Logger instance.
        cache_stats:  Optional dict {"reused": N, "computed": M} from sr_step3.
    """
    total          = len(all_files)
    fn_dup_groups  = len(dup_groups)
    fn_kept        = fn_dup_groups                           # one keeper per group
    fn_skipped     = sum(len(g["entries"]) - 1 for g in dup_groups)
    hash_dup_groups = len(hash_groups)
    hash_skipped   = sum(1 for g in hash_groups for e in g["entries"] if e["decision"] == "skip")
    unique_after_a = total - fn_skipped
    est_unique     = total - len(skip_paths)
    phase1_rate    = 0.87   # seconds per photo from CANDID test run
    est_hours      = est_unique * phase1_rate / 3600

    w = 48

    # ── scan_report.txt ───────────────────────────────────────────────────────
    lines = [
        "-" * w,
        f" SCAN REPORT -- {scan_folder}",
        "-" * w,
        f" Total files found          : {total:>8,}",
        f" Unique filenames           : {total - fn_skipped:>8,}",
        f" Filename duplicates found  : {fn_skipped:>8,}",
        f"   -> kept (highest res)    : {fn_kept:>8,}",
        f"   -> marked duplicate      : {fn_skipped:>8,}",
        f" Hash duplicates found      : {hash_skipped:>8,}",
        *(
            [f"   -> {cache_stats['reused']} reused / {cache_stats['computed']} computed fresh"]
            if cache_stats and (cache_stats["reused"] or cache_stats["computed"])
            else []
        ),
        "-" * w,
        " FOLDER BREAKDOWN",
        "-" * w,
    ]
    for folder, count in sorted(folder_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  /{folder:<38}: {count:>6,}")

    lines += [
        "-" * w,
        " DUPLICATE DETAIL (filename dupes)",
        "-" * w,
    ]
    for g in dup_groups:
        lines.append(f" {g['filename']} -> found in {len(g['entries'])} folders:")
        for e in g["entries"]:
            tag  = "KEEP" if e["decision"] == "keep" else "SKIP"
            lines.append(f"   {tag}   {e['folder']:<32} {e['res']}  ({e['size_mb']}MB)")

    if hash_groups:
        lines += ["-" * w, " DUPLICATE DETAIL (hash dupes, top 20)", "-" * w]
        for g in hash_groups[:20]:
            lines.append(f" Near-duplicate cluster ({len(g['entries'])} photos):")
            for e in g["entries"]:
                tag = "KEEP" if e["decision"] == "keep" else "SKIP"
                lines.append(f"   {tag}   {e['folder']:<32} {e['res']}  ({e['size_mb']}MB)")

    lines += [
        "-" * w,
        f" ESTIMATED UNIQUE PHOTOS AFTER DEDUP : {est_unique:>8,}",
        f" ESTIMATED PHASE 1 RUNTIME           : ~{est_hours:.1f} hours",
        "-" * w,
    ]

    txt_body = "\n".join(lines)
    with open(config.SCAN_REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(txt_body + "\n")
    print("\n" + txt_body)

    # ── scan_report.json ─────────────────────────────────────────────────────
    report = {
        "version":               1,
        "scan_folder":           scan_folder,
        "generated":             datetime.now().isoformat(timespec="seconds"),
        "total_files":           total,
        "skip":                  sorted(skip_paths),
        "keep":                  sorted(keep_paths),
        "folder_counts":         folder_counts,
        "filename_dup_groups":   dup_groups,
        "hash_dup_groups":       hash_groups,
    }
    with open(config.SCAN_REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(
        "SR step 4 -- wrote %s and %s",
        config.SCAN_REPORT_TXT, config.SCAN_REPORT_JSON,
    )
    print(f"\nWrote: {config.SCAN_REPORT_TXT}")
    print(f"Wrote: {config.SCAN_REPORT_JSON}")


def run_scan_report(scan_folder: str, logger) -> None:
    """Orchestrate the full scan-report pipeline and write output files.

    Runs sr_step1 (collect) -> sr_step2 (filename dedup) ->
    sr_step3 (hash dedup) -> sr_step4 (write reports).
    Does NOT run the normal phase-1 filter pipeline.

    Args:
        scan_folder: Root directory to scan (INPUT_FOLDER or --input override).
        logger:      Logger instance.
    """
    start = time.time()
    logger.info("=== Scan report started  folder=%s ===", scan_folder)
    print(f"\nScan report mode -- scanning: {scan_folder}")

    all_files, folder_counts = sr_step1_collect(scan_folder, logger)
    if not all_files:
        print("No supported image files found.")
        return

    keepers_a, dupes_a, dup_groups = sr_step2_filename_dedup(all_files, logger)
    final_keep_paths, hash_dupe_paths, hash_groups, cache_stats = sr_step3_hash_dedup(
        keepers_a, logger
    )

    # Build unified skip set (filename dupes + hash dupes)
    skip_paths: set[str] = {f["path"] for f in dupes_a} | hash_dupe_paths

    sr_step4_write(
        scan_folder, all_files, folder_counts,
        dup_groups, hash_groups,
        skip_paths, final_keep_paths,
        logger,
        cache_stats=cache_stats,
    )

    elapsed = time.time() - start
    logger.info("=== Scan report complete in %.1fs ===", elapsed)
    print(f"\nScan report complete in {elapsed:.1f}s")


# ── Pipeline steps ────────────────────────────────────────────────────────────

def step1_ingest(input_folder: str, logger, limit: int = None) -> list:
    """Recursively scan input_folder for all supported image files.

    Counts and logs RAW files separately from JPEG/PNG so the breakdown
    is visible at the start of the run.

    Args:
        input_folder: Root directory to scan (may be on an external drive).
        logger: Logger instance for this phase.
        limit: If set, stop after collecting this many files (test mode).

    Returns:
        List of absolute path strings for every matched file.
    """
    logger.info("Step 1 -- Scanning: %s", input_folder)
    all_paths = []
    raw_count = 0
    jpeg_count = 0

    for dirpath, _, filenames in os.walk(input_folder):
        for fname in filenames:
            ext = os.path.splitext(fname)[1]
            if ext in SUPPORTED_SET:
                full_path = os.path.abspath(os.path.join(dirpath, fname))
                all_paths.append(full_path)
                if ext in RAW_EXTENSIONS:
                    raw_count += 1
                else:
                    jpeg_count += 1
            if limit and len(all_paths) >= limit:
                break
        if limit and len(all_paths) >= limit:
            break

    if limit:
        logger.info("TEST MODE -- capped at %d files", limit)

    # Pre-filter known duplicates from scan_report.json if it exists
    if os.path.exists(config.SCAN_REPORT_JSON):
        try:
            with open(config.SCAN_REPORT_JSON, encoding="utf-8") as f:
                sr = json.load(f)
            skip_set = set(sr.get("skip", []))
            before = len(all_paths)
            all_paths = [p for p in all_paths if p not in skip_set]
            skipped = before - len(all_paths)
            logger.info(
                "scan_report.json loaded -- pre-filtered %d known duplicates "
                "(%d filename-dupes + hash-dupes)",
                skipped, skipped,
            )
            print(f"  scan_report.json: pre-filtered {skipped} known duplicates")
        except Exception as exc:
            logger.warning("Could not load scan_report.json: %s", exc)

    logger.info(
        "Step 1 complete -- %d total (RAW: %d, JPEG/PNG: %d)",
        len(all_paths), raw_count, jpeg_count,
    )
    print(f"\nStep 1 -- Found {len(all_paths)} files  "
          f"(RAW: {raw_count}  JPEG/PNG: {jpeg_count})"
          + ("  [TEST MODE]" if limit else ""))
    return all_paths


def step_process_all(paths: list, logger) -> list:
    """Load each image once and compute all quality metrics in a single pass.

    Replaces the old step2_blur + step3_exposure + hashing portion of
    step4_duplicates.  Each photo is loaded exactly once; early-exit logic
    in process_single_photo skips unnecessary computation for rejected photos.

    Photos with status "error" are logged individually after the loop so
    the tqdm bar is not polluted during processing.

    Args:
        paths: List of absolute image paths from step1_ingest.
        logger: Logger instance for this phase.

    Returns:
        List of complete record dicts, one per path, with phash already
        populated for all surviving photos.
    """
    logger.info("Processing %d photos (single-pass load)", len(paths))
    records = []

    for path in tqdm(paths, desc="Processing photos", unit="photo"):
        records.append(process_single_photo(path))

    # Log errors after the bar completes so output stays clean
    for r in records:
        if r["status"] == "error":
            logger.error("Failed to process: %s", r["path"])

    hard_blur   = sum(1 for r in records if r["status"] == "rejected_blur")
    soft_blur   = sum(1 for r in records if r["status"] == "soft_blur_surviving")
    dark        = sum(1 for r in records if r["status"] == "rejected_dark")
    bright      = sum(1 for r in records if r["status"] == "rejected_bright")
    bae         = sum(1 for r in records if r["status"] == "rejected_blur_and_exposure")
    surviving   = sum(1 for r in records if r["status"] == "surviving")
    errors      = sum(1 for r in records if r["status"] == "error")

    logger.info(
        "Processing complete -- surviving: %d sharp + %d soft_blur | "
        "rejected: %d blur, %d dark, %d bright, %d blur+exp | errors: %d",
        surviving, soft_blur, hard_blur, dark, bright, bae, errors,
    )
    return records


def step_cluster(records: list, logger) -> list:
    """Cluster near-duplicate photos and mark the best keeper per cluster.

    Pure in-memory operation — no image I/O.  All phash values are already
    populated by step_process_all.  Wraps cluster_duplicates() with logging.

    Args:
        records: Output from step_process_all.
        logger: Logger instance for this phase.

    Returns:
        Same list with cluster_id and is_cluster_best updated on every
        surviving record.
    """
    hashable = sum(
        1 for r in records
        if r["status"] in ("surviving", "soft_blur_surviving") and r["phash"]
    )
    logger.info("Clustering %d hashable survivors for duplicate detection", hashable)

    records = cluster_duplicates(records)

    dup_count = sum(1 for r in records if r["status"] == "rejected_duplicate")
    logger.info("Clustering complete -- rejected as duplicates: %d", dup_count)
    return records


def step5_output(records: list, logger, output_path: str = "surviving_photos.json") -> None:
    """Persist results to output_path and print a summary table.

    Args:
        records: Final list of photo dicts after all pipeline steps.
        logger: Logger instance for this phase.
        output_path: Destination JSON file (default: surviving_photos.json).
    """
    save_json(output_path, records)
    logger.info("Saved %s (%d records)", output_path, len(records))

    total           = len(records)
    hard_blur       = sum(1 for r in records if r["status"] == "rejected_blur")
    soft_blur_kept  = sum(1 for r in records if r["status"] == "soft_blur_surviving")
    rejected_dark   = sum(1 for r in records if r["status"] == "rejected_dark")
    rejected_bright = sum(1 for r in records if r["status"] == "rejected_bright")
    rejected_bae    = sum(1 for r in records if r["status"] == "rejected_blur_and_exposure")
    rejected_dup    = sum(1 for r in records if r["status"] == "rejected_duplicate")
    sharp_surviving = sum(1 for r in records if r["status"] == "surviving")
    errors          = sum(1 for r in records if r["status"] == "error")
    total_surviving = sharp_surviving + soft_blur_kept

    w = 46
    lines = [
        "-" * w,
        " Phase 1 -- Filter Summary",
        "-" * w,
        f" Total scanned              : {total:>6}",
        "-" * w,
        f" Hard rejected (blur)       : {hard_blur:>6}",
        f" Soft blur kept             : {soft_blur_kept:>6}",
        f" Rejected dark              : {rejected_dark:>6}",
        f" Rejected bright            : {rejected_bright:>6}",
        f" Rejected blur+exposure     : {rejected_bae:>6}",
        f" Rejected duplicate         : {rejected_dup:>6}",
        "-" * w,
        f" Total surviving            : {total_surviving:>6}",
        f"   -> sharp                 : {sharp_surviving:>6}",
        f"   -> soft blur             : {soft_blur_kept:>6}",
        f" Errors (could not process) : {errors:>6}",
        "-" * w,
    ]
    summary = "\n".join(lines)
    print("\n" + summary)
    logger.info("\n%s", summary)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    """Orchestrate Phase 1: ingest -> process (single-pass) -> cluster -> output."""
    parser = argparse.ArgumentParser(
        description="Phase 1 -- Filter photos by blur, exposure, and duplicates."
    )
    parser.add_argument(
        "--input",
        default=config.INPUT_FOLDER,
        help="Override INPUT_FOLDER from config.py (default: %(default)s)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Dry-run mode: process only the first 20 photos, "
             "save to surviving_photos_test.json",
    )
    parser.add_argument(
        "--input-override",
        default=None,
        dest="input_override",
        help="Scan a specific subfolder instead of INPUT_FOLDER. "
             f"Output saves to surviving_photos{config.SAMPLE_OUTPUT_SUFFIX}.json",
    )
    parser.add_argument(
        "--scan-report",
        action="store_true",
        dest="scan_report",
        help="Scan INPUT_FOLDER (or --input override), write scan_report.txt "
             "and scan_report.json, then exit.  Does not run the filter pipeline.",
    )
    args = parser.parse_args()

    # --scan-report: run scan only, skip filter pipeline
    if args.scan_report:
        scan_folder = args.input_override or args.input
        logger = setup_logging("phase1_filter")
        run_scan_report(scan_folder, logger)
        return

    # Determine scan folder and output path — override wins over test wins over default
    if args.input_override:
        scan_folder = args.input_override
        output_path = f"surviving_photos{config.SAMPLE_OUTPUT_SUFFIX}.json"
        test_limit  = None
    elif args.test:
        scan_folder = args.input
        output_path = "surviving_photos_test.json"
        test_limit  = 20
    else:
        scan_folder = args.input
        output_path = "surviving_photos.json"
        test_limit  = None

    logger = setup_logging("phase1_filter")
    start  = time.time()

    if args.input_override:
        print(f"\nOVERRIDE MODE -- scanning: {scan_folder}")
        logger.info("OVERRIDE MODE -- scanning: %s", scan_folder)
        logger.info("=== Phase 1 started  mode=override  input=%s ===", scan_folder)
    else:
        logger.info("=== Phase 1 started  input=%s  test=%s ===", scan_folder, args.test)

    paths = step1_ingest(scan_folder, logger, limit=test_limit)
    if not paths:
        logger.warning("No supported images found in %s -- exiting.", scan_folder)
        return

    records = step_process_all(paths, logger)
    records = step_cluster(records, logger)
    step5_output(records, logger, output_path=output_path)

    elapsed = time.time() - start
    logger.info("=== Phase 1 complete in %.1fs ===", elapsed)
    print(f"\nPhase 1 complete in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
