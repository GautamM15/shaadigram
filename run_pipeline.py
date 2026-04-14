"""
run_pipeline.py — End-to-end pipeline runner for Wedding Photo Curator.

Executes all phases in order, passing standard arguments.  Each phase is
run as a subprocess so logging, imports, and GPU memory are isolated.
All phases write their own logs to logs/.

Pipeline order:
    1.  phase1_filter.py   — blur, exposure, duplicate clustering
    2.  phase2_enrich.py   — EXIF moments, face detection, shot classification
    3.  phase1b_burst.py   — burst-group limiter (needs enriched output)
    4.  phase3_score.py    — NIMA + LLaVA aesthetic scoring
    5.  phase4_select.py   — moment-balanced top-800 selection
    6.  phase5_review.py   — keyboard-driven review UI (interactive)
    7.  phase6_export.py   — copy approved photos to output folders

Usage:
    python run_pipeline.py                    # run all phases
    python run_pipeline.py --from phase3      # resume from phase 3
    python run_pipeline.py --only phase1      # run a single phase
    python run_pipeline.py --skip-review      # skip interactive phase 5
    python run_pipeline.py --dry-run          # print commands, do not execute
"""

import argparse
import subprocess
import sys
import time


# Ordered phase registry
_PHASES = [
    {
        "name":    "phase1",
        "script":  "phase1_filter.py",
        "label":   "Phase 1  — Filter (blur / exposure / duplicates)",
        "interactive": False,
    },
    {
        "name":    "phase2",
        "script":  "phase2_enrich.py",
        "label":   "Phase 2  — Enrich (EXIF moments / faces / shot types)",
        "interactive": False,
    },
    {
        "name":    "phase1b",
        "script":  "phase1b_burst.py",
        "label":   "Phase 1b — Burst limiter",
        "interactive": False,
    },
    {
        "name":    "phase3",
        "script":  "phase3_score.py",
        "label":   "Phase 3  — Score (BRISQUE / LLaVA)",
        "interactive": False,
    },
    {
        "name":    "phase4",
        "script":  "phase4_select.py",
        "label":   "Phase 4  — Select top-800",
        "interactive": False,
    },
    {
        "name":    "phase5",
        "script":  "phase5_review.py",
        "label":   "Phase 5  — Review UI (interactive)",
        "interactive": True,
    },
    {
        "name":    "phase6",
        "script":  "phase6_export.py",
        "label":   "Phase 6  — Export finals",
        "interactive": False,
    },
]

_PHASE_NAMES = [p["name"] for p in _PHASES]


def _print_banner(text: str) -> None:
    """Print a visible section banner."""
    w = 60
    print("\n" + "=" * w)
    print(f"  {text}")
    print("=" * w)


def _run_phase(phase: dict, dry_run: bool) -> bool:
    """Run a single phase script as a subprocess.

    Args:
        phase:   Phase dict from _PHASES.
        dry_run: If True, print the command but do not execute.

    Returns:
        True if the phase succeeded (or dry_run), False on non-zero exit.
    """
    cmd = [sys.executable, phase["script"]]
    _print_banner(phase["label"])

    if dry_run:
        print(f"  [DRY RUN] would run: {' '.join(cmd)}")
        return True

    start = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\nERROR: {phase['script']} exited with code {result.returncode}")
        return False

    print(f"\n  {phase['label']} — done in {elapsed:.1f}s")
    return True


def main() -> None:
    """Parse arguments and run the requested pipeline phases."""
    parser = argparse.ArgumentParser(
        description="Run the wedding-photo-curator pipeline end-to-end."
    )
    parser.add_argument(
        "--from",
        dest="from_phase",
        choices=_PHASE_NAMES,
        default=None,
        metavar="PHASE",
        help=f"Resume from this phase (choices: {', '.join(_PHASE_NAMES)})",
    )
    parser.add_argument(
        "--only",
        dest="only_phase",
        choices=_PHASE_NAMES,
        default=None,
        metavar="PHASE",
        help="Run only this single phase",
    )
    parser.add_argument(
        "--skip-review",
        action="store_true",
        dest="skip_review",
        help="Skip the interactive Phase 5 review UI",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Print the commands that would run without executing them",
    )
    args = parser.parse_args()

    # Determine which phases to run
    if args.only_phase:
        phases_to_run = [p for p in _PHASES if p["name"] == args.only_phase]
    elif args.from_phase:
        start_idx = _PHASE_NAMES.index(args.from_phase)
        phases_to_run = _PHASES[start_idx:]
    else:
        phases_to_run = list(_PHASES)

    if args.skip_review:
        phases_to_run = [p for p in phases_to_run if not p["interactive"]]

    if not phases_to_run:
        print("No phases selected.")
        return

    _print_banner("Wedding Photo Curator — Pipeline")
    print(f"  Phases to run: {', '.join(p['name'] for p in phases_to_run)}")
    if args.dry_run:
        print("  [DRY RUN mode — no commands will be executed]")

    pipeline_start = time.time()
    for phase in phases_to_run:
        ok = _run_phase(phase, dry_run=args.dry_run)
        if not ok:
            print(f"\nPipeline aborted at {phase['name']}.")
            sys.exit(1)

    elapsed = time.time() - pipeline_start
    _print_banner(f"Pipeline complete in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
