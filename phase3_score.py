"""
phase3_score.py — Phase 3: NIMA aesthetic scoring, LLaVA emotional scoring,
weighted combination with penalty/bonus modifiers.

Reads enriched_photos_candid.json (phase 2 output), scores every surviving
record, and writes scored_photos_candid.json.

Usage:
    python phase3_score.py
    python phase3_score.py --input enriched_photos_candid.json --output scored_photos_candid.json
    python phase3_score.py --test
    python phase3_score.py --test --skip-nima --skip-llava
"""

import argparse
import base64
import concurrent.futures
import json
import os
import statistics
import time
import urllib.request

import numpy as np
import requests
from PIL import Image
from tqdm import tqdm

import config
from utils import load_json, save_json, setup_logging

# ── Optional heavy imports ────────────────────────────────────────────────────

try:
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNet
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.models import Model
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False

try:
    import cv2
    from imquality import brisque as _brisque_mod
    _BRISQUE_AVAILABLE = True
except ImportError:
    _BRISQUE_AVAILABLE = False

# ── Constants ─────────────────────────────────────────────────────────────────

# Statuses that receive full scoring (matches phase 1 / 2 convention)
_SURVIVING = {"surviving", "soft_blur_surviving"}

# Active weight sum — LIGHTING_WEIGHT not used in phase 3
# NIMA 0.25 + EMOTION 0.40 + MEMORABILITY 0.15 = 0.80
# base_score divided by this to produce true 0-1 scale
_WEIGHT_SUM = config.NIMA_WEIGHT + config.EMOTION_WEIGHT + config.MEMORABILITY_WEIGHT

# NIMA MobileNet weights
_NIMA_WEIGHTS_URL = (
    "https://github.com/titu1994/neural-image-assessment"
    "/releases/download/v0.5/mobilenet_weights.h5"
)
_NIMA_WEIGHTS_PATH = os.path.join(
    os.path.expanduser("~"), ".deepface", "weights", "mobilenet_nima_weights.h5"
)

# LLaVA prompts
_LLAVA_PROMPT = (
    "Rate this wedding photo on a scale of 1-10 for each:\n"
    "(1) emotional moment or authentic expression,\n"
    "(2) memorability - would this be cherished in 20 years?\n"
    'Reply ONLY in JSON with no other text:\n{"emotion": X, "memorability": X}'
)
_LLAVA_RETRY_PROMPT = (
    "Respond with valid JSON only, no other text.\n"
    '{"emotion": X, "memorability": X}\n'
    "where X is a number 1-10 rating this wedding photo."
)

# Null scoring block applied to rejected / non-surviving records
_NULL_SCORE = {
    "nima_score":         None,
    "nima_method":        None,
    "emotion_score":      None,
    "memorability_score": None,
    "llava_fallback":     False,
    "base_score":         None,
    "final_score":        None,
    "score_components":   {},
}


# ── Step 1 — NIMA model load ──────────────────────────────────────────────────

def load_nima_model(logger) -> tuple:
    """Load the NIMA MobileNet aesthetic scoring model.

    Downloads weights from the titu1994 GitHub release on first run,
    cached at ~/.deepface/weights/mobilenet_nima_weights.h5.
    Falls back to BRISQUE if TensorFlow is unavailable or loading fails.

    Args:
        logger: Logger instance.

    Returns:
        Tuple (model_or_None, method_str) where method_str is "nima" or
        "brisque".
    """
    if not _TF_AVAILABLE:
        logger.warning("TensorFlow not available -- using BRISQUE")
        return None, "brisque"

    try:
        if not os.path.exists(_NIMA_WEIGHTS_PATH):
            logger.info("Downloading NIMA MobileNet weights...")
            os.makedirs(os.path.dirname(_NIMA_WEIGHTS_PATH), exist_ok=True)
            urllib.request.urlretrieve(_NIMA_WEIGHTS_URL, _NIMA_WEIGHTS_PATH)
            logger.info("NIMA weights saved to %s", _NIMA_WEIGHTS_PATH)

        base = MobileNet(
            include_top=False,
            pooling="avg",
            input_shape=(224, 224, 3),
            weights=None,
        )
        predictions = Dense(10, activation="softmax")(base.output)
        model = Model(inputs=base.input, outputs=predictions)
        model.load_weights(_NIMA_WEIGHTS_PATH)
        logger.info("NIMA model loaded (MobileNet + 10-class softmax head)")
        return model, "nima"

    except Exception as exc:
        logger.warning("NIMA load failed (%s) -- falling back to BRISQUE", exc)
        return None, "brisque"


def score_nima(pil_img: Image.Image, model) -> float:
    """Score one PIL image with the NIMA model.

    Resizes to 224x224, normalises pixels to [0, 1], runs inference, and
    returns the expected value of the 1-10 rating distribution normalised
    to [0, 1].

    Args:
        pil_img: PIL Image (RGB).
        model:   Loaded Keras NIMA model.

    Returns:
        Float in [0, 1].
    """
    img = pil_img.resize((224, 224), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)           # (1, 224, 224, 3)
    probs = model.predict(arr, verbose=0)[0]    # (10,)
    mean_rating = float(np.sum(probs * np.arange(1, 11)))
    return mean_rating / 10.0


def score_brisque(image_path: str) -> float:
    """Score one image with BRISQUE (inverted, normalised to [0, 1]).

    BRISQUE: 0 = perfect quality, ~100+ = poor.  Inverted so higher = better,
    consistent with the NIMA convention.

    Args:
        image_path: Path to image file.

    Returns:
        Float in [0, 1].  Returns 0.5 if BRISQUE is unavailable or fails.
    """
    if not _BRISQUE_AVAILABLE:
        return 0.5
    try:
        img = cv2.imread(image_path)
        if img is None:
            return 0.5
        raw = _brisque_mod.score(img)
        return float(max(0.0, min(1.0, 1.0 - raw / 100.0)))
    except Exception:
        return 0.5


def step1_nima(records: list, logger, model, nima_method: str,
               skip_nima: bool) -> list:
    """Run NIMA or BRISQUE aesthetic scoring on all surviving photos.

    Non-surviving records receive nima_score=None, nima_method=None.

    Args:
        records:     All records (surviving + rejected).
        logger:      Logger instance.
        model:       Loaded NIMA model, or None when using BRISQUE.
        nima_method: "nima" or "brisque" from load_nima_model().
        skip_nima:   If True, force BRISQUE regardless of model state.

    Returns:
        Records with nima_score and nima_method populated for surviving.
    """
    effective_method = "brisque" if skip_nima else nima_method
    surviving = [r for r in records if r["status"] in _SURVIVING]

    # Resume: skip records that already have a NIMA/BRISQUE score
    already_done = [r for r in surviving if r.get("nima_score") is not None]
    todo         = [r for r in surviving if r.get("nima_score") is None]
    if already_done:
        logger.info(
            "Step 1 RESUME: %d/%d already scored, scoring remaining %d",
            len(already_done), len(surviving), len(todo),
        )
        print(
            f"\nStep 1 RESUME: {len(already_done)}/{len(surviving)} already scored, "
            f"scoring remaining {len(todo)}"
        )
    else:
        logger.info(
            "Step 1 -- NIMA scoring  method=%s  photos=%d",
            effective_method, len(surviving),
        )
        print(f"\nStep 1: NIMA scoring  method={effective_method}  photos={len(surviving)}")

    for record in tqdm(todo, desc="NIMA scoring", unit="photo"):
        try:
            if skip_nima or model is None:
                nima_score       = score_brisque(record["path"])
                nima_method_used = "brisque"
            else:
                pil_img          = Image.open(record["path"]).convert("RGB")
                nima_score       = score_nima(pil_img, model)
                nima_method_used = "nima"
        except Exception as exc:
            logger.warning(
                "NIMA failed for %s: %s -- BRISQUE fallback",
                os.path.basename(record["path"]), exc,
            )
            nima_score       = score_brisque(record["path"])
            nima_method_used = "brisque"

        record["nima_score"]  = round(float(nima_score), 6)
        record["nima_method"] = nima_method_used

    for record in records:
        if record["status"] not in _SURVIVING:
            record["nima_score"]  = None
            record["nima_method"] = None

    logger.info("Step 1 complete")
    return records


# ── Step 2 — LLaVA emotional scoring ─────────────────────────────────────────

def _call_llava(image_path: str, logger) -> tuple:
    """Send one photo to LLaVA and return normalised emotion/memorability scores.

    Attempts the full prompt first; retries once with a simplified prompt if
    the response cannot be parsed as JSON.  Falls back to neutral (0.5, 0.5)
    on second failure.

    Args:
        image_path: Path to image file.
        logger:     Logger instance.

    Returns:
        Tuple (emotion_score, memorability_score, llava_fallback) where
        scores are floats in [0, 1].
    """
    filename = os.path.basename(image_path)

    try:
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
    except Exception as exc:
        logger.warning("LLaVA: could not read %s: %s", filename, exc)
        return 0.5, 0.5, True

    def _post(prompt: str) -> dict:
        """POST to Ollama and parse JSON, with a hard wall-clock timeout.

        Uses a ThreadPoolExecutor so future.result(timeout=90) fires on actual
        elapsed time, not TCP-level activity.  Ollama can hold a connection
        open for minutes while computing without sending any data; requests
        timeout= alone does not fire in that case.
        """
        def _do_post():
            resp = requests.post(
                f"{config.OLLAMA_URL}/api/generate",
                json={
                    "model":  config.OLLAMA_MODEL,
                    "prompt": prompt,
                    "images": [img_b64],
                    "stream": False,
                },
                timeout=120,   # TCP-level safety net (backup to thread timeout)
            )
            resp.raise_for_status()
            raw_text = resp.json()["response"].strip()
            # Strip markdown code fences if model wraps response in ```json...```
            if raw_text.startswith("```"):
                parts    = raw_text.split("```")
                raw_text = (
                    parts[1].lstrip("json").strip() if len(parts) >= 3
                    else parts[-1]
                )
            return json.loads(raw_text)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_do_post)
            try:
                return future.result(timeout=90)
            except concurrent.futures.TimeoutError:
                raise TimeoutError(
                    f"Ollama wall-clock timeout (90s) for {filename}"
                )

    # First attempt — full prompt
    try:
        data         = _post(_LLAVA_PROMPT)
        emotion      = float(data["emotion"])      / 10.0
        memorability = float(data["memorability"]) / 10.0
        return (
            max(0.0, min(1.0, emotion)),
            max(0.0, min(1.0, memorability)),
            False,
        )
    except Exception as exc:
        logger.warning(
            "LLaVA first attempt failed for %s: %s -- retrying", filename, exc
        )

    # Retry — simplified prompt
    try:
        data         = _post(_LLAVA_RETRY_PROMPT)
        emotion      = float(data["emotion"])      / 10.0
        memorability = float(data["memorability"]) / 10.0
        return (
            max(0.0, min(1.0, emotion)),
            max(0.0, min(1.0, memorability)),
            False,
        )
    except Exception as exc:
        logger.warning(
            "LLaVA retry failed for %s: %s -- using fallback scores",
            filename, exc,
        )
        return 0.5, 0.5, True


def step2_llava(records: list, logger, skip_llava: bool,
                output_path: str) -> list:
    """Run LLaVA emotional scoring on all surviving photos.

    Skips records that already have emotion_score set (resume support).
    Saves a checkpoint to output_path every LLAVA_BATCH_SIZE photos so
    progress is not lost if the run is interrupted.

    Args:
        records:     All records (surviving + rejected).
        logger:      Logger instance.
        skip_llava:  If True, assign neutral 0.5 scores without Ollama calls.
        output_path: Destination JSON — written incrementally as checkpoint.

    Returns:
        Records with emotion_score, memorability_score, llava_fallback
        populated for surviving records.
    """
    surviving = [r for r in records if r["status"] in _SURVIVING]

    # Resume: skip records that already have LLaVA scores
    already_done = [r for r in surviving if r.get("emotion_score") is not None]
    todo         = [r for r in surviving if r.get("emotion_score") is None]

    logger.info(
        "Step 2 -- LLaVA scoring  photos=%d  already_done=%d  todo=%d  skip=%s",
        len(surviving), len(already_done), len(todo), skip_llava,
    )

    if already_done:
        logger.info(
            "Step 2 RESUME: %d/%d already scored, scoring remaining %d",
            len(already_done), len(surviving), len(todo),
        )
        print(
            f"\nStep 2 RESUME: {len(already_done)}/{len(surviving)} already scored, "
            f"scoring remaining {len(todo)}"
        )

    if skip_llava:
        msg = "LLaVA skipped -- neutral scores (0.5) assigned to unscored photos"
        print(f"\nStep 2: {msg}")
        logger.info(msg)
        for record in todo:
            record["emotion_score"]      = 0.5
            record["memorability_score"] = 0.5
            record["llava_fallback"]     = False
    else:
        print(
            f"\nStep 2: LLaVA scoring  todo={len(todo)}"
            f"  batch_size={config.LLAVA_BATCH_SIZE}"
        )
        for idx, record in enumerate(
            tqdm(todo, desc="LLaVA scoring", unit="photo")
        ):
            emotion, memorability, fallback = _call_llava(record["path"], logger)
            record["emotion_score"]      = round(emotion, 6)
            record["memorability_score"] = round(memorability, 6)
            record["llava_fallback"]     = fallback

            if (idx + 1) % config.LLAVA_BATCH_SIZE == 0:
                time.sleep(config.LLAVA_BATCH_PAUSE)
                # Incremental checkpoint — survive any future interruption
                save_json(output_path, records)
                logger.info(
                    "Checkpoint saved after %d LLaVA photos  file=%s",
                    len(already_done) + idx + 1, output_path,
                )

    for record in records:
        if record["status"] not in _SURVIVING:
            record["emotion_score"]      = None
            record["memorability_score"] = None
            record["llava_fallback"]     = False

    logger.info("Step 2 complete")
    return records


# ── Step 3 — Combined scoring ─────────────────────────────────────────────────

def compute_final_score(record: dict) -> dict:
    """Compute base and final weighted scores for one record.

    base_score is normalised by dividing by _WEIGHT_SUM (0.80) so scores
    are on a true 0-1 scale despite LIGHTING_WEIGHT not being used.

    Modifiers applied in order after normalisation:
        "candid" in shot_type  -> multiply by CANDID_BONUS_MULTIPLIER
        has_closed_eyes        -> multiply by CLOSED_EYES_PENALTY
        blur_tier == soft_blur -> multiply by SOFT_BLUR_PENALTY

    Args:
        record: Record dict with nima_score, emotion_score,
                memorability_score, shot_type, has_closed_eyes, blur_tier.

    Returns:
        Dict with base_score, final_score, score_components.
    """
    nima         = record.get("nima_score")         or 0.0
    emotion      = record.get("emotion_score")      or 0.0
    memorability = record.get("memorability_score") or 0.0

    raw_base   = (
        nima         * config.NIMA_WEIGHT +
        emotion      * config.EMOTION_WEIGHT +
        memorability * config.MEMORABILITY_WEIGHT
    )
    base_score  = raw_base / _WEIGHT_SUM   # normalise to true 0-1 scale
    final_score = base_score

    shot_type      = record.get("shot_type") or []
    candid_bonus   = "candid" in shot_type
    closed_eye_pen = bool(record.get("has_closed_eyes"))
    soft_blur_pen  = record.get("blur_tier") == "soft_blur"

    if candid_bonus:
        final_score *= config.CANDID_BONUS_MULTIPLIER
    if closed_eye_pen:
        final_score *= config.CLOSED_EYES_PENALTY
    if soft_blur_pen:
        final_score *= config.SOFT_BLUR_PENALTY

    return {
        "base_score":  round(base_score,  6),
        "final_score": round(final_score, 6),
        "score_components": {
            "nima":                round(nima         * config.NIMA_WEIGHT         / _WEIGHT_SUM, 6),
            "emotion":             round(emotion      * config.EMOTION_WEIGHT      / _WEIGHT_SUM, 6),
            "memorability":        round(memorability * config.MEMORABILITY_WEIGHT / _WEIGHT_SUM, 6),
            "candid_bonus":        candid_bonus,
            "closed_eyes_penalty": closed_eye_pen,
            "soft_blur_penalty":   soft_blur_pen,
        },
    }


def step3_combine(records: list, logger) -> list:
    """Apply weighted scoring and modifiers to all surviving records.

    Non-surviving records receive _NULL_SCORE values.

    Args:
        records: All records with NIMA and LLaVA scores populated.
        logger:  Logger instance.

    Returns:
        All records with base_score, final_score, score_components added.
    """
    surviving = [r for r in records if r["status"] in _SURVIVING]
    logger.info("Step 3 -- combining scores  photos=%d", len(surviving))
    print(f"\nStep 3: combining scores  photos={len(surviving)}")

    for record in surviving:
        record.update(compute_final_score(record))

    for record in records:
        if record["status"] not in _SURVIVING:
            record.update(_NULL_SCORE)

    logger.info("Step 3 complete")
    return records


# ── Step 4 — Output + summary ─────────────────────────────────────────────────

def step4_output(records: list, logger, output_path: str) -> None:
    """Save scored records to JSON and print a summary table.

    Args:
        records:     Fully scored record list.
        logger:      Logger instance.
        output_path: Destination JSON file path.
    """
    save_json(output_path, records)
    logger.info("Saved %s (%d records)", output_path, len(records))

    surviving = [r for r in records if r["status"] in _SURVIVING]
    total     = len(records)
    n_scored  = len(surviving)

    nima_ct    = sum(1 for r in surviving if r.get("nima_method") == "nima")
    brisque_ct = sum(1 for r in surviving if r.get("nima_method") == "brisque")
    fallback_ct = sum(1 for r in surviving if r.get("llava_fallback"))

    final_scores = [
        r["final_score"] for r in surviving
        if r.get("final_score") is not None
    ]
    s_min    = min(final_scores)             if final_scores else 0.0
    s_max    = max(final_scores)             if final_scores else 0.0
    s_mean   = statistics.mean(final_scores)   if final_scores else 0.0
    s_median = statistics.median(final_scores) if final_scores else 0.0

    candid_ct    = sum(
        1 for r in surviving
        if r.get("score_components", {}).get("candid_bonus")
    )
    closed_ct    = sum(
        1 for r in surviving
        if r.get("score_components", {}).get("closed_eyes_penalty")
    )
    soft_blur_ct = sum(
        1 for r in surviving
        if r.get("score_components", {}).get("soft_blur_penalty")
    )

    ranked = sorted(
        [r for r in surviving if r.get("final_score") is not None],
        key=lambda r: r["final_score"],
        reverse=True,
    )[:10]

    w = 54
    lines = [
        "-" * w,
        " Phase 3 -- Scoring Summary",
        "-" * w,
        f" Total records              : {total:>6}",
        f" Scored (surviving)         : {n_scored:>6}",
        f" Skipped (rejected)         : {total - n_scored:>6}",
        "-" * w,
        f" NIMA (neural aesthetic)    : {nima_ct:>6}",
        f" BRISQUE (fallback)         : {brisque_ct:>6}",
        f" LLaVA fallbacks            : {fallback_ct:>6}",
        "-" * w,
        " Score distribution (final_score, true 0-1 scale):",
        f"   min                      : {s_min:>9.4f}",
        f"   max                      : {s_max:>9.4f}",
        f"   mean                     : {s_mean:>9.4f}",
        f"   median                   : {s_median:>9.4f}",
        "-" * w,
        " Modifiers applied:",
        f"   candid bonus (+10%)      : {candid_ct:>6}",
        f"   closed-eyes penalty (-15%): {closed_ct:>5}",
        f"   soft-blur penalty (-5%)  : {soft_blur_ct:>6}",
        "-" * w,
        " Note: base_score = weighted sum / 0.80 (NIMA 0.25 +",
        "       EMOTION 0.40 + MEMORABILITY 0.15). LIGHTING_WEIGHT",
        "       not used in phase 3. Scores are true 0-1 scale.",
        "-" * w,
        " Top 10 photos by final_score:",
    ]
    for i, r in enumerate(ranked, 1):
        fname = os.path.basename(r["path"])
        lines.append(f"   {i:>2}. {fname:<32} {r['final_score']:.4f}")
    lines.append("-" * w)

    summary = "\n".join(lines)
    print("\n" + summary)
    logger.info("\n%s", summary)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    """Orchestrate Phase 3: NIMA -> LLaVA -> combine scores -> output."""
    parser = argparse.ArgumentParser(
        description="Phase 3 -- Score surviving photos with NIMA + LLaVA."
    )
    parser.add_argument(
        "--input",
        default=config.PHASE3_INPUT,
        help="Phase 2 JSON to read (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default=config.PHASE3_OUTPUT,
        help="Scored JSON to write (default: %(default)s)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Process first 20 surviving records only",
    )
    parser.add_argument(
        "--skip-nima",
        action="store_true",
        dest="skip_nima",
        help="Use BRISQUE instead of NIMA (faster for testing)",
    )
    parser.add_argument(
        "--skip-llava",
        action="store_true",
        dest="skip_llava",
        help="Assign neutral LLaVA scores (0.5) without calling Ollama",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume a previous run: load records from output file (if it exists) "
            "instead of the input file, skipping already-scored photos"
        ),
    )
    args = parser.parse_args()

    # Auto-derive test output path when --output is the default
    output_path = args.output
    if args.test and args.output == config.PHASE3_OUTPUT:
        output_path = config.PHASE3_OUTPUT.replace(".json", "_test.json")

    logger = setup_logging("phase3_score")
    start  = time.time()
    logger.info(
        "=== Phase 3 started  input=%s  output=%s  test=%s  "
        "skip_nima=%s  skip_llava=%s  resume=%s ===",
        args.input, output_path, args.test,
        args.skip_nima, args.skip_llava, args.resume,
    )

    # --resume: load from existing output so already-scored records are preserved
    if args.resume and os.path.exists(output_path):
        records = load_json(output_path)
        logger.info(
            "RESUME MODE: loaded %d records from existing output %s",
            len(records), output_path,
        )
        print(f"RESUME MODE: loaded {len(records)} records from {output_path}")
    else:
        records = load_json(args.input)
        if not records:
            logger.error("Could not load input file: %s -- exiting.", args.input)
            return
        logger.info("Loaded %d records from %s", len(records), args.input)

    if args.test:
        total_surviving = sum(1 for r in records if r["status"] in _SURVIVING)
        records = [r for r in records if r["status"] in _SURVIVING][:20]
        msg = f"TEST MODE -- processing 20 of {total_surviving} surviving records"
        print(msg)
        logger.info(msg)

    # Step 1 — NIMA / BRISQUE
    if args.skip_nima:
        model, nima_method = None, "brisque"
    else:
        model, nima_method = load_nima_model(logger)
    records = step1_nima(records, logger, model, nima_method, args.skip_nima)

    # Save BRISQUE checkpoint immediately — allows resuming before LLaVA
    if not args.test:
        save_json(output_path, records)
        logger.info("BRISQUE checkpoint saved to %s", output_path)

    # Step 2 — LLaVA
    records = step2_llava(records, logger, args.skip_llava, output_path)

    # Step 3 — Combine scores
    records = step3_combine(records, logger)

    # Step 4 — Save + summary
    step4_output(records, logger, output_path)

    elapsed = time.time() - start
    logger.info("=== Phase 3 complete in %.1fs ===", elapsed)
    print(f"\nPhase 3 complete in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
