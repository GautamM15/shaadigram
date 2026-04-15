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

try:
    import torch as _torch
    import torch.nn as _nn
    _TORCH_AVAILABLE = True

    class _LAIONPredictor(_nn.Module):
        """Single linear aesthetic predictor (LAION aesthetic-predictor, vit_b_32 variant).

        Expects unit-normalised CLIP ViT-B/32 embeddings (512-dim) as input.
        Outputs a single score (1-10 scale before external normalisation).

        Weight file: sa_0_4_vit_b_32_linear.pth from
        https://github.com/LAION-AI/aesthetic-predictor
        Auto-downloaded to aesthetic_model.pth on first run.
        """
        def __init__(self):
            super().__init__()
            self.linear = _nn.Linear(512, 1)

        def forward(self, x):
            """Forward pass — single linear layer."""
            return self.linear(x)

except ImportError:
    _TORCH_AVAILABLE = False

try:
    from transformers import CLIPModel as _CLIPFullModel, CLIPImageProcessor as _CLIPImageProcessor
    _CLIP_LAION_AVAILABLE = True
except ImportError:
    _CLIP_LAION_AVAILABLE = False

# ── Constants ─────────────────────────────────────────────────────────────────

# Statuses that receive full scoring (matches phase 1 / 2 convention)
_SURVIVING = {"surviving", "soft_blur_surviving"}

# New formula weights sum to 1.0 — no normalisation needed
# LAION(0.25) + emotion(0.35) + memorability(0.15) + rot(0.10) + prominence(0.15) = 1.00
_WEIGHT_SUM = 1.0   # kept for reference; no longer used as divisor

# Gaussian sigma for rule-of-thirds scoring (fraction of min image dimension)
_ROT_SIGMA_FRAC = 0.12

# Saliency distance threshold for distraction penalty (fraction of image diagonal)
_SALIENCY_DIST_THRESH = 0.20

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
    "laion_score":         None,
    "laion_method":        None,
    "nima_score":          None,   # kept for backward compat
    "nima_method":         None,   # kept for backward compat
    "rot_score":           None,
    "prominence_score":    None,
    "distraction_penalty": 1.0,
    "burst_rank":          None,
    "emotion_score":       None,
    "memorability_score":  None,
    "llava_fallback":      False,
    "base_score":          None,
    "final_score":         None,
    "score_components":    {},
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


# ── LAION aesthetic model ─────────────────────────────────────────────────────

def _load_laion_model(logger) -> tuple:
    """Load LAION aesthetic predictor — two-tier with auto-download.

    Tier 1 (pip): Try ``from aesthetic_predictor import predict_aesthetic``.
        Returns (predict_aesthetic_fn, None, None, None).  score_laion_single
        detects this by checking ``clip_processor is None``.

    Tier 2 (manual): Auto-download sa_0_4_vit_b_32_linear.pth from
        config.LAION_MODEL_URL on first run.  Load CLIP ViT-B/32 (512-dim,
        consistent with clip_embeddings.npz from phase 2) + nn.Linear(512, 1).
        Returns (clip_model, processor, mlp, device).

    Falls through to BRISQUE (returns four Nones) if both tiers fail.

    Args:
        logger: Logger instance.

    Returns:
        4-tuple consumed by score_laion_single / step1_nima.
        GPU memory freed by the caller after scoring completes.
    """
    # ── Tier 1: pip aesthetic-predictor package ───────────────────────────────
    try:
        from aesthetic_predictor import predict_aesthetic  # noqa: PLC0415
        logger.info("LAION: using aesthetic-predictor pip package")
        return predict_aesthetic, None, None, None
    except ImportError:
        pass

    # ── Tier 2: manual weights + CLIP ViT-B/32 ───────────────────────────────
    if not _TORCH_AVAILABLE or not _CLIP_LAION_AVAILABLE:
        logger.warning("LAION: torch/transformers unavailable -- BRISQUE fallback")
        return None, None, None, None

    laion_path = config.LAION_MODEL_PATH

    # Auto-download on first run
    if not os.path.exists(laion_path):
        import urllib.request
        url = config.LAION_MODEL_URL
        logger.info("LAION: downloading weights from %s ...", url)
        print(f"  Downloading LAION weights from GitHub ...")
        try:
            urllib.request.urlretrieve(url, laion_path)
            size_mb = os.path.getsize(laion_path) / 1_048_576
            logger.info("LAION: downloaded %.1f MB → %s", size_mb, laion_path)
            print(f"  Downloaded {size_mb:.1f} MB → {laion_path}")
        except Exception as exc:
            logger.warning("LAION: download failed (%s) -- BRISQUE fallback", exc)
            return None, None, None, None

    try:
        device = "cuda" if _torch.cuda.is_available() else "cpu"
        logger.info("Loading CLIP ViT-B/32 for LAION aesthetic scoring on %s ...", device)
        processor  = _CLIPImageProcessor.from_pretrained(config.CLIP_MODEL)
        clip_model = _CLIPFullModel.from_pretrained(config.CLIP_MODEL).to(device)
        clip_model.eval()

        mlp   = _LAIONPredictor()
        state = _torch.load(laion_path, map_location="cpu", weights_only=True)
        # Weights file is a plain OrderedDict {"linear.weight": ..., "linear.bias": ...}
        # Older checkpoints may use "weight"/"bias" keys without the "linear." prefix
        if "weight" in state and "linear.weight" not in state:
            state = {"linear.weight": state["weight"], "linear.bias": state["bias"]}
        mlp.load_state_dict(state)
        mlp.to(device).eval()

        logger.info("LAION aesthetic model loaded (CLIP ViT-B/32 + Linear(512,1))")
        return clip_model, processor, mlp, device
    except Exception as exc:
        logger.warning("LAION load failed (%s) -- BRISQUE fallback", exc)
        if os.path.exists(laion_path) and os.path.getsize(laion_path) < 1_000_000:
            os.remove(laion_path)   # remove likely-corrupt small file
        return None, None, None, None


def score_laion_single(path: str, clip_model, clip_processor,
                        laion_mlp, device: str) -> float:
    """Score one image with the LAION aesthetic predictor.

    Supports two modes depending on how _load_laion_model() was called:

    * pip mode   (clip_processor is None): clip_model is the predict_aesthetic
                 callable from the aesthetic-predictor package.
    * manual mode: clip_model is CLIPModel ViT-B/32; encodes the image to
                   512-dim and passes through Linear(512, 1).

    Args:
        path:            Image file path.
        clip_model:      CLIPModel (ViT-B/32) or predict_aesthetic callable.
        clip_processor:  CLIPImageProcessor, or None in pip mode.
        laion_mlp:       Loaded _LAIONPredictor, or None in pip mode.
        device:          "cuda" or "cpu", or None in pip mode.

    Returns:
        Float in [0, 1].  Returns 0.5 on any failure.
    """
    try:
        img = Image.open(path).convert("RGB")

        # ── pip mode ──────────────────────────────────────────────────────────
        if clip_processor is None:
            raw = clip_model(img)   # predict_aesthetic returns Tensor shape (1,1), 1-10 scale
            return float(max(0.0, min(1.0, float(raw.item()) / 10.0)))

        # ── manual mode: CLIP ViT-B/32 + Linear(512, 1) ──────────────────────
        inputs = clip_processor(images=[img], return_tensors="pt")
        pv     = inputs["pixel_values"].to(device)
        with _torch.no_grad():
            vis_out = clip_model.vision_model(pixel_values=pv)
            feats   = clip_model.visual_projection(vis_out.pooler_output)  # (1, 512)
            feats   = feats / feats.norm(dim=-1, keepdim=True)
            score   = laion_mlp(feats).squeeze().item()
        return float(max(0.0, min(1.0, score / 10.0)))
    except Exception:
        return 0.5


def step1_nima(records: list, logger, model, nima_method: str,
               skip_nima: bool,
               laion_components: tuple = None) -> list:
    """Run aesthetic scoring (LAION → NIMA → BRISQUE fallback chain) on all surviving photos.

    Priority: LAION aesthetic predictor (if laion_components provided and
    skip_nima is False) → NIMA (if model loaded) → BRISQUE (fallback).

    Populates both laion_score/laion_method (new formula fields) and
    nima_score/nima_method (kept for backward compatibility).

    Args:
        records:          All records (surviving + rejected).
        logger:           Logger instance.
        model:            Loaded NIMA Keras model, or None.
        nima_method:      "nima" or "brisque".
        skip_nima:        If True, skip LAION+NIMA and force BRISQUE.
        laion_components: Tuple (clip_model, clip_proc, laion_mlp, device)
                          from _load_laion_model(), or None.

    Returns:
        Records with laion_score, laion_method, nima_score, nima_method populated.
    """
    use_laion = (
        not skip_nima
        and laion_components is not None
        and laion_components[0] is not None
    )
    effective_method = "brisque" if skip_nima else ("laion" if use_laion else nima_method)

    surviving = [r for r in records if r["status"] in _SURVIVING]

    # Resume: skip records that already have a laion_score
    already_done = [r for r in surviving if r.get("laion_score") is not None]
    todo         = [r for r in surviving if r.get("laion_score") is None]
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
            "Step 1 -- aesthetic scoring  method=%s  photos=%d",
            effective_method, len(surviving),
        )
        print(f"\nStep 1: aesthetic scoring  method={effective_method}  photos={len(surviving)}")

    clip_model, clip_proc, laion_mlp, laion_device = (
        laion_components if laion_components else (None, None, None, None)
    )

    for record in tqdm(todo, desc="Aesthetic scoring", unit="photo"):
        try:
            if skip_nima or (not use_laion and model is None):
                score_val   = score_brisque(record["path"])
                method_used = "brisque"
            elif use_laion:
                score_val   = score_laion_single(
                    record["path"], clip_model, clip_proc, laion_mlp, laion_device
                )
                method_used = "laion"
            else:
                pil_img     = Image.open(record["path"]).convert("RGB")
                score_val   = score_nima(pil_img, model)
                method_used = "nima"
        except Exception as exc:
            logger.warning(
                "Aesthetic scoring failed for %s: %s -- BRISQUE fallback",
                os.path.basename(record["path"]), exc,
            )
            score_val   = score_brisque(record["path"])
            method_used = "brisque"

        record["laion_score"]  = round(float(score_val), 6)
        record["laion_method"] = method_used
        # Backward compat: keep nima_score mirroring laion_score
        record["nima_score"]   = record["laion_score"]
        record["nima_method"]  = method_used

    # Also mirror already-done records that have laion_score but not nima_score
    for record in already_done:
        if record.get("nima_score") is None:
            record["nima_score"]  = record.get("laion_score")
            record["nima_method"] = record.get("laion_method")

    for record in records:
        if record["status"] not in _SURVIVING:
            record["laion_score"]  = None
            record["laion_method"] = None
            record["nima_score"]   = None
            record["nima_method"]  = None

    # Free LAION GPU memory now — LLaVA needs the VRAM next
    if use_laion and _TORCH_AVAILABLE:
        import gc
        del clip_model, clip_proc, laion_mlp
        try:
            _torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()
        logger.info("LAION CLIP model freed from GPU memory")

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


# ── Step 2b — Composition scoring ────────────────────────────────────────────

def _compute_rot_score(face_bboxes: list, img_w: int, img_h: int) -> float:
    """Score how well the dominant face aligns with rule-of-thirds intersections.

    Computes the Gaussian-weighted minimum distance from any face centroid to
    the four rule-of-thirds intersection points.  A face centred on an
    intersection point scores ~1.0; a face far from all intersections scores
    near 0.0.

    Args:
        face_bboxes: List of dicts with keys 'x', 'y', 'w', 'h' (pixel coords).
                     An empty list returns the neutral score 0.5 (no faces →
                     don't penalise decor/venue shots).
        img_w:       Image width in pixels.
        img_h:       Image height in pixels.

    Returns:
        Float in [0, 1].
    """
    if not face_bboxes:
        return 0.5   # neutral — no faces, no penalty

    sigma = _ROT_SIGMA_FRAC * min(img_w, img_h)

    # Rule-of-thirds intersection points
    rot_points = [
        (img_w / 3, img_h / 3),
        (2 * img_w / 3, img_h / 3),
        (img_w / 3, 2 * img_h / 3),
        (2 * img_w / 3, 2 * img_h / 3),
    ]

    best = 0.0
    for bbox in face_bboxes:
        cx = bbox.get("x", 0) + bbox.get("w", 0) / 2
        cy = bbox.get("y", 0) + bbox.get("h", 0) / 2
        for rx, ry in rot_points:
            dist2 = (cx - rx) ** 2 + (cy - ry) ** 2
            score = float(np.exp(-dist2 / (2 * sigma ** 2)))
            if score > best:
                best = score
    return round(best, 6)


def _compute_prominence(face_bboxes: list, img_w: int, img_h: int) -> float:
    """Score the dominant face's size relative to the total image area.

    Args:
        face_bboxes: List of face bbox dicts ('x', 'y', 'w', 'h').
                     Empty list returns 0.5 (neutral for no-face shots).
        img_w:       Image width in pixels.
        img_h:       Image height in pixels.

    Returns:
        Float in [0, 1].  0.5 when no faces present.
    """
    if not face_bboxes:
        return 0.5

    total_area = max(img_w * img_h, 1)
    max_frac   = max(
        (b.get("w", 0) * b.get("h", 0)) / total_area
        for b in face_bboxes
    )
    return round(float(min(max_frac, 1.0)), 6)


def _compute_distraction_penalty(img_path: str, face_bboxes: list,
                                   img_w: int, img_h: int) -> float:
    """Return DISTRACTION_PENALTY if saliency peak is far from all faces.

    Uses OpenCV SpectralResidual saliency.  If opencv-contrib is unavailable
    or the saliency computation fails, returns 1.0 (no penalty).

    Args:
        img_path:    Path to the image file.
        face_bboxes: List of face bbox dicts.  Empty → no penalty (1.0).
        img_w:       Image width in pixels.
        img_h:       Image height in pixels.

    Returns:
        config.DISTRACTION_PENALTY (< 1.0) or 1.0.
    """
    if not face_bboxes:
        return 1.0   # no faces to compare against

    if not _BRISQUE_AVAILABLE:   # cv2 not available
        return 1.0

    try:
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            return 1.0
        # Resize to 300px wide for speed
        scale = 300 / max(img_bgr.shape[1], 1)
        small = cv2.resize(img_bgr, None, fx=scale, fy=scale)

        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        ok, sal_map = saliency.computeSaliency(small)
        if not ok:
            return 1.0

        # Peak location in full-image coordinates
        peak_loc = np.unravel_index(np.argmax(sal_map), sal_map.shape)
        peak_y   = peak_loc[0] / scale
        peak_x   = peak_loc[1] / scale

        diagonal = np.sqrt(img_w ** 2 + img_h ** 2)
        thresh   = _SALIENCY_DIST_THRESH * diagonal

        # Check if any face centroid is close to the saliency peak
        for bbox in face_bboxes:
            cx   = bbox.get("x", 0) + bbox.get("w", 0) / 2
            cy   = bbox.get("y", 0) + bbox.get("h", 0) / 2
            dist = np.sqrt((cx - peak_x) ** 2 + (cy - peak_y) ** 2)
            if dist <= thresh:
                return 1.0   # peak near a face — no distraction penalty

        return float(config.DISTRACTION_PENALTY)
    except Exception:
        return 1.0


def step_composition(records: list, logger) -> list:
    """Compute rule-of-thirds, prominence, and distraction scores for all surviving photos.

    Reads face bbox data stored in the record by DeepFace (phase 2).  If bbox
    data is missing, _compute_rot_score and _compute_prominence return 0.5
    (neutral) so faceless photos are not penalised.

    Checkpoints are saved every 100 photos in case of interruption.

    Args:
        records: All records with phase-2 enrichment.
        logger:  Logger instance.

    Returns:
        Records with rot_score, prominence_score, distraction_penalty added.
    """
    surviving = [r for r in records if r.get("status") in _SURVIVING]
    already   = [r for r in surviving if r.get("rot_score") is not None]
    todo      = [r for r in surviving if r.get("rot_score") is None]

    logger.info(
        "Step 2b -- composition scoring  photos=%d  already=%d",
        len(surviving), len(already),
    )
    print(f"\nStep 2b: composition scoring  photos={len(todo)}")

    for record in tqdm(todo, desc="Composition", unit="photo"):
        try:
            pil_img  = Image.open(record["path"])
            img_w, img_h = pil_img.size

            # Phase 2 stores face regions in repr_results; we stored faces_detected count
            # but not bboxes in the current schema.  We reconstruct from DeepFace quickly
            # at a reduced size to get bboxes cheaply.
            face_bboxes = []
            if record.get("faces_detected", 0) > 0 and _BRISQUE_AVAILABLE:
                try:
                    img_bgr  = cv2.imread(record["path"])
                    if img_bgr is not None:
                        scale    = min(1.0, 600 / max(img_bgr.shape[1], 1))
                        small    = cv2.resize(img_bgr, None, fx=scale, fy=scale)
                        # Use OpenCV Haar for cheap bbox detection (not identity)
                        face_cascade = cv2.CascadeClassifier(
                            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                        )
                        gray     = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                        detects  = face_cascade.detectMultiScale(
                            gray, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20)
                        )
                        for (fx, fy, fw, fh) in detects:
                            face_bboxes.append({
                                "x": int(fx / scale),
                                "y": int(fy / scale),
                                "w": int(fw / scale),
                                "h": int(fh / scale),
                            })
                except Exception:
                    pass   # fall back to empty bboxes → neutral scores

            record["rot_score"]          = _compute_rot_score(face_bboxes, img_w, img_h)
            record["prominence_score"]   = _compute_prominence(face_bboxes, img_w, img_h)
            record["distraction_penalty"] = _compute_distraction_penalty(
                record["path"], face_bboxes, img_w, img_h
            )
        except Exception as exc:
            logger.warning(
                "Composition failed for %s: %s", os.path.basename(record["path"]), exc
            )
            record["rot_score"]           = 0.5
            record["prominence_score"]    = 0.5
            record["distraction_penalty"] = 1.0

    for record in records:
        if record.get("status") not in _SURVIVING:
            record.setdefault("rot_score",           None)
            record.setdefault("prominence_score",    None)
            record.setdefault("distraction_penalty", 1.0)

    logger.info("Step 2b complete")
    return records


# ── Step 2c — LLaVA burst comparison ─────────────────────────────────────────

_BURST_GRID_CELL = 400   # pixel width per grid cell when tiling burst groups

_BURST_GRID_PROMPT = (
    "These are {n} wedding photos from the same moment. "
    "Rank them {nums} from best to worst composition and expression. "
    "Return ONLY a JSON array of ranks like {example}."
)


def _make_burst_grid(paths: list) -> Image.Image:
    """Tile up to 4 images into a 2-column grid with index labels.

    Args:
        paths: List of image file paths (1-4).

    Returns:
        Composite PIL Image with labelled cells.
    """
    cell_w, cell_h = _BURST_GRID_CELL, int(_BURST_GRID_CELL * 0.75)
    cols   = min(len(paths), 2)
    rows   = (len(paths) + 1) // 2
    canvas = Image.new("RGB", (cols * cell_w, rows * cell_h), (240, 240, 240))

    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(canvas)

    for idx, path in enumerate(paths):
        col, row = idx % 2, idx // 2
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail((cell_w, cell_h), Image.LANCZOS)
            x = col * cell_w + (cell_w - img.width)  // 2
            y = row * cell_h + (cell_h - img.height) // 2
            canvas.paste(img, (x, y))
        except Exception:
            pass
        # Index label
        label_x = col * cell_w + 4
        label_y = row * cell_h + 4
        draw.text((label_x + 1, label_y + 1), str(idx + 1), fill=(0, 0, 0))
        draw.text((label_x, label_y), str(idx + 1), fill=(255, 255, 0))

    return canvas


def step_burst_compare(records: list, logger, skip_llava: bool) -> list:
    """Use LLaVA to rank photos within each burst group and adjust emotion scores.

    For each burst_group_id, tiles up to 4 photos into a grid and sends to
    LLaVA for comparative ranking.  Multipliers from config are applied to
    emotion_score:
        rank 1 → × BURST_RANK_BONUS[0]  (1.15)
        rank 2 → × BURST_RANK_BONUS[1]  (1.05)
        rank 3+ → × BURST_RANK_PENALTY  (0.95)

    Skipped entirely when skip_llava is True or no records have a burst_group_id.

    Args:
        records:    All records with emotion_score and burst_group_id populated.
        logger:     Logger instance.
        skip_llava: Mirror the --skip-llava flag; when True, skips all LLaVA calls.

    Returns:
        Records with burst_rank populated and emotion_score adjusted.
    """
    # Collect burst groups
    from collections import defaultdict
    burst_groups: dict = defaultdict(list)
    for r in records:
        gid = r.get("burst_group_id")
        if gid is not None and r.get("status") in _SURVIVING:
            burst_groups[gid].append(r)

    if not burst_groups:
        logger.info("Step 2c -- no burst groups found, skipping")
        print("\nStep 2c: no burst groups -- skipping")
        for r in records:
            r.setdefault("burst_rank", None)
        return records

    if skip_llava:
        logger.info("Step 2c -- burst compare skipped (--skip-llava)")
        print("\nStep 2c: burst compare skipped (--skip-llava)")
        for r in records:
            r.setdefault("burst_rank", None)
        return records

    logger.info("Step 2c -- LLaVA burst comparison  groups=%d", len(burst_groups))
    print(f"\nStep 2c: LLaVA burst comparison  groups={len(burst_groups)}")

    import base64
    import io

    adjusted = 0
    errors   = 0

    for gid, group in tqdm(burst_groups.items(), desc="Burst compare", unit="group"):
        # Only compare groups of 2-4; skip singletons and very large groups
        if len(group) < 2 or len(group) > 8:
            for r in group:
                r.setdefault("burst_rank", None)
            continue

        cap    = min(len(group), 4)
        subset = group[:cap]
        paths  = [r["path"] for r in subset]

        try:
            grid    = _make_burst_grid(paths)
            buf     = io.BytesIO()
            grid.save(buf, format="JPEG", quality=85)
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            n       = cap
            nums    = "1 to " + str(n)
            example = "[" + ",".join(str(i + 1) for i in range(n)) + "]"
            prompt  = _BURST_GRID_PROMPT.format(n=n, nums=nums, example=example)

            def _do_post():
                resp = requests.post(
                    f"{config.OLLAMA_URL}/api/generate",
                    json={
                        "model":  config.OLLAMA_MODEL,
                        "prompt": prompt,
                        "images": [img_b64],
                        "stream": False,
                    },
                    timeout=120,
                )
                resp.raise_for_status()
                raw = resp.json()["response"].strip()
                if raw.startswith("```"):
                    parts = raw.split("```")
                    raw   = parts[1].lstrip("json").strip() if len(parts) >= 3 else parts[-1]
                return json.loads(raw)

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                fut  = ex.submit(_do_post)
                rank_list = fut.result(timeout=90)

            if not isinstance(rank_list, list) or len(rank_list) != cap:
                raise ValueError(f"Unexpected rank response: {rank_list}")

            bonuses = list(config.BURST_RANK_BONUS) + [
                config.BURST_RANK_PENALTY
            ] * (cap - len(config.BURST_RANK_BONUS))

            for order_idx, img_rank in enumerate(rank_list):
                r_idx = int(img_rank) - 1   # convert 1-based rank to 0-based index
                if 0 <= r_idx < cap:
                    multiplier = bonuses[order_idx]
                    subset[r_idx]["emotion_score"] = round(
                        float(subset[r_idx].get("emotion_score") or 0.5) * multiplier, 6
                    )
                    subset[r_idx]["burst_rank"] = order_idx + 1
                    adjusted += 1

        except Exception as exc:
            logger.warning("Burst compare failed for group %s: %s", gid, exc)
            for r in subset:
                r.setdefault("burst_rank", None)
            errors += 1

    for r in records:
        r.setdefault("burst_rank", None)

    logger.info(
        "Step 2c complete -- %d groups  %d emotion_scores adjusted  %d errors",
        len(burst_groups), adjusted, errors,
    )
    print(f"\n  Burst compare: {adjusted} adjustments  {errors} errors")
    return records


# ── Step 3 — Combined scoring ─────────────────────────────────────────────────

def compute_final_score(record: dict) -> dict:
    """Compute base and final weighted scores for one record.

    New formula (weights sum to 1.0 — no normalisation needed):
        base_score = laion      × LAION_WEIGHT       (0.25)
                   + emotion    × EMOTION_WEIGHT      (0.35)
                   + memorab.   × MEMORABILITY_WEIGHT (0.15)
                   + rot        × ROT_WEIGHT          (0.10)
                   + prominence × PROMINENCE_WEIGHT   (0.15)

    Multipliers applied after base_score:
        "candid" in shot_type  → × CANDID_BONUS_MULTIPLIER  (1.10)
        has_closed_eyes        → × CLOSED_EYES_PENALTY      (0.85)
        blur_tier == soft_blur → × SOFT_BLUR_PENALTY        (0.95)
        distraction_penalty    → × DISTRACTION_PENALTY      (0.90) or 1.0

    When laion_score is None (old records without LAION run), falls back to
    nima_score to stay compatible with pre-upgrade scored JSONs.

    Args:
        record: Enriched + scored record dict.

    Returns:
        Dict with base_score, final_score, score_components.
    """
    # Aesthetic: prefer laion_score, fall back to nima_score for old records
    laion        = record.get("laion_score") or record.get("nima_score") or 0.0
    emotion      = record.get("emotion_score")      or 0.0
    memorability = record.get("memorability_score") or 0.0
    rot          = record.get("rot_score")
    prominence   = record.get("prominence_score")
    distraction  = record.get("distraction_penalty") or 1.0

    # Neutral 0.5 when composition fields are absent (old runs without step 2b)
    rot        = 0.5 if rot        is None else float(rot)
    prominence = 0.5 if prominence is None else float(prominence)

    base_score = (
        laion        * config.LAION_WEIGHT        +
        emotion      * config.EMOTION_WEIGHT      +
        memorability * config.MEMORABILITY_WEIGHT +
        rot          * config.ROT_WEIGHT          +
        prominence   * config.PROMINENCE_WEIGHT
    )
    final_score = base_score

    shot_type      = record.get("shot_type") or []
    candid_bonus   = "candid" in shot_type
    closed_eye_pen = bool(record.get("has_closed_eyes"))
    soft_blur_pen  = record.get("blur_tier") == "soft_blur"
    distract_pen   = distraction < 1.0

    if candid_bonus:
        final_score *= config.CANDID_BONUS_MULTIPLIER
    if closed_eye_pen:
        final_score *= config.CLOSED_EYES_PENALTY
    if soft_blur_pen:
        final_score *= config.SOFT_BLUR_PENALTY
    final_score *= distraction   # 0.90 or 1.0

    return {
        "base_score":  round(base_score,  6),
        "final_score": round(final_score, 6),
        "score_components": {
            "laion":               round(laion        * config.LAION_WEIGHT,        6),
            "emotion":             round(emotion      * config.EMOTION_WEIGHT,      6),
            "memorability":        round(memorability * config.MEMORABILITY_WEIGHT, 6),
            "rot":                 round(rot          * config.ROT_WEIGHT,          6),
            "prominence":          round(prominence   * config.PROMINENCE_WEIGHT,   6),
            "candid_bonus":        candid_bonus,
            "closed_eyes_penalty": closed_eye_pen,
            "soft_blur_penalty":   soft_blur_pen,
            "distraction_penalty": distract_pen,
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

    laion_ct    = sum(1 for r in surviving if r.get("laion_method") == "laion")
    brisque_ct  = sum(1 for r in surviving if r.get("laion_method") == "brisque")
    fallback_ct = sum(1 for r in surviving if r.get("llava_fallback"))
    burst_ct    = sum(1 for r in surviving if r.get("burst_rank") is not None)
    distract_ct = sum(
        1 for r in surviving
        if r.get("score_components", {}).get("distraction_penalty")
    )

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
        f" LAION aesthetic            : {laion_ct:>6}",
        f" BRISQUE (fallback)         : {brisque_ct:>6}",
        f" LLaVA fallbacks            : {fallback_ct:>6}",
        f" Burst-ranked photos        : {burst_ct:>6}",
        "-" * w,
        " Score distribution (final_score, 0-1 scale):",
        f"   min                      : {s_min:>9.4f}",
        f"   max                      : {s_max:>9.4f}",
        f"   mean                     : {s_mean:>9.4f}",
        f"   median                   : {s_median:>9.4f}",
        "-" * w,
        " Modifiers applied:",
        f"   candid bonus (+10%)      : {candid_ct:>6}",
        f"   closed-eyes penalty (-15%): {closed_ct:>5}",
        f"   soft-blur penalty (-5%)  : {soft_blur_ct:>6}",
        f"   distraction penalty (-10%): {distract_ct:>5}",
        "-" * w,
        " Formula: LAION(0.25)+emotion(0.35)+memorab(0.15)",
        "          +rot(0.10)+prominence(0.15) = 1.00",
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
    """Orchestrate Phase 3: aesthetic → composition → LLaVA → burst compare → combine → output."""
    parser = argparse.ArgumentParser(
        description="Phase 3 -- Score surviving photos with LAION + composition + LLaVA."
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
        help=(
            "Skip LAION aesthetic predictor and use BRISQUE instead "
            "(same user-facing behaviour as the old --skip-nima flag)"
        ),
    )
    parser.add_argument(
        "--skip-llava",
        action="store_true",
        dest="skip_llava",
        help="Assign neutral LLaVA scores (0.5) without calling Ollama; also skips burst compare",
    )
    parser.add_argument(
        "--skip-composition",
        action="store_true",
        dest="skip_composition",
        help="Skip composition scoring step (rot_score, prominence_score, distraction_penalty)",
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
        "skip_nima=%s  skip_llava=%s  skip_composition=%s  resume=%s ===",
        args.input, output_path, args.test,
        args.skip_nima, args.skip_llava,
        args.skip_composition, args.resume,
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

    # Step 1 — LAION aesthetic (or BRISQUE fallback)
    if args.skip_nima:
        model, nima_method  = None, "brisque"
        laion_components    = None
    else:
        laion_components    = _load_laion_model(logger)
        model, nima_method  = load_nima_model(logger)

    records = step1_nima(
        records, logger, model, nima_method, args.skip_nima,
        laion_components=laion_components,
    )

    # Save aesthetic checkpoint immediately — allows resuming before LLaVA
    if not args.test:
        save_json(output_path, records)
        logger.info("Aesthetic checkpoint saved to %s", output_path)

    # Step 2b — Composition scoring
    if not args.skip_composition:
        records = step_composition(records, logger)
        if not args.test:
            save_json(output_path, records)
            logger.info("Composition checkpoint saved to %s", output_path)

    # Step 2 — LLaVA individual scoring
    records = step2_llava(records, logger, args.skip_llava, output_path)

    # Step 2c — LLaVA burst comparison
    records = step_burst_compare(records, logger, args.skip_llava)

    # Step 3 — Combine scores
    records = step3_combine(records, logger)

    # Step 4 — Save + summary
    step4_output(records, logger, output_path)

    elapsed = time.time() - start
    logger.info("=== Phase 3 complete in %.1fs ===", elapsed)
    print(f"\nPhase 3 complete in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
