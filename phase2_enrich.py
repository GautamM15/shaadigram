"""
phase2_enrich.py — Phase 2: EXIF moment grouping, face detection,
person matching, shot classification, smile and closed-eye detection.

Reads surviving_photos_candid.json (phase 1 output), enriches every
surviving record in a single pass, and writes enriched_photos_candid.json.

Usage:
    python phase2_enrich.py
    python phase2_enrich.py --input surviving_photos.json --output enriched_photos.json
"""

import argparse
import glob
import os
import pickle
import time
from datetime import datetime, timedelta

import cv2
import numpy as np
from deepface import DeepFace
from PIL import Image
from tqdm import tqdm

import config
from utils import get_exif_timestamp, load_json, save_json, setup_logging

# ── Optional CLIP imports (phase 2 step 6) ────────────────────────────────────
try:
    import torch
    from transformers import CLIPModel as _CLIPFullModel, CLIPImageProcessor, CLIPTokenizerFast
    _CLIP_AVAILABLE = True
except ImportError:
    _CLIP_AVAILABLE = False

# Statuses that go through full enrichment
_SURVIVING = {"surviving", "soft_blur_surviving"}

# Null enrichment block applied to non-surviving / error records
_NULL_ENRICHMENT = {
    "capture_time":     None,
    "moment_id":        None,
    "moment_label":     "unknown",
    "moment_start":     None,
    "moment_end":       None,
    "faces_detected":   0,
    "persons_matched":  [],
    "is_group_photo":   False,
    "has_gautam":       False,
    "has_siddharth":    False,
    "shot_type":        [],
    "primary_shot_type": "unknown",
    "max_smile_score":  None,
    "has_closed_eyes":  False,
    "confusion_warning": False,
    "clip_event_tags":   [],
}

# OpenCV eye cascade — bundled with opencv-python
_EYE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_eye.xml"


# ── Step 1 — Enrollment loading ───────────────────────────────────────────────

def load_enrollments(logger) -> dict:
    """Load all enrolled face embeddings from ENROLLED_FACES_FOLDER.

    Scans for files matching *_face.pkl, unpickles each, and returns a
    mapping of person name to unit-normalised numpy embedding.

    Args:
        logger: Logger instance for this phase.

    Returns:
        Dict mapping person name (str) to embedding (np.ndarray).
        Empty dict if no .pkl files found — face matching will be skipped.
    """
    pattern = os.path.join(config.ENROLLED_FACES_FOLDER, "*_face.pkl")
    pkl_files = glob.glob(pattern)

    if not pkl_files:
        logger.warning(
            "No enrolled faces found in '%s' — person matching will be skipped.",
            config.ENROLLED_FACES_FOLDER,
        )
        return {}

    enrollments = {}
    for pkl_path in sorted(pkl_files):
        try:
            with open(pkl_path, "rb") as f:
                record = pickle.load(f)
            name = record["name"]
            emb  = np.array(record["embedding"])
            enrollments[name] = emb
            logger.info(
                "Loaded enrollment: %s  embedding shape=%s", name, emb.shape
            )
        except Exception as exc:
            logger.error("Failed to load enrollment %s: %s", pkl_path, exc)

    logger.info("Enrollments loaded: %d person(s)", len(enrollments))
    return enrollments


# ── Step 2 — EXIF moment grouping ────────────────────────────────────────────

def step2_exif_moments(records: list, logger) -> list:
    """Extract EXIF timestamps, cluster time-based moments, and apply subfolder fallback.

    Pass 1 — EXIF photos:
        Sorted by capture time.  A new moment begins whenever the gap between
        consecutive shots exceeds MOMENT_GAP_MINUTES.  Each photo gets a
        capture_time field (ISO string) in addition to moment_id/label.

    Pass 2 — no-EXIF photos (MOMENT_SUBFOLDER_FALLBACK=True):
        The immediate parent directory of the file becomes the moment label
        (lowercased, spaces replaced with underscores).  A stable moment_id
        is assigned per unique subfolder name (starting at 10001 so it never
        collides with EXIF-based IDs which start at 1).
        moment_start and moment_end remain None.

    If MOMENT_SUBFOLDER_FALLBACK=False, no-EXIF photos keep moment_id=None
    and moment_label="unknown" (original behaviour).

    Args:
        records: List of phase-1 record dicts.
        logger:  Logger instance for this phase.

    Returns:
        Same list with capture_time, moment_id, moment_label, moment_start,
        moment_end added to every record.
    """
    logger.info("Step 2 -- Extracting EXIF timestamps and clustering moments")

    # ── Pass 1: attach timestamps to surviving records ────────────────────────
    timestamped   = []   # (datetime, record) for records with EXIF
    no_ts_records = []   # records with no EXIF timestamp
    non_surviving = []   # rejected/error records — skip EXIF entirely

    for r in records:
        if r["status"] not in _SURVIVING:
            non_surviving.append(r)
            r["capture_time"] = None
            continue
        ts = None
        try:
            ts = get_exif_timestamp(r["path"])
        except Exception as exc:
            logger.warning("EXIF read failed for %s: %s", r["path"], exc)
        if ts is not None:
            r["capture_time"] = ts.isoformat()
            timestamped.append((ts, r))
        else:
            r["capture_time"] = None
            no_ts_records.append(r)

    # ── Sort and cluster EXIF photos ──────────────────────────────────────────
    timestamped.sort(key=lambda x: x[0])

    gap            = timedelta(minutes=config.MOMENT_GAP_MINUTES)
    moment_id      = 1
    moment_start   = None
    prev_ts        = None
    moment_members = []
    moment_summary = {}   # moment_id -> count

    def _close_moment(members, mid, mstart, mend):
        """Assign final moment fields to all records in the current moment."""
        label      = f"moment_{mid:02d}"
        mstart_str = mstart.isoformat() if mstart else None
        mend_str   = mend.isoformat()   if mend   else None
        for _, rec in members:
            rec["moment_id"]    = mid
            rec["moment_label"] = label
            rec["moment_start"] = mstart_str
            rec["moment_end"]   = mend_str
        moment_summary[mid] = len(members)

    for ts, rec in timestamped:
        if prev_ts is None:
            moment_start = ts
        elif ts - prev_ts > gap:
            _close_moment(moment_members, moment_id, moment_start, prev_ts)
            moment_id     += 1
            moment_start   = ts
            moment_members = []
        moment_members.append((ts, rec))
        prev_ts = ts

    if moment_members:
        _close_moment(moment_members, moment_id, moment_start, prev_ts)

    # ── Pass 2: subfolder fallback for no-EXIF surviving photos ──────────────
    subfolder_fallback_count = 0
    still_unknown_count      = 0

    if config.MOMENT_SUBFOLDER_FALLBACK:
        # Build stable subfolder → moment_id mapping (IDs start at 10001)
        subfolder_id_map: dict[str, int] = {}
        subfolder_id_counter = 10001

        for r in no_ts_records:
            subfolder_raw   = os.path.basename(os.path.dirname(r["path"]))
            subfolder_label = subfolder_raw.lower().replace(" ", "_") or "unknown"

            if subfolder_label not in subfolder_id_map:
                subfolder_id_map[subfolder_label] = subfolder_id_counter
                subfolder_id_counter += 1

            r["moment_id"]    = subfolder_id_map[subfolder_label]
            r["moment_label"] = subfolder_label
            r["moment_start"] = None
            r["moment_end"]   = None
            subfolder_fallback_count += 1

        if subfolder_id_map:
            for label, mid in sorted(subfolder_id_map.items(), key=lambda x: x[1]):
                count = sum(1 for r in no_ts_records if r["moment_label"] == label)
                logger.info("  subfolder moment %d (%s): %d photos", mid, label, count)
    else:
        # Original behaviour: leave as unknown
        for r in no_ts_records:
            r.update({
                "moment_id":    None,
                "moment_label": "unknown",
                "moment_start": None,
                "moment_end":   None,
            })
        still_unknown_count = len(no_ts_records)

    # Non-surviving records get null moment fields
    for r in non_surviving:
        r.update({
            "moment_id":    None,
            "moment_label": "unknown",
            "moment_start": None,
            "moment_end":   None,
        })

    total_moments = len(moment_summary)
    logger.info(
        "Step 2 complete -- %d EXIF moment(s)  |  %d EXIF-timestamped  "
        "|  %d subfolder-fallback  |  %d unknown",
        total_moments, len(timestamped),
        subfolder_fallback_count, still_unknown_count,
    )
    for mid, count in sorted(moment_summary.items()):
        logger.info("  moment_%02d: %d photos", mid, count)

    return records


# ── Helpers for step 3+4+5 ────────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1-D arrays.

    Args:
        a: First embedding vector.
        b: Second embedding vector.

    Returns:
        Similarity in [-1.0, 1.0].  Returns 0.0 if either norm is zero.
    """
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def match_face_to_persons(face_embedding: np.ndarray,
                           enrollments: dict) -> list:
    """Match a single face embedding against all enrolled persons.

    Args:
        face_embedding: 1-D numpy array for one detected face.
        enrollments:    Dict of {name: embedding} from load_enrollments().

    Returns:
        List of (name, similarity) tuples where similarity >=
        (1 - FACE_MATCH_THRESHOLD), sorted descending by similarity.
    """
    matches = []
    for name, ref_emb in enrollments.items():
        sim = cosine_similarity(face_embedding, ref_emb)
        if sim >= (1.0 - config.FACE_MATCH_THRESHOLD):
            matches.append((name, sim))
    matches.sort(key=lambda x: x[1], reverse=True)
    return matches


def confusion_check(matches: list, logger, path: str) -> tuple:
    """Resolve a single face matching both gautam and siddharth.

    If one face's match list contains both "gautam" and "siddharth", it is
    impossible for both to be correct (one face = one person).  Keep only
    the higher-confidence match and flag a confusion warning.

    Args:
        matches:  List of (name, similarity) for one face, sorted desc.
        logger:   Logger instance.
        path:     Image path — used in warning message only.

    Returns:
        Tuple (cleaned_matches: list, warning_fired: bool).
    """
    names = [n for n, _ in matches]
    if "gautam" in names and "siddharth" in names:
        logger.warning(
            "CONFUSION: single face matched both gautam+siddharth in %s "
            "— keeping higher confidence match only  scores=%s",
            os.path.basename(path),
            [(n, f"{s:.3f}") for n, s in matches],
        )
        # matches already sorted by sim desc — keep only top entry
        return [matches[0]], True
    return matches, False


def detect_closed_eyes(face_crop_bgr: np.ndarray) -> bool:
    """Detect whether eyes are closed in a cropped face image.

    Runs OpenCV haarcascade_eye on the greyscale face crop.  If fewer than
    2 eyes are found the face is flagged as potentially closed-eyed.

    Args:
        face_crop_bgr: Face region as a BGR numpy array (from DeepFace).

    Returns:
        True if fewer than 2 eyes detected (likely closed/obscured).
    """
    cascade = cv2.CascadeClassifier(_EYE_CASCADE_PATH)
    gray    = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2GRAY)
    eyes    = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
    )
    return len(eyes) < 2


def classify_shot_type(faces_detected: int,
                        blur_tier: str,
                        aspect_ratio: float) -> list:
    """Assign one or more shot-type tags to a photo.

    Rules applied in order; multiple tags are allowed:
        faces == 1                  → "solo_portrait"
        faces == 2                  → "couple"
        faces >= 3                  → "group"
        blur_tier == "soft_blur"    → "candid"  (additional tag)
        faces == 0 and AR <  1.5    → "detail"
        faces == 0 and AR >= 1.5    → "venue"

    Args:
        faces_detected: Total faces found by DeepFace.
        blur_tier:      Phase-1 blur tier string ("pass", "soft_blur", …).
        aspect_ratio:   Image width / height.

    Returns:
        List of tag strings.  First element is the primary shot type.
    """
    tags = []
    if faces_detected == 1:
        tags.append("solo_portrait")
    elif faces_detected == 2:
        tags.append("couple")
    elif faces_detected >= 3:
        tags.append("group")

    if blur_tier == "soft_blur":
        tags.append("candid")

    if faces_detected == 0:
        tags.append("detail" if aspect_ratio < 1.5 else "venue")

    return tags if tags else ["unknown"]


# ── Core per-photo enrichment ─────────────────────────────────────────────────

def enrich_single_photo(record: dict, enrollments: dict, logger,
                         include_emotion: bool = False,
                         debug_face: bool = False,
                         debug_counter: list = None) -> dict:
    """Run steps 3, 4, and 5 on one photo in a single image load.

    Loads the image once with PIL, converts to numpy for DeepFace, then:
        Step 3 — face detection + person matching + confusion check
        Step 4 — shot type classification
        Step 5 — smile score (if include_emotion) + closed-eye detection

    All exceptions are caught; on failure the record receives the null
    enrichment block and status is unchanged.

    Args:
        record:          Phase-1 record dict (already has moment fields).
        enrollments:     Dict of {name: embedding} from load_enrollments().
        logger:          Logger instance.
        include_emotion: Run DeepFace.analyze() for smile scores (slow).
        debug_face:      Print raw similarity scores for the first 5 photos
                         with faces detected.
        debug_counter:   Mutable [int] shared across calls to count debug
                         photos printed so far.  Pass [0] from the caller.

    Returns:
        Record dict with all enrichment fields populated.
    """
    enrichment = dict(_NULL_ENRICHMENT)   # start from safe defaults

    try:
        # ── Load image once ───────────────────────────────────────────────────
        pil_img    = Image.open(record["path"]).convert("RGB")
        img_array  = np.array(pil_img)          # HxWx3 RGB uint8
        aspect_ratio = pil_img.width / max(pil_img.height, 1)

        # ── Step 3 — Face detection + person matching ─────────────────────────
        repr_results = DeepFace.represent(
            img_path          = img_array,
            model_name        = "Facenet",
            enforce_detection = False,
            detector_backend  = "retinaface",
        )

        faces_detected   = len(repr_results)
        all_matched_names = []
        confusion_warning = False
        face_crops_bgr    = []   # collected for closed-eye step

        # Debug: print raw similarities for first 5 face-containing photos
        do_debug = (
            debug_face
            and debug_counter is not None
            and debug_counter[0] < 5
            and faces_detected > 0
        )

        for face_idx, face_result in enumerate(repr_results):
            face_emb = np.array(face_result["embedding"])

            if do_debug:
                thresh = 1.0 - config.FACE_MATCH_THRESHOLD
                print(
                    f"[DEBUG-FACE] {os.path.basename(record['path'])}"
                    f"  face {face_idx + 1}/{faces_detected}:"
                )
                for name, ref_emb in sorted(enrollments.items()):
                    sim   = cosine_similarity(face_emb, ref_emb)
                    match = "Yes" if sim >= thresh else "No"
                    print(
                        f"  {name:<12}: sim={sim:.4f}"
                        f"  threshold={thresh:.3f}  MATCH={match}"
                    )

            matches  = match_face_to_persons(face_emb, enrollments)
            matches, warned = confusion_check(matches, logger, record["path"])
            if warned:
                confusion_warning = True
            all_matched_names.extend(n for n, _ in matches)

            # Collect face crop for eye detection (DeepFace returns RGB face)
            face_rgb = face_result.get("face")
            if face_rgb is not None:
                face_arr = np.array(face_rgb)
                if face_arr.ndim == 3 and face_arr.shape[2] == 3:
                    # Convert float [0,1] or uint8 to uint8 BGR
                    if face_arr.dtype != np.uint8:
                        face_arr = (face_arr * 255).clip(0, 255).astype(np.uint8)
                    face_crops_bgr.append(cv2.cvtColor(face_arr, cv2.COLOR_RGB2BGR))

        if do_debug:
            debug_counter[0] += 1

        persons_matched = sorted(set(all_matched_names))
        has_gautam      = "gautam"     in persons_matched
        has_siddharth   = "siddharth"  in persons_matched
        is_group_photo  = faces_detected >= 3

        # ── Step 4 — Shot classification ──────────────────────────────────────
        blur_tier  = record.get("blur_tier", "pass")
        shot_tags  = classify_shot_type(faces_detected, blur_tier, aspect_ratio)
        primary_st = shot_tags[0]

        # ── Step 5 — Smile + closed eyes ──────────────────────────────────────
        max_smile_score = None
        has_closed_eyes = False

        if faces_detected > 0:
            if include_emotion:
                try:
                    analysis = DeepFace.analyze(
                        img_path          = img_array,
                        actions           = ["emotion"],
                        enforce_detection = False,
                        detector_backend  = "retinaface",
                    )
                    if not isinstance(analysis, list):
                        analysis = [analysis]
                    smiles = [
                        r["emotion"].get("happy", 0.0) / 100.0
                        for r in analysis
                        if "emotion" in r
                    ]
                    if smiles:
                        max_smile_score = float(max(smiles))
                except Exception as exc:
                    logger.warning(
                        "Emotion analysis failed for %s: %s",
                        os.path.basename(record["path"]), exc,
                    )

            for crop in face_crops_bgr:
                if detect_closed_eyes(crop):
                    has_closed_eyes = True
                    break

        enrichment.update({
            "faces_detected":    faces_detected,
            "persons_matched":   persons_matched,
            "is_group_photo":    is_group_photo,
            "has_gautam":        has_gautam,
            "has_siddharth":     has_siddharth,
            "shot_type":         shot_tags,
            "primary_shot_type": primary_st,
            "max_smile_score":   max_smile_score,
            "has_closed_eyes":   has_closed_eyes,
            "confusion_warning": confusion_warning,
        })

    except Exception as exc:
        logger.error(
            "Enrichment failed for %s: %s", os.path.basename(record["path"]), exc
        )

    record.update(enrichment)
    return record


# ── Step 3+4+5 combined loop ──────────────────────────────────────────────────

def step3_4_5_enrich(records: list, enrollments: dict, logger,
                      include_emotion: bool = False,
                      debug_face: bool = False) -> list:
    """Run face detection, shot classification, and smile/eye detection.

    Processes only surviving records in a single tqdm pass.  Non-surviving
    records receive the null enrichment block and are passed through
    unchanged.

    Args:
        records:         All records from phase 1 (including rejected ones).
        enrollments:     Loaded face embeddings from step 1.
        logger:          Logger instance.
        include_emotion: Pass through to enrich_single_photo().
        debug_face:      Print raw similarity scores for first 5 face photos.

    Returns:
        All records with enrichment fields populated.
    """
    surviving = [r for r in records if r["status"] in _SURVIVING]
    skipped   = [r for r in records if r["status"] not in _SURVIVING]
    logger.info(
        "Step 3+4+5 -- enriching %d surviving photos (%d skipped)  "
        "include_emotion=%s  debug_face=%s",
        len(surviving), len(skipped), include_emotion, debug_face,
    )

    # Disable person matching when only gautam is enrolled — he is tagged manually
    if set(enrollments.keys()) == {"gautam"}:
        msg = (
            "Gautam-only enrollment detected — face matching disabled. "
            "Gautam's photos will be identified manually in Phase 5 review UI."
        )
        logger.info(msg)
        print(msg)
        enrollments = {}   # empty dict -> match_face_to_persons returns [] for all faces

    # Apply null enrichment to non-surviving records
    for r in skipped:
        r.update(_NULL_ENRICHMENT)

    debug_counter = [0]   # mutable counter shared across all enrich calls
    for record in tqdm(surviving, desc="Enriching photos", unit="photo"):
        enrich_single_photo(
            record, enrollments, logger,
            include_emotion=include_emotion,
            debug_face=debug_face,
            debug_counter=debug_counter,
        )

    logger.info("Step 3+4+5 complete")
    return records


# ── Step 6 — CLIP embeddings + event tags ─────────────────────────────────────

def _load_clip_model(logger) -> tuple:
    """Load CLIP ViT-B/32 onto GPU (or CPU) via transformers.

    Args:
        logger: Logger instance.

    Returns:
        Tuple (model, processor, device) on success, or (None, None, None)
        if transformers / torch are unavailable.
    """
    if not _CLIP_AVAILABLE:
        logger.warning("CLIP unavailable (transformers or torch missing) -- step 6 skipped")
        return None, None, None
    try:
        device         = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading CLIP model %s on %s ...", config.CLIP_MODEL, device)
        img_processor  = CLIPImageProcessor.from_pretrained(config.CLIP_MODEL)
        txt_tokenizer  = CLIPTokenizerFast.from_pretrained(config.CLIP_MODEL)
        model          = _CLIPFullModel.from_pretrained(config.CLIP_MODEL).to(device)
        model.eval()
        # Store tokenizer as a convenience attribute for text encoding
        model._txt_tokenizer = txt_tokenizer
        logger.info("CLIP model loaded")
        return model, img_processor, device
    except Exception as exc:
        logger.warning("CLIP model load failed (%s) -- step 6 skipped", exc)
        return None, None, None


def _embed_batch(model, processor, pil_images: list, device: str) -> np.ndarray:
    """Encode a batch of PIL images with CLIP visual encoder.

    Falls back to CPU automatically if the requested device runs out of memory.

    Args:
        model:      CLIPModel instance.
        processor:  CLIPProcessor instance.
        pil_images: List of PIL.Image objects (RGB).
        device:     "cuda" or "cpu".

    Returns:
        Float32 numpy array of shape (B, 512), L2-normalised.
    """
    import torch

    def _run(dev):
        inputs = processor(images=pil_images, return_tensors="pt")
        pv     = inputs["pixel_values"].to(dev)
        with torch.no_grad():
            vis_out = model.vision_model(pixel_values=pv)
            feats   = model.visual_projection(vis_out.pooler_output)  # (B, 512)
        return feats.cpu().float().numpy()

    try:
        feats = _run(device)
    except (RuntimeError, torch.cuda.OutOfMemoryError):
        # CUDA OOM or similar — retry on CPU
        if device != "cpu":
            torch.cuda.empty_cache()
            feats = _run("cpu")
        else:
            raise

    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return feats / norms


def _encode_text_prompts(model, processor, prompts: list, device: str) -> np.ndarray:
    """Encode a list of text prompts with CLIP text encoder.

    Args:
        model:     CLIPModel instance.
        processor: CLIPProcessor instance.
        prompts:   List of text strings.
        device:    "cuda" or "cpu".

    Returns:
        Float32 numpy array of shape (P, 512), L2-normalised.
    """
    import torch
    tokenizer = getattr(model, "_txt_tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("model._txt_tokenizer not set — call _load_clip_model() first")
    inputs = tokenizer(text=prompts, return_tensors="pt", padding=True, truncation=True)
    ii = inputs["input_ids"].to(device)
    am = inputs["attention_mask"].to(device)
    with torch.no_grad():
        txt_out = model.text_model(input_ids=ii, attention_mask=am)
        feats   = model.text_projection(txt_out.pooler_output)  # (P, 512)
    feats = feats.cpu().float().numpy()
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return feats / norms


def _classify_clip_event(embedding: np.ndarray,
                          text_embeddings: np.ndarray,
                          prompts: list) -> list:
    """Return top-2 event labels for one image embedding.

    Args:
        embedding:       (512,) unit-normalised float32 array.
        text_embeddings: (P, 512) unit-normalised float32 array.
        prompts:         List of prompt strings (length P).

    Returns:
        List of up to 2 label strings (spaces replaced with underscores).
    """
    sims = text_embeddings @ embedding   # (P,)
    top_idx = np.argsort(sims)[::-1][:2]
    return [prompts[i].replace(" ", "_") for i in top_idx]


def step6_clip_embeddings(records: list, logger,
                           output_path: str,
                           skip_clip: bool = False) -> list:
    """Generate CLIP embeddings and zero-shot event tags for all surviving photos.

    Embeds images using CLIP ViT-B/32 in batches of CLIP_BATCH_SIZE.  Saves
    all embeddings to CLIP_EMBEDDINGS_FILE (.npz) and writes clip_event_tags
    (top-2 labels) back into each JSON record.

    Non-surviving records and error photos receive clip_event_tags=[].

    Args:
        records:     All phase-2 records (enriched through step 5).
        logger:      Logger instance.
        output_path: Path to clip_embeddings.npz output file.
        skip_clip:   If True, skip this step entirely.

    Returns:
        Records list with clip_event_tags populated for surviving photos.
    """
    if skip_clip:
        logger.info("Step 6 -- CLIP skipped (--skip-clip flag)")
        print("\nStep 6: CLIP skipped")
        for r in records:
            r.setdefault("clip_event_tags", [])
        return records

    model, processor, device = _load_clip_model(logger)
    if model is None:
        logger.warning("Step 6 -- CLIP model unavailable; clip_event_tags set to []")
        print("\nStep 6: CLIP unavailable -- skipping")
        for r in records:
            r.setdefault("clip_event_tags", [])
        return records

    surviving = [r for r in records if r.get("status") in {"surviving", "soft_blur_surviving"}]
    logger.info(
        "Step 6 -- CLIP embeddings  model=%s  photos=%d  batch=%d  device=%s",
        config.CLIP_MODEL, len(surviving), config.CLIP_BATCH_SIZE, device,
    )
    print(
        f"\nStep 6: CLIP embeddings  photos={len(surviving)}"
        f"  batch={config.CLIP_BATCH_SIZE}  device={device}"
    )

    # Encode all event-prompt text embeddings once
    text_embs = _encode_text_prompts(
        model, processor, config.CLIP_EVENT_PROMPTS, device
    )

    all_paths  = []
    all_embs   = []
    errors     = 0

    batch_size = config.CLIP_BATCH_SIZE
    for batch_start in tqdm(
        range(0, len(surviving), batch_size),
        desc="CLIP embedding",
        unit="batch",
    ):
        batch = surviving[batch_start: batch_start + batch_size]
        pil_imgs    = []
        valid_batch = []
        for r in batch:
            try:
                img = Image.open(r["path"]).convert("RGB")
                # Pre-resize to max 800px — CLIP crops to 224 anyway.
                # Prevents multi-GB numpy arrays from 42MP+ wedding photos.
                if max(img.size) > 800:
                    img.thumbnail((800, 800), Image.LANCZOS)
                pil_imgs.append(img)
                valid_batch.append(r)
            except Exception as exc:
                logger.warning("CLIP: could not open %s: %s", r["path"], exc)
                r["clip_event_tags"] = []
                errors += 1

        if not pil_imgs:
            continue

        try:
            embs = _embed_batch(model, processor, pil_imgs, device)
        except Exception as exc:
            logger.warning("CLIP batch failed at index %d: %s", batch_start, exc)
            for r in valid_batch:
                r["clip_event_tags"] = []
                errors += 1
            continue

        for r, emb in zip(valid_batch, embs):
            r["clip_event_tags"] = _classify_clip_event(
                emb, text_embs, config.CLIP_EVENT_PROMPTS
            )
            all_paths.append(r["path"])
            all_embs.append(emb)

    # Ensure non-surviving records have the field
    for r in records:
        r.setdefault("clip_event_tags", [])

    # Save .npz
    if all_embs:
        np.savez(
            output_path,
            paths=np.array(all_paths),
            embeddings=np.array(all_embs, dtype=np.float32),
        )
        logger.info("Saved clip_embeddings.npz: %d embeddings → %s", len(all_embs), output_path)

    # Free GPU memory before LLaVA / other phases
    import gc
    del model, processor
    if _CLIP_AVAILABLE:
        try:
            import torch as _t
            _t.cuda.empty_cache()
        except Exception:
            pass
    gc.collect()

    # Tag summary
    from collections import Counter
    tag_counts = Counter(
        tag
        for r in records
        for tag in (r.get("clip_event_tags") or [])
    )
    top_tags = "  ".join(f"{t}({c})" for t, c in tag_counts.most_common(5))
    logger.info(
        "Step 6 complete -- %d embedded  %d errors  top tags: %s",
        len(all_embs), errors, top_tags,
    )
    print(f"\n  CLIP: {len(all_embs)} embedded  {errors} errors")
    print(f"  Top event tags: {top_tags}")
    return records


# ── Step 7 — Output + summary ─────────────────────────────────────────────────

def step7_output(records: list, logger, output_path: str) -> None:
    """Save enriched records to JSON and print a summary table.

    Args:
        records:     Fully enriched record list.
        logger:      Logger instance.
        output_path: Destination JSON file path.
    """
    save_json(output_path, records)
    logger.info("Saved %s (%d records)", output_path, len(records))

    total      = len(records)
    processed  = sum(1 for r in records if r["status"] in _SURVIVING)
    skipped    = total - processed

    # Moment stats
    exif_ts_count = sum(1 for r in records if r.get("capture_time") is not None)
    subfolder_ct  = sum(
        1 for r in records
        if r.get("moment_id") is not None
        and r.get("capture_time") is None
        and r.get("moment_label") != "unknown"
    )
    no_ts         = sum(1 for r in records if r.get("moment_label") == "unknown")
    moment_ids    = set(
        r["moment_id"] for r in records
        if r.get("moment_id") is not None
    )
    n_moments     = len(moment_ids)
    ts_count      = exif_ts_count   # alias kept for avg calculation below
    avg_per_moment = (ts_count // max(1, len({
        r["moment_id"] for r in records
        if r.get("capture_time") is not None and r.get("moment_id") is not None
    })))

    # Face stats
    face_det   = sum(1 for r in records if (r.get("faces_detected") or 0) > 0)
    total_faces = sum(r.get("faces_detected") or 0 for r in records)
    avg_faces  = (total_faces / face_det) if face_det else 0
    gautam_ct  = sum(1 for r in records if r.get("has_gautam"))
    siddarth_ct = sum(1 for r in records if r.get("has_siddharth"))
    confusion_ct = sum(1 for r in records if r.get("confusion_warning"))

    # Shot type stats
    def _tag_count(tag):
        return sum(1 for r in records if tag in (r.get("shot_type") or []))

    solo_ct   = _tag_count("solo_portrait")
    couple_ct = _tag_count("couple")
    group_ct  = _tag_count("group")
    candid_ct = _tag_count("candid")
    detail_ct = _tag_count("detail")
    venue_ct  = _tag_count("venue")
    group_photo_ct = sum(1 for r in records if r.get("is_group_photo"))

    # Smile + eyes
    smile_ct  = sum(
        1 for r in records
        if (r.get("max_smile_score") or 0) > 0.7
    )
    eyes_ct   = sum(1 for r in records if r.get("has_closed_eyes"))
    errors    = sum(
        1 for r in records
        if r["status"] in _SURVIVING
        and (r.get("faces_detected") is None)
    )

    w = 48
    lines = [
        "-" * w,
        " Phase 2 -- Enrichment Summary",
        "-" * w,
        f" Total records              : {total:>6}",
        f" Processed (surviving)      : {processed:>6}",
        f" Skipped (rejected)         : {skipped:>6}",
        "-" * w,
        f" EXIF timestamps found      : {exif_ts_count:>6}",
        f" Subfolder-fallback photos  : {subfolder_ct:>6}",
        f" Photos still unknown       : {no_ts:>6}",
        f" Moments (EXIF-based)       : {n_moments:>6}",
        f" Avg photos/EXIF moment     : {avg_per_moment:>6}",
        "-" * w,
        f" Faces detected (photos)    : {face_det:>6}",
        f" Avg faces per photo        : {avg_faces:>9.1f}",
        f" Matched: gautam            : {gautam_ct:>6}",
        f" Matched: siddharth         : {siddarth_ct:>6}",
        f" Confusion warnings         : {confusion_ct:>6}",
        "-" * w,
        " Shot types:",
        f"   solo_portrait            : {solo_ct:>6}",
        f"   couple                   : {couple_ct:>6}",
        f"   group                    : {group_ct:>6}",
        f"   candid                   : {candid_ct:>6}",
        f"   detail                   : {detail_ct:>6}",
        f"   venue                    : {venue_ct:>6}",
        f" Group photos (3+ faces)    : {group_photo_ct:>6}",
        "-" * w,
        f" Max smile score > 0.7      : {smile_ct:>6}",
        f" Has closed eyes            : {eyes_ct:>6}",
        f" Errors (could not enrich)  : {errors:>6}",
        "-" * w,
    ]
    summary = "\n".join(lines)
    print("\n" + summary)
    logger.info("\n%s", summary)


# ── Verify-person diagnostic ──────────────────────────────────────────────────

def cmd_verify_person(image_path: str, enrollments: dict, logger) -> None:
    """Run face detection on one photo and print similarity scores for all enrolled persons.

    Intended as a diagnostic tool to validate enrollment quality or investigate
    why a person is or isn't being matched.  No JSON is read or written.

    Args:
        image_path:  Path to the image file to inspect.
        enrollments: Dict of {name: embedding} from load_enrollments().
        logger:      Logger instance.
    """
    if not enrollments:
        print("No enrolled persons found — enroll someone first with enroll_face.py")
        return

    filename = os.path.basename(image_path)
    print(f"\nVerifying: {filename}")
    print(f"Enrolled persons: {', '.join(sorted(enrollments))}")
    print(f"Threshold: {1.0 - config.FACE_MATCH_THRESHOLD:.3f}  "
          f"(FACE_MATCH_THRESHOLD={config.FACE_MATCH_THRESHOLD})\n")

    try:
        pil_img   = Image.open(image_path).convert("RGB")
        img_array = np.array(pil_img)
    except Exception as exc:
        print(f"ERROR: Could not load image: {exc}")
        logger.error("verify-person: could not load %s: %s", image_path, exc)
        return

    try:
        repr_results = DeepFace.represent(
            img_path          = img_array,
            model_name        = "Facenet",
            enforce_detection = False,
            detector_backend  = "retinaface",
        )
    except Exception as exc:
        print(f"ERROR: Face detection failed: {exc}")
        logger.error("verify-person: face detection failed for %s: %s", image_path, exc)
        return

    n_faces = len(repr_results)
    if n_faces == 0:
        print("No faces detected.")
        return

    print(f"Faces detected: {n_faces}\n")
    thresh = 1.0 - config.FACE_MATCH_THRESHOLD

    for idx, face_result in enumerate(repr_results):
        face_emb = np.array(face_result["embedding"])
        print(f"  Face {idx + 1}/{n_faces}:")
        for name in sorted(enrollments):
            sim   = cosine_similarity(face_emb, enrollments[name])
            match = "Yes" if sim >= thresh else "No"
            print(f"    {name:<14}: sim={sim:.4f}  threshold={thresh:.3f}  MATCH={match}")
        print()

    logger.info(
        "verify-person: %s  faces=%d  enrolled=%s",
        filename, n_faces, list(sorted(enrollments)),
    )


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    """Orchestrate Phase 2: load -> moments -> enrich -> output."""
    parser = argparse.ArgumentParser(
        description="Phase 2 -- Enrich surviving photos with EXIF, faces, and shot types."
    )
    parser.add_argument(
        "--input",
        default=config.PHASE2_INPUT,
        help="Phase 1 JSON to read (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default=config.PHASE2_OUTPUT,
        help="Enriched JSON to write (default: %(default)s)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Process first 20 surviving records only",
    )
    parser.add_argument(
        "--include-emotion",
        action="store_true",
        dest="include_emotion",
        help="Run emotion/smile analysis (slower; off by default)",
    )
    parser.add_argument(
        "--debug-face",
        action="store_true",
        dest="debug_face",
        help="Print raw similarity scores for first 5 photos with faces",
    )
    parser.add_argument(
        "--verify-person",
        default=None,
        dest="verify_person",
        metavar="IMAGE_PATH",
        help="Run face detection on one photo and print similarity scores against all enrolled persons",
    )
    parser.add_argument(
        "--skip-clip",
        action="store_true",
        dest="skip_clip",
        help="Skip CLIP embedding step (step 6); clip_event_tags will be [] for all photos",
    )
    parser.add_argument(
        "--resume-clip",
        action="store_true",
        dest="resume_clip",
        help=(
            "Skip steps 2-5 and resume from *_step5.json checkpoint "
            "(use when step 6 CLIP crashed after a completed enrichment run)"
        ),
    )
    args = parser.parse_args()

    logger = setup_logging("phase2_enrich")

    if args.verify_person:
        enrollments = load_enrollments(logger)
        cmd_verify_person(args.verify_person, enrollments, logger)
        return
    start  = time.time()
    logger.info(
        "=== Phase 2 started  input=%s  output=%s  test=%s  "
        "include_emotion=%s  debug_face=%s  skip_clip=%s ===",
        args.input, args.output, args.test,
        args.include_emotion, args.debug_face, args.skip_clip,
    )

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

    # Intermediate checkpoint path — lives next to the output file
    _step5_path = args.output.replace(".json", "_step5.json")

    if args.resume_clip:
        # Skip steps 2-5; load enriched records from the step5 checkpoint
        records = load_json(_step5_path)
        if not records:
            logger.error("--resume-clip: checkpoint not found: %s -- exiting.", _step5_path)
            print(f"ERROR: step5 checkpoint not found: {_step5_path}")
            return
        logger.info("--resume-clip: loaded %d records from checkpoint %s", len(records), _step5_path)
        print(f"Resuming CLIP from checkpoint ({len(records)} records)")
    else:
        enrollments = load_enrollments(logger)
        records     = step2_exif_moments(records, logger)
        records     = step3_4_5_enrich(
            records, enrollments, logger,
            include_emotion=args.include_emotion,
            debug_face=args.debug_face,
        )
        # Save intermediate checkpoint so a CLIP crash doesn't require a full re-run
        save_json(records, _step5_path)
        logger.info("Saved step5 checkpoint → %s", _step5_path)

    records     = step6_clip_embeddings(
        records, logger,
        output_path=config.CLIP_EMBEDDINGS_FILE,
        skip_clip=args.skip_clip,
    )
    step7_output(records, logger, args.output)

    elapsed = time.time() - start
    logger.info("=== Phase 2 complete in %.1fs ===", elapsed)
    print(f"\nPhase 2 complete in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
