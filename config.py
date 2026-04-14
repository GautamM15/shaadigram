# ─────────────────────────────────────────────
# config.py — all configurable values live here
# ─────────────────────────────────────────────

# Input
INPUT_FOLDER = "E:\\Shagun Sid Wedding"

# Phase 1 — filter thresholds
BLUR_THRESHOLD = 100
EXPOSURE_LOW = 30
EXPOSURE_HIGH = 225
DUPLICATE_HASH_THRESHOLD = 8
BLUR_HARD_REJECT_MULTIPLIER = 0.4   # hard reject = BLUR_THRESHOLD * this
CANDID_BONUS_MULTIPLIER = 1.10      # applied in phase 3 scoring for soft_blur photos
SAMPLE_OUTPUT_SUFFIX = "_candid"    # appended to output filename in --input-override mode
MOMENT_GAP_MINUTES = 30             # gap between shots (minutes) that starts a new moment

# Phase 2 — default I/O paths
PHASE2_INPUT  = "surviving_photos_candid.json"
PHASE2_OUTPUT = "enriched_photos_candid.json"

# Phase 4 — selection limits
MAX_PHOTOS_PER_MOMENT = 15
TOP_N_ALBUM = 800
FINAL_ALBUM = 200
TOP_N_PERSON = 100

# Phase 2 — face matching
FACE_MATCH_THRESHOLD = 0.4
MANUAL_PERSON_REVIEW = ["gautam"]   # persons identified manually in phase 5 review UI

# Phase 3 — scoring weights (must sum to 1.0)
NIMA_WEIGHT = 0.25
EMOTION_WEIGHT = 0.40
LIGHTING_WEIGHT = 0.20
MEMORABILITY_WEIGHT = 0.15

# Phase 3 — I/O paths
PHASE3_INPUT  = "enriched_photos_candid.json"
PHASE3_OUTPUT = "scored_photos_candid.json"

# Phase 4 — I/O paths
PHASE4_INPUT  = "scored_photos_candid.json"
PHASE4_OUTPUT = "selected_photos_candid.json"

# Phase 5 — review UI
PHASE5_INPUT           = "selected_photos_candid.json"
REVIEW_PROGRESS_FILE   = "review_progress.json"
PERSON_TAG_COLORS      = {"gautam": "#4A90D9"}   # letter-key → hex colour
MAX_ADDITIONAL_PERSONS = 5
ALBUM_APPROVED_FOLDER  = "output/album_approved"

# Phase 3 — score modifiers
CLOSED_EYES_PENALTY = 0.85
SOFT_BLUR_PENALTY   = 0.95

# Improvement 1 — scan report (phase1_filter.py --scan-report)
SCAN_REPORT_TXT     = "scan_report.txt"
SCAN_REPORT_JSON    = "scan_report.json"
SCAN_REPORT_WORKERS = 4           # threads for phash pass in --scan-report

# Improvement 2 — subfolder moment fallback (phase2_enrich.py)
MOMENT_SUBFOLDER_FALLBACK = True  # False = old "unknown" behavior for no-EXIF photos

# Improvement 3 — burst limiter (phase1b_burst.py)
BURST_MAX_KEEP     = 3            # max photos to keep per burst group
BURST_SEQUENCE_GAP = 5            # filename seq gap that breaks a burst group
PHASE1B_INPUT      = "enriched_photos.json"

# Phase 3 — LLaVA batching
LLAVA_BATCH_SIZE  = 10   # photos per batch
LLAVA_BATCH_PAUSE = 1    # seconds between batches

# LLaVA / Ollama
OLLAMA_MODEL = "llava:7b-v1.6-mistral-q4_K_M"
OLLAMA_URL = "http://localhost:11434"

# Folders
LOG_FOLDER = "logs"
OUTPUT_FOLDER = "output"
ALBUM_TOP800_FOLDER = "output/album_top800"
ALBUM_FINAL200_FOLDER = "output/album_final200"
PERSON_HIGHLIGHTS_FOLDER = "output/person_highlights"
ENROLLED_FACES_FOLDER = "."   # .pkl enrollment files live in project root

# Supported image extensions
SUPPORTED_EXTENSIONS = [
    ".jpg", ".jpeg", ".png",
    ".JPG", ".JPEG", ".PNG",
    ".nef", ".NEF", ".cr2", ".CR2",
    ".arw", ".ARW", ".dng", ".DNG",
]
