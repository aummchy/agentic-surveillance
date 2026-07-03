import os
import re
import json
import logging
import logging.handlers
import structlog
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"


# ── JSONC Loader ────────────────────────────────────────────────
def _load_jsonc(path: Path) -> dict:
    """Load a JSONC (JSON with comments) file by stripping comments first.
    Handles // and /* */ comments without stripping // inside strings."""
    text = path.read_text(encoding="utf-8")
    result = []
    i = 0
    in_string = False
    escape_next = False
    while i < len(text):
        ch = text[i]
        if escape_next:
            result.append(ch)
            escape_next = False
            i += 1
            continue
        if in_string:
            if ch == "\\":
                escape_next = True
                result.append(ch)
                i += 1
                continue
            if ch == '"':
                in_string = False
            result.append(ch)
            i += 1
            continue
        if ch == '"':
            in_string = True
            result.append(ch)
            i += 1
            continue
        # Check for line comment //
        if ch == "/" and i + 1 < len(text) and text[i + 1] == "/":
            # Skip to end of line
            while i < len(text) and text[i] != "\n":
                i += 1
            continue
        # Check for block comment /* */
        if ch == "/" and i + 1 < len(text) and text[i + 1] == "*":
            i += 2
            while i + 1 < len(text) and not (text[i] == "*" and text[i + 1] == "/"):
                i += 1
            i += 2  # skip */
            continue
        result.append(ch)
        i += 1
    return json.loads("".join(result))


_config_path = Path(__file__).parent / "config.jsonc"
_config = _load_jsonc(_config_path) if _config_path.exists() else {}


def _get(env_key: str, config_key: str, default, cast=str):
    """Resolve a setting: env var > config.jsonc > hardcoded default."""
    val = os.getenv(env_key)
    if val is not None:
        if cast is list:
            return [v.strip() for v in val.split(",")]
        return cast(val)
    return cast(_config.get(config_key, default))


def setup_file_logging():
    """Add rotating file handler for full system logs at DEBUG level."""
    LOG_DIR.mkdir(exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_DIR / "surveillance.log",
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(message)s")
    )
    logging.root.addHandler(file_handler)


def setup_logging():
    """Configure structlog with console (INFO+) and rotating file (DEBUG) output."""
    setup_file_logging()

    # Silence noisy PyMongo/Motor driver logs — only propagate WARNING+
    for name in ("pymongo", "pymongo.topology", "pymongo.pool",
                 "pymongo.command", "pymongo.server", "motor"):
        logging.getLogger(name).setLevel(logging.WARNING)

    # Console handler — INFO and above to reduce noise
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logging.root.addHandler(console_handler)
    logging.root.setLevel(logging.DEBUG)

    # Structlog renderer — chosen by LOG_FORMAT env var
    if os.getenv("LOG_FORMAT") == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    # ProcessorFormatter routes structlog output through standard logging handlers
    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    console_handler.setFormatter(formatter)
    # Re-apply formatter to the file handler we added earlier
    for handler in logging.root.handlers:
        if isinstance(handler, logging.handlers.RotatingFileHandler):
            handler.setFormatter(formatter)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


setup_logging()


def _log_effective_settings():
    """Print effective value and source for every tunable setting at startup."""
    log = structlog.get_logger("config")
    tunables = [
        ("MATCH_THRESHOLD", "MATCH_THRESHOLD", 0.45, float),
        ("PERSON_CONF_THRESHOLD", "PERSON_CONF_THRESHOLD", 0.40, float),
        ("FRAME_WIDTH", "FRAME_WIDTH", 1280, int),
        ("FRAME_HEIGHT", "FRAME_HEIGHT", 720, int),
        ("EMBEDDING_DET_SCORE_MIN", "EMBEDDING_DET_SCORE_MIN", 0.40, float),
        ("DET_SCORE_MIN", "DET_SCORE_MIN", 0.40, float),
        ("DET_SCORE_RELAXED", "DET_SCORE_RELAXED", 0.20, float),
        ("DEDUP_SIMILARITY_THRESHOLD", "DEDUP_SIMILARITY_THRESHOLD", 0.40, float),
    ]
    for env_key, config_key, default, cast in tunables:
        env_val = os.getenv(env_key)
        config_val = _config.get(config_key)
        if env_val is not None:
            source = "env"
            effective = cast(env_val)
        elif config_val is not None:
            source = "config.jsonc"
            effective = cast(config_val)
        else:
            source = "default"
            effective = default
        log.info("config.resolved", setting=config_key, value=effective, source=source)


_log_effective_settings()


def validate_config():
    errors = []

    if not os.getenv("MONGODB_URI"):
        errors.append("MONGODB_URI is required")

    threshold = _get("MATCH_THRESHOLD", "MATCH_THRESHOLD", 0.45, float)
    if threshold > 0.45:
        errors.append(f"MATCH_THRESHOLD={threshold} exceeds maximum 0.45")

    timeout = _get("TRACK_TIMEOUT_SECS", "TRACK_TIMEOUT_SECS", 3.0, float)
    if timeout <= 0:
        errors.append("TRACK_TIMEOUT_SECS must be > 0")

    max_track = _get("MAX_TRACK_SECS", "MAX_TRACK_SECS", 300, float)
    if max_track <= 0:
        errors.append("MAX_TRACK_SECS must be > 0")

    det_min = _get("DET_SCORE_MIN", "DET_SCORE_MIN", 0.40, float)
    if not (0 < det_min < 1):
        errors.append("DET_SCORE_MIN must be between 0 and 1")

    det_relaxed = _get("DET_SCORE_RELAXED", "DET_SCORE_RELAXED", 0.20, float)
    if not (0 < det_relaxed < 1):
        errors.append("DET_SCORE_RELAXED must be between 0 and 1")
    if det_relaxed > det_min:
        errors.append("DET_SCORE_RELAXED must be <= DET_SCORE_MIN")

    emb_min = EMBEDDING_DET_SCORE_MIN
    if not (0 < emb_min < 1):
        errors.append("EMBEDDING_DET_SCORE_MIN must be between 0 and 1")

    channels = _get("ALERT_CHANNELS", "ALERT_CHANNELS", ["console"], list)
    valid_channels = {"console", "email", "sms", "webhook"}
    for ch in channels:
        if ch.strip() not in valid_channels:
            errors.append(f"Invalid alert channel: {ch.strip()}")

    if errors:
        raise ValueError("Config validation failed:\n" + "\n".join(f"  - {e}" for e in errors))


# ══════════════════════════════════════════════════════════════════
# Settings — loaded from: env var > config.jsonc > hardcoded default
# ══════════════════════════════════════════════════════════════════

# ── Secrets (env-only, never in config.jsonc) ──────────────────
MONGODB_URI = os.getenv("MONGODB_URI", "")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "surveillance")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "faces")
MONGODB_EVENTS_COLLECTION = os.getenv("MONGODB_EVENTS_COLLECTION", "events")

CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME", "")
CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY", "")
CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET", "")

ALERT_WEBHOOK_URL = os.getenv("ALERT_WEBHOOK_URL", "")
SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
ALERT_EMAIL_TO = os.getenv("ALERT_EMAIL_TO", "")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM = os.getenv("TWILIO_FROM", "")
ALERT_SMS_TO = os.getenv("ALERT_SMS_TO", "")

# ── Model / Pipeline ───────────────────────────────────────────
YOLO_MODEL = _get("YOLO_MODEL", "YOLO_MODEL", "models/yolov8s.pt")
YOLO_DEVICE = _get("YOLO_DEVICE", "YOLO_DEVICE", "cpu")
INSIGHTFACE_MODEL = _get("INSIGHTFACE_MODEL", "INSIGHTFACE_MODEL", "buffalo_m")
INSIGHTFACE_DET_SIZE = _get("INSIGHTFACE_DET_SIZE", "INSIGHTFACE_DET_SIZE", 1280, int)
INSIGHTFACE_PROVIDER = _get("INSIGHTFACE_PROVIDER", "INSIGHTFACE_PROVIDER", "CPUExecutionProvider")

# ── Detection / Tracking ───────────────────────────────────────
PERSON_CONF_THRESHOLD = _get("PERSON_CONF_THRESHOLD", "PERSON_CONF_THRESHOLD", 0.40, float)
TRACK_TIMEOUT_SECS = _get("TRACK_TIMEOUT_SECS", "TRACK_TIMEOUT_SECS", 3.0, float)
MAX_TRACK_SECS = _get("MAX_TRACK_SECS", "MAX_TRACK_SECS", 300, float)
DET_SCORE_MIN = _get("DET_SCORE_MIN", "DET_SCORE_MIN", 0.40, float)
DET_SCORE_RELAXED = _get("DET_SCORE_RELAXED", "DET_SCORE_RELAXED", 0.20, float)
EMBEDDING_DET_SCORE_MIN = _get("EMBEDDING_DET_SCORE_MIN", "EMBEDDING_DET_SCORE_MIN", 0.40, float)

# ── Progressive Recognition ────────────────────────────────────
RECOGNITION_INTERVAL_FRAMES = _get("RECOGNITION_INTERVAL_FRAMES", "RECOGNITION_INTERVAL_FRAMES", 10, int)

# ── Quality Scoring ────────────────────────────────────────────
QUALITY_BLUR_MIN = _get("QUALITY_BLUR_MIN", "QUALITY_BLUR_MIN", 30, float)
QUALITY_BRIGHTNESS_MIN = _get("QUALITY_BRIGHTNESS_MIN", "QUALITY_BRIGHTNESS_MIN", 30, float)
QUALITY_BRIGHTNESS_MAX = _get("QUALITY_BRIGHTNESS_MAX", "QUALITY_BRIGHTNESS_MAX", 240, float)
QUALITY_FACE_AREA_MIN = _get("QUALITY_FACE_AREA_MIN", "QUALITY_FACE_AREA_MIN", 1600, float)
QUALITY_BLUR_MAX = _get("QUALITY_BLUR_MAX", "QUALITY_BLUR_MAX", 1000, float)
QUALITY_AREA_MAX = _get("QUALITY_AREA_MAX", "QUALITY_AREA_MAX", 10000, float)
QUALITY_WEIGHT_BLUR = _get("QUALITY_WEIGHT_BLUR", "QUALITY_WEIGHT_BLUR", 0.60, float)
QUALITY_WEIGHT_BRIGHT = _get("QUALITY_WEIGHT_BRIGHT", "QUALITY_WEIGHT_BRIGHT", 0.25, float)
QUALITY_WEIGHT_AREA = _get("QUALITY_WEIGHT_AREA", "QUALITY_WEIGHT_AREA", 0.15, float)

# ── JPEG Compression ──────────────────────────────────────────
JPEG_QUALITY_STORE = _get("JPEG_QUALITY_STORE", "JPEG_QUALITY_STORE", 85, int)
JPEG_QUALITY_BROADCAST = _get("JPEG_QUALITY_BROADCAST", "JPEG_QUALITY_BROADCAST", 65, int)

# ── Vector Search ──────────────────────────────────────────────
VECTOR_SEARCH_CANDIDATES = _get("VECTOR_SEARCH_CANDIDATES", "VECTOR_SEARCH_CANDIDATES", 150, int)
VECTOR_SEARCH_LIMIT = _get("VECTOR_SEARCH_LIMIT", "VECTOR_SEARCH_LIMIT", 5, int)
SCAN_LIMIT = _get("SCAN_LIMIT", "SCAN_LIMIT", 500, int)

# ── Face Matching ──────────────────────────────────────────────
MATCH_THRESHOLD = _get("MATCH_THRESHOLD", "MATCH_THRESHOLD", 0.45, float)
DEDUP_SIMILARITY_THRESHOLD = _get("DEDUP_SIMILARITY_THRESHOLD", "DEDUP_SIMILARITY_THRESHOLD", 0.40, float)

# ── Embedding History ─────────────────────────────────────────
EMBEDDING_HISTORY_CAP = _get("EMBEDDING_HISTORY_CAP", "EMBEDDING_HISTORY_CAP", 25, int)

# ── Recognition Thresholds ─────────────────────────────────────
VERY_HIGH_SIMILARITY = _get("VERY_HIGH_SIMILARITY", "VERY_HIGH_SIMILARITY", 0.90, float)
HIGH_CONFIDENCE_SIMILARITY = _get("HIGH_CONFIDENCE_SIMILARITY", "HIGH_CONFIDENCE_SIMILARITY", 0.85, float)
KNOWN_VISITOR_SIMILARITY = _get("KNOWN_VISITOR_SIMILARITY", "KNOWN_VISITOR_SIMILARITY", 0.85, float)
KNOWN_VISITOR_CONFIDENCE = _get("KNOWN_VISITOR_CONFIDENCE", "KNOWN_VISITOR_CONFIDENCE", 80, float)
BORDERLINE_FACE_QUALITY = _get("BORDERLINE_FACE_QUALITY", "BORDERLINE_FACE_QUALITY", 0.8, float)
MASK_CONFIDENCE_PENALTY = _get("MASK_CONFIDENCE_PENALTY", "MASK_CONFIDENCE_PENALTY", 0.85, float)

# ── CLAHE ──────────────────────────────────────────────────────
CLAHE_CLIP_LIMIT = _get("CLAHE_CLIP_LIMIT", "CLAHE_CLIP_LIMIT", 2.0, float)
CLAHE_TILE_SIZE = _get("CLAHE_TILE_SIZE", "CLAHE_TILE_SIZE", 8, int)

# ── Mask Detection ─────────────────────────────────────────────
MASK_DETECTION = _get("MASK_DETECTION", "MASK_DETECTION", "heuristic")
MASK_RATIO_THRESHOLD = _get("MASK_RATIO_THRESHOLD", "MASK_RATIO_THRESHOLD", 0.3, float)
LOITER_SECS = _get("LOITER_SECS", "LOITER_SECS", 30, float)

# ── Face Visibility ────────────────────────────────────────────
VISIBLE_FACE_RATIO = _get("VISIBLE_FACE_RATIO", "VISIBLE_FACE_RATIO", 0.025, float)
PARTIAL_FACE_RATIO = _get("PARTIAL_FACE_RATIO", "PARTIAL_FACE_RATIO", 0.010, float)
MIN_TRACK_FRAMES = _get("MIN_TRACK_FRAMES", "MIN_TRACK_FRAMES", 15, int)

# ── Office Hours (policy agent) ───────────────────────────────
OFFICE_HOURS_START = _get("OFFICE_HOURS_START", "OFFICE_HOURS_START", 9, int)
OFFICE_HOURS_END = _get("OFFICE_HOURS_END", "OFFICE_HOURS_END", 17, int)
OFFICE_DAYS = _get("OFFICE_DAYS", "OFFICE_DAYS", [0, 1, 2, 3, 4], list)

# ── Alerting ───────────────────────────────────────────────────
ALERT_CHANNELS = _get("ALERT_CHANNELS", "ALERT_CHANNELS", ["console", "webhook"], list)
ALERT_COOLDOWN_SECS = _get("ALERT_COOLDOWN_SECS", "ALERT_COOLDOWN_SECS", 60, float)

# ── Camera ─────────────────────────────────────────────────────
# CAMERA_SOURCE: RTSP URL string (e.g. "rtsp://192.168.1.10:554/stream")
#   OR integer device index (e.g. 0 for local webcam).
#   Takes precedence over CAMERA_INDEX if set.
CAMERA_SOURCE = _get("CAMERA_SOURCE", "CAMERA_SOURCE", "")
CAMERA_INDEX = _get("CAMERA_INDEX", "CAMERA_INDEX", 0, int)
FRAME_WIDTH = _get("FRAME_WIDTH", "FRAME_WIDTH", 1280, int)
FRAME_HEIGHT = _get("FRAME_HEIGHT", "FRAME_HEIGHT", 720, int)
CAMERA_ID = _get("CAMERA_ID", "CAMERA_ID", "cam_01")

# ── LLM (Ollama) ──────────────────────────────────────────────
OLLAMA_URL = _get("OLLAMA_URL", "OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = _get("OLLAMA_MODEL", "OLLAMA_MODEL", "gemma3:4b")
OLLAMA_TIMEOUT = _get("OLLAMA_TIMEOUT", "OLLAMA_TIMEOUT", 30, int)
