import os
import logging
import logging.handlers
import structlog
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"


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


def validate_config():
    errors = []

    if not os.getenv("MONGODB_URI"):
        errors.append("MONGODB_URI is required")

    threshold = float(os.getenv("MATCH_THRESHOLD", "0.35"))
    if threshold > 0.45:
        errors.append(f"MATCH_THRESHOLD={threshold} exceeds maximum 0.45")

    timeout = float(os.getenv("TRACK_TIMEOUT_SECS", "8.0"))
    if timeout <= 0:
        errors.append("TRACK_TIMEOUT_SECS must be > 0")

    max_track = float(os.getenv("MAX_TRACK_SECS", "300"))
    if max_track <= 0:
        errors.append("MAX_TRACK_SECS must be > 0")

    det_min = float(os.getenv("DET_SCORE_MIN", "0.50"))
    if not (0 < det_min < 1):
        errors.append("DET_SCORE_MIN must be between 0 and 1")

    det_relaxed = float(os.getenv("DET_SCORE_RELAXED", "0.20"))
    if not (0 < det_relaxed < 1):
        errors.append("DET_SCORE_RELAXED must be between 0 and 1")
    if det_relaxed > det_min:
        errors.append("DET_SCORE_RELAXED must be <= DET_SCORE_MIN")

    emb_min = float(os.getenv("EMBEDDING_DET_SCORE_MIN", "0.40"))
    if not (0 < emb_min < 1):
        errors.append("EMBEDDING_DET_SCORE_MIN must be between 0 and 1")

    channels = os.getenv("ALERT_CHANNELS", "console").split(",")
    valid_channels = {"console", "email", "sms", "webhook"}
    for ch in channels:
        if ch.strip() not in valid_channels:
            errors.append(f"Invalid alert channel: {ch.strip()}")

    if errors:
        raise ValueError("Config validation failed:\n" + "\n".join(f"  - {e}" for e in errors))


MONGODB_URI = os.getenv("MONGODB_URI", "")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "surveillance")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "faces")
MONGODB_EVENTS_COLLECTION = os.getenv("MONGODB_EVENTS_COLLECTION", "events")

CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME", "")
CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY", "")
CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET", "")

YOLO_MODEL = os.getenv("YOLO_MODEL", "models/yolov8n.pt")
YOLO_DEVICE = os.getenv("YOLO_DEVICE", "cpu")
INSIGHTFACE_MODEL = os.getenv("INSIGHTFACE_MODEL", "buffalo_l")
INSIGHTFACE_DET_SIZE = int(os.getenv("INSIGHTFACE_DET_SIZE", "640"))
INSIGHTFACE_PROVIDER = os.getenv("INSIGHTFACE_PROVIDER", "CPUExecutionProvider")

PERSON_CONF_THRESHOLD = float(os.getenv("PERSON_CONF_THRESHOLD", "0.5"))
TRACK_TIMEOUT_SECS = float(os.getenv("TRACK_TIMEOUT_SECS", "8.0"))
MAX_TRACK_SECS = float(os.getenv("MAX_TRACK_SECS", "300"))
DET_SCORE_MIN = float(os.getenv("DET_SCORE_MIN", "0.50"))
DET_SCORE_RELAXED = float(os.getenv("DET_SCORE_RELAXED", "0.20"))
EMBEDDING_DET_SCORE_MIN = float(os.getenv("EMBEDDING_DET_SCORE_MIN", "0.40"))

RECOGNITION_INTERVAL_FRAMES = int(os.getenv("RECOGNITION_INTERVAL_FRAMES", "20"))

QUALITY_BLUR_MAX = float(os.getenv("QUALITY_BLUR_MAX", "1000"))
QUALITY_AREA_MAX = float(os.getenv("QUALITY_AREA_MAX", "10000"))

MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "0.25"))
DEDUP_SIMILARITY_THRESHOLD = float(os.getenv("DEDUP_SIMILARITY_THRESHOLD", "0.40"))

MASK_DETECTION = os.getenv("MASK_DETECTION", "heuristic")
LOITER_SECS = float(os.getenv("LOITER_SECS", "30"))

VISIBLE_FACE_RATIO = float(os.getenv("VISIBLE_FACE_RATIO", "0.025"))
PARTIAL_FACE_RATIO = float(os.getenv("PARTIAL_FACE_RATIO", "0.010"))
MIN_TRACK_FRAMES = int(os.getenv("MIN_TRACK_FRAMES", "30"))

ALERT_CHANNELS = [ch.strip() for ch in os.getenv("ALERT_CHANNELS", "console").split(",")]
ALERT_WEBHOOK_URL = os.getenv("ALERT_WEBHOOK_URL", "")
ALERT_COOLDOWN_SECS = float(os.getenv("ALERT_COOLDOWN_SECS", "60"))
SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
ALERT_EMAIL_TO = os.getenv("ALERT_EMAIL_TO", "")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM = os.getenv("TWILIO_FROM", "")
ALERT_SMS_TO = os.getenv("ALERT_SMS_TO", "")

CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", "640"))
FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", "480"))
CAMERA_ID = os.getenv("CAMERA_ID", "cam_01")

# LLM (Ollama)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "30"))
