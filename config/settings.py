import os
import json
import logging
import logging.handlers
import structlog
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"


# ── ANSI Colors ─────────────────────────────────────────────────
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    BLUE = "\033[34m"
    CYAN = "\033[36m"
    MAGENTA = "\033[35m"
    WHITE = "\033[37m"

    # Event prefix colors
    PREFIX = {
        "CAM": CYAN,
        "MATCH": BLUE,
        "MEMORY": MAGENTA,
        "RECOG": CYAN,
        "POLICY": GREEN,
        "VISIT": MAGENTA,
        "FINAL": GREEN,
        "FACE": YELLOW,
        "ALERT": RED,
        "SHUTDOWN": DIM,
    }

    # Status colors
    STATUS = {
        "KNOWN": GREEN,
        "VERIFIED": GREEN,
        "AUTHORIZED": GREEN,
        "KNOWN_VISITOR": GREEN,
        "UNCERTAIN": YELLOW,
        "UNKNOWN": RED,
        "BLACKLIST": RED,
        "MASKED_UNKNOWN": RED,
        "INTENTIONALLY_HIDDEN": RED,
    }


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
            items = [v.strip() for v in val.split(",")]
            if items and default and len(default) > 0:
                elem_type = type(default[0])
                return [elem_type(v) for v in items]
            return items
        if cast is bool:
            return val.lower() not in ("false", "0", "no", "off", "")
        return cast(val)
    return cast(_config.get(config_key, default))


# ── Terminal allowlist ──
TERMINAL_ALLOWLIST = frozenset({
    "startup_complete", "camera_started", "camera_stopped", "camera_reconnected",
    "track_finalized",
    "recognition_decision",
    "match_found",
    "memory_lookup",
    "visit_recorded", "visit_suppressed_duplicate",
    "alert", "alert_dispatched", "alert_broadcast_sent", "progressive_critical_alert",
    "policy_decision",
    "duplicate_suppressed",
    "face_crop_saved",
    "pipeline_stats",
    "no_embedding_after_retries",
    "llm_connect_failed", "llm_unavailable",
    "shutdown",
})


def _fmt_opt(v):
    """Format optional float for terminal display."""
    return "---" if v is None else f"{v:.3f}"


class CompactTerminalRenderer:
    """Operator-friendly one-line renderer with ANSI colors."""

    # Map event names to prefix labels
    PREFIX_MAP = {
        "camera_started": "CAM", "camera_stopped": "CAM", "camera_reconnected": "CAM",
        "recognition_decision": "RECOG", "match_found": "MATCH", "memory_lookup": "MEMORY",
        "policy_decision": "POLICY", "visit_recorded": "VISIT", "track_finalized": "FINAL",
        "face_crop_saved": "FACE",         "no_embedding_after_retries": "FACE",
        "pipeline_stats": "PERF",
        "progressive_critical_alert": "ALERT", "alert": "ALERT", "alert_dispatched": "ALERT",
        "llm_connect_failed": "LLM", "llm_unavailable": "LLM",
        "shutdown": "SHUTDOWN",
    }

    FORMATS = {
        "camera_started": "{_pfx} source={source} backend={backend} res={resolution} yolo={yolo_model} device={yolo_device} face={face_model}",
        "camera_stopped": "{_pfx} stopped",
        "camera_reconnected": "{_pfx} reconnected res={resolution}",
        "recognition_decision": (
            "{_pfx} {_trk} {name:<10} {_status} "
            "sim={_sim} top2={_top2} gap={_gap} q={_quality} dur={_duration}s conf={confidence}"
        ),
        "match_found": "{_pfx} {_trk} {name:<10} sim={_sim} top2={_top2} gap={_gap} verified={verified} tags={tags}",
        "memory_lookup": "{_pfx} {_trk} {person_id:<12} visits={visit_count} known={is_known} boost={confidence_boost}",
        "policy_decision": "{_pfx} {_trk} {_status} alert={alert_level} {_alert_flag} vis={visit_count}",
        "visit_recorded": "{_pfx} {display_id} {visit_action} total={visit_count}",
        "track_finalized": (
            "{_pfx} {_trk} {name:<10} {_status} "
            "sim={_sim} top2={_top2} gap={_gap} conf={confidence} "
            "frames={total_frames_seen} face={frames_with_detectable_face} vis={visit_count}"
        ),
        "face_crop_saved": "{_pfx} saved track={track_id} path={path}",
        "no_embedding_after_retries": "{_pfx} no_embedding track={track_id} vis={visibility} frames={frames_seen}",
        "progressive_critical_alert": "{_pfx} CRITICAL {name} reason={reason}",
        "alert": "{_pfx} {_status} track={track_id} name={name} level={alert_level} summary={summary}",
        "alert_dispatched": "{_pfx} {_status} name={name} level={alert_level}",
        "llm_connect_failed": "{_pfx} unavailable model={model} url={url}",
        "llm_unavailable": "{_pfx} unavailable model={model} url={url}",
        "pipeline_stats": "{_pfx} FPS: {fps} | tracker(avg): {tracker_avg} ms | tracker(max): {tracker_max} ms | bt={n_bt} drawn={n_drawn}",
        "shutdown": "{_pfx} {reason}",
    }

    def __call__(self, logger, method_name, event_dict):
        event = event_dict.get("event", "")
        show = event in TERMINAL_ALLOWLIST
        if method_name in ("warning", "error", "critical"):
            show = True
        if not show:
            return ""
        return self._format(event_dict)

    def _format(self, event_dict):
        d = dict(event_dict)
        ts = d.get("timestamp", "")[:19]
        d["ts"] = ts

        event = d.get("event", "")

        # Build colored prefix
        prefix = self.PREFIX_MAP.get(event, event[:5].upper())
        pfx_color = Colors.PREFIX.get(prefix, Colors.WHITE)
        d["_pfx"] = f"{ts} {pfx_color}{prefix:<7}{Colors.RESET}"

        d["_trk"] = f"trk={d['byte_track_id']}" if "byte_track_id" in d else "trk=---"

        # Build colored status
        status = d.get("status")
        if isinstance(status, str):
            status = status.upper()
            d["status"] = status
            status_color = Colors.STATUS.get(status, Colors.WHITE)
            d["_status"] = f"{status_color}{status:<12}{Colors.RESET}"
        else:
            d["_status"] = ""

        # Format float fields
        d["_sim"] = _fmt_opt(d.get("similarity"))
        d["_top2"] = _fmt_opt(d.get("top2"))
        d["_gap"] = _fmt_opt(d.get("margin"))

        # Format face quality (default: ---)
        quality = d.get("face_quality") or d.get("quality")
        d["_quality"] = f"{quality:.2f}" if quality is not None and quality > 0 else "---"

        # Format duration (default: ---)
        duration = d.get("track_duration") or d.get("duration")
        d["_duration"] = f"{duration:.0f}" if duration is not None else "---"

        # Format visit_action (default: skipped)
        d["visit_action"] = d.get("visit_action") or "skipped"

        # Format alert flag for policy_decision
        alert_flag = "ALERT" if d.get("should_alert") else "ok"
        flag_color = Colors.RED if d.get("should_alert") else Colors.GREEN
        d["_alert_flag"] = f"{flag_color}{alert_flag}{Colors.RESET}"

        # Format display_id for visit_recorded (prefer name, fallback to short person_id)
        d["display_id"] = d.get("name") or (d.get("person_id") or "???")[:12]

        # Format visit_count (default: 0)
        d["visit_count"] = d.get("visit_count", 0)

        # Name fallback for identity events
        if event in ("track_finalized", "recognition_decision"):
            name = d.get("name")
            if not name:
                status_val = d.get("status", "")
                name = "unknown" if isinstance(status_val, str) and status_val.upper() == "UNKNOWN" else (d.get("person_id") or "???")[:8]
            d["name"] = name

        fmt = self.FORMATS.get(event)
        if fmt:
            try:
                return fmt.format(**d)
            except (KeyError, IndexError):
                pass

        # Default compact fallback
        keys = [k for k in ("status", "similarity", "top2", "margin",
                            "person_id", "error", "reason", "count", "total",
                            "track_id", "confidence", "action")
                if k in d and k != "event"]
        kv = " ".join(f"{k}={d[k]!r}" for k in keys)
        return f"{ts} {event:<14} {kv}" if kv else f"{ts} {event}"


class JSONFileRenderer:
    """Full structured JSON for forensic log file."""
    def __call__(self, logger, method_name, event_dict):
        return json.dumps(event_dict, default=str)


class _BlankFilter(logging.Filter):
    """Drop blank log records and events not in terminal allowlist."""
    def filter(self, record):
        # Get event name from structlog record (msg is a dict for structlog)
        msg = getattr(record, 'msg', None)
        event = msg.get('event', '') if isinstance(msg, dict) else None
        # If event is not in allowlist and not a warning/error/critical, suppress
        if event and event not in TERMINAL_ALLOWLIST and record.levelno < logging.WARNING:
            return False
        # Check if message is blank after rendering
        msg_str = record.getMessage()
        return bool(msg_str and msg_str.strip())


def setup_logging():
    """Configure 3-tier logging: compact terminal, JSON forensic file, verbose debug file.
    Idempotent — safe to call multiple times."""
    if getattr(logging.root, '_surveillance_logging_configured', False):
        return
    logging.root._surveillance_logging_configured = True

    LOG_DIR.mkdir(exist_ok=True)

    # ── Console handler — INFO+, compact terminal output ──
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.addFilter(_BlankFilter())
    console.setFormatter(structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            CompactTerminalRenderer(),
        ],
    ))
    logging.root.addHandler(console)

    # ── JSON file handler — DEBUG+, full structured JSON ──
    json_handler = logging.handlers.RotatingFileHandler(
        LOG_DIR / "surveillance.jsonl",
        maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8",
    )
    json_handler.setLevel(logging.DEBUG)
    json_handler.setFormatter(structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            JSONFileRenderer(),
        ],
    ))
    logging.root.addHandler(json_handler)

    # ── Debug verbose file — DEBUG+, full context ──
    debug_handler = logging.handlers.RotatingFileHandler(
        LOG_DIR / "surveillance.debug.log",
        maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8",
    )
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.dev.ConsoleRenderer(),
        ],
    ))
    logging.root.addHandler(debug_handler)

    # Silence noisy PyMongo/Motor driver logs
    for name in ("pymongo", "pymongo.topology", "pymongo.pool",
                 "pymongo.command", "pymongo.server"):
        logging.getLogger(name).setLevel(logging.WARNING)

    # Silence InsightFace model loading spam (keep warnings)
    for name in ("insightface", "insightface.utils", "insightface.utils.face_align"):
        logging.getLogger(name).setLevel(logging.WARNING)

    # Silence Cloudinary connection pool warning
    logging.getLogger("cloudinary").setLevel(logging.WARNING)

    logging.root.setLevel(logging.DEBUG)

    def add_byte_track_id(logger, method_name, event_dict):
        track_id = event_dict.get("track_id")
        if isinstance(track_id, str):
            try:
                event_dict["byte_track_id"] = str(int(track_id.rsplit("_", 1)[-1]))
            except (ValueError, IndexError):
                event_dict["byte_track_id"] = "?"
        return event_dict

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            add_byte_track_id,
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

logger = structlog.get_logger("config")


def _log_effective_settings():
    """Print effective value and source for every tunable setting at startup."""
    log = structlog.get_logger("config")
    tunables = [
        ("MATCH_THRESHOLD", "MATCH_THRESHOLD", 0.45, float),
        ("PERSON_CONF_THRESHOLD", "PERSON_CONF_THRESHOLD", 0.40, float),
        ("FRAME_WIDTH", "FRAME_WIDTH", 1280, int),
        ("FRAME_HEIGHT", "FRAME_HEIGHT", 720, int),
        ("FRAME_SKIP", "FRAME_SKIP", 2, int),
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

    # Quality weight validation — use already-loaded constants
    for name, val in [("QUALITY_WEIGHT_BLUR", QUALITY_WEIGHT_BLUR),
                      ("QUALITY_WEIGHT_BRIGHT", QUALITY_WEIGHT_BRIGHT),
                      ("QUALITY_WEIGHT_AREA", QUALITY_WEIGHT_AREA)]:
        if not (0 <= val <= 1):
            errors.append(f"{name} must be between 0 and 1")
    total = QUALITY_WEIGHT_BLUR + QUALITY_WEIGHT_BRIGHT + QUALITY_WEIGHT_AREA
    if abs(total - 1.0) > 1e-6:
        logger.warning("quality_weights_sum_invalid", total=round(total, 3))

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
try:
    SMTP_PORT = int(os.getenv("SMTP_PORT") or "587")
except (ValueError, TypeError):
    logger.warning("invalid_smtp_port", value=os.getenv("SMTP_PORT"), fallback=587)
    SMTP_PORT = 587
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
ALERT_EMAIL_TO = os.getenv("ALERT_EMAIL_TO", "")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM = os.getenv("TWILIO_FROM", "")
ALERT_SMS_TO = os.getenv("ALERT_SMS_TO", "")

# ── Model / Pipeline ───────────────────────────────────────────
# YOLO_MODEL: PyTorch (.pt) or OpenVINO IR (_openvino_model/) path.
# YOLO_DEVICE: Device string passed to Ultralytics model.track().
#   For PyTorch models: "cpu" or CUDA device (e.g. "0").
#   For OpenVINO IR models: must be "cpu" (Ultralytics requirement).
#   The actual OpenVINO accelerator is controlled by OPENVINO_DEVICE below.
# Export:  yolo export model=models/yolov8s.pt format=openvino half=True
# Override via env:  $env:YOLO_MODEL="models/yolov8s.pt"; $env:YOLO_DEVICE="cpu"
YOLO_MODEL = _get("YOLO_MODEL", "YOLO_MODEL", "models/yolov8s_openvino_model/")
YOLO_DEVICE = _get("YOLO_DEVICE", "YOLO_DEVICE", "cpu")
INSIGHTFACE_MODEL = _get("INSIGHTFACE_MODEL", "INSIGHTFACE_MODEL", "buffalo_l")
INSIGHTFACE_DET_SIZE = _get("INSIGHTFACE_DET_SIZE", "INSIGHTFACE_DET_SIZE", 1280, int)
INSIGHTFACE_PROVIDER = _get("INSIGHTFACE_PROVIDER", "INSIGHTFACE_PROVIDER", "CPUExecutionProvider")

# OPENVINO_DEVICE: OpenVINO accelerator when using an OpenVINO IR model.
#   Valid values: "CPU", "GPU", "NPU", "AUTO" (let OpenVINO decide).
#   Only applies when YOLO_MODEL points to an OpenVINO export.
#   Override via env var or .env:  OPENVINO_DEVICE=GPU
OPENVINO_DEVICE = _get("OPENVINO_DEVICE", "OPENVINO_DEVICE", "GPU")

# ── Detection / Tracking ───────────────────────────────────────
PERSON_CONF_THRESHOLD = _get("PERSON_CONF_THRESHOLD", "PERSON_CONF_THRESHOLD", 0.40, float)
TRACK_TIMEOUT_SECS = _get("TRACK_TIMEOUT_SECS", "TRACK_TIMEOUT_SECS", 3.0, float)
MAX_TRACK_SECS = _get("MAX_TRACK_SECS", "MAX_TRACK_SECS", 300, float)
DET_SCORE_MIN = _get("DET_SCORE_MIN", "DET_SCORE_MIN", 0.40, float)
DET_SCORE_RELAXED = _get("DET_SCORE_RELAXED", "DET_SCORE_RELAXED", 0.20, float)
EMBEDDING_DET_SCORE_MIN = _get("EMBEDDING_DET_SCORE_MIN", "EMBEDDING_DET_SCORE_MIN", 0.40, float)

# ── Progressive Recognition ────────────────────────────────────
RECOGNITION_INTERVAL_FRAMES = _get("RECOGNITION_INTERVAL_FRAMES", "RECOGNITION_INTERVAL_FRAMES", 20, int)
MIN_QUALITY_IMPROVEMENT = _get("MIN_QUALITY_IMPROVEMENT", "MIN_QUALITY_IMPROVEMENT", 0.10, float)
RESCAN_INTERVAL_SECS = _get("RESCAN_INTERVAL_SECS", "RESCAN_INTERVAL_SECS", 3, int)
MAX_RESCAN_ATTEMPTS = _get("MAX_RESCAN_ATTEMPTS", "MAX_RESCAN_ATTEMPTS", 3, int)

# ── Quality — Validity gates ──────────────────────────────────
QUALITY_VALID_BLUR_MIN = _get("QUALITY_VALID_BLUR_MIN", "QUALITY_VALID_BLUR_MIN", 40.0, float)
QUALITY_VALID_BRIGHTNESS_MIN = _get("QUALITY_VALID_BRIGHTNESS_MIN", "QUALITY_VALID_BRIGHTNESS_MIN", 35.0, float)
QUALITY_VALID_BRIGHTNESS_MAX = _get("QUALITY_VALID_BRIGHTNESS_MAX", "QUALITY_VALID_BRIGHTNESS_MAX", 255.0, float)
QUALITY_VALID_FACE_AREA_MIN = _get("QUALITY_VALID_FACE_AREA_MIN", "QUALITY_VALID_FACE_AREA_MIN", 1200.0, float)
REGISTRATION_QUALITY_MIN = _get("REGISTRATION_QUALITY_MIN", "REGISTRATION_QUALITY_MIN", 0.6, float)

# ── Quality — Scoring normalization ───────────────────────────
QUALITY_BLUR_MIN = _get("QUALITY_BLUR_MIN", "QUALITY_BLUR_MIN", 40.0, float)
QUALITY_BLUR_MAX = _get("QUALITY_BLUR_MAX", "QUALITY_BLUR_MAX", 350.0, float)
QUALITY_BRIGHTNESS_CENTER = _get("QUALITY_BRIGHTNESS_CENTER", "QUALITY_BRIGHTNESS_CENTER", 145.0, float)
QUALITY_BRIGHTNESS_RADIUS = _get("QUALITY_BRIGHTNESS_RADIUS", "QUALITY_BRIGHTNESS_RADIUS", 110.0, float)
QUALITY_FACE_AREA_MIN = _get("QUALITY_FACE_AREA_MIN", "QUALITY_FACE_AREA_MIN", 1500.0, float)
QUALITY_AREA_MAX = _get("QUALITY_AREA_MAX", "QUALITY_AREA_MAX", 10000.0, float)
QUALITY_WEIGHT_BLUR = _get("QUALITY_WEIGHT_BLUR", "QUALITY_WEIGHT_BLUR", 0.50, float)
QUALITY_WEIGHT_BRIGHT = _get("QUALITY_WEIGHT_BRIGHT", "QUALITY_WEIGHT_BRIGHT", 0.25, float)
QUALITY_WEIGHT_AREA = _get("QUALITY_WEIGHT_AREA", "QUALITY_WEIGHT_AREA", 0.25, float)

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

# ── Confidence Formula Weights ─────────────────────────────────
WEIGHT_SIMILARITY = _get("WEIGHT_SIMILARITY", "WEIGHT_SIMILARITY", 0.65, float)
WEIGHT_QUALITY = _get("WEIGHT_QUALITY", "WEIGHT_QUALITY", 0.15, float)
WEIGHT_TRACK = _get("WEIGHT_TRACK", "WEIGHT_TRACK", 0.10, float)
WEIGHT_MEMORY = _get("WEIGHT_MEMORY", "WEIGHT_MEMORY", 0.05, float)
WEIGHT_MARGIN = _get("WEIGHT_MARGIN", "WEIGHT_MARGIN", 0.05, float)

# ── Confidence Formula Normalization ──────────────────────────
SIM_NORM_MIN = _get("SIM_NORM_MIN", "SIM_NORM_MIN", 0.25, float)
SIM_NORM_MAX = _get("SIM_NORM_MAX", "SIM_NORM_MAX", 0.80, float)
TRACK_SATURATION_SECS = _get("TRACK_SATURATION_SECS", "TRACK_SATURATION_SECS", 1.5, float)
MEMORY_NORM_MAX = _get("MEMORY_NORM_MAX", "MEMORY_NORM_MAX", 20.0, float)
MARGIN_NORM_MAX = _get("MARGIN_NORM_MAX", "MARGIN_NORM_MAX", 0.30, float)
MASK_PENALTY_MAX = _get("MASK_PENALTY_MAX", "MASK_PENALTY_MAX", 0.15, float)
DEFAULT_FACE_QUALITY = _get("DEFAULT_FACE_QUALITY", "DEFAULT_FACE_QUALITY", 0.50, float)

# ── Confidence Status Tiers ───────────────────────────────────
CONFIDENCE_KNOWN_MIN = _get("CONFIDENCE_KNOWN_MIN", "CONFIDENCE_KNOWN_MIN", 70, int)
CONFIDENCE_UNCERTAIN_MIN = _get("CONFIDENCE_UNCERTAIN_MIN", "CONFIDENCE_UNCERTAIN_MIN", 55, int)

# ── Calculation Log ───────────────────────────────────────────
ENABLE_CALC_LOG = _get("ENABLE_CALC_LOG", "ENABLE_CALC_LOG", False, bool)
CALC_LOG_MAX_SIZE_MB = _get("CALC_LOG_MAX_SIZE_MB", "CALC_LOG_MAX_SIZE_MB", 10, int)

# ── Debug Flags ─────────────────────────────────────────────
DEBUG_RECOGNITION = _get("DEBUG_RECOGNITION", "DEBUG_RECOGNITION", False, bool)
DEBUG_FACE_CROPS = _get("DEBUG_FACE_CROPS", "DEBUG_FACE_CROPS", False, bool)
DEBUG_DUPLICATE_BOXES = _get("DEBUG_DUPLICATE_BOXES", "DEBUG_DUPLICATE_BOXES", False, bool)
PERFORMANCE_STATS = _get("PERFORMANCE_STATS", "PERFORMANCE_STATS", True, bool)

# ── CLAHE ──────────────────────────────────────────────────────
CLAHE_CLIP_LIMIT = _get("CLAHE_CLIP_LIMIT", "CLAHE_CLIP_LIMIT", 2.0, float)
CLAHE_TILE_SIZE = _get("CLAHE_TILE_SIZE", "CLAHE_TILE_SIZE", 8, int)

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
FRAME_SKIP = _get("FRAME_SKIP", "FRAME_SKIP", 2, int)
CAMERA_ID = _get("CAMERA_ID", "CAMERA_ID", "cam_01")
# CAMERA_BACKEND: Video capture backend — "dshow" (DirectShow), "msmf", or "" (auto).
# Use "dshow" on Windows if MSMF drops frames (error -1072875772).
CAMERA_BACKEND = _get("CAMERA_BACKEND", "CAMERA_BACKEND", "")

# ── LLM (Ollama) ──────────────────────────────────────────────
OLLAMA_URL = _get("OLLAMA_URL", "OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = _get("OLLAMA_MODEL", "OLLAMA_MODEL", "gemma3:4b")
OLLAMA_TIMEOUT = _get("OLLAMA_TIMEOUT", "OLLAMA_TIMEOUT", 30, int)
