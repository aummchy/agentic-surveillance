"""3-tier logging configuration: compact terminal, JSON forensic file, verbose debug file."""

import json
import logging
import logging.handlers
import structlog
from pathlib import Path
from config.status import Status, STATUS_LABELS

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

    # Status colors — keyed by Status int value
    STATUS = {
        Status.KNOWN: GREEN,
        Status.VERIFIED: GREEN,
        Status.AUTHORIZED: GREEN,
        Status.KNOWN_VISITOR: GREEN,
        Status.UNCERTAIN: YELLOW,
        Status.UNKNOWN: RED,
        Status.BLACKLIST: RED,
        Status.MASKED_UNKNOWN: RED,
        Status.HIDDEN: RED,
    }


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
        if isinstance(status, int):
            status_label = STATUS_LABELS.get(status, str(status))
            d["status"] = status_label
            status_color = Colors.STATUS.get(status, Colors.WHITE)
            d["_status"] = f"{status_color}{status_label:<12}{Colors.RESET}"
        elif isinstance(status, str):
            status = status.upper()
            d["status"] = status
            d["_status"] = f"{Colors.WHITE}{status:<12}{Colors.RESET}"
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
                name = "unknown" if isinstance(status_val, int) and status_val == Status.UNKNOWN else (d.get("person_id") or "???")[:8]
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
        msg = getattr(record, 'msg', None)
        event = msg.get('event', '') if isinstance(msg, dict) else None
        if event and event not in TERMINAL_ALLOWLIST and record.levelno < logging.WARNING:
            return False
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
                parts = track_id.split("_")
                if len(parts) >= 4:
                    event_dict["byte_track_id"] = str(int(parts[-2]))
                elif len(parts) >= 3:
                    event_dict["byte_track_id"] = str(int(parts[-1]))
                else:
                    event_dict["byte_track_id"] = track_id
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
