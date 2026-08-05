"""Centralized status definitions — single source of truth.

All status values are integers (IntEnum). Higher = more trusted.
Maps to/from display strings for logs and API.

Usage:
    from config.status import Status, STATUS_LABELS
    if decision.status >= Status.KNOWN:
        ...
    print(STATUS_LABELS[decision.status])  # "known"
"""

from enum import IntEnum


class Status(IntEnum):
    """Numeric status bands. Higher value = more trusted."""
    UNKNOWN = 1           # No match or low confidence
    UNCERTAIN = 2         # Matched but mid-range confidence
    KNOWN = 3             # Matched with high confidence
    KNOWN_VISITOR = 4     # Auto-registered self-match or memory-confirmed
    VERIFIED = 5          # Operator-verified
    AUTHORIZED = 6        # Authorized person
    BLACKLIST = 7         # Blacklisted person
    MASKED_UNKNOWN = 8    # Masked/partial visibility unknown
    HIDDEN = 9            # Intentionally hidden


# Display labels for logs, API, and frontend
STATUS_LABELS: dict[int, str] = {
    Status.UNKNOWN: "unknown",
    Status.UNCERTAIN: "uncertain",
    Status.KNOWN: "known",
    Status.KNOWN_VISITOR: "known_visitor",
    Status.VERIFIED: "verified",
    Status.AUTHORIZED: "authorized",
    Status.BLACKLIST: "blacklist",
    Status.MASKED_UNKNOWN: "masked_unknown",
    Status.HIDDEN: "intentionally_hidden",
}

# Reverse lookup: string label → Status enum
LABEL_TO_STATUS: dict[str, Status] = {v: k for k, v in STATUS_LABELS.items()}

# Statuses that resolve a track — skip further recognition
RESOLVED_STATUSES: frozenset[int] = frozenset({
    Status.VERIFIED, Status.KNOWN_VISITOR, Status.AUTHORIZED,
})

# Statuses that trigger per-track alert dedup
UNVERIFIED_STATUSES: frozenset[int] = frozenset({
    Status.UNKNOWN, Status.MASKED_UNKNOWN, Status.UNCERTAIN, Status.HIDDEN,
})

# Threshold: status >= this value means "known" for memory is_known logic
IS_KNOWN_THRESHOLD: int = Status.KNOWN
