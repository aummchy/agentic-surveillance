/**
 * Status constants — must match config/status.py Status(IntEnum) exactly.
 * Backend returns integers; frontend uses these for comparisons.
 */
export const Status = Object.freeze({
  UNKNOWN: 1,
  UNCERTAIN: 2,
  KNOWN: 3,
  KNOWN_VISITOR: 4,
  VERIFIED: 5,
  AUTHORIZED: 6,
  BLACKLIST: 7,
  MASKED_UNKNOWN: 8,
  HIDDEN: 9,
})

/** Map status int → display label. */
export const STATUS_LABELS = {
  [Status.UNKNOWN]: 'Unverified',
  [Status.UNCERTAIN]: 'Uncertain',
  [Status.KNOWN]: 'Known',
  [Status.KNOWN_VISITOR]: 'Known Visitor',
  [Status.VERIFIED]: 'Verified',
  [Status.AUTHORIZED]: 'Authorized',
  [Status.BLACKLIST]: 'Blacklist',
  [Status.MASKED_UNKNOWN]: 'Masked Unknown',
  [Status.HIDDEN]: 'Hidden',
}
