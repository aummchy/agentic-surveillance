"""
Migration: String statuses → Numeric Status(IntEnum) values

Maps old string statuses in MongoDB to new int values from config.status.Status.
Run once after deploying the status refactor.

Usage:
    python scripts/migrate_status_ints.py [--dry-run]

Collections affected:
    - visit_memory: last_status, best_status, status_history[].status
    - events: status
    - faces: (no status field — unaffected)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pymongo import MongoClient
from config import settings
from config.status import Status, LABEL_TO_STATUS

# Old string → new int mapping (authoritative source)
STRING_TO_INT = {label: status for label, status in LABEL_TO_STATUS.items()}
# Add intentionally_hidden → HIDDEN (9)
STRING_TO_INT["intentionally_hidden"] = 9


def migrate_collection(collection, field_path, dry_run=False):
    """Update all documents where field matches any old string status."""
    results = {"matched": 0, "updated": 0, "errors": 0}

    for old_status, new_status in STRING_TO_INT.items():
        query = {field_path: old_status}
        count = collection.count_documents(query)
        if count == 0:
            continue

        results["matched"] += count
        print(f"  {field_path}={old_status!r} -> {new_status} : {count} docs")

        if not dry_run:
            try:
                r = collection.update_many(query, {"$set": {field_path: new_status}})
                results["updated"] += r.modified_count
            except Exception as e:
                print(f"    ERROR: {e}")
                results["errors"] += 1

    return results


def migrate_status_history(collection, dry_run=False):
    """Update status_history[].status array elements."""
    results = {"matched": 0, "updated": 0, "errors": 0}

    for old_status, new_status in STRING_TO_INT.items():
        query = {"status_history.status": old_status}
        count = collection.count_documents(query)
        if count == 0:
            continue

        results["matched"] += count
        print(f"  status_history[].status={old_status!r} -> {new_status} : {count} docs")

        if not dry_run:
            try:
                r = collection.update_many(
                    query,
                    {"$set": {"status_history.$[elem].status": new_status}},
                    array_filters=[{"elem.status": old_status}],
                )
                results["updated"] += r.modified_count
            except Exception as e:
                print(f"    ERROR: {e}")
                results["errors"] += 1

    return results


def main():
    dry_run = "--dry-run" in sys.argv

    print(f"{'[DRY RUN] ' if dry_run else ''}Migrating string statuses to ints")
    print(f"MongoDB URI: {settings.MONGODB_URI[:30]}...")
    print()

    client = MongoClient(settings.MONGODB_URI)
    db = client[settings.MONGODB_DATABASE]

    total = {"matched": 0, "updated": 0, "errors": 0}

    # ── visit_memory collection ──
    print("=== visit_memory ===")
    for field in ("last_status", "best_status"):
        r = migrate_collection(db.visit_memory, field, dry_run)
        for k in total:
            total[k] += r[k]

    r = migrate_status_history(db.visit_memory, dry_run)
    for k in total:
        total[k] += r[k]

    # ── events collection ──
    print("\n=== events ===")
    r = migrate_collection(db.events, "status", dry_run)
    for k in total:
        total[k] += r[k]

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Summary:")
    print(f"  Matched: {total['matched']} documents")
    if not dry_run:
        print(f"  Updated: {total['updated']} fields")
        print(f"  Errors:  {total['errors']}")
    else:
        print("  (no changes made — run without --dry-run to apply)")

    client.close()


if __name__ == "__main__":
    main()
