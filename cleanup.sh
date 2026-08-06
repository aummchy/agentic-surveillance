#!/bin/bash
# cleanup.sh — Delete captured images, logs, and build artifacts
# Usage: ./cleanup.sh [--dry-run]

set -euo pipefail

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "[DRY RUN] No files will be deleted."
fi

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

delete_path() {
    local path="$1"
    if [[ ! -e "$path" ]]; then
        return
    fi
    if $DRY_RUN; then
        echo "  [dry-run] would delete: $path"
    else
        rm -rf "$path"
        echo "  deleted: $path"
    fi
}

echo "=== Cleaning captures/ ==="
delete_path "$PROJECT_ROOT/captures"

echo "=== Cleaning logs/ ==="
delete_path "$PROJECT_ROOT/logs"

echo "=== Cleaning build artifacts ==="
delete_path "$PROJECT_ROOT/dashboard/frontend/dist"
delete_path "$PROJECT_ROOT/dashboard/frontend/node_modules/.vite"
delete_path "$PROJECT_ROOT/__pycache__"
delete_path "$PROJECT_ROOT/.pytest_cache"

echo ""
if $DRY_RUN; then
    echo "[DRY RUN] No files were deleted. Remove --dry-run to execute."
else
    echo "Done. Captures, logs, and build artifacts removed."
fi
