# cleanup.ps1 — Delete captured images, logs, and build artifacts
# Usage: .\cleanup.ps1 [-DryRun]

param(
    [switch]$DryRun
)

$ErrorActionPreference = "SilentlyContinue"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

function Remove-Path {
    param([string]$Path)
    if (-not (Test-Path $Path)) { return }
    if ($DryRun) {
        Write-Host "  [dry-run] would delete: $Path" -ForegroundColor Yellow
    } else {
        Remove-Item -Path $Path -Recurse -Force
        Write-Host "  deleted: $Path" -ForegroundColor Green
    }
}

Write-Host "=== Cleaning captures/ ===" -ForegroundColor Cyan
Remove-Path "$ProjectRoot\captures"

Write-Host "=== Cleaning logs/ ===" -ForegroundColor Cyan
Remove-Path "$ProjectRoot\logs"

Write-Host "=== Cleaning build artifacts ===" -ForegroundColor Cyan
Remove-Path "$ProjectRoot\dashboard\frontend\dist"
Remove-Path "$ProjectRoot\dashboard\frontend\node_modules\.vite"
Remove-Path "$ProjectRoot\__pycache__"
Remove-Path "$ProjectRoot\.pytest_cache"

Write-Host ""
if ($DryRun) {
    Write-Host "[DRY RUN] No files were deleted. Remove -DryRun to execute." -ForegroundColor Yellow
} else {
    Write-Host "Done. Captures, logs, and build artifacts removed." -ForegroundColor Green
}
