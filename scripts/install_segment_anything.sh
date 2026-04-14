#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible wrapper: the repository now uses SAM2 instead of segment_anything.
exec "$(dirname "$0")/install_sam2.sh" "$@"
