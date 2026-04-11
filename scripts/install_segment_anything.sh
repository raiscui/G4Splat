#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/build_env_common.sh"

if python_import_ok "segment_anything"; then
  echo "[install-segment-anything] 检测到可导入的 segment_anything, 跳过重复安装"
  exit 0
fi

echo "[install-segment-anything] 从官方 GitHub 安装 segment-anything"
python -m pip install --no-deps "git+https://github.com/facebookresearch/segment-anything.git"
