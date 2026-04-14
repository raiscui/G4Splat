#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/build_env_common.sh"

if python_import_ok "sam2"; then
  echo "[install-sam2] 检测到可导入的 sam2, 跳过重复安装"
  exit 0
fi

repo_tmp="$(mktemp -d)"
cleanup() {
  rm -rf "$repo_tmp"
}
trap cleanup EXIT

echo "[install-sam2] 优先按官方 GitHub 源码方式安装 SAM2"
if python -m pip install --no-deps "git+https://github.com/facebookresearch/sam2.git"; then
  exit 0
fi

echo "[install-sam2] GitHub git 安装失败，回退到官方源码 zip 安装"
zip_path="$repo_tmp/sam2-main.zip"
if curl -L --retry 4 --retry-delay 3 --max-time 240 \
  https://codeload.github.com/facebookresearch/sam2/zip/refs/heads/main \
  -o "$zip_path"; then
  unzip -q "$zip_path" -d "$repo_tmp"
  python -m pip install --no-deps --no-build-isolation "$repo_tmp/sam2-main"
  exit 0
fi

echo "[install-sam2] 官方源码获取失败，最后回退到 sam2 PyPI 包"
python -m pip install --no-deps sam2
