#!/usr/bin/env bash

set -euo pipefail

if [[ $# -gt 2 ]]; then
    echo "用法: $0 [SSH主机别名] [远程目录]"
    echo "示例: $0"
    exit 1
fi

# 第一个参数：SSH 主机别名
# 未传入时默认使用 L40s-GPU
remote_host="${1:-L40s-GPU}"

# 第二个参数：服务器上的项目目录
remote_dir="${2:-/data/niebinbin/Study/ViT/}"

# 当前脚本所在的本地项目目录
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

rsync -avz \
    -e "ssh -T -o RequestTTY=no -o RemoteCommand=none" \
    --exclude '/.git/' \
    --exclude '/.venv/' \
    --exclude '/data/' \
    --exclude '/checkpoints/***' \
    --exclude '*/checkpoints/***' \
    --exclude '__pycache__/' \
    --exclude '*.py[cod]' \
    --exclude '.pytest_cache/' \
    --exclude '.mypy_cache/' \
    --exclude '.ruff_cache/' \
    --exclude '.tox/' \
    --exclude '.nox/' \
    --exclude '.ipynb_checkpoints/' \
    --exclude '.cache/' \
    --exclude '.idea/' \
    --exclude '.DS_Store' \
    --exclude '*.tmp' \
    --exclude '*.swp' \
    --exclude '*.swo' \
    --exclude '*~' \
    --exclude '*.pt' \
    --exclude '*.pth' \
    --exclude '*.ckpt' \
    --exclude '*.safetensors' \
    --exclude '*.onnx' \
    --exclude 'sync_to_server.sh' \
    --exclude 'sync_from_server.sh' \
    "${remote_host}:${remote_dir}" \
    "${project_dir}/"