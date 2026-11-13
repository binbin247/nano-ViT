#!/usr/bin/env bash

set -euo pipefail

if [[ $# -gt 2 ]]; then
    echo "用法: $0 [SSH主机别名] [远程目录]"
    echo "示例: $0"
    exit 1
fi

# 第一个参数作为 SSH 主机名；没有传参数时默认使用 L40s-GPU
remote_host="${1:-L40s-GPU}"

# 第二个参数作为服务器目录；没有传参数时使用默认目录
remote_dir="${2:-/data/niebinbin/Study/ViT/}"

# 获取当前脚本所在的本地项目目录
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

rsync -avz \
    -e "ssh -T -o RequestTTY=no -o RemoteCommand=none" \
    --exclude ".git" \
    --exclude ".venv" \
    --exclude "__pycache__" \
    --exclude "data" \
    --exclude "sync_to_server.sh" \
    --exclude "sync_from_server.sh" \
    "${project_dir}/" \
    "${remote_host}:${remote_dir}"
