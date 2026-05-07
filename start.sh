#!/bin/bash
set -e

# 检查 uv 是否可用
if ! command -v uv &>/dev/null; then
    echo "❌ uv 未安装，请先安装: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi
echo "✓ uv $(uv --version)"

# 查找 wheel 文件
WHL=$(ls dist/govio-*.whl 2>/dev/null | head -1)
if [ -z "$WHL" ]; then
    echo "❌ 未找到 wheel 文件，请先构建: uv build"
    exit 1
fi

# 安装 govio 为 uv tool（持久化）
uv tool install --from "$WHL" govio --force
echo "✓ govio 已安装"
echo ""
echo "接下来运行: govio-cli onboard"
