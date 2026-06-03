#!/bin/bash

echo "=== 检查远程配置 ==="
git remote -v

echo ""
echo "=== 清除缓存的凭据 ==="
git config --global --unset credential.helper

echo ""
echo "=== 重新推送 ==="
git push origin main
