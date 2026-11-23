
#!/usr/bin/env bash
set -euo pipefail

SOURCE="/gaojiawei/wjl/StableAnimator/"
TARGET="/gaojiawei/wjl/StableAnimator-GitHub/"

echo "========== 同步 StableAnimator 代码 =========="

rsync -av \
  --exclude='.git/' \
  --exclude='.venv/' \
  --exclude='venv/' \
  --exclude='animator/' \
  --exclude='animate/' \
  --exclude='checkpoints/' \
  --exclude='logs/' \
  --exclude='outputs/' \
  --exclude='output/' \
  --exclude='animation_data/' \
  --exclude='aligns/' \
  --exclude='basic_infer/' \
  --exclude='.gradio/' \
  --exclude='__pycache__/' \
  --exclude='*.png' \
  --exclude='*.jpg' \
  --exclude='*.jpeg' \
  --exclude='*.gif' \
  --exclude='*.mp4' \
  --exclude='*.avi' \
  --exclude='*.pth' \
  --exclude='*.pt' \
  --exclude='*.ckpt' \
  --exclude='*.safetensors' \
  --exclude='*.npy' \
  --exclude='*.npz' \
  --exclude='*.zip' \
  --exclude='*.env' \
  --exclude='.env' \
  --exclude='evaluation/metrics/syncnet_python/' \
  "$SOURCE" "$TARGET"

cd "$TARGET"

echo "========== 查看代码变化 =========="
git status --short
git diff --stat

echo
echo "同步完成。请检查文件变化，确认没有数据集、模型权重或敏感信息。"
echo "确认无误后，手动执行："
echo "  git add -A"
echo "  git diff --cached --stat"
echo "  git commit -m 'Update StableAnimator IPI'"
echo "  git push"
