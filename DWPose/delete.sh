#!/bin/bash
# delete_npy_files.sh

BASE_DIR="/gaojiawei/wjl/StableAnimator/animation_data"

echo "🔍 Searching for npy files to delete..."
echo "========================================"

# 查找所有npy文件
npy_files=$(find ${BASE_DIR} -name "poses_dense.npy" -type f)

# 计数
count=0

for file in $npy_files; do
    echo "🗑️  Deleting: $file"
    rm -f "$file"
    ((count++))
done

echo "========================================"
echo "✅ Deleted $count npy files"