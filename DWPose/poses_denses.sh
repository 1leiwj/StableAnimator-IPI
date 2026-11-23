#!/bin/bash
# /gaojiawei/wjl/StableAnimator/DWPose/batch_process_from_txt.sh

BASE_DIR="/gaojiawei/wjl/StableAnimator"
SCRIPT_PATH="${BASE_DIR}/DWPose/poses_dense.py"
TXT_FILE="${BASE_DIR}/basic_infer/ces/1.txt"
LOG_FILE="${BASE_DIR}/animation_data/batch_process.log"

echo "Starting batch processing at $(date)" > $LOG_FILE

if [ ! -f "$TXT_FILE" ]; then
    echo "❌ Error: $TXT_FILE not found!" | tee -a $LOG_FILE
    exit 1
fi

echo "📖 Reading video paths from $TXT_FILE" | tee -a $LOG_FILE

total=0
success=0
failed=0

while IFS= read -r line || [ -n "$line" ]; do
    if [[ -z "$line" || "$line" =~ ^[0-9]+$ ]]; then
        continue
    fi
    
    video_path=$(echo "$line" | xargs)
    
    if [ ! -d "$video_path" ]; then
        echo "⚠️  Warning: $video_path does not exist, skipping..." | tee -a $LOG_FILE
        continue
    fi
    
    total=$((total + 1))
    echo "🔄 [$total] Processing: $video_path" | tee -a $LOG_FILE
    
    IMAGES_DIR="${video_path}/images"
    OUTPUT_NPY="${video_path}/poses_dense133.npy"
    
    if [ ! -d "$IMAGES_DIR" ]; then
        echo "❌ Images folder not found: $IMAGES_DIR" | tee -a $LOG_FILE
        failed=$((failed + 1))
        continue
    fi
    
    # 找到第一张图片（只用来找images文件夹是否存在）
    FIRST_IMAGE=$(ls ${IMAGES_DIR}/*.png 2>/dev/null | head -1)
    
    if [ ! -f "$FIRST_IMAGE" ]; then
        echo "❌ No PNG images found in $IMAGES_DIR" | tee -a $LOG_FILE
        failed=$((failed + 1))
        continue
    fi
    
    # =========================================================
    # 🚀 核心修改：如果存在旧的 NPY 文件，不跳过，而是删除并准备覆盖！
    # =========================================================
    if [ -f "$OUTPUT_NPY" ]; then
        echo "⚠️  poses_dense.npy already exists, deleting old file and overwriting..." | tee -a $LOG_FILE
        rm -f "$OUTPUT_NPY"  # 物理删除旧毒数据，保证万无一失
    fi
    
    echo "   Extracting poses to: $OUTPUT_NPY" | tee -a $LOG_FILE
    
    # 关键修改：去掉 --ref_image_path 参数！
    python ${SCRIPT_PATH} \
        --target_image_folder_path="${IMAGES_DIR}" \
        --output_npy_path="${OUTPUT_NPY}" 2>&1 | tee -a $LOG_FILE
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "✅ Success: ${video_path}" | tee -a $LOG_FILE
        success=$((success + 1))
    else
        echo "❌ Failed: ${video_path}" | tee -a $LOG_FILE
        failed=$((failed + 1))
    fi
    
    echo "-----------------------------------" | tee -a $LOG_FILE
    
done < "$TXT_FILE"

echo -e "\n📊 ========== SUMMARY ==========" | tee -a $LOG_FILE
echo "Total videos processed: $total" | tee -a $LOG_FILE
echo "✅ Successful: $success" | tee -a $LOG_FILE
echo "❌ Failed: $failed" | tee -a $LOG_FILE
echo "📝 Log file: $LOG_FILE" | tee -a $LOG_FILE
echo "🎉 Done at $(date)" | tee -a $LOG_FILE