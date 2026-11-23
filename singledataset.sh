#!/bin/bash

# ==========================================
# 纯粹的数据预处理脚本 (全自动，无需参考图)
# ==========================================
# bash preprocess.sh /gaojiawei/wjl/StableAnimator/datavideo/01461.mp4 /gaojiawei/wjl/StableAnimator/inference/case-00002
# 1. 接收运行参数
VIDEO_PATH=$1
OUTPUT_DIR=$2
SA_DIR="/gaojiawei/wjl/StableAnimator"

# 2. 检查参数是否给全了
if [ -z "$VIDEO_PATH" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "❌ 错误: 参数缺失！"
    echo "💡 用法: bash preprocess.sh <视频绝对路径> <输出文件夹路径>"
    echo "🎯 示例: bash preprocess.sh /root/video.mp4 /root/my_dataset/00001"
    exit 1
fi

if [ ! -f "$VIDEO_PATH" ]; then echo "❌ 找不到视频: $VIDEO_PATH"; exit 1; fi

# 3. 定义子目录
IMAGES_DIR="${OUTPUT_DIR}/images"
POSES_DIR="${OUTPUT_DIR}/poses"
OUTPUT_NPY="${OUTPUT_DIR}/poses_dense.npy"

mkdir -p "$IMAGES_DIR"
mkdir -p "$POSES_DIR"
echo "📁 目标文件夹已创建: $OUTPUT_DIR"

# ==========================================
# 步骤 1: FFmpeg 提取视频帧
# ==========================================
echo -e "\n🎥 [1/4] 开始提取视频帧..."
ffmpeg -v error -i "$VIDEO_PATH" -q:v 1 -start_number 0 -y "${IMAGES_DIR}/frame_%d.png"

# 切入工作目录
cd "$SA_DIR" || exit

# ==========================================
# 步骤 2: 提取人脸掩码
# ==========================================
echo -e "\n🧑 [2/4] 提取人脸掩模 (faces/)..."
python face_mask_extraction.py --image_folder="$IMAGES_DIR"

# ==========================================
# 步骤 3: 生成密集姿态坐标数据
# ==========================================
echo -e "\n📊 [3/4] 生成坐标矩阵 (poses_dense.npy)..."
python DWPose/poses_dense.py \
    --target_image_folder_path="$IMAGES_DIR" \
    --output_npy_path="$OUTPUT_NPY"

# ==========================================
# 步骤 4: 提取可视化骨骼图像
# ==========================================
echo -e "\n🦴 [4/4] 生成可视化人体骨骼图像 (poses/)..."
# 💡 魔法在这里：直接拿刚刚抽出来的 frame_0.png 塞给它当参考图，完美闭环！
python DWPose/skeleton_extraction.py \
    --target_image_folder_path="$IMAGES_DIR" \
    --ref_image_path="${IMAGES_DIR}/frame_0.png" \
    --poses_folder_path="$POSES_DIR"

# ==========================================
echo -e "\n🎉 所有预处理完毕！完美的数据集结构已生成在: $OUTPUT_DIR"
ls -l "$OUTPUT_DIR"