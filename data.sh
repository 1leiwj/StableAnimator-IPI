#!/bin/bash

# data_step_by_step.sh - 分步执行版本 (从00501开始)

# 设置路径
BASE_DIR="/gaojiawei/wjl"
DATASET_DIR="/gaojiawei/wjl/StableAnimator/animation_data"
ANIMATION_DIR="${BASE_DIR}/1"
REC_DIR="${ANIMATION_DIR}/vec"

# 步骤1: 仅提取视频帧
step1_extract_frames() {
    echo "步骤1: 提取视频帧"
    for i in {1460..1461}; do   # 从00501开始，到01401（你有600个视频）
        video_name=$(printf "%05d.mp4" $i)
        video_path="${DATASET_DIR}/${video_name}"
        output_dir="${REC_DIR}/$(printf "%05d" $i)/images"

        # 检查文件是否存在
        if [ ! -f "${video_path}" ]; then
            echo "跳过: ${video_name} (文件不存在)"
            continue
        fi

        mkdir -p "${output_dir}"

        ffmpeg -i "${video_path}" \
               -q:v 1 \
               -start_number 0 \
               -y \
               "${output_dir}/frame_%d.png" 2>/dev/null

        echo "提取完成: ${video_name}"
    done
}

# 步骤2: 仅运行姿态提取
step2_pose_extraction() {
    echo "步骤2: 运行姿态提取"
    cd "${BASE_DIR}/StableAnimator"
    python DWPose/training_skeleton_extraction.py \
        --root_path="${ANIMATION_DIR}" \
        --name="${REC_DIR}" \
        --start=1401 \
        --end=1461
}

# 步骤3: 仅运行面部识别
step3_face_extraction() {
    echo "步骤3: 运行面部识别"
    cd "${BASE_DIR}/StableAnimator"
    for i in {1401..1461}; do
        image_folder="${REC_DIR}/$(printf "%05d" $i)/images"
        if [ -d "${image_folder}" ]; then
            python face_mask_extraction.py --image_folder="${image_folder}"
        else
            echo "跳过: ${image_folder} (目录不存在)"
        fi
    done
}
#单张图片可用：python face_mask_extraction.py --image_folder="/gaojiawei/wjl/StableAnimator/inference/case-18/images"   注意是文件夹

# 显示菜单
echo "请选择要执行的操作:"
echo "1) 仅提取视频帧"
echo "2) 仅运行姿态提取"
echo "3) 仅运行面部识别"
echo "4) 执行完整流程"
echo "5) 退出"

read -p "请输入选择 (1-5): " choice

case $choice in
    1) step1_extract_frames ;;
    2) step2_pose_extraction ;;
    3) step3_face_extraction ;;
    4) 
        step1_extract_frames
        step2_pose_extraction
        step3_face_extraction
        ;;
    5) exit 0 ;;
    *) echo "无效选择" ;;
esac


#ffmpeg -framerate 30 -i /gaojiawei/wjl/StableAnimator/basic_infer/9/frames/frame_%d.png -c:v libx264 -pix_fmt yuv420p /gaojiawei/wjl/StableAnimator/basic_infer/9/1.mp4