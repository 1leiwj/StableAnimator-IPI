#!/bin/bash

"""
BASE_DIR="/gaojiawei/wjl/StableAnimator/animation_data"

# 先清空输出文件
> "${BASE_DIR}/video_rec_path.txt"

# 遍历 vec 下的所有文件夹
for d in "${BASE_DIR}/rec/"*; do
    if [ -d "$d" ]; then
        name=$(basename "$d")
        # 判断是否是五位数字，且在 00000~01200 范围内
        if [[ $name =~ ^[0-9]{5}$ ]] && [ "$name" -ge 0 ] && [ "$name" -le 1200 ]; then
            echo "$d" >> "${BASE_DIR}/video_rec_path.txt"
        fi
    fi
done

echo "路径列表已生成：${BASE_DIR}/video_rec_path.txt"



"""
#!/bin/bash

# data_bec_clay.sh - 针对 20000 编号起的泥塑视频预处理

# ================= 1. 路径配置区 =================
BASE_DIR="/gaojiawei/wjl"

# 你上一步生成的泥塑视频所在的文件夹
INPUT_VIDEOS_DIR="/gaojiawei/wjl/StableAnimator/basic_infer/ces" 

# 根据你的截图，目标根目录是 animation_data
ANIMATION_DIR="/gaojiawei/wjl/StableAnimator/basic_infer/ces/2"
# 你想放入的子文件夹名 (bec)
TARGET_DIR_NAME="ceshi"
# 最终的输出总路径: /gaojiawei/wjl/StableAnimator/animation_data/bec
OUTPUT_ROOT="${ANIMATION_DIR}/${TARGET_DIR_NAME}"

# ================= 2. 编号配置区 =================
# 视频从 20000 开始。请把 END_IDX 改成你实际生成的最后一个视频的编号！
# 比如你生成了 100 个视频，就把 END_IDX 改成 20099
START_IDX=30555
END_IDX=30555


# =================================================

# 步骤1: 提取视频帧到 bec 文件夹
step1_extract_frames() {
    echo "步骤1: 提取视频帧"
    for ((i=START_IDX; i<=END_IDX; i++)); do
        video_name=$(printf "%05d.mp4" $i)
        video_path="${INPUT_VIDEOS_DIR}/${video_name}"
        # 输出路径: animation_data/bec/20000/images
        output_dir="${OUTPUT_ROOT}/$(printf "%05d" $i)/images"

        # 检查文件是否存在
        if [ ! -f "${video_path}" ]; then
            echo "跳过: ${video_name} (视频文件不存在)"
            continue
        fi

        mkdir -p "${output_dir}"

        # 抽帧命令
        ffmpeg -i "${video_path}" \
               -q:v 1 \
               -start_number 0 \
               -y \
               "${output_dir}/frame_%d.png" 2>/dev/null

        echo "提取完成: ${video_name} -> ${output_dir}"
    done
}

# 步骤2: 运行姿态提取 (生成 poses_dense.npy)
step2_pose_extraction() {
    echo "步骤2: 运行姿态提取"
    cd "${BASE_DIR}/StableAnimator"
    
    # 这里的参数匹配原作者代码: root_path + name 拼接出完整的 bec 目录
    python DWPose/training_skeleton_extraction.py \
        --root_path="${ANIMATION_DIR}" \
        --name="${TARGET_DIR_NAME}" \
        --start=${START_IDX} \
        --end=${END_IDX}
}

# 步骤3: 运行面部特征提取
step3_face_extraction() {
    echo "步骤3: 运行面部特征提取"
    cd "${BASE_DIR}/StableAnimator"
    for ((i=START_IDX; i<=END_IDX; i++)); do
        image_folder="${OUTPUT_ROOT}/$(printf "%05d" $i)/images"
        if [ -d "${image_folder}" ]; then
            python face_mask_extraction.py --image_folder="${image_folder}"
        else
            echo "跳过: ${image_folder} (目录不存在)"
        fi
    done
}

# 显示菜单
echo "请选择要执行的操作 (目标目录: bec, 编号: ${START_IDX}-${END_IDX}):"
echo "1) 仅提取视频帧 (将 clay_videos 抽帧到 bec)"
echo "2) 仅运行姿态提取 (生成骨骼 npy)"
echo "3) 仅运行面部识别"
echo "4) 执行完整流程 (1 -> 2 -> 3 自动连跑)"
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