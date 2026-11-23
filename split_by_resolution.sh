#!/bin/bash
set -e

# 基础路径
RAW_DIR="/gaojiawei/wjl/dataset/champ/rawdata"
OUT_DIR="/gaojiawei/wjl/dataset/champ/sorted"

# 输出文件夹
mkdir -p "${OUT_DIR}/512x512"
mkdir -p "${OUT_DIR}/576x1024"
mkdir -p "${OUT_DIR}/others"

# 遍历 1~479 视频
for i in $(seq -w 0001 0479); do
    if [ -f "${RAW_DIR}/${i}.mp4" ]; then
        VIDEO="${RAW_DIR}/${i}.mp4"
    elif [ -f "${RAW_DIR}/${i}.MP4" ]; then
        VIDEO="${RAW_DIR}/${i}.MP4"
    else
        echo "❌ 文件不存在: ${i}.mp4 或 ${i}.MP4"
        continue
    fi

    # 获取分辨率
    RES=$(ffprobe -v error -select_streams v:0 \
          -show_entries stream=width,height \
          -of csv=s=x:p=0 "$VIDEO")

    case "$RES" in
        512x512)
            echo "[$i] $VIDEO -> 512x512"
            cp "$VIDEO" "${OUT_DIR}/512x512/"
            ;;
        576x1024)
            echo "[$i] $VIDEO -> 576x1024"
            cp "$VIDEO" "${OUT_DIR}/576x1024/"
            ;;
        *)
            echo "[$i] $VIDEO -> others ($RES)"
            cp "$VIDEO" "${OUT_DIR}/others/"
            ;;
    esac
done

echo "✅ 分流完成（保留原始文件）：结果在 ${OUT_DIR}"

# 统计结果
echo ""
echo "📊 视频数量统计："
echo "512x512   : $(ls -1 ${OUT_DIR}/512x512 | wc -l)"
echo "576x1024  : $(ls -1 ${OUT_DIR}/576x1024 | wc -l)"
echo "others    : $(ls -1 ${OUT_DIR}/others | wc -l)"
