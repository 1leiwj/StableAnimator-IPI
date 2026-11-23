import os
import cv2
import numpy as np
import glob
import re

image_dir = "/gaojiawei/wjl/StableAnimator/basic_infer/ces/30008/images"
pose_path = "/gaojiawei/wjl/StableAnimator/basic_infer/ces/30008/poses_dense133.npy"
output_video_path = "d1.mp4"

# 检查文件是否存在
if not os.path.exists(pose_path):
    print(f"❌ 找不到骨架文件: {pose_path}")
    exit(1)

if not os.path.exists(image_dir):
    print(f"❌ 找不到图片目录: {image_dir}")
    exit(1)

# 加载骨架
poses = np.load(pose_path)
print(f"📊 骨架数据 shape: {poses.shape}")

# 获取图片列表
img_paths = glob.glob(os.path.join(image_dir, "*.png"))
img_paths.sort(key=lambda x: int(re.search(r'(\d+)', os.path.basename(x)).group(1)))

print(f"🖼️  找到 {len(img_paths)} 张图片")
print(f"🦴 骨架帧数: {poses.shape[0]}")

num_frames = min(poses.shape[0], len(img_paths))
print(f"🎬 将生成 {num_frames} 帧")

# 读取第一张图获取尺寸
first_img = cv2.imread(img_paths[0])
if first_img is None:
    print(f"❌ 无法读取图片: {img_paths[0]}")
    exit(1)

h, w = first_img.shape[:2]
# 确保宽高是偶数（H.264 编码要求）
w = w if w % 2 == 0 else w - 1
h = h if h % 2 == 0 else h - 1

print(f"📐 视频尺寸: {w} x {h}")

# 使用 H.264 编码（更通用）
# 尝试不同的编码器
fourcc_options = [
    cv2.VideoWriter_fourcc(*'avc1'),  # Mac 常用
    cv2.VideoWriter_fourcc(*'H264'),  # 通用
    cv2.VideoWriter_fourcc(*'X264'),  # 通用
    cv2.VideoWriter_fourcc(*'mp4v'),  # 后备方案
]

out_video = None
for fourcc in fourcc_options:
    out_video = cv2.VideoWriter(output_video_path, fourcc, 30.0, (w, h))
    if out_video.isOpened():
        print(f"✅ 使用编码器: {fourcc}")
        break
    else:
        out_video.release()
        out_video = None

if out_video is None or not out_video.isOpened():
    print(f"❌ 无法创建视频文件")
    exit(1)

print(f"⏳ 生成视频中...")

for f in range(num_frames):
    # 读取并调整图片尺寸
    img = cv2.imread(img_paths[f])
    if img is None:
        print(f"⚠️  跳过无法读取的帧: {f}")
        continue
    
    # 确保图片尺寸一致
    if img.shape[:2] != (h, w):
        img = cv2.resize(img, (w, h))
    
    current_pose = poses[f]
    
    # 绘制骨架点
    for i, point in enumerate(current_pose):
        x_norm, y_norm = point[0], point[1]
        if x_norm < 0 or y_norm < 0:
            continue
        
        # 坐标转换
        x_img = int(x_norm * w)
        y_img = int(y_norm * h)
        
        # 确保坐标在图片范围内
        x_img = max(0, min(x_img, w-1))
        y_img = max(0, min(y_img, h-1))
        
        # 绘制关键点
        cv2.circle(img, (x_img, y_img), 3, (0, 255, 0), -1)
        cv2.putText(img, str(i), (x_img+5, y_img+5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    out_video.write(img)
    
    # 进度显示
    if (f + 1) % 50 == 0:
        print(f"  处理进度: {f+1}/{num_frames}")

out_video.release()
print(f"✅ 视频生成完毕！")
print(f"📹 视频路径: {os.path.abspath(output_video_path)}")
print(f"🎬 帧率: 30 fps, 总帧数: {num_frames}")

# 验证视频文件
if os.path.exists(output_video_path):
    file_size = os.path.getsize(output_video_path) / (1024 * 1024)
    print(f"📦 文件大小: {file_size:.2f} MB")
else:
    print(f"❌ 视频文件未生成")