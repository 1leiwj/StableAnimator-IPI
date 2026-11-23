import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import cv2
from PIL import Image
import numpy as np
from transformers import CLIPProcessor, CLIPModel

def calculate_clip_i(source_image_path, gen_video_path):
    # 1. 加载 CLIP 模型 (第一次跑会自动下载，大概几十MB)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "openai/clip-vit-base-patch32"
    
    print(f"正在加载 CLIP 模型到 {device}...")
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(device)

    # 2. 读取并处理源图片 (Source Image)
    source_img = Image.open(source_image_path).convert("RGB")
    source_inputs = processor(images=source_img, return_tensors="pt").to(device)
    with torch.no_grad():
        source_features = model.get_image_features(**source_inputs)
        # 归一化特征
        source_features = source_features / source_features.norm(p=2, dim=-1, keepdim=True)

    # 3. 读取生成视频并逐帧计算
    cap = cv2.VideoCapture(gen_video_path)
    if not cap.isOpened():
        print(f"Error: 无法打开视频 {gen_video_path}")
        return None

    similarities = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # OpenCV 读进来是 BGR，转为 RGB 并变成 PIL Image 格式供 CLIP 使用
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)

        # 提取单帧的特征
        frame_inputs = processor(images=frame_pil, return_tensors="pt").to(device)
        with torch.no_grad():
            frame_features = model.get_image_features(**frame_inputs)
            frame_features = frame_features / frame_features.norm(p=2, dim=-1, keepdim=True)

        # 计算余弦相似度
        sim = torch.matmul(source_features, frame_features.T).item()
        similarities.append(sim)

    cap.release()

    if len(similarities) == 0:
        return 0.0
    return np.mean(similarities)

# --- 运行逻辑与路径配置 ---
if __name__ == "__main__":
    base_dir = os.getcwd() 
    source_dir = os.path.join(base_dir, "data", "source_images")
    gen_dir = os.path.join(base_dir, "data", "generated_videos")

    # ！！！请把下面的名字换成你实际测试用的图片和视频名字！！！
    image_name = "reference.png" # <--- 换成 source_images 里的图片名
    video_name = "duiqiyuan.mp4"     # <--- 换成 generated_videos 里的视频名

    source_path = os.path.join(source_dir, image_name)
    gen_path = os.path.join(gen_dir, video_name)

    if os.path.exists(source_path) and os.path.exists(gen_path):
        avg_clip_i = calculate_clip_i(source_path, gen_path)
        print(f"\n======== 最终结果 ========")
        print(f"视频 [{video_name}] 与 源图 [{image_name}] 的 CLIP-I 得分为: {avg_clip_i:.4f}")
    else:
        print("\n【报错】找不到文件！请检查：")
        if not os.path.exists(source_path): print(f"-> 缺图片: {source_path}")
        if not os.path.exists(gen_path): print(f"-> 缺视频: {gen_path}")