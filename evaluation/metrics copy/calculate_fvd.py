import os
import glob
import cv2
import torch
import torch.nn as nn
import torchvision
import numpy as np
from scipy.linalg import sqrtm

def load_video_tensor(video_path, num_frames=16, size=112):
    """读取视频，均匀抽 16 帧，并转为 3D 模型需要的 Tensor 格式"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        # BGR转RGB，并调整分辨率到 112x112 (R3D_18 的标准输入大小)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (size, size))
        frames.append(frame)
    cap.release()
    
    if len(frames) == 0: return None
    
    # 均匀采样 16 帧
    indices = np.linspace(0, len(frames)-1, num_frames, dtype=int)
    sampled_frames = [frames[i] for i in indices]
    
    # 转成 [C, T, H, W] 形状，并归一化到 [0, 1]
    tensor = torch.tensor(np.array(sampled_frames), dtype=torch.float32).permute(3, 0, 1, 2) / 255.0
    
    # Kinetics-400 视频数据集的标准化参数
    mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1, 1)
    std = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1, 1)
    tensor = (tensor - mean) / std
    
    return tensor.unsqueeze(0) # 加上 Batch 维度 -> [1, C, T, H, W]

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """计算两个高斯分布之间的 Fréchet 距离"""
    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)

    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))
        
    if np.iscomplexobj(covmean): covmean = covmean.real
        
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)

def get_video_features(video_dir, model, device):
    features = []
    video_paths = glob.glob(os.path.join(video_dir, "*.mp4"))
    
    print(f"-> 正在从 {os.path.basename(video_dir)} 提取特征，共 {len(video_paths)} 个视频...")
    for vp in video_paths:
        tensor = load_video_tensor(vp)
        if tensor is None: continue
        tensor = tensor.to(device)
        with torch.no_grad():
            feat = model(tensor) # 提取出 512 维的视频时序特征
        features.append(feat.cpu().numpy()[0])
    
    features = np.array(features)
    
    # 如果视频太少，给个警告
    if features.shape[0] < 2:
        print("【严重警告】视频数量小于2！算出的协方差全是0，FVD分数无效。请复制几个视频进去充数。")
        return np.mean(features, axis=0), np.zeros((features.shape[1], features.shape[1]))
        
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma

if __name__ == "__main__":
    base_dir = os.getcwd()
    gt_dir = os.path.join(base_dir, "evaluation", "data", "gt_videos")
    gen_dir = os.path.join(base_dir, "evaluation", "data", "generated_videos")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"1. 正在加载 PyTorch 官方 R3D_18 视频模型到 {device}...")
    
    # 加载预训练的视频分类模型，并去掉最后的全连接层，让它只输出特征
    model = torchvision.models.video.r3d_18(weights=torchvision.models.video.R3D_18_Weights.KINETICS400_V1)
    model.fc = nn.Identity() 
    model = model.eval().to(device)

    # 提取两组视频的特征分布
    print("\n2. 开始处理真实视频...")
    mu_gt, sigma_gt = get_video_features(gt_dir, model, device)
    
    print("\n3. 开始处理生成视频...")
    mu_gen, sigma_gen = get_video_features(gen_dir, model, device)

    # 计算最终的 FVD 分数
    print("\n4. 正在计算 Fréchet Video Distance (FVD)...")
    fvd_score = calculate_frechet_distance(mu_gt, sigma_gt, mu_gen, sigma_gen)
    
    print(f"\n======== 最终结果 ========")
    print(f"数据集的 FVD 得分为: {fvd_score:.4f} (越低越好)")