import cv2
import numpy as np
import os
import torch

try:
    import lpips
except ImportError:
    print("【报错】未安装 lpips 库，请先在终端运行: pip install lpips")
    exit(1)

def calculate_video_lpips(gt_video_path, gen_video_path, device='cuda'):
    # 加载模型 (AlexNet backbone 是官方推荐用于评估感知距离的最佳选择)
    loss_fn_alex = lpips.LPIPS(net='alex').to(device).eval()

    cap_gt = cv2.VideoCapture(gt_video_path)
    cap_gen = cv2.VideoCapture(gen_video_path)

    if not cap_gt.isOpened() or not cap_gen.isOpened():
        print(f"Error: 无法打开视频 {gt_video_path} 或 {gen_video_path}")
        return None

    lpips_scores = []
    
    while True:
        ret_gt, frame_gt = cap_gt.read()
        ret_gen, frame_gen = cap_gen.read()

        if not ret_gt or not ret_gen:
            break

        if frame_gt.shape != frame_gen.shape:
            frame_gen = cv2.resize(frame_gen, (frame_gt.shape[1], frame_gt.shape[0]))

        # 转为 RGB 格式
        rgb_gt = cv2.cvtColor(frame_gt, cv2.COLOR_BGR2RGB)
        rgb_gen = cv2.cvtColor(frame_gen, cv2.COLOR_BGR2RGB)
        
        # 转换为 Tensor，并将像素域归一化到 [-1, 1] 供 LPIPS 使用
        t_gt = torch.from_numpy(rgb_gt).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
        t_gen = torch.from_numpy(rgb_gen).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
        
        t_gt = t_gt.to(device)
        t_gen = t_gen.to(device)
        
        with torch.no_grad():
            l_val = loss_fn_alex(t_gt, t_gen).item()
        lpips_scores.append(l_val)

    cap_gt.release()
    cap_gen.release()

    if len(lpips_scores) == 0:
        return 0.0
    return np.mean(lpips_scores)

# --- 本地简单测试逻辑 ---
if __name__ == "__main__":
    # 获取上级目录 (evaluation/)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
    gt_dir = os.path.join(base_dir, "data", "gt_videos")
    gen_dir = os.path.join(base_dir, "data", "generated_videos")

    # ！！！注意：请把下面这个名字换成你放进去的真实视频的名字！！！
    video_name = "test_1.mp4" 
    
    gt_test = os.path.join(gt_dir, video_name)
    gen_test = os.path.join(gen_dir, video_name)

    if os.path.exists(gt_test) and os.path.exists(gen_test):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"找到视频了！正在计算 LPIPS (使用 {device})...")
        avg_lpips = calculate_video_lpips(gt_test, gen_test, device)
        print(f"======== 最终结果 ========")
        print(f"视频 {video_name} 的 LPIPS 得分为: {avg_lpips:.4f} (越低越好)")
    else:
        print(f"【报错】找不到文件！请检查 data/gt_videos 和 data/generated_videos 下是否有 {video_name}")
