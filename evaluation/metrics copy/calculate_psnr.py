import cv2
import numpy as np
import os

def calculate_video_psnr(gt_video_path, gen_video_path):
    cap_gt = cv2.VideoCapture(gt_video_path)
    cap_gen = cv2.VideoCapture(gen_video_path)

    if not cap_gt.isOpened() or not cap_gen.isOpened():
        print(f"Error: 无法打开视频 {gt_video_path} 或 {gen_video_path}")
        return None

    psnr_scores = []
    
    while True:
        ret_gt, frame_gt = cap_gt.read()
        ret_gen, frame_gen = cap_gen.read()

        # 如果有一个视频读完了，就结束
        if not ret_gt or not ret_gen:
            break

        # 确保两帧分辨率一致，如果不一致则强行 resize 到 gt 的尺寸
        if frame_gt.shape != frame_gen.shape:
            frame_gen = cv2.resize(frame_gen, (frame_gt.shape[1], frame_gt.shape[0]))

        # 计算单帧 PSNR
        psnr_val = cv2.PSNR(frame_gt, frame_gen)
        psnr_scores.append(psnr_val)

    cap_gt.release()
    cap_gen.release()

    # 返回整个视频的平均 PSNR
    if len(psnr_scores) == 0:
        return 0.0
    return np.mean(psnr_scores)

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
        print(f"找到视频了！正在计算 PSNR...")
        avg_psnr = calculate_video_psnr(gt_test, gen_test)
        print(f"======== 最终结果 ========")
        print(f"视频 {video_name} 的 PSNR 得分为: {avg_psnr:.4f} dB (越高越好)")
    else:
        print(f"【报错】找不到文件！请检查 data/gt_videos 和 data/generated_videos 下是否有 {video_name}")
