import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os
#越接近 1：视频越清晰、越接近原图、失真越小
#越低：模糊、块效应、伪影、抖动越严重
def calculate_video_ssim(gt_video_path, gen_video_path):
    # 打开两个视频
    cap_gt = cv2.VideoCapture(gt_video_path)
    cap_gen = cv2.VideoCapture(gen_video_path)

    if not cap_gt.isOpened() or not cap_gen.isOpened():
        print(f"Error: 无法打开视频 {gt_video_path} 或 {gen_video_path}")
        return None

    ssim_scores = []
    
    while True:
        ret_gt, frame_gt = cap_gt.read()
        ret_gen, frame_gen = cap_gen.read()

        # 如果有一个视频读完了，就结束
        if not ret_gt or not ret_gen:
            break

        # 将图像转换为灰度图计算 SSIM (业界标准做法)
        gray_gt = cv2.cvtColor(frame_gt, cv2.COLOR_BGR2GRAY)
        gray_gen = cv2.cvtColor(frame_gen, cv2.COLOR_BGR2GRAY)

        # 确保两帧分辨率一致，如果不一致则强行 resize 到 gt 的尺寸
        if gray_gt.shape != gray_gen.shape:
            gray_gen = cv2.resize(gray_gen, (gray_gt.shape[1], gray_gt.shape[0]))

        # 计算单帧 SSIM
        score, _ = ssim(gray_gt, gray_gen, full=True)
        ssim_scores.append(score)

    cap_gt.release()
    cap_gen.release()

    # 返回整个视频的平均 SSIM
    if len(ssim_scores) == 0:
        return 0.0
    return np.mean(ssim_scores)

# --- 本地简单测试逻辑 ---
if __name__ == "__main__":
    # 1. 获取当前你运行命令所在的基础路径
    base_dir = os.getcwd() 
    print(f"【诊断信息】当前终端运行目录是: {base_dir}")
    
    # 2. 拼接出文件夹的绝对路径
    gt_dir = os.path.join(base_dir, "data", "gt_videos")
    gen_dir = os.path.join(base_dir, "data", "generated_videos")
    
    # 3. 打印出这两个文件夹里到底有什么文件
    if os.path.exists(gt_dir):
        print(f"【诊断信息】gt_videos 文件夹里的文件有: {os.listdir(gt_dir)}")
    else:
        print(f"【警告】找不到文件夹: {gt_dir}")
        
    if os.path.exists(gen_dir):
        print(f"【诊断信息】generated_videos 文件夹里的文件有: {os.listdir(gen_dir)}")
    else:
        print(f"【警告】找不到文件夹: {gen_dir}")

    print("-" * 30)

    # ！！！注意：请把下面这个名字换成你放进去的真实视频的名字！！！
    # 比如你的视频叫 "demo.mp4"，就改成 video_name = "demo.mp4"
    video_name = "test_01ipinew.mp4" 
    
    gt_test = os.path.join(gt_dir, "test_01.mp4")
    gen_test = os.path.join(gen_dir, "test_1.mp4")

    # 4. 最终检查并运行
    if os.path.exists(gt_test) and os.path.exists(gen_test):
        print(f"太好了，找到视频了！正在计算 SSIM...")
        avg_ssim = calculate_video_ssim(gt_test, gen_test)
        print(f"======== 最终结果 ========")
        print(f"视频 {video_name} 的 SSIM 得分为: {avg_ssim:.4f}")
    else:
        print(f"【最终报错】还是找不到文件！")
        print(f"Python 试图寻找的完整路径是：\n1. {gt_test}\n2. {gen_test}")
        print("请对比一下上面的【诊断信息】，看看是文件名写错了，还是后缀名不对（比如是 .MP4 或者 .avi）？")