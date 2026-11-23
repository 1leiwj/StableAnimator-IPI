import os
import glob
import sys
import torch
import torchvision
import torch.nn as nn
import subprocess
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# 尝试导入 lpips
try:
    import lpips
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False

from metrics.calculate_clip_i import calculate_clip_i
from metrics.calculate_fvd import get_video_features, calculate_frechet_distance
from metrics.extract_frames_for_fid import extract_frames

def evaluate_video_pair_metrics(gt_path, gen_path, device='cuda', lpips_model=None):
    """
    一次性遍历视频帧，同时计算 SSIM, PSNR 和 LPIPS，极大地提高运行效率
    """
    cap_gt = cv2.VideoCapture(gt_path)
    cap_gen = cv2.VideoCapture(gen_path)
    
    if not cap_gt.isOpened() or not cap_gen.isOpened():
        return None, None, None
        
    ssim_scores = []
    psnr_scores = []
    lpips_scores = []
    
    while True:
        ret_gt, frame_gt = cap_gt.read()
        ret_gen, frame_gen = cap_gen.read()
        if not ret_gt or not ret_gen:
            break
            
        # 强制将生成视频的尺寸与真实视频对齐，防止报错
        if frame_gt.shape != frame_gen.shape:
            frame_gen = cv2.resize(frame_gen, (frame_gt.shape[1], frame_gt.shape[0]))
            
        # 1. 计算 PSNR (越高越好)
        psnr_val = cv2.PSNR(frame_gt, frame_gen)
        psnr_scores.append(psnr_val)
        
        # 2. 计算 SSIM (越高越好，转灰度图计算)
        gray_gt = cv2.cvtColor(frame_gt, cv2.COLOR_BGR2GRAY)
        gray_gen = cv2.cvtColor(frame_gen, cv2.COLOR_BGR2GRAY)
        score, _ = ssim(gray_gt, gray_gen, full=True)
        ssim_scores.append(score)
        
        # 3. 计算 LPIPS (越低越好，使用深度学习特征)
        if lpips_model is not None:
            # LPIPS 要求的输入格式为 RGB, 值域 [-1, 1], 形状 [1, C, H, W]
            rgb_gt = cv2.cvtColor(frame_gt, cv2.COLOR_BGR2RGB)
            rgb_gen = cv2.cvtColor(frame_gen, cv2.COLOR_BGR2RGB)
            
            t_gt = torch.from_numpy(rgb_gt).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
            t_gen = torch.from_numpy(rgb_gen).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
            
            t_gt = t_gt.to(device)
            t_gen = t_gen.to(device)
            
            with torch.no_grad():
                l_val = lpips_model(t_gt, t_gen).item()
            lpips_scores.append(l_val)
            
    cap_gt.release()
    cap_gen.release()
    
    avg_ssim = np.mean(ssim_scores) if ssim_scores else 0.0
    avg_psnr = np.mean(psnr_scores) if psnr_scores else 0.0
    avg_lpips = np.mean(lpips_scores) if lpips_scores else 0.0
    
    return avg_ssim, avg_psnr, avg_lpips if lpips_model is not None else None


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    gen_dir = os.path.join(data_dir, "generated_videos")
    gt_dir = os.path.join(data_dir, "gt_videos")
    source_dir = os.path.join(data_dir, "source_images")
    results_dir = os.path.join(base_dir, "results")
    
    gt_frame_dir = os.path.join(data_dir, "gt_frames")
    gen_frame_dir = os.path.join(data_dir, "generated_frames")

    os.makedirs(results_dir, exist_ok=True)

    gen_videos = glob.glob(os.path.join(gen_dir, "*.mp4"))
    if not gen_videos:
        print(f"【错误】没有在 {gen_dir} 找到任何 .mp4 视频文件。请先将你的生成结果放入该目录！")
        return

    results = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("🚀 开始执行自动化视频评测流程 (六合一终极版)...")
    print(f"-> 发现 {len(gen_videos)} 个生成视频，开始逐一评估")
    
    # 提前加载 LPIPS 模型，避免循环里反复加载
    lpips_model = None
    if HAS_LPIPS:
        print("-> 正在加载 LPIPS 感知模型 (AlexNet backbone)...")
        lpips_model = lpips.LPIPS(net='alex').to(device).eval()
    else:
        print("-> ⚠️ 未检测到 lpips 库，将跳过 LPIPS 计算。建议运行: pip install lpips")
        
    print("=" * 60)

    # 3. 逐个视频计算 CLIP-I, SSIM, PSNR, LPIPS
    for gen_video_path in gen_videos:
        video_name = os.path.basename(gen_video_path)
        base_name = os.path.splitext(video_name)[0]
        
        print(f"\n正在评测视频: {video_name}")
        
        clip_score, ssim_score, psnr_score, lpips_score = None, None, None, None
        
        # --- 计算 CLIP-I ---
        source_image_path = None
        for ext in ['.jpg', '.png', '.jpeg']:
            possible_path = os.path.join(source_dir, f"{base_name}{ext}")
            if os.path.exists(possible_path):
                source_image_path = possible_path
                break
                
        if source_image_path:
            try:
                clip_score = calculate_clip_i(source_image_path, gen_video_path)
                print(f"  [CLIP-I] 得分: {clip_score:.4f}")
            except Exception as e:
                print(f"  [CLIP-I] 计算失败: {e}")
        else:
            print(f"  [CLIP-I] ⚠️ 未找到对应的原图 (预期名字如 {base_name}.jpg)，跳过。")

        # --- 计算 SSIM, PSNR, LPIPS ---
        gt_video_path = os.path.join(gt_dir, video_name)
        if os.path.exists(gt_video_path):
            try:
                ssim_score, psnr_score, lpips_score = evaluate_video_pair_metrics(
                    gt_video_path, gen_video_path, device, lpips_model
                )
                print(f"  [SSIM]   得分: {ssim_score:.4f}")
                print(f"  [PSNR]   得分: {psnr_score:.4f} dB")
                if lpips_score is not None:
                    print(f"  [LPIPS]  得分: {lpips_score:.4f}")
            except Exception as e:
                print(f"  [帧级对比] 计算失败: {e}")
        else:
            print(f"  [帧级对比] ⚠️ 未找到对应的真实视频 ({video_name})，跳过 SSIM/PSNR/LPIPS。")
            
        results.append({
            "video": video_name,
            "clip_i": clip_score,
            "ssim": ssim_score,
            "psnr": psnr_score,
            "lpips": lpips_score
        })

    # 4. 计算 FVD (整体分布)
    fvd_score = None
    gt_videos = glob.glob(os.path.join(gt_dir, "*.mp4"))
    
    if len(gen_videos) >= 2 and len(gt_videos) >= 2:
        print("\n" + "=" * 60)
        print("🎥 开始计算 FVD (Fréchet Video Distance) ...")
        try:
            print(f"  [FVD] 加载 R3D_18 视频特征模型到 {device} ...")
            model = torchvision.models.video.r3d_18(weights=torchvision.models.video.R3D_18_Weights.KINETICS400_V1)
            model.fc = nn.Identity() 
            model = model.eval().to(device)

            mu_gt, sigma_gt = get_video_features(gt_dir, model, device)
            mu_gen, sigma_gen = get_video_features(gen_dir, model, device)
            
            fvd_score = calculate_frechet_distance(mu_gt, sigma_gt, mu_gen, sigma_gen)
            print(f"  [FVD] 计算完成，得分: {fvd_score:.4f}")
        except Exception as e:
            print(f"  [FVD] 计算失败: {e}")
    else:
        print("\n" + "=" * 60)
        print("⚠️ 提示: 跳过 FVD 计算 (需要 generated_videos 和 gt_videos 中至少各包含 2 个视频)。")

    # 5. 计算 FID
    fid_score_str = "N/A"
    if len(gen_videos) > 0 and len(gt_videos) > 0:
        print("\n" + "=" * 60)
        print("🖼️ 开始计算 FID (Fréchet Inception Distance) ...")
        try:
            print("  [FID] 第一步：正在从视频抽取关键帧...")
            extract_frames(gt_dir, gt_frame_dir)
            extract_frames(gen_dir, gen_frame_dir)
            
            print("  [FID] 第二步：调用 pytorch_fid 库计算分布差异 (初次运行会下载模型并显示进度条)...")
            cmd = [sys.executable, "-m", "pytorch_fid", gt_frame_dir, gen_frame_dir, "--device", device]
            
            # 使用 Popen 以便实时把进度条打印到屏幕上，防止让人以为卡死了
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            output_lines = []
            for line in process.stdout:
                sys.stdout.write("    " + line)  # 实时打印到终端
                sys.stdout.flush()
                output_lines.append(line)
            
            process.wait()
            
            if process.returncode == 0:
                output = "".join(output_lines).strip()
                for line in output.split('\n'):
                    if 'FID:' in line:
                        fid_score_str = line.split('FID:')[-1].strip()
                        break
                if fid_score_str == "N/A" and output:
                    fid_score_str = output.split('\n')[-1].replace('FID:', '').strip()
                print(f"  [FID] 计算完成，得分: {fid_score_str}")
            else:
                print(f"  [FID] 计算失败！请确保你已经在终端运行了: pip install pytorch-fid")
        except Exception as e:
            print(f"  [FID] 计算流程出错: {e}")
    else:
        print("\n" + "=" * 60)
        print("⚠️ 提示: 跳过 FID 计算 (缺少生成视频或真实视频)。")

    # 6. 输出汇总报告
    print("\n\n" + "🌟" * 35)
    print(" " * 25 + "评测最终报告 (六大权威指标)")
    print("🌟" * 35)
    
    report_lines = []
    
    header = f"{'视频名称(Video)':<22} | {'CLIP-I(↑)':<10} | {'SSIM(↑)':<10} | {'PSNR(↑)':<10} | {'LPIPS(↓)':<10}"
    divider = "-" * 75
    
    print(header)
    print(divider)
    report_lines.extend([header, divider])
    
    avg_clip, avg_ssim, avg_psnr, avg_lpips = [], [], [], []
    
    for r in results:
        v_name = r['video']
        if len(v_name) > 20: v_name = v_name[:17] + "..."
            
        c_s = f"{r['clip_i']:>10.4f}" if r['clip_i'] is not None else f"{'N/A':>10}"
        s_s = f"{r['ssim']:>10.4f}" if r['ssim'] is not None else f"{'N/A':>10}"
        p_s = f"{r['psnr']:>10.4f}" if r['psnr'] is not None else f"{'N/A':>10}"
        l_s = f"{r['lpips']:>10.4f}" if r['lpips'] is not None else f"{'N/A':>10}"
        
        line = f"{v_name:<22} | {c_s} | {s_s} | {p_s} | {l_s}"
        print(line)
        report_lines.append(line)
        
        if r['clip_i'] is not None: avg_clip.append(r['clip_i'])
        if r['ssim'] is not None: avg_ssim.append(r['ssim'])
        if r['psnr'] is not None: avg_psnr.append(r['psnr'])
        if r['lpips'] is not None: avg_lpips.append(r['lpips'])

    print(divider)
    report_lines.append(divider)
    
    m_c = f"{sum(avg_clip)/len(avg_clip):>10.4f}" if avg_clip else f"{'N/A':>10}"
    m_s = f"{sum(avg_ssim)/len(avg_ssim):>10.4f}" if avg_ssim else f"{'N/A':>10}"
    m_p = f"{sum(avg_psnr)/len(avg_psnr):>10.4f}" if avg_psnr else f"{'N/A':>10}"
    m_l = f"{sum(avg_lpips)/len(avg_lpips):>10.4f}" if avg_lpips else f"{'N/A':>10}"
    
    summary_line1 = f"{'⭐ 平均分 (Average)':<22} | {m_c} | {m_s} | {m_p} | {m_l}"
    print(summary_line1)
    report_lines.append(summary_line1)
    
    f_s = f"{fvd_score:.4f}" if fvd_score is not None else "N/A"
    summary_line2 = f"\n【全局分布指标】:\n-> FVD 得分 (↓越低越好): {f_s}\n-> FID 得分 (↓越低越好): {fid_score_str}"
    print(summary_line2)
    report_lines.append(summary_line2)
    
    report_path = os.path.join(results_dir, "evaluation_report.txt")
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))
        print(f"\n✅ 详细结果已保存至: {report_path}")
    except Exception as e:
        print(f"\n⚠️ 保存报告失败: {e}")

if __name__ == "__main__":
    main()
