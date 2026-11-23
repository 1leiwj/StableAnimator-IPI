import os
import subprocess
import argparse
from pathlib import Path

def run_cmd(cmd, cwd=None):
    """运行终端命令的辅助函数"""
    cmd_str = ' '.join(cmd)
    print(f"🚀 [执行命令]: {cmd_str}")
    try:
        subprocess.run(cmd, check=True, cwd=cwd)
    except subprocess.CalledProcessError as e:
        print(f"❌ [命令执行失败]: {e}")
        exit(1)

def main():
    parser = argparse.ArgumentParser(description="一键准备推理用的视频数据 (包含 poses 图像和 npy)")
    parser.add_argument("--video_path", type=str, required=True, help="输入的驱动视频绝对路径 (.mp4)")
    parser.add_argument("--ref_path", type=str, required=True, help="角色参考图绝对路径 (.png/.jpg)")
    parser.add_argument("--output_dir", type=str, required=True, help="输出的案例文件夹路径 (如 inference/case-01)")
    parser.add_argument("--sa_dir", type=str, default="/gaojiawei/wjl/StableAnimator", help="StableAnimator 根目录")
    args = parser.parse_args()

    video_path = Path(args.video_path)
    ref_path = Path(args.ref_path)
    output_dir = Path(args.output_dir)

    if not video_path.exists():
        print(f"❌ 找不到视频文件: {video_path}")
        return
    if not ref_path.exists():
        print(f"❌ 找不到参考图文件: {ref_path}")
        return

    # 定义所有子文件夹和文件路径
    images_dir = output_dir / "images"
    poses_dir = output_dir / "poses"
    output_npy = output_dir / "poses_dense.npy"
    
    # 创建基础文件夹
    images_dir.mkdir(parents=True, exist_ok=True)
    poses_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 目标推理文件夹已就绪: {output_dir}")

    # ==========================================
    # 步骤 1: FFmpeg 提取视频帧 -> images/
    # ==========================================
    print("\n" + "="*40)
    print("🎥 步骤 1: 开始提取视频帧...")
    ffmpeg_cmd = [
        "ffmpeg", "-v", "error", "-i", str(video_path), 
        "-q:v", "1", 
        "-start_number", "0", 
        "-y", 
        str(images_dir / "frame_%d.png")
    ]
    run_cmd(ffmpeg_cmd)
    print(f"✅ 抽帧完成，保存在: {images_dir}")

    # ==========================================
    # 步骤 2: 提取可视化骨骼图像 -> poses/
    # ==========================================
    print("\n" + "="*40)
    print("🦴 步骤 2: 生成可视化人体骨骼图像...")
    skeleton_script = Path(args.sa_dir) / "DWPose" / "skeleton_extraction.py"
    pose_img_cmd = [
        "python", str(skeleton_script),
        f"--target_image_folder_path={images_dir}",
        f"--ref_image_path={ref_path}",
        f"--poses_folder_path={poses_dir}"
    ]
    run_cmd(pose_img_cmd, cwd=args.sa_dir)
    print(f"✅ 骨骼图像提取完成，保存在: {poses_dir}")

    # ==========================================
    # 步骤 3: 提取人脸掩码 -> faces/
    # ==========================================
    print("\n" + "="*40)
    print("🧑 步骤 3: 提取人脸掩模 (Face Mask)...")
    face_script = Path(args.sa_dir) / "face_mask_extraction.py"
    face_cmd = [
        "python", str(face_script),
        f"--image_folder={images_dir}"
    ]
    run_cmd(face_cmd, cwd=args.sa_dir)
    print("✅ 人脸掩模提取完成。")

    # ==========================================
    # 步骤 4: 生成密集姿态坐标数据 -> poses_dense.npy
    # ==========================================
    print("\n" + "="*40)
    print("📊 步骤 4: 生成坐标矩阵 (poses_dense.npy)...")
    dense_script = Path(args.sa_dir) / "DWPose" / "poses_dense.py"
    dense_cmd = [
        "python", str(dense_script),
        f"--target_image_folder_path={images_dir}",
        f"--output_npy_path={output_npy}"
    ]
    run_cmd(dense_cmd, cwd=args.sa_dir)
    print(f"✅ NPY 文件生成完成: {output_npy.name}")

    # ==========================================
    # 总结
    # ==========================================
    print("\n" + "🎉"*10)
    print("所有推理数据预处理完毕！")
    print(f"最终目录结构已生成在: {output_dir}")
    subprocess.run(["ls", "-l", str(output_dir)])

if __name__ == "__main__":
    main()