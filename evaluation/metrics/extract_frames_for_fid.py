import cv2
import os
import glob
import shutil

def extract_frames(video_folder, output_folder, max_frames_per_video=30):
    """
    从文件夹里的所有视频中抽帧并保存为图片
    max_frames_per_video: 默认每个视频最多抽前30帧（防止图片太多算得慢，你可以自己改）
    """
    # 每次抽帧前先清空旧文件夹，防止旧的尺寸不一的遗留图片污染数据
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    # 找到文件夹下所有的 mp4
    video_paths = glob.glob(os.path.join(video_folder, "*.mp4"))
    if not video_paths:
        print(f"【警告】在 {video_folder} 没找到 .mp4 视频！")
        return

    total_frames = 0
    for vid_path in video_paths:
        vid_name = os.path.basename(vid_path).split('.')[0]
        cap = cv2.VideoCapture(vid_path)
        
        count = 0
        while cap.isOpened() and count < max_frames_per_video:
            ret, frame = cap.read()
            if not ret:
                break
                
            # 🌟 关键修复：强制把所有图片 resize 到统一大小 (256x256)
            # 否则 pytorch_fid 在打包不同分辨率的图片 (Dataloader collate) 时会直接崩溃报错
            frame = cv2.resize(frame, (256, 256))
                
            img_path = os.path.join(output_folder, f"{vid_name}_{count:04d}.png")
            cv2.imwrite(img_path, frame)
            count += 1
            total_frames += 1
            
        cap.release()
    print(f"成功从 {os.path.basename(video_folder)} 抽取了 {total_frames} 张图片。")

if __name__ == "__main__":
    base_dir = os.getcwd()
    
    # 视频输入路径
    gt_vid_dir = os.path.join(base_dir, "evaluation", "data", "gt_videos")
    gen_vid_dir = os.path.join(base_dir, "evaluation", "data", "test")
    
    # 图片输出路径 (会自动创建这俩文件夹)
    gt_frame_dir = os.path.join(base_dir, "evaluation", "data", "gt_frames")
    gen_frame_dir = os.path.join(base_dir, "evaluation", "data", "generated_frames")
    
    print("1. 开始抽取真实视频帧...")
    extract_frames(gt_vid_dir, gt_frame_dir)
    
    print("\n2. 开始抽取生成视频帧...")
    extract_frames(gen_vid_dir, gen_frame_dir)
    
    print("\n【抽帧完成！】")



#python evaluation/metrics/extract_frames_for_fid.py
#python -m pytorch_fid evaluation/data/gt_frames evaluation/data/generated_frames --device cuda