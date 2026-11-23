import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm
import sys
import torch

# 添加dwpose路径
sys.path.append('/gaojiawei/wjl/StableAnimator')
from dwpose.wholebody import Wholebody

class DWposeDetector:
    def __init__(self):
        # 初始化模型
        self.pose_estimation = Wholebody()

    def __call__(self, oriImg):
        # 🚀 修复1：保持 BGR 格式。OpenCV 读取的就是 BGR。
        oriImg = oriImg.copy()
        
        # 这里的 H, W 会自动获取拉伸后的 1024 和 768
        H, W, C = oriImg.shape
        
        with torch.no_grad():
            candidate, subset = self.pose_estimation(oriImg)
            
            # 安全检查：如果没检测到人
            if candidate.shape[0] == 0:
                return None

            # 🚀 修复2：防干扰逻辑。如果画面有多个人，选面积最大的（主人物）。
            if candidate.shape[0] > 1:
                areas = []
                for i in range(candidate.shape[0]):
                    valid_pts = candidate[i][subset[i] > 0.3]
                    if len(valid_pts) > 0:
                        min_x, min_y = valid_pts.min(axis=0)
                        max_x, max_y = valid_pts.max(axis=0)
                        areas.append((max_x - min_x) * (max_y - min_y))
                    else:
                        areas.append(0)
                main_person_idx = np.argmax(areas)
            else:
                main_person_idx = 0

            # 锁定主人物
            candidate = candidate[main_person_idx][np.newaxis, :, :]
            subset = subset[main_person_idx][np.newaxis, :]
            
            # 🚀 关键步：在这里执行归一化，确保输出是 [0, 1]
            # 因为除以的是 768 和 1024，拉伸带来的变形在这里被【完美抵消】！
            norm_candidate = candidate.copy()
            norm_candidate[..., 0] /= float(W)
            norm_candidate[..., 1] /= float(H)
            
            # --- 转换到你需要的 18 点格式逻辑 ---
            nums, keys, locs = norm_candidate.shape
            
            # 取全部 133 个关键点 (身体+面部+双手+脚)
            body = norm_candidate[:, :133, :].copy()
            body = body.reshape(nums * 133, locs)
            
            # 这里的 subset 是置信度
            score = subset[:, :133].copy()
            
            # 如果置信度太低，标记为 -1 (保持你原有的处理逻辑)
            for i in range(len(score)):
                for j in range(len(score[i])):
                    if score[i][j] <= 0.3:
                        body[i*133 + j] = [-1.0, -1.0]

            return body

def extract_poses_to_npy(target_image_folder_path, output_npy_path=None):
    print(f"📂 Processing folder: {target_image_folder_path}")
    
    dwpose_model = DWposeDetector()
    files = os.listdir(target_image_folder_path)
    png_files = [f for f in files if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # 保留你原本完美的严格排序逻辑
    try:
        png_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]) if 'frame_' in x else x)
    except:
        png_files.sort()
    
    print(f"📸 Found {len(png_files)} images")
    all_bodies = [] 
    
    for idx, img_file in enumerate(tqdm(png_files, desc="Detecting poses")):
        img_path = os.path.join(target_image_folder_path, img_file)
        
        # 使用 cv2.imread 直接读取 BGR
        image = cv2.imread(img_path) 
        if image is None: continue
        
        try:
            # 🚀✨ 终极魔法：强行拉伸为 3:4 (768x1024)，彻底绕过 DWPose 黑边 Bug！
            trick_image = cv2.resize(image, (768, 1024))
            
            # 把拉伸后的图片喂给模型，得到无偏移的 [0, 1] 坐标
            body_keypoints = dwpose_model(trick_image)
            
            if body_keypoints is not None:
                all_bodies.append(body_keypoints)
            else:
                # 没检测到人时，用上一帧填充，防止抖动
                if idx > 0 and len(all_bodies) > 0:
                    all_bodies.append(all_bodies[-1].copy())
                else:
                    all_bodies.append(np.zeros((133, 2)) - 1.0)
                    
        except Exception as e:
            print(f"❌ Error in frame {idx}: {e}")
            all_bodies.append(np.zeros((133, 2)) - 1.0)
    
    all_bodies = np.array(all_bodies, dtype=np.float32)  # [T, 133, 2]
    
    # 打印统计信息，验证是否为 0-1
    valid_data = all_bodies[all_bodies != -1]
    print(f"✅ Extraction Complete. Shape: {all_bodies.shape}")
    print(f"📊 Range: min={valid_data.min():.4f}, max={valid_data.max():.4f} (Should be 0~1)")

    if output_npy_path is None:
        parent_folder = os.path.dirname(target_image_folder_path)
        output_npy_path = os.path.join(parent_folder, "poses_dense133.npy")
    
    np.save(output_npy_path, all_bodies)
    print(f"💾 Saved to: {output_npy_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_image_folder_path', type=str, required=True)
    parser.add_argument('--output_npy_path', type=str, default=None)
    args = parser.parse_args()
    
    extract_poses_to_npy(args.target_image_folder_path, args.output_npy_path)

if __name__ == "__main__":
    main()