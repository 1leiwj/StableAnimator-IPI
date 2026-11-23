import os, io, csv, math, random, copy
from importlib.metadata import files
import os.path as osp

import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from einops import rearrange
import cv2
import warnings
import json
from transformers import CLIPImageProcessor

# [IPI Mod] 辅助函数：归一化 Pose 坐标

# [IPI Mod] 辅助函数：智能归一化 Pose 坐标
def normalize_pose(pose_data, width, height):
    """
    智能归一化：自动识别绝对像素坐标和 [0,1] 坐标，统一映射到 [-1, 1]
    """
    # 创建副本避免修改原数据
    pose_data = pose_data.copy()
    
    # 找到有效点 (非-1)
    valid_mask = (pose_data[..., 0] != -1) & (pose_data[..., 1] != -1)
    
    valid_poses = pose_data[valid_mask]
    if len(valid_poses) > 0:
        # 🚀 智能判断：如果最大值大于 1.5，说明输入的是绝对像素坐标（比如 512x512）
        if valid_poses.max() > 1.5:
            pose_data[valid_mask, 0] = (pose_data[valid_mask, 0] / width) * 2.0 - 1.0
            pose_data[valid_mask, 1] = (pose_data[valid_mask, 1] / height) * 2.0 - 1.0
        # 🚀 如果最大值在 0~1.5 之间，说明已经是 [0, 1] 归一化坐标了
        else:
            pose_data[valid_mask, 0] = pose_data[valid_mask, 0] * 2.0 - 1.0
            pose_data[valid_mask, 1] = pose_data[valid_mask, 1] * 2.0 - 1.0
            
    # 无效点保持-1不变
    return pose_data

def load_target_frame_pose(path, frame_indices, frame_count, image_sizes):
    """Return the detected joints in the coordinate system of their own RGB frames."""
    sequence = np.load(path, mmap_mode="r", allow_pickle=False)
    if (sequence.ndim != 3 or sequence.shape[:2] != (frame_count, 133)
            or sequence.shape[2] < 2 or sequence.dtype.kind not in "fiu"):
        raise ValueError(f"Expected {frame_count} frames with 133 joints in {path}, got {sequence.shape}")
    points = sequence[frame_indices, :, :2].astype(np.float32)
    visible = np.isfinite(points).all(axis=-1) & (points != -1).all(axis=-1)
    if visible.any() and points[visible].max() > 1.5:
        points /= np.asarray(image_sizes, dtype=np.float32)[:, None, :]
    points[~visible] = -1
    return points


def compute_body_scales(ref_pose, drv_pose):
    def dist(p1, p2):
        if p1[0]==-1 or p2[0]==-1: return -1.0
        return np.linalg.norm(p1 - p2)
    def safe_dist(ref, drv, p_idx1, p_idx2):
        d_ref = dist(ref[p_idx1], ref[p_idx2])
        d_drv = dist(drv[p_idx1], drv[p_idx2])
        if d_ref > 0 and d_drv > 0.01: return d_ref / d_drv
        return 1.0

    scales = {}
    scales["shoulder"] = safe_dist(ref_pose, drv_pose, 2, 5)
    scales["body"] = safe_dist(ref_pose, drv_pose, 1, 8)
    
    s_arm_u_l, s_arm_u_r = safe_dist(ref_pose, drv_pose, 5, 6), safe_dist(ref_pose, drv_pose, 2, 3)
    scales["arm_upper"] = (s_arm_u_l + s_arm_u_r) / 2.0 if (s_arm_u_l != 1.0 and s_arm_u_r != 1.0) else max(s_arm_u_l, s_arm_u_r)

    s_arm_l_l, s_arm_l_r = safe_dist(ref_pose, drv_pose, 6, 7), safe_dist(ref_pose, drv_pose, 3, 4)
    scales["arm_lower"] = (s_arm_l_l + s_arm_l_r) / 2.0 if (s_arm_l_l != 1.0 and s_arm_l_r != 1.0) else max(s_arm_l_l, s_arm_l_r)

    s_leg_u_l, s_leg_u_r = safe_dist(ref_pose, drv_pose, 11, 12), safe_dist(ref_pose, drv_pose, 8, 9)
    scales["leg_upper"] = (s_leg_u_l + s_leg_u_r) / 2.0 if (s_leg_u_l != 1.0 and s_leg_u_r != 1.0) else max(s_leg_u_l, s_leg_u_r)

    s_leg_l_l, s_leg_l_r = safe_dist(ref_pose, drv_pose, 12, 13), safe_dist(ref_pose, drv_pose, 9, 10)
    scales["leg_lower"] = (s_leg_l_l + s_leg_l_r) / 2.0 if (s_leg_l_l != 1.0 and s_leg_l_r != 1.0) else max(s_leg_l_l, s_leg_l_r)
    return scales

def align_body_pose_single(pose, scales):
    aligned = pose.copy()
    valid = np.isfinite(pose).all(axis=-1) & (pose != -1).all(axis=-1)
    chains = (
        (1, (2, 5), "shoulder"),
        (1, (0, 8, 11, 14, 15, 16, 17), "body"),
        (2, (3,), "arm_upper"), (5, (6,), "arm_upper"),
        (3, (4,), "arm_lower"), (6, (7,), "arm_lower"),
        (8, (9,), "leg_upper"), (11, (12,), "leg_upper"),
        (9, (10,), "leg_lower"), (12, (13,), "leg_lower"),
    )
    for parent, children, scale_name in chains:
        for child in children:
            if valid[parent] and valid[child]:
                aligned[child] = aligned[parent] + (pose[child] - pose[parent]) * scales[scale_name]

    # Dense DWPose order: left/right feet, face, left/right hands.
    details = (
        (slice(18, 21), 13, 1), (slice(21, 24), 10, 1),
        (slice(24, 92), 0, scales["body"]),
        (slice(92, 113), 7, 1), (slice(113, 133), 4, 1),
    )
    for region, anchor, scale in details:
        if valid[anchor]:
            indices = np.arange(len(pose))[region][valid[region]]
            aligned[indices] = aligned[anchor] + (pose[indices] - pose[anchor]) * scale
    return aligned

# [IPI Mod] 极速本地骨架渲染 (方案B核心: 从拉伸后的纯数值重绘彩色骨架)
def draw_bodypose_local(canvas, candidate, subset):
    H, W, C = canvas.shape
    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
               [1, 16], [16, 18], [3, 17], [6, 18]]
    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
              
    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index: continue
            
            p1_idx = int(index[0])
            p2_idx = int(index[1])
            if candidate[p1_idx, 0] == -1 or candidate[p2_idx, 0] == -1: continue
            
            Y = np.array([candidate[p1_idx, 0], candidate[p2_idx, 0]])
            X = np.array([candidate[p1_idx, 1], candidate[p2_idx, 1]])
            
            if np.max(candidate) <= 1.5:
                Y = Y * W
                X = X * H
                
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), 4), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(canvas, polygon, colors[i])

    canvas = (canvas * 0.6).astype(np.uint8)

    for i in range(18):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1 or candidate[index, 0] == -1: continue
            
            y = candidate[index, 0]
            x = candidate[index, 1]
            if np.max(candidate) <= 1.5:
                y = y * W
                x = x * H
            cv2.circle(canvas, (int(y), int(x)), 4, colors[i], thickness=-1)
    return canvas

def render_pose_frames(dwpose_raw_frames, width, height):
    """根据修改后的坐标重新画出彩色骨架Tensor, 取代硬盘上的旧图片"""
    subset = np.arange(18).reshape(1, 18)
    rendered_tensors = []
    for i in range(len(dwpose_raw_frames)):
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas = draw_bodypose_local(canvas, dwpose_raw_frames[i], subset)
        
        # 转换为 Tensor [-1, 1] 遵循原先的 Normalize 标准
        canvas_tensor = torch.from_numpy(canvas).float() / 127.5 - 1.0
        canvas_tensor = canvas_tensor.permute(2, 0, 1) # [C, H, W]
        rendered_tensors.append(canvas_tensor)
        
    return torch.stack(rendered_tensors, dim=0)


def mask_posenet_limb(points):
    """Hide one observed distal limb in the PoseNet drawing for this clip only."""
    ends = ((3, 4), (6, 7), (9, 10), (12, 13))  # elbows/wrists and knees/ankles
    visible = np.isfinite(points).all(axis=-1) & (points != -1).all(axis=-1)
    eligible = [pair for pair in ends if visible[:, pair].all(axis=1).any()]
    if not eligible:
        return points
    masked = points.copy()
    masked[:, random.choice(eligible), :] = -1
    return masked


class LargeScaleAnimationVideos(Dataset):
    # Spawned workers can receive dataset state created before this option existed.
    target_frame_pose = False
    ipi_limb_mask_prob = 0.0

    def __init__(self, root_path, txt_path, width, height, n_sample_frames, sample_frame_rate, sample_margin=30,
                 app=None, handler_ante=None, face_helper=None,
                 pose_aug_prob=0.1, pose_aug_scale=(0.95, 1.05), pose_aug_aspect=(0.85, 1.15),
                 clip_image_processor=None, target_frame_pose=False, ipi_limb_mask_prob=0.0):
        self.root_path = root_path
        self.txt_path = txt_path
        self.width = width
        self.height = height
        self.n_sample_frames = n_sample_frames
        self.sample_frame_rate = sample_frame_rate
        self.sample_margin = sample_margin
        self.target_frame_pose = target_frame_pose
        self.ipi_limb_mask_prob = ipi_limb_mask_prob

        self.video_files = self._read_txt_file_images()

        self.app = app
        self.handler_ante = handler_ante
        self.face_helper = face_helper
        
        # [IPI Mod] 初始化 CLIP Image Processor 用于处理驱动视频帧
        self.clip_image_processor = clip_image_processor if clip_image_processor is not None else CLIPImageProcessor()
        
        # [IPI Mod] 骨架增强参数 (借鉴 pose_align_withdiffaug.py / utils_aug.py)
        self.pose_aug_prob = pose_aug_prob        # 增强触发概率 (默认20%)
        self.pose_aug_scale = pose_aug_scale      # 整体缩放范围
        self.pose_aug_aspect = pose_aug_aspect     # 宽高比变化范围

    def _read_txt_file_images(self):
        with open(self.txt_path, 'r') as file:
            lines = file.readlines()
            video_files = []
            for line in lines:
                video_file = line.strip()
                # 兼容不同系统的路径
                video_files.append(video_file)
        return video_files

    def __len__(self):
        return len(self.video_files)

    def frame_count(self, frames_path):
        files = os.listdir(frames_path)
        png_files = [file for file in files if file.endswith('.png') or file.endswith('.jpg')]
        png_files_count = len(png_files)
        return png_files_count

    def find_frames_list(self, frames_path):
        files = os.listdir(frames_path)
        image_files = [file for file in files if file.endswith('.png') or file.endswith('.jpg')]
        # 更加健壮的排序逻辑
        try:
            if image_files[0].startswith('frame_'):
                image_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
            else:
                image_files.sort(key=lambda x: int(x.split('.')[0]))
        except:
            image_files.sort()
        return image_files

    # [IPI Mod] 读取 DWPose JSON 数据
    def load_dwpose_json(self, json_path, frame_indices):
        """
        从 json 文件中读取指定帧的姿态数据
        返回: (Num_Selected_Frames, 133, 2)
        """
        # 注意：这里假设你已经把姿态预处理成了 .json 或 .npy
        # 如果是 .npy 格式会更快：data = np.load(path)
        # 这里以 json 为例，假设结构是 {frame_name: [[x,y,s], ...]} 或类似结构
        # 为了高效，建议将整个视频的pose存为一个 .npy 文件 [Total_Frames, 133, 3]
        
        npy_path = json_path.replace('.json', '.npy') # 优先尝试读 npy
        # print(f"📁 Trying to load: {npy_path}")  
        if os.path.exists(npy_path):
            all_poses = np.load(npy_path) # [Total_Frames, 133, 3] (x, y, score)
            # 筛选选中的帧
            max_idx = all_poses.shape[0] - 1
            # 把所有超出 npy 长度的索引，强行限制在最后一样
            safe_frame_indices = [min(idx, max_idx) for idx in frame_indices]
            
            # 使用安全的索引切片
            selected_poses = all_poses[safe_frame_indices]
            # 只取 x, y
            return selected_poses[..., :2]
        
        # 兜底逻辑：如果没有预处理好的文件，返回全0 (避免训练中断，但要记得处理数据)
        print(f"Warning: Pose data not found for {json_path}")
        return np.zeros((len(frame_indices), 133, 2))

    # ===================================================================
    # [IPI Mod] 鲁棒性修复：NaN/Inf/丢帧/丢关键点处理
    #           借鉴 pose_align.py L60-70 的 NaN 处理逻辑
    # ===================================================================
    def sanitize_pose_data(self, pose_data):
        """
        鲁棒性处理：修复 NaN、Inf、全零帧、单点丢失
        
        策略：
        1. NaN / Inf → 标记为 -1 (无效)
        2. 整帧丢失(有效点<3) → 用相邻帧填充
        3. 单个关键点丢失 → 用前一帧同一关键点填充
        
        Args:
            pose_data: [F, 133, 2] numpy array
        Returns:
            pose_data: [F, 133, 2] 修复后的 numpy array
        """
        pose_data = pose_data.copy()
        num_frames = pose_data.shape[0]
        num_kps = pose_data.shape[1]
        
        # Step 1: NaN / Inf → -1 (标记为无效)
        nan_inf_mask = np.isnan(pose_data) | np.isinf(pose_data)
        if nan_inf_mask.any():
            pose_data[nan_inf_mask] = -1.0
        
        # Step 2: 整帧丢失检测 → 用相邻帧填充
        for f in range(num_frames):
            frame = pose_data[f]
            valid = (frame[..., 0] != -1) & (frame[..., 1] != -1) & \
                    ~((frame[..., 0] == 0) & (frame[..., 1] == 0))
            
            if valid.sum() < 3:  # 有效关键点少于3个，视为丢帧
                if f > 0:
                    pose_data[f] = pose_data[f - 1].copy()
                elif f < num_frames - 1:
                    pose_data[f] = pose_data[f + 1].copy()
        
        # Step 3: 逐关键点修复 → 前帧插值
        for f in range(1, num_frames):
            for kp in range(num_kps):
                x, y = pose_data[f, kp]
                if x == -1 or y == -1 or (x == 0 and y == 0):
                    prev_x, prev_y = pose_data[f - 1, kp]
                    if prev_x != -1 and prev_y != -1:
                        pose_data[f, kp] = pose_data[f - 1, kp].copy()
        
        return pose_data

    # ===================================================================
    # [IPI Mod] 骨架随机增强：打破模型的"静态粘性"
    #           借鉴 utils_aug.py 的 pose_aug_diff 函数
    #           核心：随机缩放 + 宽高比变化 + 中心点补偿
    # ===================================================================
    def pose_augmentation(self, pose_data):
        """
        骨架随机增强 (训练时使用)
        
        以 self.pose_aug_prob 的概率触发，对骨架施加：
        1. 随机整体缩放 (scale)
        2. 随机宽高比变化 (aspect_ratio，只影响x轴)
        3. 中心点补偿：保持1号点(脖子)不漂移
        
        Args:
            pose_data: [F, 133, 2] numpy array
        Returns:
            pose_data: [F, 133, 2] 增强后的 numpy array
        """
        if random.random() > self.pose_aug_prob:
            return pose_data  # 不增强，直接返回
        
        pose_aug = pose_data.copy()
        
        # 标记有效点 (非 -1)
        valid_mask = (pose_aug[..., 0] != -1) & (pose_aug[..., 1] != -1)
        
        if valid_mask.sum() == 0:
            return pose_data  # 全无效，无法增强
        
        # 随机增强参数
        scale = random.uniform(*self.pose_aug_scale)
        asp_ratio = random.uniform(*self.pose_aug_aspect)
        
        # 记录中心点(1号点=脖子)的原始位置，用于补偿
        # 1号点在 COCO-18 Body Keypoints 中是 neck
        center_before = pose_aug[:, 1, :].copy()  # [F, 2]
        
        # 施加宽高比变化 (只影响 x 轴)
        x_valid = valid_mask.copy()
        pose_aug[x_valid, 0] = pose_aug[x_valid, 0] * asp_ratio
        
        # 施加整体缩放
        for f in range(pose_aug.shape[0]):
            for kp in range(pose_aug.shape[1]):
                if valid_mask[f, kp]:
                    pose_aug[f, kp] = pose_aug[f, kp] * scale
        
        # 中心点补偿：保持1号点不动 (关键！防止增强后人物位置漂移)
        center_after = pose_aug[:, 1, :].copy()  # [F, 2]
        offset = center_before - center_after     # [F, 2]
        
        for f in range(pose_aug.shape[0]):
            for kp in range(pose_aug.shape[1]):
                if valid_mask[f, kp]:
                    pose_aug[f, kp] += offset[f]
        
        # 恢复无效点为 -1
        pose_aug[~valid_mask] = -1.0
        
        return pose_aug

    def __getitem__(self, idx):

        warnings.filterwarnings('ignore', category=DeprecationWarning)
        warnings.filterwarnings('ignore', category=FutureWarning)

        # 路径定义
        frames_path = osp.join(self.video_files[idx], "images")
        poses_path = osp.join(self.video_files[idx], "poses")
        face_masks_path = osp.join(self.video_files[idx], "faces")
        
        # [IPI Mod] 原始姿态数据路径 (假设存在 poses_raw 文件夹或 .npy 文件)
        # 建议结构：video_folder/poses_dense.npy
        dwpose_raw_path = osp.join(self.video_files[idx], "poses_dense133.npy") 
        if not os.path.exists(dwpose_raw_path):
             # 兼容 json 路径
             dwpose_raw_path = osp.join(self.video_files[idx], "poses_dense.npy")

        video_length = self.frame_count(frames_path)
        frames_list = self.find_frames_list(frames_path)

        # 采样逻辑 (保持不变)
        clip_length = min(video_length, (self.n_sample_frames - 1) * self.sample_frame_rate + 1)
        start_idx = random.randint(0, video_length - clip_length)
        batch_index = np.linspace(
            start_idx, start_idx + clip_length - 1, self.n_sample_frames, dtype=int
        ).tolist()
        
        # Reference Frame 采样逻辑 (保持不变)
        all_indices = list(range(0, video_length))
        available_indices = [i for i in all_indices if i not in batch_index]
        reference_frame_idx = None
        if available_indices:
            reference_frame_idx = random.choice(available_indices)
        else:
            # Extreme case handling (保持不变)
            extreme_sample_frame_rate = 2
            extreme_clip_length = min(video_length, (self.n_sample_frames - 1) * extreme_sample_frame_rate + 1)
            extreme_start_idx = random.randint(0, video_length - extreme_clip_length)
            extreme_batch_index = np.linspace(
                extreme_start_idx, extreme_start_idx + extreme_clip_length - 1, self.n_sample_frames, dtype=int
            ).tolist()
            extreme_available_indices = [i for i in all_indices if i not in extreme_batch_index]
            if extreme_available_indices:
                reference_frame_idx = random.choice(extreme_available_indices)
            else:
                reference_frame_idx = 0 # Fallback

        pose_pil_image_list = []
        tgt_pil_image_list = []
        tgt_face_masks_list = []
        
        # [IPI Mod] 存储驱动视频的原图 (PIL格式)，用于 CLIP 处理
        driving_clip_pil_list = []
        image_sizes = []

        # Reference Image 处理 (保持不变)
        reference_frame_path = osp.join(frames_path, frames_list[reference_frame_idx])
        reference_pil_image = Image.open(reference_frame_path).convert('RGB')
        reference_pil_image = reference_pil_image.resize((self.width, self.height))
        reference_pil_image_tensor = torch.from_numpy(np.array(reference_pil_image)).float()
        reference_pil_image_tensor = reference_pil_image_tensor / 127.5 - 1
        
        # Face Embedding 处理 (保持不变，省略部分细节以节省空间，逻辑照旧)
        # ... (这里是你原来的 face_helper 逻辑，保持原样即可) ...
        reference_frame_id_ante_embedding = np.zeros((512,)) # 占位，请保留你原来的逻辑

        # 循环读取帧
        for index in batch_index:
            tgt_img_path = osp.join(frames_path, frames_list[index])
            pose_name = os.path.splitext(os.path.basename(tgt_img_path))[0] + '.png'
            face_name = pose_name
            pose_path = osp.join(poses_path, pose_name)
            face_mask_path = osp.join(face_masks_path, face_name)

            # 读取目标图像
            try:
                tgt_img_pil = Image.open(tgt_img_path).convert('RGB')
                image_sizes.append(tgt_img_pil.size)
                
                # [IPI Mod] 收集原始图像用于 CLIP (resize 到 224x224 可以节省后续处理时间，或者保持原图交给 processor)
                # CLIP processor 默认会 resize 到 224，所以这里直接存原图或者 resize 后的图都可以
                # 建议 resize 到 self.width/height 保持一致性
                driving_clip_pil_list.append(tgt_img_pil.resize((self.width, self.height)))
                
            except Exception as e:
                if self.target_frame_pose:
                    raise ValueError(f"Cannot pair pose with target image {tgt_img_path}") from e
                print(f"Fail loading the image: {tgt_img_path}")
                # Fallback
                driving_clip_pil_list.append(Image.new('RGB', (self.width, self.height)))

            # 读取 Face Mask (保持不变)
            try:
                tgt_face_mask = Image.open(face_mask_path)
                tgt_face_mask = tgt_face_mask.resize((self.width, self.height))
                tgt_face_mask = torch.from_numpy(np.array(tgt_face_mask)).float() / 255
            except:
                tgt_face_mask = torch.ones(self.height, self.width, 1)
            tgt_face_masks_list.append(tgt_face_mask)

            # 处理目标图像 Tensor
            tgt_img_pil = tgt_img_pil.resize((self.width, self.height))
            tgt_img_tensor = torch.from_numpy(np.array(tgt_img_pil)).float()
            tgt_img_normalized = tgt_img_tensor / 127.5 - 1
            tgt_pil_image_list.append(tgt_img_normalized)

            # 读取 Pose 图片 (保持不变)
            try:
                pose = Image.open(pose_path).convert('RGB')
                pose = pose.resize((self.width, self.height))
                pose = torch.from_numpy(np.array(pose)).float() / 127.5 - 1
            except:
                pose = torch.zeros_like(reference_pil_image_tensor)
            pose_pil_image_list.append(pose)

        # 堆叠 Tensors (保持不变)
        pose_pil_image_list = torch.stack(pose_pil_image_list, dim=0)
        tgt_pil_image_list = torch.stack(tgt_pil_image_list, dim=0)
        
        # Rearrange (F H W C -> F C H W)
        tgt_pil_image_list = rearrange(tgt_pil_image_list, "f h w c -> f c h w")
        reference_pil_image_tensor = rearrange(reference_pil_image_tensor, "h w c -> c h w")
        pose_pil_image_list = rearrange(pose_pil_image_list, "f h w c -> f c h w")
        
        # === [数据清洗补丁：统一 Mask 维度] ===
        cleaned_masks = []
        for m in tgt_face_masks_list:
            # 如果末尾多了一个 1 通道 [512, 512, 1]，就把 1 挤掉
            if m.dim() == 3 and m.shape[-1] == 1:
                m = m.squeeze(-1)
            # 如果开头多了一个 1 通道 [1, 512, 512]，也把 1 挤掉
            elif m.dim() == 3 and m.shape[0] == 1:
                m = m.squeeze(0)
            cleaned_masks.append(m)
            
        tgt_face_masks_list = torch.stack(cleaned_masks, dim=0)
        # =======================================
        tgt_face_masks_list = torch.unsqueeze(tgt_face_masks_list, dim=-1)
        tgt_face_masks_list = rearrange(tgt_face_masks_list, "f h w c -> f c h w")

        # -------------------------------------------------------------------
        # === [IPI Mod] 1. 加载 DWPose Raw Data ===
        # -------------------------------------------------------------------
        # 假设你有一个 load_dwpose_json 函数或者直接读取 .npy
        # 这里为了演示，生成一个全0的 Tensor，你需要替换成真实的读取逻辑
        if self.target_frame_pose:
            dwpose_raw = load_target_frame_pose(dwpose_raw_path, batch_index, video_length, image_sizes)
        else:
            dwpose_raw = self.sanitize_pose_data(self.load_dwpose_json(dwpose_raw_path, batch_index))
            if reference_frame_idx is not None:
                ref_pose_raw = self.load_dwpose_json(dwpose_raw_path, [reference_frame_idx])
                ref_pose_clean = self.sanitize_pose_data(ref_pose_raw)[0]
                first_drv_pose = dwpose_raw[0]
                body_scales = compute_body_scales(ref_pose_clean, first_drv_pose)
                necks = np.stack((ref_pose_clean[1], first_drv_pose[1]))
                center_offset = (necks[0] - necks[1] if np.isfinite(necks).all() and (necks != -1).all()
                                 else np.zeros(2, dtype=dwpose_raw.dtype))
                for i in range(len(dwpose_raw)):
                    dwpose_raw[i] = align_body_pose_single(dwpose_raw[i], body_scales)
                    valid = (dwpose_raw[i] != -1).all(axis=-1)
                    dwpose_raw[i][valid] += center_offset
            dwpose_raw = self.pose_augmentation(dwpose_raw)
        
        # -------------------------------------------------------------------
        # [IPI Mod] 方案 B 核心：利用拉伸变形后的最新真实坐标重绘彩色骨架图片
        #           因为如果数值骨架和输入图像不一致，可能会混淆模型
        #           (返回的已是归一化过的 [F, C, H, W] Tensor)
        # -------------------------------------------------------------------
        spatial_points = dwpose_raw
        if self.ipi_limb_mask_prob and random.random() < self.ipi_limb_mask_prob:
            spatial_points = mask_posenet_limb(dwpose_raw)
        pose_pil_image_list = render_pose_frames(spatial_points, self.width, self.height)

        
        # 归一化
        dwpose_raw = normalize_pose(dwpose_raw, self.width, self.height)
        dwpose_raw_tensor = torch.from_numpy(dwpose_raw).float()
        # Flatten: [F, 133, 2] -> [F, 266]
        dwpose_raw_tensor = dwpose_raw_tensor.view(self.n_sample_frames, -1)

        # -------------------------------------------------------------------
        # === [IPI Mod] 2. 处理 CLIP 图像输入 ===
        # -------------------------------------------------------------------
        # 使用 CLIPImageProcessor 处理所有帧
        # output['pixel_values'] shape: [Frames, 3, 224, 224]
        clip_inputs = self.clip_image_processor(images=driving_clip_pil_list, return_tensors="pt")
        driving_clip_pixel_values = clip_inputs.pixel_values

        # -------------------------------------------------------------------
        # [魏总强力补丁] 针对雕像数据的 IPI 语义切断
        # -------------------------------------------------------------------
        if "statue" in self.video_files[idx] or "clay" in self.video_files[idx]:
            import torchvision.transforms.functional as TF
            import torchvision.transforms as T
            # 对 [F, 3, 224, 224] 的 CLIP 输入进行致盲
            # 1. 强制去色 (防止肤色干扰)
            driving_clip_pixel_values = TF.rgb_to_grayscale(driving_clip_pixel_values, num_output_channels=3)
            # 2. 强力模糊 (防止五官干扰，留存运动影子)
            blur_op = T.GaussianBlur(kernel_size=(15, 15), sigma=(5.0, 5.0))
            driving_clip_pixel_values = blur_op(driving_clip_pixel_values)

        sample = dict(
            pixel_values=tgt_pil_image_list,
            reference_image=reference_pil_image_tensor,
            pose_pixels=pose_pil_image_list,
            faceid_embeds=reference_frame_id_ante_embedding,
            tgt_face_masks=tgt_face_masks_list,
            # [IPI Mod] 新增字段
            dwpose_raw=dwpose_raw_tensor,
            driving_clip_pixel_values=driving_clip_pixel_values
        )

        return sample
