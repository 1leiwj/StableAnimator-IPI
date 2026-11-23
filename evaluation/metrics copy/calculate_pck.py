import os
import sys
import cv2
import numpy as np

# 添加项目根目录以导入 DWPose
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        from DWPose.dwpose_utils.dwpose_detector import dwpose_detector_aligned
    except ImportError:
        dwpose_detector_aligned = None

class ProportionEvaluator:
    def __init__(self):
        # COCO 18 keypoints 的骨骼连接定义
        # 确保在 DWPose body 格式下依然适用
        self.limb_pairs = {
            'L_UPPER_ARM': (5, 6),
            'L_LOWER_ARM': (6, 7),
            'R_UPPER_ARM': (2, 3),
            'R_LOWER_ARM': (3, 4),
            'L_THIGH': (11, 12),
            'L_CALF': (12, 13),
            'R_THIGH': (8, 9),
            'R_CALF': (9, 10)
        }
        
    def _get_joint(self, candidate, subset, joint_idx):
        """安全地从 DWPose 的输出中提取对应索引的坐标，并过滤低置信度噪点"""
        if len(subset) == 0:
            return None
        idx = int(subset[0][joint_idx])
        if idx == -1 or idx >= len(candidate):
            return None
            
        pt = candidate[idx]
        # B. 引入置信度过滤 (Confidence Score Filtering)
        conf = pt[2] if len(pt) > 2 else 1.0
        if conf < 0.3: # 置信度阈值，剔除因遮挡或转体导致的检测畸变
            return None
            
        return pt[:2]

    def _calculate_pose_ratios(self, pose):
        """计算给定骨架的各项肢体长度与躯干长度的比值"""
        if 'bodies' not in pose or 'candidate' not in pose['bodies'] or 'subset' not in pose['bodies']:
            return None
            
        candidate = pose['bodies']['candidate']
        subset = pose['bodies']['subset']
        if len(subset) == 0:
            return None
            
        # 躯干长度作为归一化基准: 脖子(1) 到 髋部中心点(8和11的中心)
        neck = self._get_joint(candidate, subset, 1)
        l_hip = self._get_joint(candidate, subset, 11)
        r_hip = self._get_joint(candidate, subset, 8)
        
        if neck is None or l_hip is None or r_hip is None:
            return None
            
        mid_hip = (l_hip + r_hip) / 2.0
        torso_len = np.linalg.norm(neck - mid_hip)
        
        if torso_len < 1e-5: # 防止除以零
            return None
            
        ratios = {}
        for limb_name, (j1, j2) in self.limb_pairs.items():
            p1 = self._get_joint(candidate, subset, j1)
            p2 = self._get_joint(candidate, subset, j2)
            if p1 is not None and p2 is not None:
                limb_len = np.linalg.norm(p1 - p2)
                ratios[limb_name] = limb_len / torso_len
                
        return ratios

    def extract_image_ratios(self, img_path):
        """提取参考原图（雕像）的金标准解剖学比例"""
        if dwpose_detector_aligned is None:
            return None
            
        img = cv2.imread(img_path)
        if img is None:
            return None
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            pose = dwpose_detector_aligned(img_rgb)
            return self._calculate_pose_ratios(pose)
        except Exception:
            return None

    def extract_video_ratios(self, video_path):
        """
        提取生成视频的最大肢体比例（A. 解决“透视收缩”导致的误判）
        取最大值通常代表该肢体完全展开、无透视缩短的状态，
        同时也能精准捕获模型产生的“肢体异常拉伸畸变”。
        """
        if dwpose_detector_aligned is None:
            return None
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
            
        all_ratios = {k: [] for k in self.limb_pairs.keys()}
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                pose = dwpose_detector_aligned(frame_rgb)
                ratios = self._calculate_pose_ratios(pose)
                if ratios is not None:
                    for k, v in ratios.items():
                        all_ratios[k].append(v)
            except Exception:
                pass
                
        cap.release()
        
        # 记录视频中肢体长度的最大值
        max_ratios = {}
        for k, v_list in all_ratios.items():
            if len(v_list) > 0:
                max_ratios[k] = float(np.max(v_list))
                
        return max_ratios if len(max_ratios) > 0 else None

    def calculate_proportion_consistency(self, ref_image_path, gen_video_path):
        """
        相对骨骼比例一致性得分 (Relative Proportion Consistency Score, RPCS)
        C. 从 MAE 升级到 MAPE（平均绝对百分比误差）
        """
        if dwpose_detector_aligned is None:
            raise ImportError("DWPose is not properly installed or imported.")
            
        ref_ratios = self.extract_image_ratios(ref_image_path)
        gen_ratios = self.extract_video_ratios(gen_video_path)
        
        if ref_ratios is None or gen_ratios is None:
            return None
            
        relative_errors = []
        for limb in self.limb_pairs.keys():
            if limb in ref_ratios and limb in gen_ratios:
                ref = ref_ratios[limb]
                gen = gen_ratios[limb]
                if ref > 1e-5:
                    # 使用相对误差百分比
                    relative_error = abs(ref - gen) / ref
                    relative_errors.append(relative_error)
                
        if not relative_errors:
            return None
            
        mape = np.mean(relative_errors)
        # 将 MAPE 转换为一致性得分，最高 1.0，最低 0.0
        score = max(0.0, 1.0 - mape)
        return float(score)
