import numpy as np
from scipy.signal import savgol_filter
import cv2

def smooth_pose_sequence(pose_data, window_length=7, polyorder=2):
    """
    对 DWPose 的 .npy 时序序列进行 Savitzky-Golay 平滑滤波。
    pose_data shape: (Frames, 18, 2)
    """
    # 拷贝一份避免污染原数据
    smoothed_poses = np.copy(pose_data)
    frames, num_keypoints, dims = pose_data.shape
    
    # 窗口大小必须是奇数，且不能大于总帧数
    window_length = min(window_length, frames)
    if window_length % 2 == 0:
        window_length -= 1
    if window_length <= 3:
        return smoothed_poses # 帧数太少就不平滑了
        
    for k in range(num_keypoints):
        for d in range(dims):
            signal = pose_data[:, k, d]
            
            # 🛡️ 护盾：找出有效的点 (排除掉识别失败的负数点)
            valid_mask = signal >= 0
            valid_indices = np.where(valid_mask)[0]
            valid_values = signal[valid_mask]
            
            # 只有有效点数量大于窗口大小时才进行平滑
            if len(valid_values) > window_length:
                # 🚀 核心魔法：S-G 滤波削峰填谷
                smoothed_values = savgol_filter(valid_values, window_length, polyorder)
                
                # 把平滑后的漂亮曲线塞回原位置
                smoothed_poses[valid_indices, k, d] = smoothed_values
                
    return smoothed_poses

# ================= 测试代码 =================
if __name__ == "__main__":
    # 替换为你那个看着发抖的 npy 路径
    bad_npy_path = "/gaojiawei/wjl/StableAnimator/animation_data/dec/30513/poses_dense.npy"
    
    # 1. 加载并平滑
    raw_poses = np.load(bad_npy_path)
    perfect_poses = smooth_pose_sequence(raw_poses, window_length=7) # 窗口设为5-9之间最好
    
    # 2. 保存为你喜欢的名字
    smoothed_path = bad_npy_path.replace(".npy", ".npy")
    np.save(smoothed_path, perfect_poses)
    
    print(f"🎉 平滑完成！拯救了 {raw_poses.shape[0]} 帧的帕金森骨架。")
    print(f"新文件已保存至: {smoothed_path}")
    print("快把这个新的 npy 塞进你的视频生成脚本(check_video_alignment.py)里看看效果吧！")