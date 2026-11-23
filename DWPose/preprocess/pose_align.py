import numpy as np
import argparse
import torch
import copy
import cv2
import os
import moviepy.video.io.ImageSequenceClip
import sys
from tqdm import tqdm
from pose.script.dwpose import DWposeDetector, draw_pose
from pose.script.util import size_calculate, warpAffine_kps
from utils_aug import pose_aug_diff

# ========== 🚀 新增：导入 133 点 Wholebody 依赖 ==========
sys.path.append('/gaojiawei/wjl/StableAnimator/DWPose')
from dwpose.wholebody import Wholebody
# =========================================================

'''
    Detect dwpose from img, then align it by scale parameters
    img: frame from the pose video
    detector: DWpose
    scales: scale parameters
'''
# =====================================================================
# 🚀 新增：用于提取 133 点稠密骨架的类与函数 (融入了防黑边与防干扰魔法)
# =====================================================================
class DenseWholebodyDetector:
    def __init__(self):
        self.pose_estimation = Wholebody()

    def __call__(self, oriImg):
        oriImg = oriImg.copy()
        H, W, C = oriImg.shape
        
        with torch.no_grad():
            candidate, subset = self.pose_estimation(oriImg)
            
            if candidate.shape[0] == 0:
                return None

            # 防干扰逻辑：锁定主人物
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

            candidate = candidate[main_person_idx][np.newaxis, :, :]
            subset = subset[main_person_idx][np.newaxis, :]
            
            # 归一化抵消拉伸形变
            norm_candidate = candidate.copy()
            norm_candidate[..., 0] /= float(W)
            norm_candidate[..., 1] /= float(H)
            
            nums, keys, locs = norm_candidate.shape
            body = norm_candidate[:, :133, :].copy()
            body = body.reshape(nums * 133, locs)
            score = subset[:, :133].copy()
            
            for i in range(len(score)):
                for j in range(len(score[i])):
                    if score[i][j] <= 0.3:
                        body[i*133 + j] = [-1.0, -1.0]

            return body

def extract_dense_poses_to_npy(target_image_folder_path, output_npy_path):
    print(f"\n🚀 开始从对齐后的画面提取 133 点高精度骨架...")
    dwpose_model = DenseWholebodyDetector()
    files = os.listdir(target_image_folder_path)
    png_files = [f for f in files if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    try:
        png_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]) if 'frame_' in x else x)
    except:
        png_files.sort()
    
    all_bodies = [] 
    for idx, img_file in enumerate(tqdm(png_files, desc="Detecting poses")):
        img_path = os.path.join(target_image_folder_path, img_file)
        image = cv2.imread(img_path) 
        if image is None: continue
        
        try:
            # 强行拉伸绕过黑边 Bug
            trick_image = cv2.resize(image, (768, 1024))
            body_keypoints = dwpose_model(trick_image)
            
            if body_keypoints is not None:
                all_bodies.append(body_keypoints)
            else:
                if idx > 0 and len(all_bodies) > 0:
                    all_bodies.append(all_bodies[-1].copy())
                else:
                    all_bodies.append(np.zeros((133, 2)) - 1.0)
        except Exception as e:
            print(f"❌ Error in frame {idx}: {e}")
            all_bodies.append(np.zeros((133, 2)) - 1.0)
    
    all_bodies = np.array(all_bodies, dtype=np.float32)
    valid_data = all_bodies[all_bodies != -1]
    
    np.save(output_npy_path, all_bodies)
    print(f"✅ 提取完毕! 最终数据 Shape: {all_bodies.shape}")
    print(f"📊 Range: min={valid_data.min():.4f}, max={valid_data.max():.4f}")
    print(f"💾 稠密骨架已保存至: {output_npy_path}")
# =====================================================================

def align_img(img, pose_ori, scales, detect_resolution, image_resolution):

    body_pose = copy.deepcopy(pose_ori['bodies']['candidate'])
    hands = copy.deepcopy(pose_ori['hands'])
    faces = copy.deepcopy(pose_ori['faces'])

    '''
    计算逻辑:
    0. 该函数内进行绝对变换，始终保持人体中心点 body_pose[1] 不变
    1. 先把 ref 和 pose 的高 resize 到一样，且都保持原来的长宽比。
    2. 用点在图中的实际坐标来计算。
    3. 实际计算中，把h的坐标归一化到 [0, 1],  w为[0, W/H]
    4. 由于 dwpose 的输出本来就是归一化的坐标，所以h不需要变，w要乘W/H
    注意：dwpose 输出是 (w, h)
    '''

    # h不变，w缩放到原比例
    H_in, W_in, C_in = img.shape 
    video_ratio = W_in / H_in
    body_pose[:, 0]  = body_pose[:, 0] * video_ratio
    hands[:, :, 0] = hands[:, :, 0] * video_ratio
    faces[:, :, 0] = faces[:, :, 0] * video_ratio

    # scales of 10 body parts 
    scale_neck      = scales["scale_neck"] 
    # scale_face      = scales["scale_face"]
    scale_face_left = scales["scale_face_left"]
    scale_face_right  = scales["scale_face_right"]
    scale_shoulder  = scales["scale_shoulder"]
    scale_arm_upper = scales["scale_arm_upper"]
    scale_arm_lower = scales["scale_arm_lower"]
    scale_hand      = scales["scale_hand"]
    scale_body_len  = scales["scale_body_len"]
    scale_leg_upper = scales["scale_leg_upper"]
    scale_leg_lower = scales["scale_leg_lower"]

    scale_sum = 0
    count = 0
    # scale_list = [scale_neck, scale_face, scale_shoulder, scale_arm_upper, scale_arm_lower, scale_hand, scale_body_len, scale_leg_upper, scale_leg_lower]
    scale_list = [scale_neck, scale_face_left, scale_face_right, scale_shoulder, scale_arm_upper, scale_arm_lower, scale_hand, scale_body_len, scale_leg_upper, scale_leg_lower]
    for i in range(len(scale_list)):
        if not np.isinf(scale_list[i]):
            scale_sum = scale_sum + scale_list[i]
            count = count + 1
    for i in range(len(scale_list)):
        if np.isinf(scale_list[i]):   
            scale_list[i] = scale_sum/count



    # offsets of each part 
    offset = dict()
    # offset["14_15_16_17_to_0"] = body_pose[[14,15,16,17], :] - body_pose[[0], :]
    offset["14_16_to_0"] = body_pose[[14,16], :] - body_pose[[0], :]
    offset["15_17_to_0"] = body_pose[[15,17], :] - body_pose[[0], :]
    offset["3_to_2"] = body_pose[[3], :] - body_pose[[2], :] 
    offset["4_to_3"] = body_pose[[4], :] - body_pose[[3], :] 
    offset["6_to_5"] = body_pose[[6], :] - body_pose[[5], :] 
    offset["7_to_6"] = body_pose[[7], :] - body_pose[[6], :] 
    offset["9_to_8"] = body_pose[[9], :] - body_pose[[8], :] 
    offset["10_to_9"] = body_pose[[10], :] - body_pose[[9], :] 
    offset["12_to_11"] = body_pose[[12], :] - body_pose[[11], :] 
    offset["13_to_12"] = body_pose[[13], :] - body_pose[[12], :] 
    offset["hand_left_to_4"] = hands[1, :, :] - body_pose[[4], :]
    offset["hand_right_to_7"] = hands[0, :, :] - body_pose[[7], :]

    # neck
    c_ = body_pose[1]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx,cy), 0, scale_neck)

    neck = body_pose[[0], :] 
    neck = warpAffine_kps(neck, M)
    body_pose[[0], :] = neck

    # # body_pose_up_shoulder
    # c_ = body_pose[0]
    # cx = c_[0]
    # cy = c_[1]
    # M = cv2.getRotationMatrix2D((cx,cy), 0, scale_face)

    # body_pose_up_shoulder = offset["14_15_16_17_to_0"] + body_pose[[0], :]
    # body_pose_up_shoulder = warpAffine_kps(body_pose_up_shoulder, M)
    # body_pose[[14,15,16,17], :] = body_pose_up_shoulder

    # body_pose_up_shoulder left
    c_ = body_pose[0]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx,cy), 0, scale_face_left)

    body_pose_up_shoulder = offset["14_16_to_0"] + body_pose[[0], :]
    body_pose_up_shoulder = warpAffine_kps(body_pose_up_shoulder, M)
    body_pose[[14,16], :] = body_pose_up_shoulder


    # body_pose_up_shoulder right
    c_ = body_pose[0]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx,cy), 0, scale_face_right)

    body_pose_up_shoulder = offset["15_17_to_0"] + body_pose[[0], :]
    body_pose_up_shoulder = warpAffine_kps(body_pose_up_shoulder, M)
    body_pose[[15,17], :] = body_pose_up_shoulder

    # shoulder 
    c_ = body_pose[1]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx,cy), 0, scale_shoulder)

    body_pose_shoulder = body_pose[[2,5], :] 
    body_pose_shoulder = warpAffine_kps(body_pose_shoulder, M) 
    body_pose[[2,5], :] = body_pose_shoulder

    # arm upper left
    c_ = body_pose[2]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx,cy), 0, scale_arm_upper)
 
    elbow = offset["3_to_2"] + body_pose[[2], :]
    elbow = warpAffine_kps(elbow, M)
    body_pose[[3], :] = elbow

    # arm lower left
    c_ = body_pose[3]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx,cy), 0, scale_arm_lower)
 
    wrist = offset["4_to_3"] + body_pose[[3], :]
    wrist = warpAffine_kps(wrist, M)
    body_pose[[4], :] = wrist

    # hand left
    c_ = body_pose[4]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx,cy), 0, scale_hand)
 
    hand = offset["hand_left_to_4"] + body_pose[[4], :]
    hand = warpAffine_kps(hand, M)
    hands[1, :, :] = hand

    # arm upper right
    c_ = body_pose[5]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx,cy), 0, scale_arm_upper)
 
    elbow = offset["6_to_5"] + body_pose[[5], :]
    elbow = warpAffine_kps(elbow, M)
    body_pose[[6], :] = elbow

    # arm lower right
    c_ = body_pose[6]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx,cy), 0, scale_arm_lower)
 
    wrist = offset["7_to_6"] + body_pose[[6], :]
    wrist = warpAffine_kps(wrist, M)
    body_pose[[7], :] = wrist

    # hand right
    c_ = body_pose[7]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx,cy), 0, scale_hand)
 
    hand = offset["hand_right_to_7"] + body_pose[[7], :]
    hand = warpAffine_kps(hand, M)
    hands[0, :, :] = hand

    # body len
    c_ = body_pose[1]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx,cy), 0, scale_body_len)

    body_len = body_pose[[8,11], :] 
    body_len = warpAffine_kps(body_len, M)
    body_pose[[8,11], :] = body_len

    # leg upper left
    c_ = body_pose[8]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx,cy), 0, scale_leg_upper)
 
    knee = offset["9_to_8"] + body_pose[[8], :]
    knee = warpAffine_kps(knee, M)
    body_pose[[9], :] = knee

    # leg lower left
    c_ = body_pose[9]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx,cy), 0, scale_leg_lower)
 
    ankle = offset["10_to_9"] + body_pose[[9], :]
    ankle = warpAffine_kps(ankle, M)
    body_pose[[10], :] = ankle

    # leg upper right
    c_ = body_pose[11]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx,cy), 0, scale_leg_upper)
 
    knee = offset["12_to_11"] + body_pose[[11], :]
    knee = warpAffine_kps(knee, M)
    body_pose[[12], :] = knee

    # leg lower right
    c_ = body_pose[12]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx,cy), 0, scale_leg_lower)
 
    ankle = offset["13_to_12"] + body_pose[[12], :]
    ankle = warpAffine_kps(ankle, M)
    body_pose[[13], :] = ankle

    # none part
    body_pose_none = pose_ori['bodies']['candidate'] == -1.
    hands_none = pose_ori['hands'] == -1.
    faces_none = pose_ori['faces'] == -1.

    body_pose[body_pose_none] = -1.
    hands[hands_none] = -1. 
    nan = float('nan')
    if len(hands[np.isnan(hands)]) > 0:
        print('nan')
    faces[faces_none] = -1.

    # last check nan -> -1.
    body_pose = np.nan_to_num(body_pose, nan=-1.)
    hands = np.nan_to_num(hands, nan=-1.)
    faces = np.nan_to_num(faces, nan=-1.)

    # return
    pose_align = copy.deepcopy(pose_ori)
    pose_align['bodies']['candidate'] = body_pose
    pose_align['hands'] = hands
    pose_align['faces'] = faces

    return pose_align



def run_align_video_with_filterPose_translate_smooth(args):

    vidfn=args.vidfn
    imgfn_refer=args.imgfn_refer
    outfn=args.outfn
    
    video = cv2.VideoCapture(vidfn)
    width= video.get(cv2.CAP_PROP_FRAME_WIDTH)
    height= video.get(cv2.CAP_PROP_FRAME_HEIGHT)
 
    total_frame= video.get(cv2.CAP_PROP_FRAME_COUNT)
    fps= video.get(cv2.CAP_PROP_FPS)

    print("height:", height)
    print("width:", width)
    print("fps:", fps)

    H_in, W_in  = height, width
    H_out, W_out = size_calculate(H_in,W_in,args.detect_resolution) 
    H_out, W_out = size_calculate(H_out,W_out,args.image_resolution) 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    detector = DWposeDetector(
        det_config = args.yolox_config, 
        det_ckpt = args.yolox_ckpt,
        pose_config = args.dwpose_config, 
        pose_ckpt = args.dwpose_ckpt, 
        keypoints_only=False
        )    
    detector = detector.to(device)

    #### refer_img 的精确处理 在前边直接处理完毕，就不需要考虑后续 pose 的二次处理 ####
    refer_img = cv2.imread(imgfn_refer)
    ref_height, ref_width, channels = refer_img.shape
    # print("ref_height: ", ref_height)
    # print("ref_width: ", ref_width)
    aspect_ratio = ref_height / ref_width
    # max_area = "832*480"
    max_area = "1024*576"
    # max_area = "1664*960"
    lat_h = round(
        np.sqrt(int(eval(max_area)) * aspect_ratio) // 16)
    lat_w = round(
        np.sqrt(int(eval(max_area)) / aspect_ratio) // 16)
    new_height = lat_h * 16
    new_width = lat_w * 16
    # print("new_height:", new_height)
    # print("new_width:", new_width)
    # resize_height = int(ref_height*(1.0 * new_height/ref_height))
    # resize_width = int(ref_width*(1.0 * new_width/ref_width))
    # resize_height = new_height * 2
    # resize_width = new_width * 2
    resize_height = new_height
    resize_width = new_width
    # print("resize_height:", resize_height)
    # print("resize_width:", resize_width)
    refer_img = cv2.resize(refer_img, (resize_width, resize_height), interpolation=cv2.INTER_CUBIC)
    ref_height, ref_width, channels = refer_img.shape

    output_refer, pose_refer = detector(refer_img,detect_resolution=args.detect_resolution, image_resolution=args.image_resolution, output_type='cv2',return_pose_dict=True)
    body_ref_img  = pose_refer['bodies']['candidate']
    hands_ref_img = pose_refer['hands']
    faces_ref_img = pose_refer['faces']
    output_refer = cv2.cvtColor(output_refer, cv2.COLOR_RGB2BGR)
    

    skip_frames = args.align_frame
    max_frame = args.max_frame
    pose_list, video_frame_buffer, video_pose_buffer = [], [], []

    for i in range(max_frame):
        ret, img = video.read()
        if img is None: 
            break 
        else: 
            if i < skip_frames:
                continue           
            video_frame_buffer.append(img)


       
        # estimate scale parameters by the 1st frame in the video
        if i==skip_frames:
            output_1st_img, pose_1st_img = detector(img, args.detect_resolution, args.image_resolution, output_type='cv2', return_pose_dict=True)
            body_1st_img  = pose_1st_img['bodies']['candidate']
            hands_1st_img = pose_1st_img['hands']
            faces_1st_img = pose_1st_img['faces']

            '''
            计算逻辑:
            1. 先把 ref 和 pose 的高 resize 到一样，且都保持原来的长宽比。
            2. 用点在图中的实际坐标来计算。
            3. 实际计算中，把h的坐标归一化到 [0, 1],  w为[0, W/H]
            4. 由于 dwpose 的输出本来就是归一化的坐标，所以h不需要变，w要乘W/H
            注意：dwpose 输出是 (w, h)
            '''
            
            # h不变，w缩放到原比例
            ref_H, ref_W = refer_img.shape[0], refer_img.shape[1]
            ref_ratio = ref_W / ref_H
            body_ref_img[:, 0]  = body_ref_img[:, 0] * ref_ratio
            hands_ref_img[:, :, 0] = hands_ref_img[:, :, 0] * ref_ratio
            faces_ref_img[:, :, 0] = faces_ref_img[:, :, 0] * ref_ratio

            video_ratio = width / height
            body_1st_img[:, 0]  = body_1st_img[:, 0] * video_ratio
            hands_1st_img[:, :, 0] = hands_1st_img[:, :, 0] * video_ratio
            faces_1st_img[:, :, 0] = faces_1st_img[:, :, 0] * video_ratio

            # scale
            align_args = dict()
            
            dist_1st_img = np.linalg.norm(body_1st_img[0]-body_1st_img[1])   # 0.078   
            dist_ref_img = np.linalg.norm(body_ref_img[0]-body_ref_img[1])   # 0.106
            align_args["scale_neck"] = dist_ref_img / dist_1st_img  # align / pose = ref / 1st

            # dist_1st_img = np.linalg.norm(body_1st_img[16]-body_1st_img[17])
            # dist_ref_img = np.linalg.norm(body_ref_img[16]-body_ref_img[17])
            # align_args["scale_face"] = dist_ref_img / dist_1st_img

            dist_1st_img = np.linalg.norm(body_1st_img[16]-body_1st_img[14]) + np.linalg.norm(body_1st_img[14]-body_1st_img[0])
            dist_ref_img = np.linalg.norm(body_ref_img[16]-body_ref_img[14]) + np.linalg.norm(body_ref_img[14]-body_ref_img[0])
            align_args["scale_face_left"] = dist_ref_img / dist_1st_img

            dist_1st_img = np.linalg.norm(body_1st_img[17]-body_1st_img[15]) + np.linalg.norm(body_1st_img[15]-body_1st_img[0])
            dist_ref_img = np.linalg.norm(body_ref_img[17]-body_ref_img[15]) + np.linalg.norm(body_ref_img[15]-body_ref_img[0])
            align_args["scale_face_right"] = dist_ref_img / dist_1st_img

            dist_1st_img = np.linalg.norm(body_1st_img[2]-body_1st_img[5])  # 0.112
            dist_ref_img = np.linalg.norm(body_ref_img[2]-body_ref_img[5])  # 0.174
            align_args["scale_shoulder"] = dist_ref_img / dist_1st_img  

            dist_1st_img = np.linalg.norm(body_1st_img[2]-body_1st_img[3])  # 0.895
            dist_ref_img = np.linalg.norm(body_ref_img[2]-body_ref_img[3])  # 0.134
            s1 = dist_ref_img / dist_1st_img
            dist_1st_img = np.linalg.norm(body_1st_img[5]-body_1st_img[6])
            dist_ref_img = np.linalg.norm(body_ref_img[5]-body_ref_img[6])
            s2 = dist_ref_img / dist_1st_img
            align_args["scale_arm_upper"] = (s1+s2)/2 # 1.548

            dist_1st_img = np.linalg.norm(body_1st_img[3]-body_1st_img[4])
            dist_ref_img = np.linalg.norm(body_ref_img[3]-body_ref_img[4])
            s1 = dist_ref_img / dist_1st_img
            dist_1st_img = np.linalg.norm(body_1st_img[6]-body_1st_img[7])
            dist_ref_img = np.linalg.norm(body_ref_img[6]-body_ref_img[7])
            s2 = dist_ref_img / dist_1st_img
            align_args["scale_arm_lower"] = (s1+s2)/2

            # hand
            dist_1st_img = np.zeros(10)
            dist_ref_img = np.zeros(10)      
             
            dist_1st_img[0] = np.linalg.norm(hands_1st_img[0,0]-hands_1st_img[0,1])
            dist_1st_img[1] = np.linalg.norm(hands_1st_img[0,0]-hands_1st_img[0,5])
            dist_1st_img[2] = np.linalg.norm(hands_1st_img[0,0]-hands_1st_img[0,9])
            dist_1st_img[3] = np.linalg.norm(hands_1st_img[0,0]-hands_1st_img[0,13])
            dist_1st_img[4] = np.linalg.norm(hands_1st_img[0,0]-hands_1st_img[0,17])
            dist_1st_img[5] = np.linalg.norm(hands_1st_img[1,0]-hands_1st_img[1,1])
            dist_1st_img[6] = np.linalg.norm(hands_1st_img[1,0]-hands_1st_img[1,5])
            dist_1st_img[7] = np.linalg.norm(hands_1st_img[1,0]-hands_1st_img[1,9])
            dist_1st_img[8] = np.linalg.norm(hands_1st_img[1,0]-hands_1st_img[1,13])
            dist_1st_img[9] = np.linalg.norm(hands_1st_img[1,0]-hands_1st_img[1,17])

            dist_ref_img[0] = np.linalg.norm(hands_ref_img[0,0]-hands_ref_img[0,1])
            dist_ref_img[1] = np.linalg.norm(hands_ref_img[0,0]-hands_ref_img[0,5])
            dist_ref_img[2] = np.linalg.norm(hands_ref_img[0,0]-hands_ref_img[0,9])
            dist_ref_img[3] = np.linalg.norm(hands_ref_img[0,0]-hands_ref_img[0,13])
            dist_ref_img[4] = np.linalg.norm(hands_ref_img[0,0]-hands_ref_img[0,17])
            dist_ref_img[5] = np.linalg.norm(hands_ref_img[1,0]-hands_ref_img[1,1])
            dist_ref_img[6] = np.linalg.norm(hands_ref_img[1,0]-hands_ref_img[1,5])
            dist_ref_img[7] = np.linalg.norm(hands_ref_img[1,0]-hands_ref_img[1,9])
            dist_ref_img[8] = np.linalg.norm(hands_ref_img[1,0]-hands_ref_img[1,13])
            dist_ref_img[9] = np.linalg.norm(hands_ref_img[1,0]-hands_ref_img[1,17])

            ratio = 0   
            count = 0
            total_iters = 0 # 10
            for i in range (total_iters):
                if dist_1st_img[i] != 0:
                    ratio = ratio + dist_ref_img[i]/dist_1st_img[i]
                    count = count + 1
            if count!=0:
                align_args["scale_hand"] = (ratio/count+align_args["scale_arm_upper"]+align_args["scale_arm_lower"])/3
            else:
                align_args["scale_hand"] = (align_args["scale_arm_upper"]+align_args["scale_arm_lower"])/2

            # body 
            dist_1st_img = np.linalg.norm(body_1st_img[1] - (body_1st_img[8] + body_1st_img[11])/2 )
            dist_ref_img = np.linalg.norm(body_ref_img[1] - (body_ref_img[8] + body_ref_img[11])/2 )
            align_args["scale_body_len"]=dist_ref_img / dist_1st_img

            dist_1st_img = np.linalg.norm(body_1st_img[8]-body_1st_img[9])
            dist_ref_img = np.linalg.norm(body_ref_img[8]-body_ref_img[9])
            s1 = dist_ref_img / dist_1st_img
            dist_1st_img = np.linalg.norm(body_1st_img[11]-body_1st_img[12])
            dist_ref_img = np.linalg.norm(body_ref_img[11]-body_ref_img[12])
            s2 = dist_ref_img / dist_1st_img
            align_args["scale_leg_upper"] = (s1+s2)/2

            dist_1st_img = np.linalg.norm(body_1st_img[9]-body_1st_img[10])
            dist_ref_img = np.linalg.norm(body_ref_img[9]-body_ref_img[10])
            s1 = dist_ref_img / dist_1st_img
            dist_1st_img = np.linalg.norm(body_1st_img[12]-body_1st_img[13])
            dist_ref_img = np.linalg.norm(body_ref_img[12]-body_ref_img[13])
            s2 = dist_ref_img / dist_1st_img
            align_args["scale_leg_lower"] = (s1+s2)/2

            ####################
            ####################
            # need adjust nan
            for k,v in align_args.items():
                if np.isnan(v):
                    align_args[k]=1

            # centre offset (the offset of key point 1)
            offset = body_ref_img[1] - body_1st_img[1]
        
    
        # pose align
        pose_img, pose_ori = detector(img, args.detect_resolution, args.image_resolution, output_type='cv2', return_pose_dict=True)
        video_pose_buffer.append(pose_img)
        pose_align = align_img(img, pose_ori, align_args, args.detect_resolution, args.image_resolution)
        

        # add centre offset
        pose = pose_align
        pose['bodies']['candidate'] = pose['bodies']['candidate'] + offset
        pose['hands'] = pose['hands'] + offset
        pose['faces'] = pose['faces'] + offset


        # h不变，w从绝对坐标缩放回0-1 注意这里要回到ref的坐标系
        pose['bodies']['candidate'][:, 0] = pose['bodies']['candidate'][:, 0] / ref_ratio
        pose['hands'][:, :, 0] = pose['hands'][:, :, 0] / ref_ratio
        pose['faces'][:, :, 0] = pose['faces'][:, :, 0] / ref_ratio
        pose_list.append(pose)

    # stack
    body_list  = [pose['bodies']['candidate'][:18] for pose in pose_list]
    body_list_subset = [pose['bodies']['subset'][:1] for pose in pose_list]
    hands_list = [pose['hands'][:2] for pose in pose_list]
    faces_list = [pose['faces'][:1] for pose in pose_list]
   
    body_seq         = np.stack(body_list       , axis=0)
    body_seq_subset  = np.stack(body_list_subset, axis=0)
    hands_seq        = np.stack(hands_list      , axis=0)
    faces_seq        = np.stack(faces_list      , axis=0)
    # =======================================================
    # 🚀 你的专属补丁：只保存“对齐且稳定”的骨架，不要加了干扰的 aug 骨架！
    save_npy_path = args.outfn.replace('.mp4', '.npy')
    np.save(save_npy_path, body_seq)
    print(f"\n✅ 完美适配雕像比例的稳定骨架已保存至: {save_npy_path}")
    print(f"✅ 请将此 .npy 文件传入大模型推理脚本！")
    # =======================================================
    # =======================================================
    # 🚀 新增：计算用于原视频画面的全局仿射变换矩阵
    # 根据之前对 pose 的坐标变换逻辑，推导出原图到目标画布 (ref_W*2, ref_H*2) 的映射矩阵
    # =======================================================
    out_w, out_h = int(ref_W * 2), int(ref_H * 2)
    scale_factor = out_h / height
    tx = offset[0] * out_h
    ty = offset[1] * out_h
    
    M_img = np.array([
        [scale_factor, 0, tx],
        [0, scale_factor, ty]
    ], dtype=np.float32)
    
    warped_video_buffer = [] # 用于存储拉伸/平移后的原视频帧
    # =======================================================
    # concatenate and paint results
    # H = 768 # paint height
    H = ref_H # paint height
    W1 = int((H/ref_H * ref_W)//2 *2)
    W2 = int((H/height * width)//2 *2)
    result_demo = [] # = Writer(args, None, H, 3*W1+2*W2, outfn, fps)
    result_pose_only = [] # Writer(args, None, H, W1, args.outfn_align_pose_video, fps)
    for i in range(len(body_seq)):
        pose_t={}
        pose_t["bodies"]={}
        pose_t["bodies"]["candidate"]=body_seq[i]
        pose_t["bodies"]["subset"]=body_seq_subset[i]
        pose_t["hands"]=hands_seq[i]
        pose_t["faces"]=faces_seq[i]

        ref_img = cv2.cvtColor(refer_img, cv2.COLOR_RGB2BGR)
        ref_img = cv2.resize(ref_img, (W1, H))
        ref_pose= cv2.resize(output_refer, (W1, H))
        
        # output_transformed = draw_pose(
        #     pose_t, 
        #     int(H_in*1024/W_in), 
        #     1024, 
        #     draw_face=False,
        #     )
        # output_transformed = cv2.cvtColor(output_transformed, cv2.COLOR_BGR2RGB)
        # output_transformed = cv2.resize(output_transformed, (W1, H))
        
        output_transformed = draw_pose(     # single.mp4
            pose_t,
            ref_H*2,
            ref_W*2,
            draw_face=False,
            )
        output_transformed = cv2.cvtColor(output_transformed, cv2.COLOR_BGR2RGB)
        # output_transformed = cv2.resize(output_transformed, (W1, H), interpolation=cv2.INTER_CUBIC)
        
        output_transformed_1 = draw_pose(   # all.mp4
            pose_t,
            ref_H,
            ref_W,
            draw_face=False,
            )
        output_transformed_1 = cv2.cvtColor(output_transformed_1, cv2.COLOR_BGR2RGB)
        # output_transformed_1 = cv2.resize(output_transformed_1, (W1, H), interpolation=cv2.INTER_CUBIC)
        # =======================================================
        # 🚀 新增：对原视频帧进行仿射变换（全局拉伸与平移）
        # =======================================================
        orig_frame_bgr = video_frame_buffer[i]
        warped_frame_bgr = cv2.warpAffine(
            orig_frame_bgr, 
            M_img, 
            (out_w, out_h), 
            flags=cv2.INTER_CUBIC, 
            borderMode=cv2.BORDER_CONSTANT, 
            borderValue=(0,0,0)
        )
        warped_frame_rgb = cv2.cvtColor(warped_frame_bgr, cv2.COLOR_BGR2RGB)
        warped_video_buffer.append(warped_frame_rgb)
        # =======================================================
        video_frame = cv2.resize(video_frame_buffer[i], (W2, H), interpolation=cv2.INTER_CUBIC)
        video_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
        video_pose  = cv2.resize(video_pose_buffer[i], (W2, H), interpolation=cv2.INTER_CUBIC)

        res = np.concatenate([ref_img, ref_pose, output_transformed_1, video_frame, video_pose], axis=1) # all.mp4
        result_demo.append(res) # all.mp4
        result_pose_only.append(output_transformed) # single.mp4

    print(f"pose_list len: {len(pose_list)}")
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(result_demo, fps=fps)
    clip.write_videofile(outfn, fps=fps, codec="libx264") # all.mp4
    
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(result_pose_only, fps=fps)
    clip.write_videofile(args.outfn_align_pose_video, fps=fps, codec="libx264") # single.mp4
    # =======================================================
    # 🚀 新增：保存全局拉伸和平移后的原视频
    # =======================================================
    outfn_aligned_orig_video = args.outfn.replace('.mp4', '_aligned_orig.mp4')
    clip_orig = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(warped_video_buffer, fps=fps)
    clip_orig.write_videofile(outfn_aligned_orig_video, fps=fps, codec="libx264")
    print(f"✅ 已保存全局拉伸对齐后的原视频至: {outfn_aligned_orig_video}")
    # =======================================================
    # ========== 新增：保存图片序列到 poses 文件夹 ==========
    output_video_dir = os.path.dirname(args.outfn_align_pose_video)
    poses_folder = os.path.join(output_video_dir, "poses")
    os.makedirs(poses_folder, exist_ok=True)

    # 确定文件名中数字的位数（例如总帧数小于1000用3位）
    num_digits = len(str(len(result_pose_only)))
    for idx, frame_rgb in enumerate(result_pose_only):
        # OpenCV 保存需要 BGR 顺序，而 moviepy 使用的是 RGB
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        filename = f"frame_{str(idx).zfill(num_digits)}.png"
        save_path = os.path.join(poses_folder, filename)
        cv2.imwrite(save_path, frame_bgr)
    print(f"✅ 已保存 {len(result_pose_only)} 张 pose 图片至: {poses_folder}")
    # ======================================================
    # ======================================================
    # ========== 🚀 再次新增：保存拉伸后的原视频帧序列到 images 文件夹 ==========
    # ======================================================
    images_folder = os.path.join(output_video_dir, "images")
    os.makedirs(images_folder, exist_ok=True)

    num_digits_img = len(str(len(warped_video_buffer)))
    for idx, frame_rgb in enumerate(warped_video_buffer):
        # OpenCV 保存需要 BGR 顺序，而 buffer 中是 RGB 顺序
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        filename = f"frame_{str(idx).zfill(num_digits_img)}.png"
        save_path = os.path.join(images_folder, filename)
        cv2.imwrite(save_path, frame_bgr)
    print(f"✅ 已保存 {len(warped_video_buffer)} 张拉伸对齐后的原视频图片至: {images_folder}")
    # ======================================================
    # ... 前面是我们上次加的保存 images 文件夹的代码 ...
    
    # ======================================================
    # 🚀 终极融合：读取刚才保存的 images 文件夹，提取 133 点 Dense NPY！
    # ======================================================
    # 我们将稠密骨架命名为 xxx_dense133.npy 方便区分原来的 18点 npy
    final_dense_npy_path = args.outfn.replace('.mp4', '_dense133.npy')
    extract_dense_poses_to_npy(images_folder, final_dense_npy_path)
    # ======================================================

    print('pose align done')



def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--detect_resolution', type=int, default=512, help='detect_resolution')
    # parser.add_argument('--image_resolution', type=int, default=720, help='image_resolution')
    parser.add_argument('--detect_resolution', type=int, default=1024, help='detect_resolution')
    parser.add_argument('--image_resolution', type=int, default=720, help='image_resolution')

    parser.add_argument("--yolox_config",  type=str, default=f"{os.path.dirname(__file__)}/pose/config/yolox_l_8xb8-300e_coco.py")
    parser.add_argument("--dwpose_config", type=str, default=f"{os.path.dirname(__file__)}/pose/config/dwpose-l_384x288.py")
    parser.add_argument("--yolox_ckpt",  type=str, default="/gaojiawei/wjl/StableAnimator/checkpoints/DWPose/yolox_l_8x8_300e_coco.pth")
    parser.add_argument("--dwpose_ckpt", type=str, default="/gaojiawei/wjl/StableAnimator/checkpoints/DWPose/dw-ll_ucoco_384.pth")


    parser.add_argument('--align_frame', type=int, default=0, help='the frame index of the video to align')
    parser.add_argument('--max_frame', type=int, default=300, help='maximum frame number of the video to align')
    parser.add_argument('--imgfn_refer', type=str, default="/gaojiawei/wjl/StableAnimator/animation_data/others/1.png", help='refer image path')
    parser.add_argument('--vidfn', type=str, default="/gaojiawei/wjl/dataset/ceshi/10001.mp4", help='Input video path')
    parser.add_argument('--outfn_align_pose_video', type=str, default=None, help='output path of the aligned video of the refer img')
    parser.add_argument('--outfn', type=str, default=None, help='Output path of the alignment visualization')
    args = parser.parse_args()
    
    # if not os.path.exists("./aligns/poses/align"):
    #     # os.makedirs("./aligns/poses/")
    #     os.makedirs("./aligns/poses/align")
    #     os.makedirs("./aligns/poses/align_demo")
        
    img_name = os.path.basename(args.imgfn_refer).split('.')[0]
    video_name = os.path.basename(args.vidfn).split('.')[0]
    if args.outfn_align_pose_video is None:
        args.outfn_align_pose_video = "./aligns/poses/align/img_{}_video_{}.mp4".format(img_name, video_name)
    if args.outfn is None:
        args.outfn = "./aligns/poses/align_demo/img_{}_video_{}.mp4".format(img_name, video_name)

    os.makedirs(os.path.dirname(args.outfn), exist_ok=True)
    os.makedirs(os.path.dirname(args.outfn_align_pose_video), exist_ok=True)

    run_align_video_with_filterPose_translate_smooth(args)


    
if __name__ == '__main__':
    main()
