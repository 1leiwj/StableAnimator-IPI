import os

# ================= 配置区 =================
# 你的 bec 文件夹路径
TARGET_ROOT = "/gaojiawei/wjl/StableAnimator/animation_data/rec"

# 设定预期的帧数 (StableAnimator 16帧/8帧复制法通常是16)
EXPECTED_FRAMES = 16

# 必备文件夹
SUBDIRS = ["images", "faces", "poses"]
# 必备文件
REQUIRED_FILE = "poses_dense133.npy"

# ==========================================

print(f"🔍 正在深度扫描数据集: {TARGET_ROOT}")
print(f"预期每项数据应包含 {EXPECTED_FRAMES} 个文件\n")

all_folders = sorted([f for f in os.listdir(TARGET_ROOT) if f.isdigit()])
total_count = len(all_folders)
corrupt_folders = []

for folder in all_folders:
    folder_path = os.path.join(TARGET_ROOT, folder)
    errors = []

    # 1. 检查子文件夹内容数量
    for subdir in SUBDIRS:
        subdir_path = os.path.join(folder_path, subdir)
        if not os.path.isdir(subdir_path):
            errors.append(f"缺失目录: {subdir}")
            continue
        
        # 获取文件夹内文件数量
        files = [f for f in os.listdir(subdir_path) if not f.startswith('.')]
        count = len(files)
        
        if count == 0:
            errors.append(f"目录为空: {subdir}")
        elif count < EXPECTED_FRAMES:
            errors.append(f"数量不足: {subdir} (仅有 {count}/{EXPECTED_FRAMES} 个文件)")

    # 2. 检查 poses_dense.npy
    npy_path = os.path.join(folder_path, REQUIRED_FILE)
    if not os.path.isfile(npy_path):
        errors.append(f"缺失核心文件: {REQUIRED_FILE}")
    elif os.path.getsize(npy_path) < 500: # 正常的 npy 不应该太小
        errors.append(f"文件异常: {REQUIRED_FILE} (体积过小)")

    # 输出错误报告
    if errors:
        print(f"❌ 编号 [{folder}] 发现隐患:")
        for err in errors:
            print(f"    - {err}")
        corrupt_folders.append(folder)

# ================= 结算汇总 =================
print("\n" + "="*40)
print(f"📊 扫描报告汇总:")
print(f"✅ 完美可用: {total_count - len(corrupt_folders)} / {total_count}")
print(f"🚨 存在问题: {len(corrupt_folders)} / {total_count}")

if corrupt_folders:
    print(f"\n需要重跑或检查的编号清单:")
    print(", ".join(corrupt_folders))
else:
    print("\n🎉 恭喜！所有文件夹内容完整，数量达标，可以安心炼丹！")
print("="*40)