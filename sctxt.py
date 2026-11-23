import os

# 1. 配置区
# 你存放泥塑数据的根目录 (如果你叫 bec 就改成 bec)
data_dir = "/gaojiawei/wjl/StableAnimator/animation_data/rec"

# 你想生成的 txt 文件路径和名字
output_txt = "/gaojiawei/wjl/StableAnimator/animation_data/video_rrec_path.txt"

print(f"🔍 正在扫描 {data_dir} 下的文件夹...")

folders = []
# 检查目录是否存在
if os.path.exists(data_dir):
    # 遍历该目录下的所有内容
    for item in os.listdir(data_dir):
        full_path = os.path.join(data_dir, item)
        # 只保留文件夹（比如 20000, 20001 这种目录）
        if os.path.isdir(full_path):
            folders.append(full_path)
else:
    print(f"❌ 找不到目录: {data_dir}，请检查路径是否正确！")
    exit()

# 按照数字顺序排个队，治愈强迫症
folders.sort()

# 2. 写入 txt 文件
with open(output_txt, "w", encoding="utf-8") as f:
    for folder in folders:
        f.write(f"{folder}\n")

print(f"\n🎉 搞定！共找到 {len(folders)} 个有效数据文件夹。")
print(f"📄 所有的路径已经完美写入到: {output_txt}")