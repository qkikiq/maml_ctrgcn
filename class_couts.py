import os
from collections import defaultdict

skeleton_dir = '/dadaY/xinyu/dataset/nturgbd_skeletons_s001_to_s017/nturgb+d_skeletons/'  # 包含所有 skeleton 文件的路径
class_counts = defaultdict(int)

for fname in os.listdir(skeleton_dir):
    if not fname.endswith('.skeleton'):
        continue
    # 提取动作编号 A001 ~ A060
    action_code = fname.split('A')[1][:3]  # 取 '001'
    action_index = int(action_code) - 1    # 转换为 0-based index
    class_counts[action_index] += 1

# 打印统计结果
for cls in sorted(class_counts.keys()):
    print(f"Class {cls+1:02d}: {class_counts[cls]} samples")
