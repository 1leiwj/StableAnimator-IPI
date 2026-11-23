# 1. 随便找一个npy文件，看看内容

import numpy as np
import glob

npy_files = glob.glob('/gaojiawei/wjl/StableAnimator/animation_data/rec/00002/poses_dense.npy', recursive=True)
if npy_files:
    print(f'Found {len(npy_files)} npy files')
    data = np.load(npy_files[0])
    print(f'First file: {npy_files[0]}')
    print(f'Shape: {data.shape}')
    print(f'Min: {data.min():.4f}, Max: {data.max():.4f}')
    print(f'Mean: {data.mean():.4f}, Std: {data.std():.4f}')
    print(f'Unique values: {np.unique(data)[:10]}')
else:
    print('No npy files found!')
