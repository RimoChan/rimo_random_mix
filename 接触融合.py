import json

import torch
import numpy as np
from tqdm import tqdm
from safetensors.numpy import load_file, save_file


a_name = 'Theater_372_12_01450ca2'
b_name = 'Theater_9074_10_65a6892a'

rank = 4    # 越高越接近B

模型文件夹 = 'R:/stable-diffusion-webui-master/models/Stable-diffusion'
a = load_file(f'{模型文件夹}/{a_name}.safetensors')
b = load_file(f'{模型文件夹}/{b_name}.safetensors')


新模型 = {}
for k in tqdm(set(a) & set(b)):
    ak = a[k].astype(np.float32)
    bk = b[k].astype(np.float32)
    diff = bk - ak
    if len(diff.shape) > 1:
        conv2d = len(diff.shape) == 4
        conv2d_3x3 = conv2d and diff.shape[2:4] != (1, 1)
        if conv2d_3x3:
            diff = diff.reshape(diff.shape[0], -1)
        elif conv2d:
            diff = diff.squeeze()
        u, s, vh = torch.linalg.svd(torch.from_numpy(diff).to('cuda'), full_matrices=False)
        diff = (u[:, :rank] @ torch.diag(s[:rank]) @ vh[:rank, :]).cpu().numpy()
        if conv2d_3x3 or conv2d:
            diff = diff.reshape(ak.shape)
    新模型[k] = (ak + diff).astype(np.float16)
    del a[k], b[k]


save_file(新模型, f'{模型文件夹}/ConfusionXL_R{rank}B.safetensors', metadata={'contact_fusion': json.dumps({'a': a_name, 'b': b_name, 'rank': rank})})
