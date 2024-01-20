import sys
import json
import time

import numpy as np
from bayes_opt import BayesianOptimization
from safetensors.numpy import load_file, save_file

sys.path.append('R:/e')
from common import 上网, 服务器地址
from 评测多标签 import 评测模型


模型文件夹 = 'R:/stable-diffusion-webui-master/models/Stable-diffusion'


原模型 = [
    'rimochan_random_mix_1.1',
    'novelailatest-pruned',
    'calicomix_v75',
    'bluePencil_v10',
]

记录文件名 = f'记录{int(time.time())}.txt'
记录 = []

def 算点(**d):
    ds = sum(d.values())
    d = {int(k): v / ds for k, v in d.items()}
    文件名 = f'rimo_tmp_{int(time.time())}'
    for i, k in enumerate(原模型):
        t = load_file(f'{模型文件夹}/{k}.safetensors')
        if i == 0:
            新模型 = {层: d[i] * t[层].astype(np.float32) for 层 in t}
        else:
            新模型 = {层: 新模型[层] + d[i] * t[层].astype(np.float32) for 层 in set(新模型) & set(t)}
        del t
    save_file(新模型, f'{模型文件夹}/{文件名}.safetensors')
    del 新模型

    上网(f'{服务器地址}/sdapi/v1/refresh-checkpoints', method='post')
    结果 = 评测模型(文件名, 'blessed2.vae.safetensors', 32, n_iter=70, use_tqdm=False, savedata=False)
    m = []
    for dd in 结果:
        m.extend(dd['分数'])
    a = np.array(m)
    acc = (a > 0.001).sum() / len(a.flatten())
    记录.append({
        '参数': d,
        '文件名': 文件名,
        'acc': acc,
    })
    with open(记录文件名, 'w', encoding='utf8') as f:
        json.dump(记录, f, indent=2)
    return acc


optimizer = BayesianOptimization(
    f=算点,
    pbounds={
        '0': (0.5, 0.8),
        '1': (-0.2, 0.2),
        '2': (-0.05, 0.5),
        '3': (-0.05, 0.5),
    },
    random_state=1,
)

optimizer.maximize(
    init_points=4,
    n_iter=1000,
)
