import sys
import json
import time
import hashlib

import numpy as np
from bayes_opt import BayesianOptimization
from safetensors.numpy import load_file, save_file

sys.path.append('R:/e')
from common import 上网, 服务器地址
from 评测多标签 import 评测模型


模型文件夹 = 'R:/stable-diffusion-webui-master/models/Stable-diffusion'

a = load_file(f'{模型文件夹}/Rimo.safetensors')
b = load_file(f'{模型文件夹}/b.safetensors')
c = load_file(f'{模型文件夹}/c.safetensors')


all_k = set(a) & set(b) & set(c)

记录文件名 = f'记录{int(time.time())}.txt'
记录 = []

def 融合识别(s: str) -> str:
    nm = {
        'x': 'model.diffusion_model.input_blocks.',
        'y': 'model.diffusion_model.middle_block.',
        'z': 'model.diffusion_model.output_blocks.',
    }
    for k, v in nm.items():
        if s.startswith(v):
            n = int(s.removeprefix(v).split('.')[0])
            return f'{k}_{n//3}'
    return 'r'


def 名字(kw: dict):
    s = sorted(kw.items())
    md5 = hashlib.md5(str(''.join(f'{k}{v:.2f}' for k, v in s)).encode()).hexdigest()
    return f'R3XL_{md5[:8]}'


def 烙(**kw):
    文件名 = 名字(kw)
    新模型 = {}
    for k in all_k:
        qk = 融合识别(k)
        新模型[k] = a[k].astype(np.float32) * (1-kw['b'+qk]-kw['c'+qk]) + \
            b[k].astype(np.float32) * kw['b'+qk] + \
            c[k].astype(np.float32) * kw['c'+qk]
    save_file(新模型, f'{模型文件夹}/{文件名}.safetensors')
    del 新模型
    上网(f'{服务器地址}/sdapi/v1/refresh-checkpoints', method='post')
    结果 = 评测模型(文件名, 'sdxl_vae_0.9.safetensors', 32, n_iter=80, use_tqdm=False, savedata=False, seed=22987, tags_seed=2223456, 计算相似度=False, width=576, height=576)
    m = []
    for dd in 结果:
        m.extend(dd['分数'])
    mm = np.array(m)
    acc = (mm > 0.001).sum() / len(mm.flatten())
    记录.append({
        '文件名': 文件名,
        'acc': acc,
    })
    print(文件名, acc, mm.shape)
    with open(记录文件名, 'w', encoding='utf8') as f:
        json.dump(记录, f, indent=2)
    return acc

识别结果 = set([融合识别(k) for k in all_k])
所有参数 = sorted(['b'+k for k in 识别结果] + ['c'+k for k in 识别结果])

optimizer = BayesianOptimization(
    f=烙,
    pbounds={k: (-0.2, 0.55) for k in 所有参数},
    random_state=1,
)
optimizer.probe(params={k: 0 for k in 所有参数})
optimizer.maximize(
    init_points=4,
    n_iter=1000,
)
