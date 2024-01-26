import sys
import json
import time

import numpy as np
from bayes_opt import BayesianOptimization
from safetensors.numpy import load_file, save_file

sys.path.append('R:/e')
from common import 上网, 服务器地址
from 评测多标签 import 评测模型


from safetensors.numpy import load_file, save_file
import numpy as np


模型文件夹 = 'R:/stable-diffusion-webui-master/models/Stable-diffusion'

a = load_file(f'{模型文件夹}/rimochan_random_mix_1.1.safetensors')
b = load_file(f'{模型文件夹}/b.safetensors')
c = load_file(f'{模型文件夹}/c.safetensors')
d = load_file(f'{模型文件夹}/d.safetensors')


all_k = set(a) & set(b) & set(c) & set(d)

记录文件名 = f'记录{int(time.time())}.txt'
记录 = []

nm = {
    'x': 'model.diffusion_model.input_blocks',
    'y': 'model.diffusion_model.middle_block',
    'z': 'model.diffusion_model.output_blocks',
}

def 烙(**kw):
    新模型 = {}
    for k in all_k:
        qk = None
        for pk in 'xyz':
            if k.startswith(nm[pk]):
                qk = pk
        if qk is None:
            新模型[k] = a[k]
        else:
            新模型[k] = a[k].astype(np.float32) * (1-kw['b'+qk]-kw['c'+qk]-kw['d'+qk]) + \
                b[k].astype(np.float32) * kw['b'+qk] + \
                c[k].astype(np.float32) * kw['c'+qk] + \
                d[k].astype(np.float32) * kw['d'+qk]
    kw = {k: round(v, 3) for k, v in kw.items()}
    文件名 = f'central_dogma_{kw["bx"]}_{kw["by"]}_{kw["bz"]}_{kw["cx"]}_{kw["cy"]}_{kw["cz"]}_{kw["dx"]}_{kw["dy"]}_{kw["dz"]}'
    save_file(新模型, f'{模型文件夹}/{文件名}.safetensors')
    上网(f'{服务器地址}/sdapi/v1/refresh-checkpoints', method='post')
    结果 = 评测模型(文件名, 'blessed2.vae.safetensors', 32, n_iter=70, use_tqdm=False, savedata=False)
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


optimizer = BayesianOptimization(
    f=烙,
    pbounds={
        'bx': (-0.05, 0.25),
        'by': (-0.05, 0.25),
        'bz': (-0.05, 0.25),
        'cx': (-0.05, 0.25),
        'cy': (-0.05, 0.25),
        'cz': (-0.05, 0.25),
        'dx': (-0.05, 0.25),
        'dy': (-0.05, 0.25),
        'dz': (-0.05, 0.25),
    },
    random_state=1,
)

optimizer.probe(
    params={
        'bx': 0,
        'by': 0,
        'bz': 0,
        'cx': 0,
        'cy': 0,
        'cz': 0,
        'dx': 0,
        'dy': 0,
        'dz': 0,
    },
    lazy=True,
)

optimizer.maximize(
    init_points=4,
    n_iter=1000,
)
