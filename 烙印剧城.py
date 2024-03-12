import sys
import json
import time
import random
import hashlib

import numpy as np
from bayes_opt import BayesianOptimization
from safetensors.numpy import load_file, save_file

sys.path.append('R:/e')
from common import 上网, 服务器地址
from 评测多标签 import 评测模型


from safetensors.numpy import load_file, save_file
import numpy as np


rd = random.Random(int(time.time()))


模型文件夹 = 'R:/stable-diffusion-webui-master/models/Stable-diffusion'

所有模型 = [
    'AOM3A1',
    'Counterfeit-V2.2',
    'Counterfeit-V3.0_fp16',
    'anyloraCheckpoint_novaeFp16',
    'bluePencil_v10',
    'calicomix_v75',
    'cetusMix_v4',
    'cuteyukimixAdorable_specialchapter',
    'sweetMix_v22Flat',
    'petitcutie_v15',
    'kaywaii_v70',
    'kaywaii_v90',
    'rimochan_random_mix_1.1',
    'superInvincibleAnd_v2',
    'sakuramochimix_v10',
    'sweetfruit_melon.safetensors_v1.0',
    'AnythingV5Ink_ink',
    'rabbit_v7',
    'rainbowsweets_v20',
    'himawarimix_v100',
    'koji_v21',
    'yetanotheranimemodel_v20',
    'irismix_v90',
    'theWondermix_v12',
]


融合识别缩写 = {}
def 融合识别(s: str) -> str:
    sp = s.split('.')[:3]
    for ss in sp:
        if ss == 'model':
            融合识别缩写[ss] = 'M'
        else:
            融合识别缩写[ss] = ss[0]
    assert len(融合识别缩写.values()) == len(set(融合识别缩写.values()))
    blocks = ''.join([融合识别缩写[i] for i in sp])
    n = int(s.split('.')[3])
    return f'{blocks}#{n//2}'


def 烙(**kw):
    新模型 = {}
    for k in 所有层:
        if k not in 好层:
            新模型[k] = a[k]
        else:
            新模型[k] = a[k].astype(np.float32) * (1 - kw[融合识别(k)]) + b[k].astype(np.float32) * kw[融合识别(k)]
    文件名 = 名字(kw)
    save_file(新模型, f'{模型文件夹}/{文件名}.safetensors')
    上网(f'{服务器地址}/sdapi/v1/refresh-checkpoints', method='post')
    结果 = 评测模型(文件名, 'blessed2.vae.safetensors', 32, n_iter=100, use_tqdm=False, savedata=False, seed=random.randint(1000, 9000), tags_seed=random.randint(1000, 9000), 计算相似度=False)
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


def 名字(kw: dict):
    s = sorted(kw.items())
    md5 = hashlib.md5(str(''.join(f'{k}{v:.2f}' for k, v in s)).encode()).hexdigest()
    return f'Theater_{当前标记}_{md5[:8]}'


当前模型 = 'rimochan_random_mix_3.2'


标记 = rd.randint(0, 10000)

for i in range(100):
    当前标记 = f'{标记}_{i}'
    记录文件名 = f'记录_烙印剧城_{当前标记}_{int(time.time())}.txt'
    记录 = []
    b名 = rd.choice(所有模型)
    print('融合', 当前模型, b名)
    a = load_file(f'{模型文件夹}/{当前模型}.safetensors')
    b = load_file(f'{模型文件夹}/{rd.choice(所有模型)}.safetensors')

    所有层 = set(a) & set(b)
    好层 = {i for i in 所有层 if i.startswith('model.diffusion_model.input_blocks.') or i.startswith('model.diffusion_model.middle_block.') or i.startswith('model.diffusion_model.output_blocks.')}
    好层识别 = {融合识别(i) for i in 好层}
    print('好层识别为:', 好层识别)

    optimizer = BayesianOptimization(
        f=烙,
        pbounds={i: (-0.05, 0.25) for i in 好层识别},
        random_state=666,
    )

    optimizer.probe(
        params={i: 0 for i in 好层识别},
    )

    optimizer.maximize(
        init_points=4,
        n_iter=25,
    )

    当前模型 = 名字(optimizer.max['params'])
