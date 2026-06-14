import os
import time
import json
import random
import hashlib
from pathlib import Path

import fire
import torch
import requests
import numpy as np
from PIL import Image
from bayes_opt import BayesianOptimization
from safetensors.torch import load_file, save_file

from tqdm import tqdm

from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline
from compel import Compel, ReturnedEmbeddingsType
from tensorboardX import SummaryWriter


测试用prompt = [
    '1girl, Alice in glitterworld, blonde hair, twintails, blue dress, white apron, smile, closed mouth, tachi-e, fullbody, white background, half-closed eyes, konya karasue, nonddu, kome cola, fukemachi, ask (askzy)',
    '1girl, black twintails, school uniform, outdoors, street, fullbody, black pantyhose, holding phone, looking at phone, > <, hoji(hooooooooji1029), kani biimu',
    '1girl, twintails, cat ears, maid, maid headdress, holding tray, white pantyhose, indoors, kitchen, melailai, kedama milk, efuri (riarea00), mairo',
    '1girl, twintails, blunt bangs, samurai, japanese armor, seiza, indoors, holding cup of milk, ogipote, niliu chahui, fuzichoco',
]


global_step = 1


class 超StableDiffusionXLPipeline:
    def __init__(self, path, 串行化vae=True):
        p = {}
        for _ in range(100):
            try:
                self._pipe = StableDiffusionXLPipeline.from_single_file(
                    path,
                    torch_dtype=torch.float16,
                    **p,
                ).to("cuda")
            except requests.exceptions.RequestException as e:
                print(f'没连上，休息1下: {e}')
                time.sleep(60)
            else:
                break
        self._pipe.scheduler = DPMSolverMultistepScheduler.from_config(self._pipe.scheduler.config)
        self._pipe.set_progress_bar_config(disable=True)
        if 串行化vae:
            self._pipe.enable_vae_slicing()

        self.compel = Compel(truncate_long_prompts=False, tokenizer=[self._pipe.tokenizer, self._pipe.tokenizer_2], text_encoder=[self._pipe.text_encoder, self._pipe.text_encoder_2], returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, requires_pooled=[False, True])

    def __call__(
        self,
        prompt: list[str],
        negative_prompt: list[str],
        **d,
    ):
        conditioning, pooled = self.compel(prompt)
        negative_embed, negative_pooled = self.compel(negative_prompt)
        [conditioning, negative_embed] = self.compel.pad_conditioning_tensors_to_same_length([conditioning, negative_embed])
        return self._pipe(
            prompt_embeds=conditioning,
            pooled_prompt_embeds=pooled,
            negative_prompt_embeds=negative_embed,
            negative_pooled_prompt_embeds=negative_pooled,
            **d,
        )


def 评测pipeline(pipe, n_iter: int, min_tags, max_tags, seed=0, guidance_scale=7) -> float:
    from benchmarker.common import 要测的标签, ml_danbooru标签2

    rd = random.Random(seed)
    
    要测的标签_打乱 = []
    for _ in range(3):
        t = 要测的标签.copy()
        rd.shuffle(t)
        要测的标签_打乱.extend(t)
    所有得分 = []
    for _ in tqdm(range(n_iter), desc='评测'):
        标签个数 = rd.randint(min_tags, max_tags)
        标签组 = 要测的标签_打乱[:标签个数]
        要测的标签_打乱 = 要测的标签_打乱[标签个数:]
        assert '_' not in str(标签组)
        images = pipe(
            prompt=f'1 girl, {", ".join(标签组)}',
            negative_prompt='',
            generator=torch.Generator(device='cuda').manual_seed(rd.randint(0, 2**16)),
            num_inference_steps=18+rd.randint(0, 4),
            guidance_scale=guidance_scale,
            width=640+rd.randint(0, 7)*64,
            height=640+rd.randint(0, 7)*64,
        ).images
        预测标签 = ml_danbooru标签2(images)[0]
        下划线标签组 = [i.strip().replace(' ', '_') for i in 标签组]
        得分 = len(set(下划线标签组) & set(预测标签)) / len(set(下划线标签组))
        所有得分.append(得分)
    return sum(所有得分) / len(所有得分)


def 哈(x, 长度=4) -> str:
    return hashlib.md5(str(x).encode()).hexdigest().upper()[:长度]


def 融合识别(s: str) -> str:
    nm = {
        'x': 'model.diffusion_model.input_blocks.',
        'y': 'model.diffusion_model.middle_block.',
        'z': 'model.diffusion_model.output_blocks.',
    }
    for k, v in nm.items():
        if s.startswith(v):
            n = int(s.removeprefix(v).split('.')[0])
            if k == 'x':
                for z, s in enumerate(((0, 1, 2, 3), (4, 5, 6), (7,), (8,))):
                    if n in s:
                        真n = z
            elif k == 'y':
                真n = 0
            elif k == 'z':
                for z, s in enumerate(((0,), (1,), (2,), (3, 4, 5), (6, 7, 8))):
                    if n in s:
                        真n = z
            else:
                raise Exception('啊？')
            return f'{k}_{真n}'
    return 'r'


# 大小 = {'r': 1637, 'x_0': 5, 'x_1': 85, 'x_2': 702, 'x_3': 715, 'y_0': 774, 'z_0': 749, 'z_1': 749, 'z_2': 761, 'z_3': 158, 'z_4': 14}
def 融合(所有层, 当前模型: dict, 其他模型: list[dict], 保存文件名, **kw) -> float:
    新模型 = {}
    a = 当前模型
    负数权重 = 0
    n = 0
    for k in 所有层:
        识别k = 融合识别(k)
        if 识别k.endswith('x_0'):
            大小k = 120
        elif 识别k.endswith('x_1'):
            大小k = 6
        elif 识别k.endswith('z_3'):
            大小k = 4 * 2
        elif 识别k.endswith('z_4'):
            大小k = 60 * 4
        elif 识别k.endswith('r'):
            大小k = 0.5
        else:
            大小k = 1
        aw = 1
        for i, _ in enumerate(其他模型):
            aw -= kw[f'{i}_{识别k}']
        n += a[k].numel() * 大小k
        if aw < 0:
            负数权重 += aw * a[k].numel() * 大小k
        新模型[k] = a[k].clone().cuda().to(torch.float32) * aw
        for i, b in enumerate(其他模型):
            bw = kw[f'{i}_{识别k}']
            n += b[k].numel() * 大小k
            if bw < 0:
                负数权重 += bw * b[k].numel() * 大小k
            新模型[k].add_(b[k].cuda().to(torch.float32) * bw)
        新模型[k] = 新模型[k].to(torch.float16).cpu()
    save_file(新模型, 保存文件名)
    return 负数权重 / n


def 烙(output_dir, 所有层, 当前模型: dict, 其他模型: list[dict], 标记: str, summary_writer, eval_n_iter: int, neg_penalty: float, 记录: list, 保存: dict, eval_min_tags, eval_max_tags, **kw):
    global global_step

    文件名 = f'{output_dir}/{标记}_step{global_step}.safetensors'
    平均负数 = 融合(所有层, 当前模型, 其他模型, 文件名, **kw)

    pipe = 超StableDiffusionXLPipeline(文件名)

    images = []
    for i, prompt in enumerate(tqdm(测试用prompt)):
        images.append(pipe(
            prompt=prompt,
            negative_prompt='',
            generator=torch.Generator(device='cuda').manual_seed(i),
            num_inference_steps=25,
            guidance_scale=7,
            width=640,
            height=1280,
        ).images[0])
    new_image = Image.new('RGB', (640 * len(images), 1280))
    for i, image in enumerate(images):
        new_image.paste(image, (i * 640, 0))
    summary_writer.add_image('图', np.asarray(new_image), global_step, dataformats='HWC')

    惩罚 = 平均负数 * neg_penalty

    acc = 评测pipeline(pipe, n_iter=eval_n_iter, min_tags=eval_min_tags, max_tags=eval_max_tags)
    记录.append({
        '文件名': 文件名,
        'acc': acc,
        '惩罚': 惩罚,
        'kw': kw,
    })
    print(文件名, f'acc={acc}', f'惩罚={惩罚}')
    with open(f'{output_dir}/{标记}.txt', 'w', encoding='utf8') as f:
        json.dump(记录, f, indent=2, ensure_ascii=False, default=float)

    总分 = acc + 惩罚

    summary_writer.add_scalar('acc', torch.tensor(acc), global_step=global_step)
    summary_writer.add_scalar('平均负数', torch.tensor(平均负数), global_step=global_step)
    summary_writer.add_scalar('惩罚', torch.tensor(惩罚), global_step=global_step)
    summary_writer.add_scalar('总分', torch.tensor(总分), global_step=global_step)

    保存[文件名] = 总分
    for k, _ in sorted(保存.items(), key=lambda x: -x[1])[3:]:
        if os.path.exists(k):
            os.remove(k)
    global_step += 1
    return 总分


def ember(models: list[str], output_dir: str = './savedata', bound=(-0.3, 0.8), n_iter=100, eval_n_iter=100, eval_min_tags=5, eval_max_tags=20, seed=9, neg_penalty=0.05):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    特征 = f'{哈(models)}_B{bound[0]},{bound[1]}_e{eval_n_iter}_{eval_min_tags}_{eval_max_tags}_n{neg_penalty}_s{seed}'
    summary_writer = SummaryWriter(logdir=f'{output_dir}/log/{特征}')

    random.seed(seed)

    当前模型 = load_file(models[0])
    其他模型 = [load_file(i) for i in models[1:]]

    所有层 = set(当前模型)
    for i in 其他模型:
        所有层 &= set(i)

    识别结果 = {融合识别(i) for i in 所有层}

    所有参数 = []
    for i, _ in enumerate(其他模型):
        for j in 识别结果:
            所有参数.append(f'{i}_{j}')
    所有参数.sort()

    print('所有参数为:', 所有参数)
    print('参数数量:', len(所有参数))

    标记 = f'{特征}_{random.randint(9999, 99999)}'
    记录 = []
    保存 = {}
    optimizer = BayesianOptimization(
        f=lambda **kw: 烙(output_dir=output_dir, 当前模型=当前模型, 其他模型=其他模型, 所有层=所有层, 标记=标记, 记录=记录, 保存=保存, summary_writer=summary_writer, eval_n_iter=eval_n_iter, eval_min_tags=eval_min_tags, eval_max_tags=eval_max_tags, neg_penalty=neg_penalty, **kw),
        pbounds={i: bound for i in 所有参数},
        random_state=seed,
    )
    for 初始i in ['无'] + [*range(len(其他模型))]:
        params = {i: 0 for i in 所有参数}
        for k, v in [*params.items()]:
            if k.startswith(f'{初始i}_'):
                params[k] = 1
        # print(params)
        optimizer.probe(
            params=params,
        )
    # for d in json.load(open('记忆.json')):
    #     optimizer.probe(params=d)
    optimizer.maximize(
        init_points=2*len(models),
        n_iter=n_iter,
    )

# python .\ro_merge.py --output_dir="C:/Users/Administrator/Desktop/RO_savedata" --models="['C:/Users/Administrator/Desktop/models/waiIllustriousSDXL_v160.safetensors','C:/Users/Administrator/Desktop/models/waiNSFWIllustrious_v150.safetensors','C:/Users/Administrator/Desktop/models/waiNSFWIllustrious_v140.safetensors']" --n_iter=300 --eval_n_iter=100 --bound="(-0.8,1.5)" --neg_penalty=0.1 --seed=7


# set PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync,expandable_segments:True
# set PYTORCH_ALLOC_CONF=backend:cudaMallocAsync,expandable_segments:True
# python .\ro_merge.py --output_dir="C:/Users/Administrator/Desktop/RO_savedata" --models="['C:/Users/Administrator/Desktop/models/waiIllustriousSDXL_v160.safetensors','C:/Users/Administrator/Desktop/models/novaAnimeXL_ilV120.safetensors','C:/Users/Administrator/Desktop/models/oneObsession_v18.safetensors']" --n_iter=600 --eval_n_iter=100 --bound="(-0.8,1.5)" --neg_penalty=0.2 --seed=8


# tensorboard --logdir=log --host 0.0.0.0 --samples_per_plugin "scalars=1000,images=1000"


if __name__ == '__main__':
    fire.Fire(ember)
