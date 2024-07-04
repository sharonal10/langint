from langint.third_party.deepfloyd.deepfloyd_if.modules import IFStageI, IFStageII
from langint.third_party.deepfloyd.deepfloyd_if.modules.t5 import T5Embedder
from langint.third_party.deepfloyd.deepfloyd_if.pipelines import dream, style_transfer, super_resolution, inpainting
import os
# os.environ['FORCE_MEM_EFFICIENT_ATTN'] = "1"

import torch.nn.functional as F
import random
import torchvision.transforms as T
import numpy as np
import requests
from typing import List
from PIL import Image
import torch
import re

CACHE_DIR = 'cache/IF_'
os.makedirs(CACHE_DIR, exist_ok=True)
# model_name = 'IF-I-XL-v1.0'
# model_name = 'IF-I-M-v1.0'


class Pipeline:
    def __init__(
            self, model_name='IF-I-M-v1.0', if_I_kwargs=None, t5_kwargs=None, model_II_name=None, if_II_kwargs=None
        ):
        if os.getenv('IF_I_FLOAT16') == '1':
            self.if_I = IFStageI(model_name, device='cuda', cache_dir=CACHE_DIR, model_kwargs={'precision': 16}, **(if_I_kwargs or {}))
            if model_II_name is not None:
                self.i_II = IFStageII(model_II_name, device='cuda', cache_dir=CACHE_DIR, model_kwargs={'precision': 16}, **(if_II_kwargs or {}))
            else:
                self.i_II = None
        else:
            self.if_I = IFStageI(model_name, device='cuda', cache_dir=CACHE_DIR, model_kwargs={'precision': 32}, **(if_I_kwargs or {}))
            if model_II_name is not None:
                self.i_II = IFStageII(model_II_name, device='cuda', cache_dir=CACHE_DIR, model_kwargs={'precision': 32}, **(if_II_kwargs or {}))
            else:
                self.i_II = None

        if os.getenv('T5_BFLOAT16') == '1':
            self.t5 = T5Embedder(device='cuda', cache_dir=CACHE_DIR, torch_dtype=torch.bfloat16, **(t5_kwargs or {}))
        else: 
            self.t5 = T5Embedder(device='cuda', cache_dir=CACHE_DIR, torch_dtype=torch.float32, **(t5_kwargs or {}))


    def dream(self, prompt: str, count: int = 4, return_all=False) -> List[Image.Image]:
        result = dream(
            t5=self.t5, if_I=self.if_I, if_II=self.i_II, if_III=None,
            prompt=[prompt] * count,
            if_I_kwargs={
                "guidance_scale": 7.0,
                "sample_timestep_respacing": "smart100",
            },
            if_II_kwargs={
                "guidance_scale": 4.0,
                "sample_timestep_respacing": "smart50",
            },
            disable_watermark=True,
        )
        if return_all:
            return result
        if self.i_II is not None:
            return result['II']
        return result['I']
