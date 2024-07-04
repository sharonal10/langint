from langint.utils.deepfloyd_no_diffusers import CACHE_DIR
import torch.nn as nn
from langint.third_party.deepfloyd.deepfloyd_if.model.nn import mean_flat
from langint.third_party.deepfloyd.deepfloyd_if.modules import IFStageI
from langint.third_party.deepfloyd.deepfloyd_if.model.gaussian_diffusion import GaussianDiffusion
from langint.third_party.deepfloyd.deepfloyd_if.modules.t5 import T5Embedder
from langint.third_party.deepfloyd.deepfloyd_if.model.gaussian_diffusion import LossType, ModelVarType, ModelMeanType
from PIL import Image
from typing import Union, List, Optional, Dict, Any
import torch
import os

MODEL_NAME = 'IF-I-M-v1.0'


class InvertDeepFloyd:

    def __init__(self):
        if os.getenv('IF_I_FLOAT16') == '1':
            self.if_I = IFStageI(MODEL_NAME, device='cuda', cache_dir=CACHE_DIR, model_kwargs={'precision': 16})
        else:
            self.if_I = IFStageI(MODEL_NAME, device='cuda', cache_dir=CACHE_DIR, model_kwargs={'precision': 32})
        self.t5 = None

        self.if_I.model.requires_grad_(False)
        self.diffusion = self.if_I.get_diffusion('')

    def __call__(self, out: Dict[str, torch.Tensor], data, writer=None) -> Dict[str, Any]:
        terms = dict()
        diffusion: GaussianDiffusion = self.diffusion
        unet: nn.Module = self.if_I.model
        unet.eval()

        embeddings = out['embeddings']
        batch_size = embeddings.shape[0]

        model_kwargs = dict(
            text_emb=embeddings.to(unet.dtype),
            timestep_text_emb=None,
            use_cache=False,
        )

        t = torch.randint(0, diffusion.num_timesteps, (batch_size,), device='cuda', dtype=torch.long)

        x_start = data['image'].to(unet.dtype)
        noise = torch.randn_like(x_start)

        # https://github.com/openai/glide-text2im/blob/69b530740eb6cef69442d6180579ef5ba9ef063e/glide_text2im/gaussian_diffusion.py#L180
        # x_t = \sqrt{\hat{\alpha_t}} x_0 + \sqrt{1 - \hat{\alpha_t}} \epsilon is a sample from q(x_t | x_0)
        x_t = diffusion.q_sample(x_start, t, noise=noise)
        # model outputs is noise \epsilon_\theta(x_t, t)
        model_output = unet(x_t, diffusion._scale_timesteps(t), **model_kwargs)
        assert model_output.shape == (batch_size, 3 * 2, 64, 64)

        model_output, model_var_values = torch.split(model_output, [3, 3], dim=1)
        assert diffusion.loss_type == LossType.RESCALED_MSE and diffusion.model_var_type == ModelVarType.LEARNED_RANGE
        assert diffusion.model_mean_type == ModelMeanType.EPSILON

        # Learn the variance using the variational bound, but don't let
        # it affect our mean prediction.
        frozen_out = torch.cat([model_output.detach(), model_var_values], dim=1)
        terms['vb'] = diffusion._vb_terms_bpd(
            model=lambda *args, r=frozen_out: r,
            x_start=x_start,
            x_t=x_t,
            t=t,
            clip_denoised=False,
        )['output']
        # Divide by 1000 for equivalence with initial implementation.
        # Without a factor of 1/1000, the VB term hurts the MSE term.
        terms['vb'] *= diffusion.num_timesteps / 1000.0

        # https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/guided_diffusion/gaussian_diffusion.py#L809
        target = noise
        assert model_output.shape == target.shape

        assert not torch.isnan(model_output).any() and not torch.isnan(target).any(), (torch.isnan(model_output).any(), torch.isnan(target).any())

        terms['mse'] = mean_flat((target - model_output) ** 2)
        assert terms['mse'].shape == terms['vb'].shape, (terms['mse'].shape, terms['vb'].shape)

        terms['loss'] = terms['mse'] + terms['vb']
        writer.add_scalar('loss_terms/fruit_blip_loss', out['fruit_blip_loss'], out['iteration'])
        writer.add_scalar('loss_terms/color_blip_loss', out['color_blip_loss'], out['iteration'])
        writer.add_scalar('loss_terms/diff_loss', terms['loss'].mean().float(), out['iteration'])

        return terms['loss'].mean() + out['fruit_blip_loss'] + out['color_blip_loss']


    def visualize(self, out: Dict[str, torch.Tensor], data) -> Dict[str, Any]:
        if_I = self.if_I
        embeddings = out['embeddings']
        negative_embeddings = out.get('negative_embeddings', None)
        stageI_generations, _ = if_I.embeddings_to_image(
            t5_embs=embeddings,
            negative_t5_embs=negative_embeddings,
            guidance_scale=7.0,
            sample_timestep_respacing='smart50',
            progress=False,
        )
        return {'image': stageI_generations}

    def dream(self, prompt: Union[str, List[str]], negative_prompt: Optional[Union[str, List[str]]] = None):
        if self.t5 is None:
            if os.getenv('T5_BFLOAT16') == '1':
                self.t5 = T5Embedder(device='cuda', cache_dir=CACHE_DIR, torch_dtype=torch.bfloat16)
            else: 
                self.t5 = T5Embedder(device='cuda', cache_dir=CACHE_DIR, torch_dtype=torch.float32)
            self.t5.model.requires_grad_(False)

        if_I = self.if_I
        t5 = self.t5
        if isinstance(prompt, str):
            prompt = [prompt]
        t5_embs = t5.get_text_embeddings(prompt)

        if_I_kwargs = dict()
        if_I_kwargs['t5_embs'] = t5_embs

        if_I_kwargs['guidance_scale'] = 7.0
        if_I_kwargs['sample_timestep_respacing'] = 'smart100'
        if_I_kwargs['aspect_ratio'] = '1:1'
        if_I_kwargs['progress'] = False

        if negative_prompt is not None:
            if isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt]
            negative_t5_embs = t5.get_text_embeddings(negative_prompt)
            if_I_kwargs['negative_t5_embs'] = negative_t5_embs

        stageI_generations, _ = if_I.embeddings_to_image(**if_I_kwargs)
        image: List[Image.Image] = if_I.to_images(stageI_generations, disable_watermark=True)
        return image
