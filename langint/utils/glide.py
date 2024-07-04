from PIL import Image
import os
from langint.utils.glide_modifier import change_attention, change_attention_debug
from typing import List, Callable, Optional
import torch.nn.functional as F
import clip
import torchvision.transforms as T
import torch
from glide_text2im.clip.model_creation import create_clip_model
from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)


class GLIDEPipeline:
    def __init__(self, options=None):
        if options is None:
            options = model_and_diffusion_defaults()
            options['timestep_respacing'] = '100' # use 100 diffusion steps for fast sampling
            options['use_fp16'] = True
        if os.getenv('FAST_ATTENTION') == '1':
            options['use_fp16'] = False

        model, diffusion = create_model_and_diffusion(**options)
        if options['use_fp16']:
            model.convert_to_fp16()
        for p in model.parameters():
            p.requires_grad = False
        # model.eval()
        model.cuda()
        if options['inpaint'] is True:
            model.load_state_dict(load_checkpoint('base-inpaint', 'cuda'))
        else:
            model.load_state_dict(load_checkpoint('base', 'cuda'))

        self.options = options
        self.model = model
        self.diffusion = diffusion

        if os.getenv('FAST_ATTENTION') == '1':
            if os.getenv('DEBUG') == '1':
                self.model.apply(change_attention_debug)
            else:
                self.model.apply(change_attention)

    def sample(self, prompt, batch_size=4):
        # Create the text tokens to feed to the model.
        guidance_scale = 3.0

        options = self.options
        model = self.model
        diffusion = self.diffusion

        self.model.eval()

        tokens = model.tokenizer.encode(prompt)
        tokens, mask = model.tokenizer.padded_tokens_and_mask(
            tokens, options['text_ctx']
        )

        # Create the classifier-free guidance tokens (empty)
        full_batch_size = batch_size * 2
        uncond_tokens, uncond_mask = model.tokenizer.padded_tokens_and_mask(
            [], options['text_ctx']
        )

        # Pack the tokens together into model kwargs.
        model_kwargs = dict(
            tokens=torch.tensor([tokens] * batch_size + [uncond_tokens] * batch_size, device='cuda'),
            mask=torch.tensor(
                [mask] * batch_size + [uncond_mask] * batch_size,
                dtype=torch.bool,
                device='cuda',
                ),
        )

        # Create a classifier-free guidance sampling function
        def model_fn(x_t, ts, **kwargs):
            half = x_t[: len(x_t) // 2]
            combined = torch.cat([half, half], dim=0)
            model_out = model(combined, ts, **kwargs)
            eps, rest = model_out[:, :3], model_out[:, 3:]
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)
            return torch.cat([eps, rest], dim=1)

        # Sample from the base model.
        model.del_cache()
        samples = diffusion.p_sample_loop(
            model_fn,
            (full_batch_size, 3, options["image_size"], options["image_size"]),
            device='cuda',
            clip_denoised=True,
            progress=False,#True,
            model_kwargs=model_kwargs,
            cond_fn=None,
        )[:batch_size]
        model.del_cache()
        return samples


class GLIDEPipelineCLIPGuided(GLIDEPipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create CLIP model.
        clip_model = create_clip_model(device='cuda')
        clip_model.image_encoder.load_state_dict(load_checkpoint('clip/image-enc', 'cuda'))
        clip_model.text_encoder.load_state_dict(load_checkpoint('clip/text-enc', 'cuda'))
        self.clip_model = clip_model

        # convert to range [0, 1]
        n_px = self.options['image_size']
        self.image_preprocess = T.Compose([
            T.Resize(n_px, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            T.CenterCrop(n_px),
            T.ToTensor(),
            T.Normalize(0.5, 0.5),
        ])

    def sample(self, prompt):
        # Setup guidance function for CLIP model.
        guidance_scale = 3.0

        def meta_cond_fn(prompts, images):
            return self.clip_model.cond_fn(prompts, grad_scale=guidance_scale)

        return self.sample_core(prompt=prompt, image=None, meta_cond_fn=meta_cond_fn)

    def sample_image_guided(self, prompt, image: Image):
        # guidance_scale = 0.1
        guidance_scale = 3.0

        def meta_cond_fn(prompts, images):
            return self.get_cond_fn_image_guided(images=images, grad_scale=guidance_scale)

        return self.sample_core(prompt=prompt, image=image, meta_cond_fn=meta_cond_fn)

    def sample_core(self, prompt: str, image: Optional[Image.Image],
                    meta_cond_fn: Callable[[List[str], List[Image.Image]], Callable[..., torch.Tensor]], batch_size=4):
        self.model.eval()

        model = self.model
        diffusion = self.diffusion
        options = self.options

        tokens = model.tokenizer.encode(prompt)
        tokens, mask = model.tokenizer.padded_tokens_and_mask(
            tokens, options['text_ctx']
        )

        # Pack the tokens together into model kwargs.
        model_kwargs = dict(
            tokens=torch.tensor([tokens] * batch_size, device='cuda'),
            mask=torch.tensor([mask] * batch_size, dtype=torch.bool, device='cuda'),
        )

        # Setup guidance function for CLIP model.
        cond_fn = meta_cond_fn(prompts=[prompt] * batch_size, images=[image] * batch_size)

        # Sample from the base model.
        model.del_cache()
        samples = diffusion.p_sample_loop(
            model,
            (batch_size, 3, options["image_size"], options["image_size"]),
            device='cuda',
            clip_denoised=True,
            progress=False,#True,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
        )
        model.del_cache()
        return samples

    def get_cond_fn_image_guided(self, images: List[Image.Image], grad_scale: float):
        images = torch.stack([self.image_preprocess(image) for image in images]).cuda()
        # images: (b, c, h, w), range [-1, 1]
        clip_model = self.clip_model
        with torch.no_grad():
            # z_t = clip_model.image_embeddings(images, t=torch.ones(images.shape[0], device=images.device, dtype=torch.long) * (self.options['diffusion_steps'] - 1))
            # TODO try images + noise, with t > 0, as image_embeddings inputs
            z_t = clip_model.image_embeddings(images, t=torch.zeros(images.shape[0], device='cuda', dtype=torch.long))

        def cond_fn(x, t, grad_scale=grad_scale, **kwargs):
            with torch.enable_grad():
                x_var = x.detach().requires_grad_(True)
                z_i = clip_model.image_embeddings(x_var, t)
                loss = torch.exp(clip_model.logit_scale) * (z_t * z_i).sum()
                grad = torch.autograd.grad(loss, x_var)[0].detach()
            return grad * grad_scale

        return cond_fn


class VanillaCLIPModel:
    def __init__(self):
        # model, _ = clip.load('ViT-B/32', 'cuda', download_root='checkpoints/clip')
        model, _ = clip.load('ViT-L/14@336px', 'cuda', download_root='checkpoints/clip')
        self.clip_model = model

        # assume inputs are squared
        # https://github.com/openai/CLIP/blob/a9b1bf5920416aaeaec965c25dd9e8f98c864f16/clip/clip.py#L79
        n_px = model.visual.input_resolution
        pixel_mean = torch.tensor((0.48145466, 0.4578275, 0.40821073), dtype=torch.float32, device='cuda')
        pixel_std = torch.tensor((0.26862954, 0.26130258, 0.27577711), dtype=torch.float32, device='cuda')

        def preprocess(t: torch.Tensor) -> torch.Tensor:
            # t: (b, c, h, w), range [-1, 1]
            assert t.size(2) == t.size(3), t.shape
            t = F.interpolate(t, size=n_px, mode='bicubic', align_corners=False)
            t = t * .5 + .5  # [-1, 1] -> [0, 1]
            t = (t - pixel_mean[:, None, None]) / pixel_std[:, None, None]
            return t

        self.clip_preprocess = preprocess

    def text_embeddings(self, prompts: List[str]) -> torch.Tensor:
        tokens = torch.cat([clip.tokenize(c) for c in prompts]).cuda()
        z_t = self.clip_model.encode_text(tokens)
        return z_t / (torch.linalg.norm(z_t, dim=-1, keepdim=True) + 1e-12)

    def image_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        images = self.clip_preprocess(images)
        z_i = self.clip_model.encode_image(images)
        return z_i / (torch.linalg.norm(z_i, dim=-1, keepdim=True) + 1e-12)

    def cond_fn(self, prompts: List[str], grad_scale: float) -> Callable[..., torch.Tensor]:
        with torch.no_grad():
            z_t = self.text_embeddings(prompts)

        def cond_fn(x, t, grad_scale=grad_scale, **kwargs):
            with torch.enable_grad():
                x_var = x.detach().requires_grad_(True)
                z_i = self.image_embeddings(x_var)
                loss = torch.exp(self.clip_model.logit_scale) * (z_t * z_i).sum()
                grad = torch.autograd.grad(loss, x_var)[0].detach()
            return grad * grad_scale

        return cond_fn

    def cond_fn_image_guided(self, images: torch.Tensor, grad_scale: float):
        with torch.no_grad():
            z_t = self.image_embeddings(images)

        def cond_fn(x, t, grad_scale=grad_scale, **kwargs):
            with torch.enable_grad():
                x_var = x.detach().requires_grad_(True)
                z_i = self.image_embeddings(x_var)
                loss = torch.exp(self.clip_model.logit_scale) * (z_t * z_i).sum()
                grad = torch.autograd.grad(loss, x_var)[0].detach()
            return grad * grad_scale

        return cond_fn


class GLIDEPipelineVanillaCLIPGuided(GLIDEPipelineCLIPGuided):
    def __init__(self):
        GLIDEPipeline.__init__(self)
        self.clip_model = VanillaCLIPModel()

        # process guiding image to be the same as generated (may be noisy) ones
        n_px = self.options['image_size']
        self.image_preprocess = T.Compose([
            T.Resize(n_px, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(n_px),
            T.ToTensor(),
            T.Normalize(0.5, 0.5),
        ])

    def get_cond_fn_image_guided(self, images: List[Image.Image], grad_scale: float):
        images = torch.stack([self.image_preprocess(image) for image in images]).cuda()
        return self.clip_model.cond_fn_image_guided(images, grad_scale)


class GLIDEInpaintingPipeline:
    def __init__(self, options=None):
        super().__init__()
        if options is None:
            options = model_and_diffusion_defaults()
            options['timestep_respacing'] = '100'
            options['inpaint'] = True
        GLIDEPipeline.__init__(self, options)

        self.image_preprocess = T.Compose([
            T.Resize(options['image_size'], interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            T.CenterCrop(options['image_size']),
            T.ToTensor(),
            T.Normalize(0.5, 0.5),
        ])
        self.mask_preprocess = T.Compose([
            T.Resize(options['image_size'], interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            T.CenterCrop(options['image_size']),
            T.ToTensor(),
        ])

    def sample(self, input_image: Image.Image, input_mask: Image.Image,
               prompt: str, negative_prompt: Optional[str] = None):
        # input image has size (64, 64, 3)
        # input mask has size (64, 64)

        batch_size = 4
        guidance_scale = 5.0

        options = self.options
        model = self.model
        diffusion = self.diffusion

        tokens = model.tokenizer.encode(prompt)
        tokens, mask = model.tokenizer.padded_tokens_and_mask(
            tokens, options['text_ctx']
        )

        # Create the classifier-free guidance tokens (empty)
        full_batch_size = batch_size * 2
        if negative_prompt is not None:
            uncond_tokens = model.tokenizer.encode(negative_prompt)
            uncond_tokens, uncond_mask = model.tokenizer.padded_tokens_and_mask(uncond_tokens, options['text_ctx'])
        else:
            uncond_tokens, uncond_mask = model.tokenizer.padded_tokens_and_mask([], options['text_ctx'])

        input_image = self.image_preprocess(input_image)[None]
        input_mask = self.mask_preprocess(input_mask)[None]

        # Pack the tokens together into model kwargs.
        model_kwargs = dict(
            tokens=torch.tensor(
                [tokens] * batch_size + [uncond_tokens] * batch_size, device='cuda'
            ),
            mask=torch.tensor(
                [mask] * batch_size + [uncond_mask] * batch_size,
                dtype=torch.bool,
                device='cuda',
                ),

            # Masked inpainting image
            inpaint_image=(input_image * input_mask).repeat(full_batch_size, 1, 1, 1).cuda(),
            inpaint_mask=input_mask.repeat(full_batch_size, 1, 1, 1).cuda(),
        )

        # Create an classifier-free guidance sampling function
        def model_fn(x_t, ts, **kwargs):
            half = x_t[: len(x_t) // 2]
            combined = torch.cat([half, half], dim=0)
            model_out = model(combined, ts, **kwargs)
            eps, rest = model_out[:, :3], model_out[:, 3:]
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)
            return torch.cat([eps, rest], dim=1)

        def denoised_fn(x_start):
            # Force the model to have the exact right x_start predictions
            # for the part of the image which is known.
            return (
                    x_start * (1 - model_kwargs['inpaint_mask'])
                    + model_kwargs['inpaint_image'] * model_kwargs['inpaint_mask']
            )

        # Sample from the base model.
        model.del_cache()
        samples = diffusion.p_sample_loop(
            model_fn,
            (full_batch_size, 3, options["image_size"], options["image_size"]),
            device='cuda',
            clip_denoised=True,
            progress=False,
            model_kwargs=model_kwargs,
            cond_fn=None,
            denoised_fn=denoised_fn,
        )#[:batch_size]
        model.del_cache()
        return samples