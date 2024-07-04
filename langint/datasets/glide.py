import torch
from tu.utils.config import build_from_config
import os
from PIL import Image
from typing import List, Dict
import numpy as np
import torchvision.transforms.functional as TF
from langint.utils.dataset import imagenet_templates_small
import logging
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from collections import Counter
from transformers import CLIPVisionModel
import kornia
import random
from collections import defaultdict

logger = logging.getLogger(__name__)


placeholder_words_list = [f'mytoken{i}' for i in range(500)]


class Synthetic(torch.utils.data.Dataset):
    def __init__(self, data_root: str, num_placeholder_words: int):
        super().__init__()

        self.data_root = data_root.replace('_', " ")
        self.templates = imagenet_templates_small

        pipeline = GLIDEPipeline()
        ground_truth_words = data_root.split(',')
        placeholder_words = placeholder_words_list[:num_placeholder_words]
        assert len(placeholder_words) == num_placeholder_words, (placeholder_words, num_placeholder_words)
        assert len(placeholder_words) == 1 or len(ground_truth_words) == len(placeholder_words), (ground_truth_words, placeholder_words)
        if len(placeholder_words) == 1:
            placeholder_words = placeholder_words * len(ground_truth_words)
        images_all: List[torch.Tensor] = []
        ph_words_all: List[str] = []
        for ind in range(len(ground_truth_words)):
            gt_word = ground_truth_words[ind]
            ph_word = placeholder_words[ind]
            prompt = self.templates[0].format(gt_word)
            images = pipeline.sample(prompt).cpu()
            images_all.append(images)
            ph_words_all.extend([ph_word] * len(images))
        self.images: torch.Tensor = torch.cat(images_all)
        self.placeholder_words: List[str] = ph_words_all
        del pipeline

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        # GLIDE expects range [-1, 1]
        image: torch.Tensor = self.images[item]
        if np.random.rand() < .5:
            image = TF.hflip(image)

        ph_word = self.placeholder_words[item]
        prompt = self.templates[np.random.choice(len(self.templates))].format(ph_word)

        return {'image': image, 'prompt': prompt}


def glide_sample_prompt(pipeline, prompt: str, num_repeats=4):
    images: torch.Tensor = pipeline.sample(prompt, batch_size=num_repeats).cpu()
    return images


def glide_sample_prompts(prompts: List[str], num_repeats=4) -> torch.Tensor:
    from langint.utils.glide import GLIDEPipeline
    # return (bs, 3, 64, 64) in pixel range [-1, 1]
    pipeline = GLIDEPipeline()
    images_all: List[torch.Tensor] = []
    for prompt in prompts:
        images = glide_sample_prompt(pipeline, prompt, num_repeats=num_repeats)
        images_all.append(images)
    del pipeline
    return torch.cat(images_all)


def deepfloyd_sample_prompts(prompts: List[str], num_repeats=4, model=None, processor=None, blip_fruit_q=None, blip_color_q=None, real_image=None):
    from langint.utils.deepfloyd_no_diffusers import Pipeline
    pipeline = Pipeline()
    images_all: List[Image.Image] = []
    blip_common_fruits = []
    blip_common_colors = []
    for prompt in prompts:
        if real_image is None:
            images: List[Image.Image] = pipeline.dream(prompt, count=num_repeats)
        else:
            img = Image.open(real_image)
            img = img.convert('RGB')
            img = img.resize((64, 64), Image.ANTIALIAS)
            images = [img.copy() for _ in range(num_repeats)]
        images_all.extend(images)

        if model is not None:
            blip_fruits = []
            blip_colors = []
            
            for image_i in range(len(images)):
                image = images[image_i]
                assert image.mode == 'RGB', image.mode

                inputs = processor(image, blip_fruit_q, return_tensors="pt").to("cuda", torch.float16)
                blip_out = model.generate(**inputs)
                blip_fruits.append(processor.decode(blip_out[0], skip_special_tokens=True))

                inputs = processor(image, blip_color_q, return_tensors="pt").to("cuda", torch.float16)
                blip_out = model.generate(**inputs)
                blip_colors.append(processor.decode(blip_out[0], skip_special_tokens=True))
            blip_fruit_counter = Counter(blip_fruits)
            
            blip_common_fruit = blip_fruit_counter.most_common(1)[0][0]
            blip_common_fruits.append(blip_common_fruit)


            blip_color_counter = Counter(blip_colors)
            
            blip_common_color = blip_color_counter.most_common(1)[0][0]
            blip_common_colors.append(blip_common_color)
    
    if model is not None:
        assert len(prompts) == len(blip_common_colors) == len(blip_common_fruits), (len(prompts), len(blip_common_colors), len(blip_common_fruits))
    del pipeline
    return torch.stack([TF.to_tensor(image) * 2 - 1 for image in images_all]), blip_common_fruits, blip_common_colors, images_all

def cache_deepfloyd_samples(prompts: List[str], num_repeats=4) -> torch.Tensor:
    cache_dir = 'cache/deepfloyd'
    image_paths_all = []
    pipeline = None
    for prompt in prompts:
        cache_subdir = prompt.replace(" ", "_")
        os.makedirs(os.path.join(cache_dir, cache_subdir), exist_ok=True)
        image_paths = [os.path.join(cache_dir, cache_subdir, f"{ind:02d}.png") for ind in range(num_repeats)]
        if not all(os.path.exists(path) for path in image_paths):
            if pipeline is None:
                from langint.utils.deepfloyd_no_diffusers import Pipeline
                pipeline = Pipeline()
            images: List[Image.Image] = pipeline.dream(prompt, count=num_repeats)
            for ind in range(num_repeats):
                images[ind].save(image_paths[ind])
        image_paths_all.extend(image_paths)
    if pipeline is not None:
        del pipeline
    return image_paths_all


def load_deepfloyd_samples(prompts: List[str], num_repeats=4):
    logger.info('loading deepfloyd samples from cache...')
    image_paths = cache_deepfloyd_samples(prompts, num_repeats)
    images: List[Image.Image] = []
    for path in image_paths:
        image = Image.open(path)
        images.append(image)
    return torch.stack([TF.to_tensor(image) * 2 - 1 for image in images])

class HookFunction:
    def __init__(self):
        self.layer_outputs = []

    def hook_layers(self, model):
        # https://github.com/openai/CLIP/blob/a1d071733d7111c9c014f024669f959182114e33/clip/model.py#L206
        layer_counts = 0
        for layer in model.transformer.resblocks:
            layer_counts += 1
            assert layer.__class__.__name__ == 'ResidualAttentionBlock'
            layer.register_forward_hook(self.save_output)
        assert layer_counts > 0

    def save_output(self, module, input, output):
        self.layer_outputs.append(output.detach())

    def clear_outputs(self):
        self.layer_outputs = []

class SyntheticBiLevel(torch.utils.data.Dataset):
    def __init__(self, data_root: str,
                 templates: Dict,
                 num_data_per_prompt: int = 8, num_data_copies: int = 1, num_tokens_per_word: int = 1,
                 num_placeholder_words: int = 1, num_placeholder_groups: int = 2, shared_tokens=0, real_image=None): 
        
        assert shared_tokens in [0, 1], shared_tokens
        num_placeholder_words = 1

        device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
        model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl", load_in_8bit=True, device_map="auto")
        
        # we limit it to words which are known to have corresponding t5 embeddings which are one token long
        blip_fruit_question = "What is the species of the fruit in the photo, in singular form?"
        blip_color_question = "What is the color of the fruit in the photo?"




        self.templates = build_from_config(templates)

        ground_truth_words = data_root.split(",")
        ground_truth_words = [word.replace('_', " ") for word in ground_truth_words]
        ground_truth_words = [word.split('-') for word in ground_truth_words] # [['apple', 'green'], ['apple', 'red'], ...]
        assert len(ground_truth_words) == num_placeholder_words, (ground_truth_words, len(ground_truth_words), num_placeholder_words)
        ground_truth_prompt_args = [[] for i in range(num_placeholder_groups)]
        for split_word in ground_truth_words:
            assert len(split_word) == num_placeholder_groups
            for i in range(num_placeholder_groups):
                ground_truth_prompt_args[i].append(split_word[i])
                # now we have [apple, apple, banana, banana] and [green, red, green, yellow]

        self.ground_truth_prompt_args = ground_truth_prompt_args
        self.unique_gt_words = ground_truth_words
        for gt_word in self.unique_gt_words:
            assert len(gt_word) == num_placeholder_groups

        unique_prompts = []
        for ind in range(num_placeholder_words):
            curr_prompt_words = []
            for ground_truth_prompt_arg in ground_truth_prompt_args:
                curr_prompt_words.append(ground_truth_prompt_arg[ind])
            prompt = self.templates[0].format(*curr_prompt_words)
            unique_prompts.append(prompt)
        self.gt_prompts: List[str] = unique_prompts
        self.images, self.blip_colors, self.blip_fruits, pil_images = deepfloyd_sample_prompts(unique_prompts, num_repeats=num_data_per_prompt, model=model, processor=processor, blip_fruit_q=blip_fruit_question, blip_color_q=blip_color_question, real_image=real_image)

        del processor
        del model
        torch.cuda.empty_cache()

        clip_vision = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14").to("cuda").requires_grad_(False)
        def clip_preprocess(x):
            x = kornia.geometry.resize(
                x, (clip_vision.config.image_size, clip_vision.config.image_size), interpolation='bicubic', align_corners=True, antialias=False
            )
            x = (x + 1.) / 2.
            # renormalize according to clip
            x = kornia.enhance.normalize(x, torch.Tensor([0.48145466, 0.4578275, 0.40821073]), torch.Tensor([0.26862954, 0.26130258, 0.27577711]))
            return x

        preprocessed_images = self.images.to("cuda")
        preprocessed_images = clip_preprocess(preprocessed_images)

        run_count = 0
        self.clip_features = []
        for img in preprocessed_images:
            with torch.no_grad():
                result = clip_vision(pixel_values=img.unsqueeze(0).expand(3, -1, -1, -1), output_hidden_states=True)
                result = result.hidden_states[1:][1::2]
                result = [clip_vision.vision_model.post_layernorm(hidden_states[:, 0, :]).to("cpu") for hidden_states in result]
                self.clip_features.append(result)
                run_count += 1

        self.clip_features = [torch.stack(clip_feature, dim=1).type(torch.float32).to("cpu") for clip_feature in self.clip_features]

        for i in range(len(self.clip_features)):
            assert self.clip_features[i].shape == (3, 12, 1024)
        assert len(self.clip_features) == len(self.images), (len(self.clip_features), len(self.images))

        inference_input = 'city-oil'
        inf_data = inference_input.split(",")
        inf_data = [word.replace('_', " ") for word in inf_data]
        inf_data = [word.split('-') for word in inf_data] # [['apple', 'green'], ['apple', 'red'], ...]

        inf_ph_tokens = [[f'mytoken{2*i}', f'mytoken{2*i + 1}'] for i in range(len(inf_data))]

        assert len(inf_data) == len(inf_ph_tokens), (len(inf_data), len(inf_ph_tokens), inf_data, inf_ph_tokens)

        self.inf_gt_prompts = [self.templates[0].format(*pair) for pair in inf_data]

        self.inf_prompts = [self.templates[0].format(*inf_ph_tokens[i]) for i in range(len(inf_data))]

        self.inf_fruit_prompts = [imagenet_templates_small[0].format(pair[0]) for pair in inf_ph_tokens]
        self.inf_color_prompts = ['an artwork made with {}'.format(pair[1]) for pair in inf_ph_tokens]

        self.inf_images, _, _, inf_pil_images = deepfloyd_sample_prompts(self.inf_gt_prompts, num_repeats=1)

        preprocessed_images = self.inf_images.to("cuda")
        preprocessed_images = clip_preprocess(preprocessed_images)

        run_count = 0
        self.inf_clip_features = []
        for img in preprocessed_images:
            with torch.no_grad():
                result = clip_vision(pixel_values=img.unsqueeze(0).expand(3, -1, -1, -1), output_hidden_states=True)
                result = result.hidden_states[1:][1::2]
                result = [clip_vision.vision_model.post_layernorm(hidden_states[:, 0, :]).to("cpu") for hidden_states in result]
                self.inf_clip_features.append(result)
                run_count += 1

        self.inf_clip_features = [torch.stack(clip_feature, dim=1).type(torch.float32).to("cpu") for clip_feature in self.inf_clip_features]
        for i in range(len(self.inf_clip_features)):
            assert self.inf_clip_features[i].shape == (3, 12, 1024)
        assert len(self.inf_clip_features) == len(self.inf_images), (len(self.inf_clip_features), len(self.inf_images))

        self.inf_dict = {
            'image': [img for img in self.inf_images],
            'prompt': self.inf_prompts,
            'gt_prompt': self.inf_gt_prompts,
            'fruit_prompt': self.inf_fruit_prompts,
            'color_prompt': self.inf_color_prompts,
            'clip_feature': [feat for feat in self.inf_clip_features],
        }


        del clip_vision
        
        torch.cuda.empty_cache()

        self.num_placeholder_words = num_placeholder_words*num_tokens_per_word
        num_placeholder_tokens = num_placeholder_words*num_tokens_per_word*num_placeholder_groups
        placeholder_words = [] #['mytoken0', 'mytoken1', 'mytoken0', ...., ]
        fruit_dict = {}
        fruit_count = 0
        color_count = 1
        for i in range(len(ground_truth_words)):
            curr_fruit = ground_truth_words[i][0]
            # category tokens are shared amongst all instances of the same category
            if shared_tokens == 1:
                if curr_fruit not in fruit_dict:
                    fruit_dict[curr_fruit] = f'mytoken{fruit_count}'
                    fruit_count += 2
                placeholder_words.append(fruit_dict[curr_fruit])
            else:
                placeholder_words.append(f'mytoken{fruit_count}')
                fruit_count += 2
            placeholder_words.append(f'mytoken{color_count}')
            color_count += 2
        placeholder_words = np.split(np.array(placeholder_words), num_placeholder_words*num_placeholder_groups)
        placeholder_words_prompt_args = np.transpose(np.split(np.array(placeholder_words), num_placeholder_words), (1,0,2))

        assert len(placeholder_words_prompt_args) == len(ground_truth_prompt_args), (len(placeholder_words_prompt_args), len(ground_truth_prompt_args))
        for placeholder_words_prompt_arg in placeholder_words_prompt_args:
            assert len(placeholder_words_prompt_arg) == num_placeholder_words, (placeholder_words_prompt_arg, num_placeholder_words)
        
        self.ph_words_all = []
        for ind in range(num_placeholder_words):
            curr_ph_words = []
            for placeholder_words_prompt_arg in placeholder_words_prompt_args:
                curr_ph_words.append(placeholder_words_prompt_arg[ind])
            self.ph_words_all.extend([curr_ph_words] * num_data_per_prompt)
        self.placeholder_words_prompt_args = placeholder_words_prompt_args
        unique_ph_words = [] # [['mytoken0', 'mytoken1'], ['mytoken2', 'mytoken3']]
        num_placeholder_words = len(self.placeholder_words_prompt_args[0])
        for ind in range(num_placeholder_words):
            curr_ph_words = []
            for placeholder_words_prompt_arg in self.placeholder_words_prompt_args:
                curr_ph_words.append(placeholder_words_prompt_arg[ind])
            unique_ph_words.append([''.join(word) for word in curr_ph_words])

        blip_fruit_for_each_ph = defaultdict(list)
        assert len(unique_ph_words) == len(self.blip_fruits), (len(unique_ph_words), len(self.blip_fruits), unique_ph_words, self.blip_fruits)
        for i in range(len(ground_truth_words)):
            ph_fruit = unique_ph_words[i][0]
            blip_fruit = self.blip_fruits[i]

            blip_fruit_for_each_ph[ph_fruit].append(blip_fruit)

        common_blip_fruit_for_each_ph = {}
        for ph_fruit in blip_fruit_for_each_ph:
            blip_fruit_counter = Counter(blip_fruit_for_each_ph[ph_fruit])
            blip_common_fruit = blip_fruit_counter.most_common(1)[0][0]
            common_blip_fruit_for_each_ph[ph_fruit] = blip_common_fruit

        self.blip_fruits = [common_blip_fruit_for_each_ph[ph_pair[0]] for ph_pair in unique_ph_words]
        assert len(unique_ph_words) == len(self.blip_fruits), (len(unique_ph_words), len(self.blip_fruits), unique_ph_words, self.blip_fruits)
        self.num_data_copies = num_data_copies
        self.num_data_per_prompt = num_data_per_prompt

    def __len__(self):
        return len(self.images) * self.num_data_copies

    def __getitem__(self, item):
        item = item % len(self.images)
        # GLIDE expects range [-1, 1]
        image: torch.Tensor = self.images[item]

        curr_ph_words = self.ph_words_all[item]

        template = self.templates[np.random.choice(len(self.templates))]
        prompt = template.format(*[''.join(word) for word in curr_ph_words])

        clip_feature = self.clip_features[item]
        blip_color = self.blip_colors[item//self.num_data_per_prompt]
        blip_fruit = self.blip_fruits[item//self.num_data_per_prompt]

        return {'image': image, 'prompt': prompt, 'gt_prompt': self.gt_prompts[item//self.num_data_per_prompt], 'clip_feature': clip_feature, 'blip_color': blip_color, 'blip_fruit': blip_fruit}


class SyntheticBiLevelEval(SyntheticBiLevel):
    def __init__(self, data_root: str, num_placeholder_words: int, templates: Dict, ref_dataset: SyntheticBiLevel):

        unique_ph_words = [] # [['mytoken0', 'mytoken1'], ['mytoken2', 'mytoken3']]
        num_placeholder_words = len(ref_dataset.placeholder_words_prompt_args[0])
        for ind in range(num_placeholder_words):
            curr_ph_words = []
            for placeholder_words_prompt_arg in ref_dataset.placeholder_words_prompt_args:
                curr_ph_words.append(placeholder_words_prompt_arg[ind])
            unique_ph_words.append([''.join(word) for word in curr_ph_words])
        
        self.val_batch_size = 4
        assert len(ref_dataset.images) == 1, len(ref_dataset.images)
        self.image = ref_dataset.images[0].clone()
        self.gt_word_pairs = ref_dataset.unique_gt_words
        self.ph_word_pairs = unique_ph_words
        self.full_template = ref_dataset.templates[0]
        self.fruit_template = imagenet_templates_small[0]
        self.color_template0 = 'the color {}'
        self.color_template1 = 'an artwork made with {}'
        self.color_template2 = '{}'
        

        self.all_gt_colors = [word_pair[1] for word_pair in self.gt_word_pairs]
        self.all_ph_colors = [word_pair[1] for word_pair in self.ph_word_pairs]
        self.all_colors = [word_pair[1] for word_pair in ref_dataset.unique_gt_words]
        self.blip_colors = ref_dataset.blip_colors
        self.blip_fruits = ref_dataset.blip_fruits

        assert len(self.all_ph_colors) == len(self.all_gt_colors), (len(self.all_ph_colors), len(self.all_gt_colors))

        self.inf_dict = ref_dataset.inf_dict
        

    def __len__(self):
        return len(self.gt_word_pairs) * self.val_batch_size


    def __getitem__(self, item):
        gt_word_pair = self.gt_word_pairs[item//self.val_batch_size]
        ph_word_pair = self.ph_word_pairs[item//self.val_batch_size]
        gt_prompt = self.full_template.format(*gt_word_pair)
        prompt = self.full_template.format(*ph_word_pair)

        random.seed(item)
        assert len(self.all_gt_colors) == len(self.all_ph_colors), (len(self.all_gt_colors), len(self.all_ph_colors), self.all_gt_colors, self.all_ph_colors)
        if item < self.val_batch_size:
            indices = list(range(len(self.all_gt_colors)))
        else:
            indices = sorted(random.sample((list(range(len(self.all_gt_colors)))), 10))
        
        return {
            'image': self.image,
            'prompt': prompt,
            'gt_prompt': gt_prompt,
            'gt_fruit': gt_word_pair[0],
            'gt_color': gt_word_pair[1],
            'ph_fruit': ph_word_pair[0],
            'ph_color': ph_word_pair[1],
            'full_template': self.full_template,
            'fruit_template': self.fruit_template,
            'color_template0': self.color_template0,
            'color_template1': self.color_template1,
            'color_template2': self.color_template2,
            'all_gt_colors': [self.all_gt_colors[i] for i in indices],
            'all_ph_colors': [self.all_ph_colors[i] for i in indices],
            'all_colors': self.all_colors,
            'blip_color': self.blip_colors[item//self.val_batch_size],
            'blip_fruit': self.blip_fruits[item//self.val_batch_size],
            'inf': self.inf_dict
        }
