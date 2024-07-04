import torch.nn as nn
import os
from langint.datasets.glide import placeholder_words_list
import torch
from langint.utils.deepfloyd_no_diffusers import CACHE_DIR
from langint.third_party.deepfloyd.deepfloyd_if.modules.t5 import T5Embedder
from typing import List, Dict, Optional
import logging
from datetime import datetime

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)


class Embeddings(nn.Module):
    def __init__(self, num_placeholder_words: int, initializer_word: Optional[str], num_placeholder_groups: int = 2, shared_tokens=0, gt_init=0, blip_guidance=-1, fruit_blip_coeff=0, color_blip_coeff=0):
        super().__init__()
        self.blip_guidance = blip_guidance
        self.fruit_blip_coeff = fruit_blip_coeff
        self.color_blip_coeff = color_blip_coeff
        if os.getenv('T5_BFLOAT16') == '1':
            self.t5 = t5 = T5Embedder(device='cuda', cache_dir=CACHE_DIR, torch_dtype=torch.bfloat16)
        else:
            self.t5 = t5 = T5Embedder(device='cuda', cache_dir=CACHE_DIR, torch_dtype=torch.float32)
        tokenizer = t5.tokenizer
        text_encoder = t5.model

        assert shared_tokens in [0, 1], shared_tokens
        assert gt_init in [0, 1], gt_init
        assert num_placeholder_groups == 2, num_placeholder_groups

        num_placeholder_tokens = num_placeholder_words * num_placeholder_groups
        print('num_placeholder_words, groups, tokens', num_placeholder_words, num_placeholder_groups, num_placeholder_tokens)
        assert initializer_word is None, initializer_word

        # get placeholder token (string) and token id (indices in the embedding)
        placeholder_tokens: List[str] = placeholder_words_list[:num_placeholder_tokens]
        for placeholder_token in placeholder_tokens:
            assert t5.text_preprocessing(placeholder_token) == placeholder_token, (placeholder_token, t5.text_preprocessing(placeholder_token))

        # https://huggingface.co/docs/transformers/v4.29.1/en/internal/tokenization_utils#transformers.SpecialTokensMixin.add_tokens
        num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
        if num_added_tokens != len(placeholder_tokens):
            raise ValueError(f'Expected to add {len(placeholder_tokens)} tokens, got {num_added_tokens}')
        # Resize the token embeddings as we are adding new special tokens to the tokenizer
        text_encoder.resize_token_embeddings(len(tokenizer))

        placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)
        logger.info(f'placeholder tokens: {str(placeholder_tokens)}')
        logger.info(f'placeholder tokens encoded: {str(tokenizer.encode(" ".join(placeholder_tokens)))}')
        logger.info(f'placeholder tokens encoded decoded: {str(tokenizer.decode(tokenizer.encode(" ".join(placeholder_tokens))))}')
        logger.info(f'placeholder token ids: {str(placeholder_token_ids)}')
        logger.info(f'placeholder tokens recon: {str(tokenizer.convert_ids_to_tokens(placeholder_token_ids))}')

        # get initializing token (string) and token id (indices in the embedding)
        assert initializer_word is None
        initializer_embs = []
        
        if initializer_word is None:
            embs_mean = text_encoder.get_input_embeddings().weight.data.mean(0)
            self.embs_mean = embs_mean.clone()

            initializer_embs = [
                embs_mean.clone() + torch.randn_like(embs_mean) * 0.1
                for _ in range(num_placeholder_tokens)
            ]

            for emb in initializer_embs:
                assert emb.shape == (4096,), emb.shape

            assert len(initializer_embs) == num_placeholder_tokens, (num_placeholder_tokens, len(initializer_embs), initializer_embs)
        else:
            initializer_tokens: List[str] = initializer_word.split(',')
            if len(initializer_tokens) == 1:
                initializer_tokens = initializer_tokens * num_placeholder_tokens
            assert len(initializer_tokens) == num_placeholder_tokens, (len(initializer_tokens), num_placeholder_tokens)
            initializer_token_ids: List[List[int]] = [tokenizer.encode(initializer_tokens[ind], add_special_tokens=False) for ind in range(num_placeholder_tokens)]
            assert len(initializer_token_ids) == len(initializer_tokens), (len(initializer_token_ids), len(initializer_tokens))
            logger.info(f'init tokens: {str(initializer_tokens)}')
            logger.info(f'init tokens encoded: {str(tokenizer.encode(" ".join(initializer_tokens)))}')
            logger.info(f'init tokens encoded decoded: {str(tokenizer.decode(tokenizer.encode(" ".join(initializer_tokens))))}')
            logger.info(f'init token ids: {str(initializer_token_ids)}')
            for ind in range(num_placeholder_tokens):
                assert len(initializer_token_ids[ind]) == 1, (initializer_tokens[ind], len(initializer_token_ids[ind]))

            initializer_embs = [
                text_encoder.get_input_embeddings().weight.data[initializer_token_ids[ind]].clone()
                for ind in range(num_placeholder_tokens)
            ]

            
        self.usage_counts = [0]

        text_encoder.requires_grad_(False)

        self.trainable_embeddings = nn.ParameterDict({
            placeholder_tokens[ind]:
                nn.Parameter(initializer_embs[ind])
            for ind in range(num_placeholder_tokens)
        })
        self.initial_embeddings = nn.ParameterDict({
            placeholder_tokens[ind]:
                nn.Parameter(initializer_embs[ind].clone()) 
            for ind in range(num_placeholder_tokens)
        })
        self.placeholder_token_to_id = {
            placeholder_tokens[ind]: placeholder_token_ids[ind]
            for ind in range(num_placeholder_tokens)
        }

        self.num_placeholder_groups = num_placeholder_groups
        self.num_placeholder_words = num_placeholder_words

        self.iteration = 0

        self.pos_linear_fruit = nn.ModuleList([nn.Linear(1024, 1024) for _ in range(12)])
        self.dropout_fruit = nn.Dropout(0.2)
        self.act_fruit = nn.LeakyReLU()
        self.final_linear_fruit = nn.Linear(1024, 4096)

        self.pos_linear_color = nn.ModuleList([nn.Linear(1024, 1024) for _ in range(12)])
        self.dropout_color = nn.Dropout(0.2)
        self.act_color = nn.LeakyReLU()
        self.final_linear_color = nn.Linear(1024, 4096)

        self.blip_mse_loss = nn.MSELoss()



    def _pass_through_layers(self, x, data):
        outputs = []
        for i in range(12):
            outputs.append(self.dropout_fruit(self.pos_linear_fruit[i](x[:, :, i, :])))
        x_fruit = torch.stack(outputs, dim=2)
        
        x_fruit = x_fruit.mean(dim=1)
        x_fruit = x_fruit.mean(dim=1)

        x_fruit = self.act_fruit(x_fruit)

        x_fruit = self.final_linear_fruit(x_fruit)

        outputs = []
        for i in range(12):
            outputs.append(self.dropout_color(self.pos_linear_color[i](x[:, :, i, :])))
        x_color = torch.stack(outputs, dim=2)
        
        x_color = x_color.mean(dim=1)
        x_color = x_color.mean(dim=1)

        x_color = self.act_color(x_color)

        x_color = self.final_linear_color(x_color)

        x = torch.stack((x_fruit, x_color), dim=1)
        return x


    def _process_clip_features(self, data, tokenizer, text_encoder):
        blip_word_dict = {}
        assert len(data['blip_color']) == len(data['prompt']), (data['blip_color'], data['prompt'])

        ph_words_by_prompt = []
        for i in range(len(data['prompt'])):
            prompt = data['prompt'][i]
            words: List[str] = prompt.split(' ')
            ph_words_for_curr_prompt = [None, None]
            for word in words:
                if word in self.trainable_embeddings:
                    _, number = word.split("mytoken")
                    if int(number) % 2 == 0: 
                        ph_words_for_curr_prompt[0] = word
                        blip_fruit_token = tokenizer.encode(data['blip_fruit'][i], add_special_tokens=False)
                        blip_fruit_emb = text_encoder.get_input_embeddings().weight.data[blip_fruit_token].clone()
                        assert len(blip_fruit_emb) == 1, (len(blip_fruit_emb), data['blip_fruit'][i], word)
                        blip_word_dict[word] = blip_fruit_emb[0]
                    else:
                        ph_words_for_curr_prompt[1] = word
                        blip_color_token = tokenizer.encode(data['blip_color'][i], add_special_tokens=False)
                        blip_color_emb = text_encoder.get_input_embeddings().weight.data[blip_color_token].clone()
                        assert len(blip_color_emb) == 1, (len(blip_color_emb), data['blip_color'][i], word)
                        blip_word_dict[word] = blip_color_emb[0]
            assert ph_words_for_curr_prompt[0] != None, prompt
            assert ph_words_for_curr_prompt[1] != None, prompt 
            ph_words_by_prompt.append(ph_words_for_curr_prompt)
        assert len(ph_words_by_prompt) == len(data['clip_feature']), (len(ph_words_by_prompt), len(data['clip_feature']))

        assert len(data['clip_feature']) == len(data['prompt']), (len(data['clip_feature']), len(data['prompt']))
        x = data['clip_feature'].clone().to('cuda')
        x = self._pass_through_layers(x, data)
        
        assert len(ph_words_by_prompt) == len(x), (len(ph_words_by_prompt), len(x))

        temp_ph_word_dict = {}

        for ph_words, emb in zip(ph_words_by_prompt, x):
            assert len(emb) == len(ph_words) == 2, (len(emb), len(ph_words), ph_words)
            ph_fruit, ph_color = ph_words
            emb_fruit, emb_color = emb

            assert len(self.trainable_embeddings[ph_fruit]) == len(emb_fruit), (len(self.trainable_embeddings[ph_fruit]), len(emb_fruit))
            self.trainable_embeddings[ph_fruit] = nn.Parameter(emb_fruit.to(torch.bfloat16).clone())
            temp_ph_word_dict[ph_fruit] = emb_fruit.to(torch.bfloat16)

            assert len(self.trainable_embeddings[ph_color]) == len(emb_color), (len(self.trainable_embeddings[ph_color]), len(emb_color))
            self.trainable_embeddings[ph_color] = nn.Parameter(emb_color.to(torch.bfloat16).clone())
            temp_ph_word_dict[ph_color] = emb_color.to(torch.bfloat16)

        return temp_ph_word_dict, blip_word_dict

    def _get_text_embeddings(self, texts, t5, tokenizer, model, data):
        texts = [t5.text_preprocessing(text) for text in texts]
        text_tokens_and_mask = tokenizer(
            texts,
            max_length=77,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        text_tokens_and_mask['input_ids'] = text_tokens_and_mask['input_ids']
        text_tokens_and_mask['attention_mask'] = text_tokens_and_mask['attention_mask']

        # https://github.com/huggingface/transformers/blob/4b6aecb48e4961efef9edb8062dbbdd1f3d9385e/src/transformers/models/t5/modeling_t5.py#L1900
        input_ids = text_tokens_and_mask['input_ids'].cuda()  # (bs, max_length=77)
        with torch.no_grad():
            # https://github.com/huggingface/transformers/blob/4b6aecb48e4961efef9edb8062dbbdd1f3d9385e/src/transformers/models/t5/modeling_t5.py#L943
            inputs_embeds = model.shared(input_ids)


        if self.training:
            temp_ph_word_dict, blip_word_dict = self._process_clip_features(data, tokenizer, model)
        fruit_blip_loss = []
        color_blip_loss = []
        for i in range(len(self.placeholder_token_to_id)):
            token = f'mytoken{i}'
            assert token in self.placeholder_token_to_id, (token, self.placeholder_token_to_id)
            token_id = self.placeholder_token_to_id[token]
            if token_id in input_ids:

                if self.training:
                    inputs_embeds[input_ids == token_id] = temp_ph_word_dict[token]
                    if self.blip_guidance == 0:
                        if i % 2 == 0:
                            fruit_blip_loss.append(self.blip_mse_loss(temp_ph_word_dict[token], blip_word_dict[token]))
                        if i % 2 == 1:
                            color_blip_loss.append(self.blip_mse_loss(temp_ph_word_dict[token], blip_word_dict[token]))
                else:
                    inputs_embeds[input_ids == token_id] = self.trainable_embeddings[token]

        text_encoder_embs = model(
            inputs_embeds=inputs_embeds,
            input_ids=None,
            attention_mask=text_tokens_and_mask['attention_mask'].cuda(),
        )['last_hidden_state']
        fruit_blip_loss = (sum(fruit_blip_loss) / len(fruit_blip_loss)).float() if fruit_blip_loss else 0.0
        color_blip_loss = (sum(color_blip_loss) / len(color_blip_loss)).float() if color_blip_loss else 0.0

        return text_encoder_embs, text_tokens_and_mask, fruit_blip_loss, color_blip_loss

    def forward(self, data, return_all=False, return_pre_gumbel=False) -> Dict[str, torch.Tensor]:
        if self.training:
            self.iteration += 1
        
        t5 = self.t5
        tokenizer = t5.tokenizer
        model = t5.model
        # follow t5.get_text_embeddings
        texts: List[str] = data['prompt']
        ret = {}

        text_encoder_embs, text_tokens_and_mask, fruit_blip_loss, color_blip_loss = self._get_text_embeddings(texts, t5, tokenizer, model, data)

        assert not torch.isnan(text_encoder_embs).any(), (torch.isnan(text_encoder_embs).any(), text_encoder_embs)

        ret['embeddings'] = text_encoder_embs
        ret['fruit_blip_loss'] = fruit_blip_loss * self.fruit_blip_coeff 
        ret['color_blip_loss'] = color_blip_loss * self.color_blip_coeff 
        ret['iteration'] = self.iteration
        if return_all:
            input_ids_list = []
            for ind in range(len(data['prompt'])):
                input_ids = text_tokens_and_mask['input_ids'][ind]
                attention_mask = text_tokens_and_mask['attention_mask'][ind]
                input_ids = input_ids[:attention_mask.sum()]
                input_ids_list.append(input_ids)
            ret['input_ids']: List[torch.Tensor] = input_ids_list
            ret['processed_prompts']: List[str] = texts


        return ret
    
    def _inference_forward(self, data):
        
        t5 = self.t5
        tokenizer = t5.tokenizer
        model = t5.model
        # follow t5.get_text_embeddings
        texts: List[str] = data['prompt']
        ret = {}

        texts = [t5.text_preprocessing(text) for text in texts]
        text_tokens_and_mask = tokenizer(
            texts,
            max_length=77,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        text_tokens_and_mask['input_ids'] = text_tokens_and_mask['input_ids']
        text_tokens_and_mask['attention_mask'] = text_tokens_and_mask['attention_mask']

        # https://github.com/huggingface/transformers/blob/4b6aecb48e4961efef9edb8062dbbdd1f3d9385e/src/transformers/models/t5/modeling_t5.py#L1900
        input_ids = text_tokens_and_mask['input_ids'].cuda()  # (bs, max_length=77)
        with torch.no_grad():
            # https://github.com/huggingface/transformers/blob/4b6aecb48e4961efef9edb8062dbbdd1f3d9385e/src/transformers/models/t5/modeling_t5.py#L943
            inputs_embeds = model.shared(input_ids)

        ph_words_by_prompt = []
        for prompt in data['prompt']:
            words: List[str] = prompt.split(' ')
            ph_words_for_curr_prompt = [None, None]
            for word in words:
                if word in self.trainable_embeddings:
                    _, number = word.split("mytoken")
                    ph_words_for_curr_prompt[int(number) % 2] = word
            ph_words_by_prompt.append(ph_words_for_curr_prompt)
        assert len(ph_words_by_prompt) == len(data['clip_feature']), (len(ph_words_by_prompt), len(data['clip_feature']))

        assert len(data['clip_feature']) == len(data['prompt']), (len(data['clip_feature']), len(data['prompt']))
        x = data['clip_feature'].clone().to('cuda')
        x = self._pass_through_layers(x, data)

        assert len(ph_words_by_prompt) == len(x), (len(ph_words_by_prompt), len(x))

        temp_ph_word_dict = {}

        for ph_word, emb in zip(ph_words_by_prompt, x):
            assert len(ph_word) == len(emb) == 2, (len(ph_word), len(emb), ph_word)
            ph_fruit, ph_color = ph_word 
            emb_fruit, emb_color = emb
            if ph_fruit != None:
                temp_ph_word_dict[ph_fruit] = emb_fruit.to(torch.bfloat16)
            if ph_color != None:
                temp_ph_word_dict[ph_color] = emb_color.to(torch.bfloat16)
        
        for i in range(len(self.placeholder_token_to_id)):
            token = f'mytoken{i}'
            token_id = self.placeholder_token_to_id[token]
            if token_id in input_ids:
                inputs_embeds[input_ids == token_id] = temp_ph_word_dict[token]
        text_encoder_embs = model(
            inputs_embeds=inputs_embeds,
            input_ids=None,
            attention_mask=text_tokens_and_mask['attention_mask'].cuda(),
        )['last_hidden_state']
        assert not torch.isnan(text_encoder_embs).any(), (torch.isnan(text_encoder_embs).any(), text_encoder_embs)

        ret['embeddings'] = text_encoder_embs

        return ret