from tu.trainers.simple_trainer import BaseTrainer
from tu.utils.training import process_batch
import numpy as np
import torch
from tu.utils.visualize import dump_helper, dump_row_helper
import numpy as np

class Trainer(BaseTrainer):
    def _visualize_core(self, data, prefix=None):
        if 'dataset' in data:
            dataset = data['dataset']
            dump_helper(self, 'full_dataset', dataset.images * .5 + .5, prefix=prefix)
        
        model = self.modules[self.module_key]

        model.eval()
        with torch.no_grad():
            if 'val' in prefix:
                zero_image = torch.zeros_like(data['image'][0]).unsqueeze(0)
                images = [data['image']]
                out = model({'prompt': data['prompt']}, return_all=True)
                fake_data = {'image': zero_image.repeat(len(data['prompt']), 1, 1, 1), 'prompt': data['prompt']}
                images.append(self.loss_modules['textual_inversion'].visualize({'embeddings': out['embeddings']}, fake_data)['image'])

                fruit_prompts = []
                color_prompts0 = []
                color_prompts1 = []
                color_prompts2 = []
                
                for ph_fruit, ph_color, fruit_template, color_template0, color_template1, color_template2 in zip(data['ph_fruit'], data['ph_color'], data['fruit_template'], data['color_template0'], data['color_template1'], data['color_template2']):
                    fruit_prompts.append(fruit_template.format(ph_fruit))
                    color_prompts0.append(color_template0.format(ph_color))
                    color_prompts1.append(color_template1.format(ph_color))
                    color_prompts2.append(color_template2.format(ph_color))
                    
                out = model({'prompt': fruit_prompts}, return_all=True)
                fake_data = {'image': zero_image.repeat(len(fruit_prompts), 1, 1, 1), 'prompt': fruit_prompts}
                images.append(self.loss_modules['textual_inversion'].visualize({'embeddings': out['embeddings']}, fake_data)['image'])
                out = model({'prompt': color_prompts1}, return_all=True)
                fake_data = {'image': zero_image.repeat(len(color_prompts1), 1, 1, 1), 'prompt': color_prompts1}
                images.append(self.loss_modules['textual_inversion'].visualize({'embeddings': out['embeddings']}, fake_data)['image'])

                # input is a list containing ~5 sublists, each sublist contains tensor of shape (8, 3, 64, 64). The output will be (40, 3, 64, 64)
                # they are interleaved, so if the 5 sublists were abcde we would have a0,b0,c0,d0,e0,a1...
                nrow = len(images)
                images = [tensor.unsqueeze(1) for tensor in images]
                images = torch.cat(images, dim=1)
                reshape_size = [-1] + list(images[0].shape[1:])
                images = images.view(*reshape_size)

                #prints out the columns of images 
                dump_helper(self, f'reconstruction, approx. iteration = {out["iteration"]}', images * .5 + .5, prefix=prefix, nrow=nrow)
                
                comp_fruits = [] 
                comp_imgs = [] 
                gt_colors = []
                gt_comp_prompts = []
                data['all_gt_colors'] = ['charcoal', 'oil', 'paint', 'acrylic', 'crayon', 'pastel']
                data['all_ph_colors'] = [f'mytoken{x}' for x in [49, 51, 53, 55, 57, 59]]
                for i in range(len(data['all_gt_colors'])):
                    curr_gt_color = data['all_gt_colors'][i]
                    curr_ph_color = data['all_ph_colors'][i]
                    gt_colors.append(curr_gt_color)
                    gt_comp_prompts.append(data['full_template'][0].format(data['gt_fruit'][0], curr_gt_color))
                    comp_fruits.append(data['full_template'][0].format(data['ph_fruit'][0], curr_ph_color))
                out = model({'prompt': comp_fruits}, return_all=True)
                num_cross_sets = 4
                for i in range(num_cross_sets):
                    fake_data = {'image': zero_image.repeat(len(comp_fruits), 1, 1, 1), 'prompt': comp_fruits}
                    comp_imgs.append(self.loss_modules['textual_inversion'].visualize({'embeddings': out['embeddings']}, fake_data)['image'])
                nrow = len(comp_fruits)
                comp_imgs = torch.cat(comp_imgs, dim=0)
                
                dump_helper(self, 'composition with attribute', comp_imgs * .5 + .5, prefix=prefix, nrow=nrow)

                comp_colors = [] 
                comp_imgs = [] 
                gt_fruits = []
                gt_comp_prompts = []
                data['all_gt_fruits'] = ['tree', 'apple', 'cat', 'dog', 'face', 'city', 'hill']
                data['all_ph_fruits'] = [f'mytoken{x}' for x in [2, 4, 6, 8, 10, 12, 14]]
                for i in range(len(data['all_gt_fruits'])):
                    curr_gt_fruit = data['all_gt_fruits'][i]
                    curr_ph_fruit = data['all_ph_fruits'][i]
                    gt_fruits.append(curr_gt_fruit)
                    gt_comp_prompts.append(data['full_template'][0].format(curr_gt_fruit, data['gt_color'][0]))
                    comp_colors.append(data['full_template'][0].format(curr_ph_fruit, data['ph_color'][0]))

                out = model({'prompt': comp_colors}, return_all=True)
                num_cross_sets = 4
                for i in range(num_cross_sets):
                    fake_data = {'image': zero_image.repeat(len(comp_colors), 1, 1, 1), 'prompt': comp_colors}
                    comp_imgs.append(self.loss_modules['textual_inversion'].visualize({'embeddings': out['embeddings']}, fake_data)['image'])

                nrow = len(comp_colors)
                comp_imgs = torch.cat(comp_imgs, dim=0)
                
                dump_helper(self, 'composition with category', comp_imgs * .5 + .5, prefix=prefix, nrow=nrow)
                
            elif 'inference' in prefix:
                data['image'] = torch.stack(data['image'])
                data['clip_feature'] = torch.stack(data['clip_feature'])
                images = []
                
                images.append(data['image'])
                for _ in range(4):
                    out = model._inference_forward(data)
                    images.append(self.loss_modules['textual_inversion'].visualize({'embeddings': out['embeddings']}, data)['image'])

                for _ in range(4):
                    fake_data = {'clip_feature': data['clip_feature'], 'prompt': data['fruit_prompt'], 'image': data['image']}
                    out = model._inference_forward(fake_data)
                    images.append(self.loss_modules['textual_inversion'].visualize({'embeddings': out['embeddings']}, fake_data)['image'])

                for _ in range(4):
                    fake_data = {'clip_feature': data['clip_feature'], 'prompt': data['color_prompt'], 'image': data['image']}
                    out = model._inference_forward(fake_data)
                    images.append(self.loss_modules['textual_inversion'].visualize({'embeddings': out['embeddings']}, fake_data)['image'])

                images = torch.cat(images, dim=0)

                dump_helper(self, '', images * .5 + .5, prefix=prefix, nrow=len(data['gt_prompt']))
                


            else:
                dump_helper(self, 'input', data['image'] * .5 + .5, prefix=prefix)
                out = model(data, return_all=True)
                for k, v in self.loss_modules['textual_inversion'].visualize(out, data).items():
                    if isinstance(v, torch.Tensor) and v.ndim == 4:
                        dump_helper(self, k, v * .5 + .5, prefix=prefix)
                    elif isinstance(v, list) and isinstance(v[0], torch.Tensor):
                        if len(v) > 10:
                            indices = np.linspace(0, len(v), 10, endpoint=False, dtype=int)
                        else:
                            indices = np.arange(len(v))
                        v = [v[i] * .5 + .5 for i in indices]
                        dump_row_helper(self, [str(i) for i in indices], v, prefix=prefix)


    def validate(self, dataloader):
        for ind, batch in enumerate(dataloader):
            data = process_batch(batch)
            with torch.no_grad():
                self._visualize_core(data, prefix=f"val/{data['gt_fruit'][0]}-{data['gt_color'][0]}")

        
