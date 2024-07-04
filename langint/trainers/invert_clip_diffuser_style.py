from tu.trainers.simple_trainer import BaseTrainer
from tu.utils.training import process_batch
import numpy as np
import torch
from tu.utils.visualize import dump_helper, dump_row_helper

class Trainer(BaseTrainer):
    def _visualize_core(self, data, prefix=None):
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
                mat_prompts = []
                color_prompts = []
                
                for ph_fruit, ph_mat, ph_color, fruit_template, mat_template, color_template in zip(data['ph_fruit'], data['ph_mat'], data['ph_color'], data['fruit_template'], data['mat_template'], data['color_template']):
                    fruit_prompts.append(fruit_template.format(ph_fruit))
                    mat_prompts.append(mat_template.format(ph_mat))
                    color_prompts.append(color_template.format(ph_color))

                out = model({'prompt': fruit_prompts}, return_all=True)
                fake_data = {'image': zero_image.repeat(len(fruit_prompts), 1, 1, 1), 'prompt': fruit_prompts}
                images.append(self.loss_modules['textual_inversion'].visualize({'embeddings': out['embeddings']}, fake_data)['image'])
                out = model({'prompt': mat_prompts}, return_all=True)
                fake_data = {'image': zero_image.repeat(len(mat_prompts), 1, 1, 1), 'prompt': mat_prompts}
                images.append(self.loss_modules['textual_inversion'].visualize({'embeddings': out['embeddings']}, fake_data)['image'])
                out = model({'prompt': color_prompts}, return_all=True)
                fake_data = {'image': zero_image.repeat(len(color_prompts), 1, 1, 1), 'prompt': color_prompts}
                images.append(self.loss_modules['textual_inversion'].visualize({'embeddings': out['embeddings']}, fake_data)['image'])


                # input is a list containing ~5 sublists, each sublist contains tensor of shape (8, 3, 64, 64). The output will be (40, 3, 64, 64)
                # they are interleaved, so if the 5 sublists were abcde we would have a0,b0,c0,d0,e0,a1...
                nrow = len(images)
                images = [tensor.unsqueeze(1) for tensor in images]
                images = torch.cat(images, dim=1)
                reshape_size = [-1] + list(images[0].shape[1:])
                images = images.view(*reshape_size)

                # prints out the columns of images 
                dump_helper(self, 'Reconstruction', images * .5 + .5, prefix='', nrow=nrow)

                # select already-trained tokens to combine with the new tokens
                rand_gt_fruits = ['chair', 'bench', 'table', 'chair', 'bench', 'table']
                rand_ph_fruits = [f'mytoken{x}' for x in [0, 6, 3, 0, 6, 3]]
                rand_gt_mats = ['plastic', 'plastic', 'wood', 'wood', 'metal', 'metal']
                rand_ph_mats = [f'mytoken{x}' for x in [49, 142, 4, 7, 25, 28]]
                rand_gt_colors = ['green', 'blue', 'black', 'white', 'red', 'orange']
                rand_ph_colors = [f'mytoken{x}' for x in [11, 14, 20, 23, 170, 5]]
                perm_length = 6

                assert perm_length == len(rand_gt_fruits) == len(rand_ph_fruits) == len(rand_gt_mats) == len(rand_ph_mats) == len(rand_gt_colors) == len(rand_ph_colors)

                new_gt_fruits = [data['gt_fruit'][0]] * perm_length
                new_ph_fruits = [data['ph_fruit'][0]] * perm_length
                new_gt_mats = [data['gt_mat'][0]] * perm_length
                new_ph_mats = [data['ph_mat'][0]] * perm_length
                new_gt_colors = [data['gt_color'][0]] * perm_length
                new_ph_colors = [data['ph_color'][0]] * perm_length

                perms = [
                    [rand_gt_fruits, rand_gt_mats, new_gt_colors, rand_ph_fruits, rand_ph_mats, new_ph_colors],
                    [rand_gt_fruits, new_gt_mats, rand_gt_colors, rand_ph_fruits, new_ph_mats, rand_ph_colors],
                    [new_gt_fruits, rand_gt_mats, rand_gt_colors, new_ph_fruits, rand_ph_mats, rand_ph_colors],
                ]

                for perm in perms:
                    gt_comp_prompts = []
                    ph_comp_prompts = []
                    comp_imgs = []
                    for gt_fruit, gt_mat, gt_color, ph_fruit, ph_mat, ph_color in zip(*perm):
                        gt_comp_prompts.append(data['full_template'][0].format(gt_fruit, gt_mat, gt_color))
                        ph_comp_prompts.append(data['full_template'][0].format(ph_fruit, ph_mat, ph_color))

                    out = model({'prompt': ph_comp_prompts}, return_all=True)
                    num_cross_sets = 4
                    for i in range(num_cross_sets):
                        fake_data = {'image': zero_image.repeat(len(ph_comp_prompts), 1, 1, 1), 'prompt': ph_comp_prompts}
                        comp_imgs.append(self.loss_modules['textual_inversion'].visualize({'embeddings': out['embeddings']}, fake_data)['image'])
                    nrow = len(ph_comp_prompts)
                    comp_imgs = torch.cat(comp_imgs, dim=0)
                    
                    dump_helper(self, 'Extrapolation', comp_imgs * .5 + .5, prefix='', nrow=nrow)
            else:
                dump_helper(self, 'Input', data['image'] * .5 + .5, prefix='')
                out = model(data, return_all=True)
                for k, v in self.loss_modules['textual_inversion'].visualize(out, data).items():
                    if isinstance(v, torch.Tensor) and v.ndim == 4:
                        dump_helper(self, "Output", v * .5 + .5, prefix='')
                    elif isinstance(v, list) and isinstance(v[0], torch.Tensor):
                        if len(v) > 10:
                            indices = np.linspace(0, len(v), 10, endpoint=False, dtype=int)
                        else:
                            indices = np.arange(len(v))
                        v = [v[i] * .5 + .5 for i in indices]
                        dump_row_helper(self, [str(i) for i in indices], v, prefix='')

                if self.writer is not None:
                    self.vi_helper.dump_table(self.vi, [[str(out['iteration'])]],
                                              table_name='', col_names=['Epoch #'])

    def validate(self, dataloader):
        for ind, batch in enumerate(dataloader):
            data = process_batch(batch)
            with torch.no_grad():
                self._visualize_core(data, prefix="val")

        
