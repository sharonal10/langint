### Setup
```
conda create --name langint python=3.10 -y
conda activate langint
cd langint/third_party
git clone https://github.com/deep-floyd/IF.git deepfloyd
cd ../..
pip install deepfloyd_if==1.0.2rc0 
pip install xformers==0.0.16 tensorboard==2.17.0 imageio==2.34.2 kornia==0.7.0 bitsandbytes==0.43.1 openai-clip==1.0.1 triton==2.3.1 Pillow==9.0.1
pip install git+https://github.com/huggingface/transformers # will cause conflicts, ignore them
pip install transformers[torch]
pip install -e .
cd tu
pip install -e .
cd ..

pip install huggingface_hub --upgrade
python
>> from huggingface_hub import login
>> login() # create an account and token at https://huggingface.co/settings/tokens
```

### Variations
Each branch in this git repo is a different variation.

#### Training a new model:
two-axes-training - learns two axes: fruit and color.
```
python scripts/train_clip_inversion.py -c configs/train_deepfloyd_inversion.yaml -t fruits-two-train -d 'cherry-red,cherry-blue,cherry-green,apple-purple,apple-black,apple-yellow,banana-blue,banana-yellow,mango-purple,mango-yellow,mango-orange,strawberry-red,strawberry-blue,pineapple-yellow,pineapple-blue,pineapple-green,lemon-black,lemon-orange,lemon-green,raspberry-green,raspberry-red,raspberry-yellow' training.optimizers.embeddings.kwargs.lr=0.02 shared_tokens=1 gt_init=0 fruit_blip_coeff=0.0001 color_blip_coeff=0.001 blip_guidance=0 num_placeholder_words=22
```

three-axes-training - learns three axes: clothes, season, and color.
```
python scripts/train_clip_inversion.py -c configs/train_deepfloyd_inversion.yaml -t clothes-three-train -d 'shirt-spring-red,shirt-spring-yellow,shirt-spring-green,shirt-spring-purple,shirt-spring-white,shirt-spring-cream,shirt-summer-red,shirt-summer-yellow,shirt-summer-green,shirt-summer-purple,shirt-summer-white,shirt-summer-cream,shirt-fall-red,shirt-fall-yellow,shirt-fall-green,shirt-fall-purple,shirt-fall-white,shirt-fall-cream,shirt-winter-red,shirt-winter-yellow,shirt-winter-green,shirt-winter-purple,shirt-winter-white,shirt-winter-cream,pants-spring-red,pants-spring-yellow,pants-spring-green,pants-spring-purple,pants-spring-white,pants-spring-cream,pants-summer-red,pants-summer-yellow,pants-summer-green,pants-summer-purple,pants-summer-white,pants-summer-cream,pants-fall-red,pants-fall-yellow,pants-fall-green,pants-fall-purple,pants-fall-white,pants-fall-cream,pants-winter-red,pants-winter-yellow,pants-winter-green,pants-winter-purple,pants-winter-white,pants-winter-cream,shoes-spring-red,shoes-spring-yellow,shoes-spring-green,shoes-spring-purple,shoes-spring-white,shoes-spring-cream,shoes-summer-red,shoes-summer-yellow,shoes-summer-green,shoes-summer-purple,shoes-summer-white,shoes-summer-cream,shoes-fall-red,shoes-fall-yellow,shoes-fall-green,shoes-fall-purple,shoes-fall-white,shoes-fall-cream,shoes-winter-red,shoes-winter-yellow,shoes-winter-green,shoes-winter-purple,shoes-winter-white,shoes-winter-cream,dress-spring-red,dress-spring-yellow,dress-spring-green,dress-spring-purple,dress-spring-white,dress-spring-cream,dress-summer-red,dress-summer-yellow,dress-summer-green,dress-summer-purple,dress-summer-white,dress-summer-cream,dress-fall-red,dress-fall-yellow,dress-fall-green,dress-fall-purple,dress-fall-white,dress-fall-cream,dress-winter-red,dress-winter-yellow,dress-winter-green,dress-winter-purple,dress-winter-white,dress-winter-cream,cap-spring-red,cap-spring-yellow,cap-spring-green,cap-spring-purple,cap-spring-white,cap-spring-cream,cap-summer-red,cap-summer-yellow,cap-summer-green,cap-summer-purple,cap-summer-white,cap-summer-cream,cap-fall-red,cap-fall-yellow,cap-fall-green,cap-fall-purple,cap-fall-white,cap-fall-cream,cap-winter-red,cap-winter-yellow,cap-winter-green,cap-winter-purple,cap-winter-white,cap-winter-cream' training.optimizers.embeddings.kwargs.lr=0.02 shared_tokens=1 gt_init=0 fruit_blip_coeff=0.0001 mat_blip_coeff=0.001 color_blip_coeff=0.001 blip_guidance=0 num_placeholder_groups=3 num_placeholder_words=120
```

#### Visual concept extraction:
two-axes-extraction - using a model checkpoint which has learned two axes (art style and subject) + an input image, extracts the visual concepts from the input image along the two axes. Checkpoint can be found [here](https://drive.google.com/drive/folders/1JK0D9Z6KcFTGzxPztxGWHaqJ9OMFUSpp?usp=drive_link) and saved as checkpoints/art/art.pt.
```
python scripts/train_clip_inversion.py -c configs/train_deepfloyd_inversion.yaml -t art-two-extract -d 'cat-charcoal' training.optimizers.embeddings.kwargs.lr=0.001 shared_tokens=1 gt_init=0 real_image='real_imgs/cat_charcoal.jpg'
```

three-axes-extraction - using a model checkpoint which has learned three axes (furniture type, material, and color) + an input image, extracts the visual concepts from the input image along the three axes. Checkpoint can be found [here](https://drive.google.com/drive/folders/1JK0D9Z6KcFTGzxPztxGWHaqJ9OMFUSpp?usp=drive_link) and placed in checkpoint/furniture/furniture.pt.
```
python scripts/train_clip_inversion.py -c configs/train_deepfloyd_inversion.yaml -t furniture-three-extract -d 'chair-plastic-gray' training.optimizers.embeddings.kwargs.lr=0.001 shared_tokens=1 gt_init=0 num_placeholder_groups=3 num_placeholder_words=72 real_image='real_imgs/chair-plastic-gray.jpg'
```

### Notes
New logs will be created in logs/{exp_name} as html files, and new checkpoints will be saved in logs/{exp_name}/checkpoints.

Generation of images, prompts, and CLIP features associated with each image are in langint\datasets\glide.py.

Our model architecture to train Deepfloyd embeddings are in langint\models\textual_inversion_embeddings_deepfloyd.py.

Diffusion loss is calculated in langint\loss\invert_deepfloyd.py.

Results are formatted in langint\trainers\invert_clip_diffuser_style.py.