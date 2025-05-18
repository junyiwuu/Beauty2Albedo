# This file is modified from the original implementation of "PBR_Boost_3DGen"
# https://github.com/snowflakewang/PBR_Boost_3DGen
# Original license: Apache License 2.0
# Modifications by Junyi Wu, 2025




import torch
from PIL import Image

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import UNet2DConditionModel, AutoencoderKL, DDIMScheduler

from albedo_pipeline import MaterialPipeline  # your pipeline definition


import json

import argparse
import logging
import math
import os
import random
import shutil
from pathlib import Path

import accelerate
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import diffusers
from albedo_pipeline import MaterialPipeline
import pdb


def single_inference(src_path, dst_path):

    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    current_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = os.path.join(current_dir, "../output/train/checkpoint/latest/unet/diffusion_pytorch_model.bin")


    # Load the U-Net I trained

    with open("../output/train/checkpoint/latest/unet/config.json") as f:
        unet_cfg = json.load(f)

    unet = UNet2DConditionModel(**unet_cfg)
    state_dict = torch.load(ckpt_path, map_location=device)
    unet.load_state_dict(state_dict)
    
    # Load others
    vae_albedo   = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="vae")
    vae_beauty   = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="vae")
    text_encoder = CLIPTextModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    tokenizer    = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    scheduler    = DDIMScheduler.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="scheduler")
    # vae_beauty.to(device)
    # vae_albedo.to(device)
    # unet.to(device)
    # text_encoder.to(device)
    # scheduler.to(device)


    pipeline = MaterialPipeline(
            unet=unet,
            vae_albedo=vae_albedo,
            vae_beauty=vae_beauty,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
).to(device)

    image = Image.open(src_path)

    res = pipeline(input_image=image, denoising_steps=30, ensemble_size=1)
    albedo_color = res.albedo_pil
    albedo_color.save(dst_path)

if __name__ == '__main__':
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    # ckpts_path = os.path.join(current_dir, "../output/train/checkpoint/latest")
  
    
    single_inference(
        src_path='../../infer_images/rgb_images/test.png',
        dst_path='../../infer_images/infer_image.png',

    )