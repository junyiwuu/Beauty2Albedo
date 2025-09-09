# This file is modified from the original implementation of "PBR_Boost_3DGen"
# https://github.com/snowflakewang/PBR_Boost_3DGen
# Original license: Apache License 2.0
# Modifications by Junyi Wu, 2025




import torch
from PIL import Image
import argparse

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler

from albedo_pipeline import MaterialPipeline


import json
import os
from pathlib import Path

import accelerate
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import diffusers
from albedo_pipeline import MaterialPipeline
import pdb
from safetensors.torch import save_file


# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def single_inference(src_path, dst_path, input_weights=None):

    if(input_weights):
        # if provide input checkpoint( download from online )
        pipeline = MaterialPipeline.from_pretrained(
            pretrained_model_name_or_path=input_weights
        ).to(device)

    else: #load from my own trained weight

        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_weights = os.path.join(current_dir, "../output/train/checkpoint/latest/unet/diffusion_pytorch_model.bin")

        # Load the U-Net I trained
        with open("../output/train/checkpoint/latest/unet/config.json") as f:
            unet_cfg = json.load(f)

        unet = UNet2DConditionModel(**unet_cfg)
        state_dict = torch.load(model_weights, map_location=device)
        unet.load_state_dict(state_dict)

        # Load others
        vae_albedo   = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="vae")
        vae_beauty   = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="vae")
        text_encoder = CLIPTextModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        tokenizer    = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        scheduler    = DDPMScheduler.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="scheduler")


        pipeline = MaterialPipeline(
                unet=unet,
                vae_albedo=vae_albedo,
                vae_beauty=vae_beauty,
                scheduler=scheduler,
                text_encoder=text_encoder,
                tokenizer=tokenizer,).to(device)
        
        # save out trained weights
        # pipeline.save_pretrained("../../safetensors/", safe_serialization=True, max_shard_size="2GB")

        pipeline._encode_empty_text()




    # create generator
    generator = torch.Generator(device=device)
    generator.manual_seed(2023)

    image = Image.open(src_path)

    res = pipeline(input_image=image, denoising_steps=30, ensemble_size=1, generator=generator)

    albedo_color = res.albedo_pil
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    
    albedo_color.save(dst_path)

if __name__ == '__main__':
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # input_weights = os.path.join(current_dir, "../../safetensors")
  
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_path", type=str, required=True, help="The image you want to predict from")
    parser.add_argument("--dst_path", type=str, required=True, help="The path where to write predicted Albedo image")
    parser.add_argument("--weights" , type=str, required=False, default=None, help="The path of downloaded weights")

    args = parser.parse_args()

    if (args.weights):
        input_weights = args.weights
    else:
        input_weights = None
    
    single_inference(
        src_path=args.src_path,
        dst_path=args.dst_path,
        input_weights=input_weights
    )

