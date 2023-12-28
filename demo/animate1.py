import argparse
import datetime
import inspect
import os
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from collections import OrderedDict

import torch

from diffusers import AutoencoderKL, DDIMScheduler, UniPCMultistepScheduler

from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from magicanimate.models.unet_controlnet import UNet3DConditionModel
from magicanimate.models.controlnet import ControlNetModel
from magicanimate.models.appearance_encoder import AppearanceEncoderModel
from magicanimate.models.mutual_self_attention import ReferenceAttentionControl
from magicanimate.pipelines.pipeline_animation import AnimationPipeline
from magicanimate.utils.util import save_videos_grid
from accelerate.utils import set_seed

from magicanimate.utils.videoreader import VideoReader

from einops import rearrange, repeat

import csv, pdb, glob
from safetensors import safe_open
import math
from pathlib import Path

import imageio
import gradio as gr

class MagicAnimate():
    def __init__(self, config="configs/prompts/animation.yaml") -> None:
        print("Initializing MagicAnimate Pipeline...")
        *_, func_args = inspect.getargvalues(inspect.currentframe())
        func_args = dict(func_args)
        
        config  = OmegaConf.load(config)
        
        inference_config = OmegaConf.load(config.inference_config)
            
        motion_module = config.motion_module
       
        ### >>> create animation pipeline >>> ###
        tokenizer = CLIPTokenizer.from_pretrained(config.pretrained_model_path, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(config.pretrained_model_path, subfolder="text_encoder")
        if config.pretrained_unet_path:
            unet = UNet3DConditionModel.from_pretrained_2d(config.pretrained_unet_path, unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs))
        else:
            unet = UNet3DConditionModel.from_pretrained_2d(config.pretrained_model_path, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs))
        self.appearance_encoder = AppearanceEncoderModel.from_pretrained(config.pretrained_appearance_encoder_path, subfolder="appearance_encoder").cuda()
        self.reference_control_writer = ReferenceAttentionControl(self.appearance_encoder, do_classifier_free_guidance=True, mode='write', fusion_blocks=config.fusion_blocks)
        self.reference_control_reader = ReferenceAttentionControl(unet, do_classifier_free_guidance=True, mode='read', fusion_blocks=config.fusion_blocks)
        if config.pretrained_vae_path is not None:
            vae = AutoencoderKL.from_pretrained(config.pretrained_vae_path)
        else:
            vae = AutoencoderKL.from_pretrained(config.pretrained_model_path, subfolder="vae")

        ### Load controlnet
        controlnet   = ControlNetModel.from_pretrained(config.pretrained_controlnet_path)

        vae.to(torch.float16)
        unet.to(torch.float16)
        text_encoder.to(torch.float16)
        controlnet.to(torch.float16)
        self.appearance_encoder.to(torch.float16)
        
        unet.enable_xformers_memory_efficient_attention()
        self.appearance_encoder.enable_xformers_memory_efficient_attention()
        controlnet.enable_xformers_memory_efficient_attention()

        self.pipeline = AnimationPipeline(
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, controlnet=controlnet,
            scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)),
            # NOTE: UniPCMultistepScheduler
        ).to("cuda")

        # 1. unet ckpt
        # 1.1 motion module
        motion_module_state_dict = torch.load(motion_module, map_location="cpu")
        if "global_step" in motion_module_state_dict: func_args.update({"global_step": motion_module_state_dict["global_step"]})
        motion_module_state_dict = motion_module_state_dict['state_dict'] if 'state_dict' in motion_module_state_dict else motion_module_state_dict
        try:
            # extra steps for self-trained models
            state_dict = OrderedDict()
            for key in motion_module_state_dict.keys():
                if key.startswith("module."):
                    _key = key.split("module.")[-1]
                    state_dict[_key] = motion_module_state_dict[key]
                else:
                    state_dict[key] = motion_module_state_dict[key]
            motion_module_state_dict = state_dict
            del state_dict
            missing, unexpected = self.pipeline.unet.load_state_dict(motion_module_state_dict, strict=False)
            assert len(unexpected) == 0
        except:
            _tmp_ = OrderedDict()
            for key in motion_module_state_dict.keys():
                if "motion_modules" in key:
                    if key.startswith("unet."):
                        _key = key.split('unet.')[-1]
                        _tmp_[_key] = motion_module_state_dict[key]
                    else:
                        _tmp_[key] = motion_module_state_dict[key]
            missing, unexpected = unet.load_state_dict(_tmp_, strict=False)
            assert len(unexpected) == 0
            del _tmp_
        del motion_module_state_dict

        self.pipeline.to("cuda")
        self.L = config.L
        
        print("Initialization Done!")
        
    def __call__(self, source_image, motion_sequence, random_seed, step, guidance_scale, size=512):
            prompt = n_prompt = ""
            random_seed = int(random_seed)
            step = int(step)
            guidance_scale = float(guidance_scale)
            samples_per_video = []
            # manually set random seed for reproduction
            if random_seed != -1: 
                torch.manual_seed(random_seed)
                set_seed(random_seed)
            else:
                torch.seed()

            if motion_sequence.endswith('.mp4'):
                control = VideoReader(motion_sequence).read()
                if control[0].shape[0] != size:
                    control = [np.array(Image.fromarray(c).resize((size, size))) for c in control]
                control = np.array(control)
            
            if source_image.shape[0] != size:
                source_image = np.array(Image.fromarray(source_image).resize((size, size)))
            H, W, C = source_image.shape
            
            init_latents = None
            original_length = control.shape[0]
            if control.shape[0] % self.L > 0:
                control = np.pad(control, ((0, self.L-control.shape[0] % self.L), (0, 0), (0, 0), (0, 0)), mode='edge')
            generator = torch.Generator(device=torch.device("cuda:0"))
            generator.manual_seed(torch.initial_seed())
            sample = self.pipeline(
                prompt,
                negative_prompt         = n_prompt,
                num_inference_steps     = step,
                guidance_scale          = guidance_scale,
                width                   = W,
                height                  = H,
                video_length            = len(control),
                controlnet_condition    = control,
                init_latents            = init_latents,
                generator               = generator,
                appearance_encoder       = self.appearance_encoder, 
                reference_control_writer = self.reference_control_writer,
                reference_control_reader = self.reference_control_reader,
                source_image             = source_image,
            ).videos

            source_images = np.array([source_image] * original_length)
            source_images = rearrange(torch.from_numpy(source_images), "t h w c -> 1 c t h w") / 255.0
            samples_per_video.append(source_images)
            
            control = control / 255.0
            control = rearrange(control, "t h w c -> 1 c t h w")
            control = torch.from_numpy(control)
            samples_per_video.append(control[:, :, :original_length])

            samples_per_video.append(sample[:, :, :original_length])

            samples_per_video = torch.cat(samples_per_video)

            time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            savedir = f"demo/outputs"
            animation_path = f"{savedir}/{time_str}.mp4"

            os.makedirs(savedir, exist_ok=True)
            save_videos_grid(samples_per_video, animation_path)
            
            return animation_path
            
animator = MagicAnimate()

def animate(reference_image, motion_sequence_state, seed, steps, guidance_scale):
    return animator(reference_image, motion_sequence_state, seed, steps, guidance_scale)

with gr.Blocks(title='AICoverGenWebUI') as app:
  #gr.Label('AICoverGen WebUI created with ❤️', show_label=False)
  animation = gr.Video(format="mp4", label="Animation Results", autoplay=True)
  with gr.Row():
    reference_image  = gr.Image(label="Reference Image")
    motion_sequence  = gr.Video(format="mp4", label="Motion Sequence")
    with gr.Column():
            random_seed         = gr.Textbox(label="Random seed", value=1, info="default: -1")
            sampling_steps      = gr.Textbox(label="Sampling steps", value=25, info="default: 25")
            guidance_scale      = gr.Textbox(label="Guidance scale", value=7.5, info="default: 7.5")
            submit              = gr.Button("Animate")

    def read_video(video):
        reader = imageio.get_reader(video)
        fps = reader.get_meta_data()['fps']
        return video
    
    def read_image(image, size=512):
        return np.array(Image.fromarray(image).resize((size, size)))
    
    # when user uploads a new video
    motion_sequence.upload(
        read_video,
        motion_sequence,
        motion_sequence
    )
    # when `first_frame` is updated
    reference_image.upload(
        read_image,
        reference_image,
        reference_image
    )
    # when the `submit` button is clicked
    submit.click(
        animate,
        [reference_image, motion_sequence, random_seed, sampling_steps, guidance_scale], 
        animation
    )

    # Examples
    gr.Markdown("## Examples")
    gr.Examples(
        examples=[
            ["inputs/applications/source_image/monalisa.png", "inputs/applications/driving/densepose/running.mp4"], 
            ["inputs/applications/source_image/demo4.png", "inputs/applications/driving/densepose/demo4.mp4"],
            ["inputs/applications/source_image/dalle2.jpeg", "inputs/applications/driving/densepose/running2.mp4"],
            ["inputs/applications/source_image/dalle8.jpeg", "inputs/applications/driving/densepose/dancing2.mp4"],
            ["inputs/applications/source_image/multi1_source.png", "inputs/applications/driving/densepose/multi_dancing.mp4"],
        ],
        inputs=[reference_image, motion_sequence],
        outputs=animation,
    )
#demo.launch(share=True)
    app.launch(
        share=True,
        enable_queue=True,
        #server_name=None if not args.listen else (args.listen_host or '0.0.0.0'),
        #server_port=args.listen_port,
    )
