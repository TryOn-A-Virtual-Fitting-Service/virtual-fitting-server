from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).absolute().parents[0].absolute()
sys.path.insert(0, str(PROJECT_ROOT))
import os
import torch
import random
import time
import threading
import gc

from pipelines_ootd.pipeline_ootd import OotdPipeline
from pipelines_ootd.unet_garm_2d_condition import UNetGarm2DConditionModel
from pipelines_ootd.unet_vton_2d_condition import UNetVton2DConditionModel
from diffusers import UniPCMultistepScheduler
from diffusers import AutoencoderKL

from transformers import AutoProcessor, CLIPVisionModelWithProjection
from transformers import CLIPTextModel, CLIPTokenizer

VIT_PATH = "./checkpoints/clip-vit-large-patch14"
VAE_PATH = "./checkpoints/ootd"
UNET_PATH = "./checkpoints/ootd/ootd_dc/checkpoint-36000"
MODEL_PATH = "./checkpoints/ootd"

class OOTDiffusionDC:

    def __init__(self, gpu_id, accelerator):
        self.accelerator = accelerator
        self.gpu_id = self.accelerator.device
        # self.accelerator = Accelerator(mixed_precision='fp16')

        print("Current working directory:", os.getcwd())
        vae = AutoencoderKL.from_pretrained(
            VAE_PATH,
            subfolder="vae",
            torch_dtype=torch.float16,
        )

        unet_garm = UNetGarm2DConditionModel.from_pretrained(
            UNET_PATH,
            subfolder="unet_garm",
            torch_dtype=torch.float16,
            use_safetensors=True,
        )
        unet_vton = UNetVton2DConditionModel.from_pretrained(
            UNET_PATH,
            subfolder="unet_vton",
            torch_dtype=torch.float16,
            use_safetensors=True,
        )

        unet_vton.enable_xformers_memory_efficient_attention()
        unet_garm.enable_xformers_memory_efficient_attention()

        vae, unet_garm, unet_vton = self.accelerator.prepare(vae, unet_garm, unet_vton)

        self.pipe = OotdPipeline.from_pretrained(
            MODEL_PATH,
            unet_garm=unet_garm,
            unet_vton=unet_vton,
            vae=vae,
            torch_dtype=torch.float16,
            # variant="fp16",
            use_safetensors=True,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(self.accelerator.device)

        # self.pipe.enable_attention_slicing()

        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        
        self.auto_processor = AutoProcessor.from_pretrained(VIT_PATH)
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(VIT_PATH)

        self.tokenizer = CLIPTokenizer.from_pretrained(
            MODEL_PATH,
            subfolder="tokenizer",
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            MODEL_PATH,
            subfolder="text_encoder",
        )

        # self.pipe.unet_garm = torch.compile(self.pipe.unet_garm)
        # self.pipe.unet_vton = torch.compile(self.pipe.unet_vton)

    def tokenize_captions(self, captions, max_length):
        inputs = self.tokenizer(
            captions, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids


    def __call__(self,
                model_type='hd',
                category='upperbody',
                image_garm=None,
                image_vton=None,
                mask=None,
                image_ori=None,
                num_samples=1,
                num_steps=20,
                image_scale=1.0,
                seed=-1,
    ):
        if seed == -1:
            random.seed(time.time())
            seed = random.randint(0, 2147483647)
        print('Initial seed: ' + str(seed))
        generator = torch.manual_seed(seed)

        with torch.no_grad():
            prompt_image = self.auto_processor(images=image_garm, return_tensors="pt")
            prompt_image = self.image_encoder(prompt_image.data['pixel_values']).image_embeds
            prompt_image = prompt_image.unsqueeze(1)
            if model_type == 'hd':
                prompt_embeds = self.text_encoder(self.tokenize_captions([""], 2))[0]
                prompt_embeds[:, 1:] = prompt_image[:]
            elif model_type == 'dc':
                prompt_embeds = self.text_encoder(self.tokenize_captions([category], 3))[0]
                prompt_embeds = torch.cat([prompt_embeds, prompt_image], dim=1)
            else:
                raise ValueError("model_type must be \'hd\' or \'dc\'!")

            images = self.pipe(prompt_embeds=prompt_embeds,
                        image_garm=image_garm,
                        image_vton=image_vton, 
                        mask=mask,
                        image_ori=image_ori,
                        num_inference_steps=num_steps,
                        image_guidance_scale=image_scale,
                        num_images_per_prompt=num_samples,
                        generator=generator,
            ).images
        
        def clear_vram_cache():
            gc.collect()
            torch.cuda.empty_cache()

        clear_cache_thread = threading.Thread(target=clear_vram_cache, daemon=True)
        clear_cache_thread.start()

        return images