from pathlib import Path
from PIL import Image
import sys
import os
import gc
from run.utils_ootd import get_mask_location
from preprocess.openpose.run_openpose import OpenPose
from preprocess.humanparsing.run_parsing import Parsing
from ootd.inference_ootd_hd import OOTDiffusionHD
from ootd.inference_ootd_dc import OOTDiffusionDC

import time

def run_ootd(model_path, cloth_path, accelerator, gpu_id=0, model_type="hd", category=0, scale=2.0, step=40, sample=1, seed=-1):
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"Loading model: openpose")
    openpose_model = OpenPose(gpu_id)
    print(f"Models loaded")

    print(f"Loading model: human parsing")
    parsing_model = Parsing(gpu_id)
    print(f"Models loaded")

    category_dict = ['upperbody', 'lowerbody', 'dress']
    category_dict_utils = ['upper_body', 'lower_body', 'dresses']

    print(f"Loading model: OOTD")
    if model_type == "hd":
        model = OOTDiffusionHD(gpu_id, accelerator)
    elif model_type == "dc":
        model = OOTDiffusionDC(gpu_id, accelerator)
    else:
        raise ValueError("model_type must be 'hd' or 'dc'!")

    if model_type == 'hd' and category != 0:
        raise ValueError("model_type 'hd' requires category == 0 (upperbody)!")
    print(f"Models loaded")

    print(f"Resizing images")
    cloth_img = Image.open(cloth_path).resize((768, 1024))
    model_img = Image.open(model_path).resize((768, 1024))
    print(f"Images resized")

    print(f"Running openpose and human parsing")
    keypoints = openpose_model(model_img.resize((384, 512)))
    model_parse, _ = parsing_model(model_img.resize((384, 512)))
    print(f"Openpose and human parsing finished")

    print(f"Getting mask location")
    mask, mask_gray = get_mask_location(model_type, category_dict_utils[category], model_parse, keypoints)
    mask = mask.resize((768, 1024), Image.NEAREST)
    mask_gray = mask_gray.resize((768, 1024), Image.NEAREST)
    print(f"Mask location obtained")

    print(f"Creating masked image")
    masked_vton_img = Image.composite(mask_gray, model_img, mask)
    # masked_vton_img.save('./results/mask.jpg')
    print(f"Masked image created")

    print(f"Running OOTD model")
    images = model(
        model_type=model_type,
        category=category_dict[category],
        image_garm=cloth_img,
        image_vton=masked_vton_img,
        mask=mask,
        image_ori=model_img,
        num_samples=sample,
        num_steps=step,
        image_scale=scale,
        seed=seed,
    )
    print(f"OOTD model finished")

    model.cleanup()

    image = None
    if type(images) == list:
        image = images[0]
    if len(images) == 1:
        image = images[0]

    torch.cuda.empty_cache()
    gc.collect()

    return image