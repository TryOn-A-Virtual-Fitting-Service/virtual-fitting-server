import pdb
import config
from pathlib import Path
import sys

# Set the project root directory and add it to the system path
PROJECT_ROOT = Path(__file__).absolute().parents[0].absolute()
sys.path.insert(0, str(PROJECT_ROOT))
import os

import cv2
import einops
import numpy as np
import random
import time
import json

# from pytorch_lightning import seed_everything
from preprocess.openpose.annotator.util import resize_image, HWC3
from preprocess.openpose.annotator.openpose import OpenposeDetector

import argparse
from PIL import Image
import torch
from torch.cuda.amp import autocast  # Import autocast for AMP

# Optionally, set which CUDA devices to use
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

class OpenPose:
    def __init__(self, gpu_id: int):
        """
        Initialize the OpenPose model on the specified GPU.

        Args:
            gpu_id (int): The ID of the GPU to use.
        """
        self.gpu_id = gpu_id
        torch.cuda.set_device(gpu_id)  # Set the current device
        self.preprocessor = OpenposeDetector()  # Initialize the OpenPose detector
        
        # Optionally convert the model to half-precision if supported
        # Uncomment the following line if your model supports it
        # self.preprocessor.to(torch.float16)
    
    def __call__(self, input_image, resolution=384):
        """
        Perform pose estimation on the input image.

        Args:
            input_image (PIL.Image.Image or str): The input image or path to the image.
            resolution (int, optional): The resolution to resize the image. Defaults to 384.

        Returns:
            dict: A dictionary containing the 2D pose keypoints.
        """
        torch.cuda.set_device(self.gpu_id)  # Ensure the correct GPU is set

        # Handle different types of input images
        if isinstance(input_image, Image.Image):
            input_image = np.asarray(input_image)
        elif isinstance(input_image, str):
            input_image = np.asarray(Image.open(input_image))
        else:
            raise ValueError("Input must be a PIL Image or a file path string.")

        with torch.no_grad():  # Disable gradient computation for inference
            # Enable AMP for mixed-precision operations
            with autocast(enabled=True):
                input_image = HWC3(input_image)  # Ensure the image has 3 channels
                input_image = resize_image(input_image, resolution)  # Resize the image
                H, W, C = input_image.shape
                assert (H == 512 and W == 384), 'Incorrect input image shape'

                # Convert the input image to a tensor and move it to the specified GPU
                input_tensor = torch.from_numpy(input_image).permute(2, 0, 1).unsqueeze(0).to(self.gpu_id)

                # Perform pose detection
                pose, detected_map = self.preprocessor(input_tensor, hand_and_face=False)

                # Process the detected pose data
                candidate = pose['bodies']['candidate']
                subset = pose['bodies']['subset'][0][:18]
                for i in range(18):
                    if subset[i] == -1:
                        candidate.insert(i, [0, 0])
                        for j in range(i, 18):
                            if subset[j] != -1:
                                subset[j] += 1
                    elif subset[i] != i:
                        candidate.pop(i)
                        for j in range(i, 18):
                            if subset[j] != -1:
                                subset[j] -= 1

                candidate = candidate[:18]

                # Scale the keypoints to the original image size
                for i in range(18):
                    candidate[i][0] *= 384
                    candidate[i][1] *= 512

                keypoints = {"pose_keypoints_2d": candidate}

                # Optional: Save keypoints and output image
                # with open("/home/aigc/ProjectVTON/OpenPose/keypoints/keypoints.json", "w") as f:
                #     json.dump(keypoints, f)
                #
                # output_image = cv2.resize(cv2.cvtColor(detected_map, cv2.COLOR_BGR2RGB), (768, 1024))
                # cv2.imwrite('/home/aigc/ProjectVTON/OpenPose/keypoints/out_pose.jpg', output_image)

        return keypoints  # Return the detected keypoints


if __name__ == '__main__':
    """
    Main entry point for the script.
    Initializes the OpenPose model and performs pose estimation on a specified image.
    """
    # Parse command-line arguments if necessary
    # Uncomment and modify the following lines if you want to accept GPU ID as an argument
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use')
    # args = parser.parse_args()
    #
    # gpu_id = args.gpu_id

    gpu_id = 0  # Set the GPU ID manually or parse from arguments
    model = OpenPose(gpu_id)  # Initialize the OpenPose model on the specified GPU

    # Path to the input image
    input_image_path = './images/bad_model.jpg'

    # Measure inference time
    start_time = time.time()
    keypoints = model(input_image_path)  # Perform pose estimation
    end_time = time.time()

    print(f"Keypoints: {keypoints}")  # Print the detected keypoints
    print(f"Inference Time: {end_time - start_time} seconds")  # Print the inference time

    # Optional: Further processing or saving results can be done here
