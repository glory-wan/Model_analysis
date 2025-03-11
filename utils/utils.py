import os
import cv2
import json
import random
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import time
import matplotlib.pyplot as plt

__all__ = (
    'compress_to_single_channel',
    'load_image',
    'get_feature_map',
    'visualize_feature_map',
    'set_seed',
)


def set_seed(seed=142):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compress_to_single_channel(tensor):
    return tensor.mean(dim=1, keepdim=True)


def load_image(image_path, resize_shape=None):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Resize(resize_shape),
    ])
    image = Image.open(image_path)
    return preprocess(image).unsqueeze(0)


def get_feature_map(model, x):
    layers = list(model.children())[:-2]
    feature_extractor = torch.nn.Sequential(*layers)
    start_time = time.time()
    output_tensor = feature_extractor(x)
    inference_time = time.time() - start_time
    inference_time = inference_time * 1e3

    return output_tensor, inference_time


def visualize_feature_map(feature_map, save_path, infer_time):
    feature_map = feature_map.squeeze(0)
    plt.figure(figsize=(10, 10))
    plt.imshow(feature_map[0].cpu(), cmap='viridis')
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f'save {os.path.basename(save_path)}, inference_time = {infer_time}(ms)')
