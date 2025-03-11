import os
import json
import torch

from utils.utils import *
from utils.get_model_info import get_model_stat
from model.model_zoo import get_model


def process_img(images_dir, model_name, model, device, output_dir='./vis_result'):
    json_path = None
    model_info_path = None
    infer_time = 0.0

    basedir = os.path.join(output_dir, model_name)
    if not os.path.exists(basedir):
        os.makedirs(basedir)
        print(f"Directory {basedir} created.")

    for filename in os.listdir(images_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(images_dir, filename)

            nameImg = os.path.splitext(os.path.basename(image_path))[0]
            save_name = f"{nameImg}_{model_name}.png"
            save_path = os.path.join(basedir, save_name)
            json_path = os.path.join(basedir, f"{model_name}_stats.json")
            model_info_path = os.path.join(basedir, f"{model_name}_info.txt")

            input_tensor = load_image(image_path)
            input_tensor = input_tensor.to(device)

            with torch.no_grad():
                feature_map, inference_time = get_feature_map(
                    model=model,
                    x=input_tensor,
                )

            feature_map = compress_to_single_channel(feature_map)
            visualize_feature_map(feature_map,
                                  save_path=save_path,
                                  infer_time=inference_time,
                                  )
            infer_time += inference_time

    return json_path, model_info_path, infer_time


def computing_info(images_dir,
                   output_dir='./vis_result',
                   model=None, device='cuda', model_name=None, shape=(1, 3, 224, 224)):
    set_seed(142)
    if device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    print(f'device is {device}')

    model_ = None
    if model_name is not None:
        model_ = get_model(model_name)

    if model_ is not None:
        model = model_

    model.to(device)

    stats = {'model': model_name}

    json_path, model_info_path, infer_time = process_img(
        images_dir=images_dir,
        model_name=model_name,
        model=model,
        device=device,
        output_dir=output_dir,
    )

    input_tensor = torch.randn(shape).to(device)
    flops, params = get_model_stat(model, input_tensor,
                                   output_path=model_info_path,
                                   device=str(device)
                                   )
    stats["Parameters(M)"] = params / 1e6
    stats["FLOPs(G)"] = flops / 1e9

    average_time = infer_time / len(os.listdir(images_dir))
    stats["average_time(ms)"] = average_time
    if device == 'cpu':
        stats["device"] = str(device)
    else:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 2  # covert to MiB
        device_info = f"CUDA: ({gpu_name}, {int(gpu_memory)}MiB)"
        stats["device"] = device_info

    with open(json_path, 'w') as json_file:
        json.dump(stats, json_file, indent=4)
    for k, v in stats.items():
        print(f'{k}: {v}')



