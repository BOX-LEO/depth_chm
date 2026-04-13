"""Stage 2 — inference with trained DepthAnything model(s) over a directory of images.

For each variant listed in `inference.variants`, loads the model at
`${paths.model_dir}_<variant>` and writes predictions (float32 .npy, meters)
to `${paths.predictions_dir}_<variant>`.
"""

import argparse
import os
import sys
from pathlib import Path
from os.path import isdir

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, DepthAnythingForDepthEstimation

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rgb2chm.config import add_config_arg, load_config


def run_inference(model_path, image_dir, output_dir, max_depth=40.0, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    os.makedirs(output_dir, exist_ok=True)

    print(f'Loading model from {model_path}...')
    is_local = isdir(model_path)
    processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=is_local)
    model = DepthAnythingForDepthEstimation.from_pretrained(model_path, local_files_only=is_local)
    model = model.to(device).eval()

    image_files = sorted(f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg')))
    print(f'Found {len(image_files)} images')

    with torch.no_grad():
        for image_file in tqdm(image_files, desc='Processing images'):
            image = Image.open(os.path.join(image_dir, image_file)).convert('RGB')
            original_size = image.size  # (w, h)
            inputs = processor(images=image, return_tensors='pt')
            pixel_values = inputs['pixel_values'].to(device)

            outputs = model(pixel_values)
            pred_scaled = outputs.predicted_depth * max_depth
            pred_resized = F.interpolate(
                pred_scaled.unsqueeze(0),
                size=(original_size[1], original_size[0]),
                mode='bilinear', align_corners=True,
            ).squeeze().cpu().numpy()

            output_name = os.path.splitext(image_file)[0] + '.npy'
            np.save(os.path.join(output_dir, output_name), pred_resized.astype(np.float32))

    print(f'Inference complete. Results saved to: {output_dir}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arg(parser)
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg['paths']
    inf = cfg['inference']

    model_base = paths['model_dir']
    pred_base = paths['predictions_dir']
    image_dir = paths['image_dir']
    max_depth = inf['max_depth']

    for variant in inf['variants']:
        model_path = f'{model_base}_{variant}'
        output_dir = os.path.join(pred_base, variant)
        if not os.path.isdir(model_path):
            print(f'[skip] model not found: {model_path}')
            continue
        run_inference(model_path, image_dir, output_dir, max_depth=max_depth)


if __name__ == '__main__':
    main()
