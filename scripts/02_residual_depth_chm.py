"""Stage 1 — fuse vanilla depth predictions with CHM into high-resolution pseudo GT."""

import argparse
import os
import sys
from pathlib import Path
import cv2
import numpy as np
import rasterio
from PIL import Image
from skimage.exposure import match_histograms
from scipy.ndimage import gaussian_filter, gaussian_gradient_magnitude

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rgb2chm.config import add_config_arg, load_config


def read_tif_height(file_path):
    chm = rasterio.open(file_path)
    chm = chm.read(1)
    chm = chm.astype(np.float32)
    # flip vertically
    chm = np.flipud(chm)
    return chm


def regularized_depth(org_depth,target_depth):
    """
    Regularize the depth map using the fused CHM.
    :param org_depth: Original depth map
    :param target_depth: Target depth map
    :return: Regularized depth map
    """
    # Normalize the original depth map to match the scale of the fused CHM
    org_depth = org_depth / np.nanmax(org_depth) * np.nanmax(target_depth)
    
    # Calculate the residual depth
    residual_depth = target_depth - org_depth
    
    # smooth the residual depth with a Gaussian filter
    residual_depth = gaussian_filter(residual_depth, sigma=5)
    gradient_mag = gaussian_gradient_magnitude(residual_depth, sigma=5)
    alpha = 1.0  # control sharpness of transition
    weight = np.exp(-alpha * (gradient_mag / gradient_mag.max()))
    residual_depth = residual_depth * weight
    residual_depth  = gaussian_filter(residual_depth, sigma=31)
    
    # Blend the residual depth with the original depth map
    regularized_depth = org_depth + residual_depth

    # m2: clip the regularized depth to be between 0 and the max of the target depth
    regularized_depth = np.clip(regularized_depth, 0, np.nanmax(target_depth))

    
    return regularized_depth


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arg(parser)
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg['paths']

    fused_chm_path = paths['chm_dir']
    depth_pred_path = paths['vanilla_depth_dir']
    image_path = paths['image_dir']
    output_path = paths['pseudo_gt_dir']
    os.makedirs(output_path, exist_ok=True)

    fused_chm_files = sorted(os.path.join(fused_chm_path, f) for f in os.listdir(fused_chm_path)
                              if f.endswith(('.tif', '.npy')))
    depth_pred_files = sorted(os.path.join(depth_pred_path, f) for f in os.listdir(depth_pred_path)
                               if f.endswith('.npy'))
    image_files = sorted(os.path.join(image_path, f) for f in os.listdir(image_path)
                          if f.endswith(('.png', '.jpg')))

    for fused_chm_file, depth_pred_file, image_file in zip(fused_chm_files, depth_pred_files, image_files):
        image = Image.open(image_file)
        chm = np.load(fused_chm_file) if fused_chm_file.endswith('.npy') else read_tif_height(fused_chm_file)
        chm = chm- np.nanmin(chm)  # Normalize to start from 0
        depth_pred = np.load(depth_pred_file)
        # interpolate fused_chm to match the depth_pred shape
        fused_chm = cv2.resize(chm, (depth_pred.shape[1], depth_pred.shape[0]), interpolation=cv2.INTER_NEAREST)
        fused_chm = match_histograms(fused_chm, chm, channel_axis=None)

        depth_pred = np.nanmax(depth_pred) - depth_pred  # Convert metric depth to height map


        regularized_pred = regularized_depth(depth_pred, fused_chm)
        regularized_pred_m = match_histograms(regularized_pred, fused_chm, channel_axis=None)
        regularized_gt = regularized_depth(regularized_pred, regularized_pred_m)

        out_name = os.path.basename(fused_chm_file).replace('.tif', '.npy').replace('.png', '.npy')
        np.save(os.path.join(output_path, out_name), regularized_gt)


if __name__ == '__main__':
    main()