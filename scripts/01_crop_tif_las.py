"""Stage 1 — crop aligned TIF + LAS into tiles and generate per-tile CHM rasters.

Reads paths and crop parameters from a YAML config (default: configs/default.yaml).
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import laspy
import numpy as np
import rasterio
from PIL import Image
from pyproj import CRS, Transformer
from scipy.interpolate import griddata
from scipy.stats import binned_statistic_2d
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rgb2chm.config import add_config_arg, load_config


def get_las_crs(input):
    infile = laspy.read(input) if isinstance(input, str) else input
    crs = infile.header.parse_crs()
    try:
        return crs.sub_crs_list[0].to_epsg()
    except Exception:
        return crs


def crop_las(las_path, top_left, bottom_right, output_path):
    x_min, y_max = top_left
    x_max, y_min = bottom_right
    las = laspy.read(las_path)
    mask = (las.x >= x_min) & (las.x <= x_max) & (las.y >= y_min) & (las.y <= y_max)
    if np.sum(mask) == 0:
        print('no points in the bounding box')
        return
    cropped = laspy.LasData(las.header)
    cropped.points = las.points[mask]
    cropped.update_header()
    cropped.write(output_path)


def canopy_height_model(file_path, top_left, bottom_right, grid_num,
                        output_CHM_file=None, smooth=True, crs=None):
    infile = laspy.read(file_path)
    x, y, z = infile.x, infile.y, infile.z
    classification = infile.classification

    ground_mask = (classification == 2)
    ground_x, ground_y, ground_z = x[ground_mask], y[ground_mask], z[ground_mask]
    x_min, y_max = top_left
    x_max, y_min = bottom_right
    grid_res_x = (x_max - x_min) / (grid_num - 1)
    grid_res_y = (y_max - y_min) / (grid_num - 1)
    x_grid = np.arange(x_min, x_max + 1e-4, grid_res_x)
    y_grid = np.arange(y_min, y_max + 1e-4, grid_res_y)
    x_grid[-1] = x_max
    y_grid[-1] = y_max
    X, Y = np.meshgrid(x_grid, y_grid)
    assert len(x_grid) == len(y_grid) == grid_num, \
        f'grid number {grid_num} does not match {len(x_grid)}, {len(y_grid)}'

    if len(ground_x) > 0:
        print('found ground points')
        DTM = griddata((ground_x, ground_y), ground_z, (X, Y), method='nearest')
    else:
        DTM = np.zeros_like(X)

    stat, _, _, _ = binned_statistic_2d(x, y, z, statistic='max', bins=[grid_num, grid_num])
    DSM = stat.T
    print('max-min DSM:', np.nanmax(DSM) - np.nanmin(DSM))
    DSM = np.nan_to_num(DSM, nan=np.nanmin(DSM))
    CHM = DSM - DTM

    if smooth:
        CHM_smooth = cv2.GaussianBlur(CHM, (3, 3), 0)
        CHM = np.maximum(CHM, CHM_smooth)
    print('max-min CHM:', np.nanmax(CHM) - np.nanmin(CHM))

    if output_CHM_file is not None:
        transform = rasterio.transform.from_origin(x.min(), y.max(), grid_res_x, grid_res_y)
        with rasterio.open(output_CHM_file, 'w', driver='GTiff',
                           height=CHM.shape[0], width=CHM.shape[1],
                           count=1, dtype=CHM.dtype, crs=crs, transform=transform) as dst:
            dst.write(CHM, 1)
    return CHM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arg(parser)
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg['paths']
    crop_cfg = cfg['crop']

    input_tif = paths['input_tif']
    input_las = paths['input_las']
    proj_file = paths['proj_file']
    crop_tif_path = paths['image_dir']
    crop_las_path = paths['lidar_dir']
    chm_path = paths['chm_dir']

    crop_size = crop_cfg['crop_size']
    crop_step_size = crop_cfg['crop_step_size']
    grid_num = crop_cfg['grid_num']
    smooth = crop_cfg['smooth']
    save_image = crop_cfg['save_image']
    save_chm = crop_cfg['save_chm']

    for p in (crop_tif_path, crop_las_path, chm_path):
        os.makedirs(p, exist_ok=True)

    with rasterio.open(input_tif) as src:
        tif_crs = CRS.from_user_input(src.crs)
        print('Axis units:', tif_crs.axis_info[0].unit_name, '/', tif_crs.axis_info[1].unit_name)

    if proj_file and os.path.exists(proj_file):
        with open(proj_file, 'r') as f:
            las_crs = CRS.from_wkt(f.read())
    else:
        las_crs = get_las_crs(input_las)
        print('las_crs:', las_crs)

    try:
        transformer_tif2las = Transformer.from_crs(tif_crs, las_crs, always_xy=True)
    except Exception:
        raise ValueError(f'TIF CRS {tif_crs} and LAS CRS {las_crs} are not compatible')

    las = laspy.read(input_las)
    las_top_left = (las.x.min(), las.y.max())
    las_bottom_right = (las.x.max(), las.y.min())

    with rasterio.open(input_tif) as src:
        h, w = src.height, src.width
        print('tif_h:', h, 'tif_w:', w)

    progress = tqdm(total=(w + 1) * (h + 1) // (crop_step_size ** 2), desc='Cropping', unit='crop')
    with rasterio.open(input_tif) as src:
        for c in range(0, w + 1, crop_step_size):
            for r in range(0, h + 1, crop_step_size):
                output_image_file = f'crop_{c}_{r}.png'
                window = ((r, r + crop_size), (c, c + crop_size))
                pix2coor = src.window_transform(window)
                tif_tl = transformer_tif2las.transform(*(pix2coor * (0, 0)))
                tif_br = transformer_tif2las.transform(*(pix2coor * (crop_size, crop_size)))

                if not (tif_tl[0] >= las_top_left[0] and tif_br[0] <= las_bottom_right[0]
                        and tif_tl[1] <= las_top_left[1] and tif_br[1] >= las_bottom_right[1]):
                    progress.update(1)
                    continue

                data = src.read(window=window)
                if data.shape[0] == 4:
                    rgb = data[:3]
                elif data.shape[0] == 3:
                    rgb = data
                else:
                    raise ValueError('the tif file should be rgb or rgb infrared')

                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
                rgb = np.moveaxis(rgb, 0, -1)
                # Skip crops that are mostly white (edges of orthomosaic)
                if np.sum(rgb == 255) > 3000:
                    progress.update(1)
                    continue

                if save_image:
                    Image.fromarray(rgb).save(os.path.join(crop_tif_path, output_image_file))

                las_output_file = os.path.join(crop_las_path, output_image_file[:-4] + '.las')
                crop_las(input_las, tif_tl, tif_br, las_output_file)

                if save_chm:
                    chm_output_file = os.path.join(chm_path, output_image_file[:-4] + '.tif')
                    canopy_height_model(las_output_file, tif_tl, tif_br, grid_num,
                                        output_CHM_file=chm_output_file, smooth=smooth, crs=las_crs)

                progress.update(1)
    progress.close()


if __name__ == '__main__':
    main()
