"""
COMPREHENSIVE GROUND ANALYSIS SCRIPT

This script performs comprehensive analysis of LiDAR-derived data focusing on:
1. Sample distribution analysis across different %ground thresholds
2. Ground threshold analysis with CHM downsampling using different strategies for ground vs non-ground regions

The script analyzes how model predictions (metric depth) perform against ground truth (CHM) 
under different conditions related to ground coverage (%ground).

Key Features:
- Analyzes scale=1 samples only (individual tiles)
- Tests thresholds from 10% to 0.1% ground coverage
- Implements smart downsampling: max pooling for vegetation, mean pooling for ground
- Provides comprehensive statistics and insights
- Saves results to multiple CSV files

"""

import os
import argparse
import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn.functional as F
from sklearn.metrics import r2_score
from tqdm import tqdm
import re
import matplotlib.pyplot as plt
import seaborn as sns

def read_tif_height(file_path):
    """
    Read TIF file and return height data
    
    Args:
        file_path (str): Path to the TIF file
        
    Returns:
        np.ndarray: Height data as float32 array, vertically flipped
    """
    chm = rasterio.open(file_path)
    chm = chm.read(1)
    chm = chm.astype(np.float32)
    # flip vertically to match coordinate system
    chm = np.flipud(chm)
    return chm


def smart_downsample(height_map, filter_size):
    """
    Smart downsampling of height_map using different strategies for ground vs non-ground regions
    
    Ground regions (height_map < 5m): Use mean pooling to preserve average ground height
    Non-ground regions (height_map >= 5m): Use max pooling to preserve canopy structure
    
    Args:
        height_map (np.ndarray): height_map array to downsample
        filter_size (int): Size of the pooling filter
        
    Returns:
        np.ndarray: Downsampled height_map array
    """
    if filter_size == 1:
        return height_map  # No downsampling
    
    h, w = height_map.shape
    
    # Calculate output dimensions
    out_h = h // filter_size
    out_w = w // filter_size
    
    # Create output array
    downsampled = np.zeros((out_h, out_w), dtype=np.float32)
    
    for i in range(out_h):
        for j in range(out_w):
            # Extract the patch
            start_h = i * filter_size
            end_h = start_h + filter_size
            start_w = j * filter_size
            end_w = start_w + filter_size
            
            patch = height_map[start_h:end_h, start_w:end_w]
            downsampled[i, j] = np.mean(patch)
    return downsampled


def extract_coordinates(filename):
    """
    Extract x, y coordinates from filename crop_<x>_<y>.ext

    Args:
        filename (str): Filename in format crop_<x>_<y>.ext

    Returns:
        tuple: (x, y) coordinates as integers, or (None, None) if parsing fails
    """
    match = re.match(r'crop_(\d+)_(\d+)\.', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def resize_to_match(pred, target):
    """
    Resize prediction to match target dimensions using bilinear interpolation

    Args:
        pred (np.ndarray): Prediction array to resize
        target (np.ndarray): Target array whose dimensions to match

    Returns:
        np.ndarray: Resized prediction array
    """
    pred_tensor = torch.from_numpy(pred).unsqueeze(0)
    target_h, target_w = target.shape[-2:]
    resized = F.interpolate(pred_tensor[:, None], size=(target_h, target_w),
                            mode='bilinear', align_corners=True).squeeze().numpy()
    return resized


def upsample_to_size(data, target_size):
    """
    Upsample data to target size using bilinear interpolation

    Args:
        data (np.ndarray): 2D array to upsample
        target_size (tuple): Target (height, width) dimensions

    Returns:
        np.ndarray: Upsampled array
    """
    if data.shape[0] == target_size[0] and data.shape[1] == target_size[1]:
        return data  # No upsampling needed

    data_tensor = torch.from_numpy(data.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    upsampled = F.interpolate(data_tensor, size=target_size,
                              mode='bilinear', align_corners=True).squeeze().numpy()
    return upsampled


def check_files_exist(base_path, coordinates, file_type):
    """
    Check if all files exist for given coordinates

    Args:
        base_path (str): Base directory path
        coordinates (list): List of (x, y) coordinate tuples
        file_type (str): File extension ('tif' or 'npy')

    Returns:
        bool: True if all files exist, False otherwise
    """
    for x, y in coordinates:
        filename = f"crop_{x}_{y}.{file_type}"
        filepath = os.path.join(base_path, filename)
        if not os.path.exists(filepath):
            return False
    return True


def load_data_for_coordinates(base_path, coordinates, file_type):
    """
    Load data for given coordinates

    Args:
        base_path (str): Base directory path
        coordinates (list): List of (x, y) coordinate tuples
        file_type (str): File extension ('tif' or 'npy')

    Returns:
        list: List of loaded data arrays
    """
    data = []
    for x, y in coordinates:
        filename = f"crop_{x}_{y}.{file_type}"
        filepath = os.path.join(base_path, filename)

        if file_type == 'tif':
            data.append(read_tif_height(filepath))
        elif file_type == 'npy':
            data.append(np.load(filepath))

    return data


def compute_metrics(pred, chm, pseudo_gt):
    """
    Compute all required metrics for a sample

    Args:
        pred (np.ndarray): Prediction array
        chm (np.ndarray): CHM ground truth array
        pseudo_gt (np.ndarray): Pseudo ground truth array

    Returns:
        dict: Dictionary containing all computed metrics
    """
    # Resize prediction to match CHM dimensions
    pred_resized_chm = resize_to_match(pred, chm)

    # Resize prediction to match pseudo GT dimensions (if needed)
    if pred.shape != pseudo_gt.shape:
        pred_resized_pseudo = resize_to_match(pred, pseudo_gt)
    else:
        pred_resized_pseudo = pred

    # Calculate R² scores
    r2_chm = r2_score(pred_resized_chm.flatten(), chm.flatten())
    r2_pseudo = r2_score(pred_resized_pseudo.flatten(), pseudo_gt.flatten())

    # Calculate ground percentage (CHM < 5m)
    ground_pixels = np.sum(chm < 5)
    total_pixels = chm.size
    ground_percentage = (ground_pixels / total_pixels) * 100

    # Calculate CHM statistics
    chm_min = np.min(chm)
    chm_max = np.max(chm)
    chm_std = np.std(chm)

    return {
        'r2_chm': r2_chm,
        'r2_pseudo': r2_pseudo,
        '%ground': ground_percentage,
        'chm_min': chm_min,
        'chm_max': chm_max,
        'chm_std': chm_std
    }


def analyze_predictions(root_path, pred_root_path, sub_path_list, analysis_dir=None):
    """
    Analyze predictions at scale 1 (individual tiles)

    This function performs the complete analysis pipeline:
    1. Discovers all available files and extracts coordinates
    2. Analyzes individual tiles (Scale 1)
    3. Computes metrics for each sample
    4. Saves results to CSV file
    5. Provides summary statistics

    Args:
        root_path (str): Root path containing chm and pseudo_gt directories
        pred_root_path (str): Root path containing prediction subdirectories
        sub_path_list (list): List of prediction subdirectory names to analyze
    """
    chm_path = os.path.join(root_path, 'chm')
    pseudo_gt_path = os.path.join(root_path, 'pseudo_gt')
    out_dir = analysis_dir if analysis_dir is not None else pred_root_path
    os.makedirs(out_dir, exist_ok=True)

    for s in sub_path_list:
        pred_path = os.path.join(pred_root_path, s)

        # Get all available files
        chm_files = [f for f in os.listdir(chm_path) if f.endswith('.tif')]
        pseudo_gt_files = [f for f in os.listdir(pseudo_gt_path) if f.endswith('.npy')]
        pred_files = [f for f in os.listdir(pred_path) if f.endswith('.npy')]

        print(f"Found {len(chm_files)} CHM files, {len(pseudo_gt_files)} pseudo GT files, {len(pred_files)} prediction files")

        # Extract coordinates from filenames
        coordinates_set = set()
        for filename in chm_files:
            x, y = extract_coordinates(filename)
            if x is not None and y is not None:
                coordinates_set.add((x, y))

        coordinates_list = sorted(list(coordinates_set))
        print(f"Found {len(coordinates_list)} unique coordinates")

        results = []

        # Analyze scale 1 (individual tiles)
        print("Analyzing scale 1 (individual tiles)...")
        for x, y in tqdm(coordinates_list):
            # Check if all required files exist
            if not check_files_exist(chm_path, [(x, y)], 'tif'):
                continue
            if not check_files_exist(pseudo_gt_path, [(x, y)], 'npy'):
                continue
            if not check_files_exist(pred_path, [(x, y)], 'npy'):
                continue

            # Load data
            chm_data = load_data_for_coordinates(chm_path, [(x, y)], 'tif')[0]
            pseudo_gt_data = load_data_for_coordinates(pseudo_gt_path, [(x, y)], 'npy')[0]
            pred_data = load_data_for_coordinates(pred_path, [(x, y)], 'npy')[0]

            # Convert metric depth to height map
            max_depth = 40
            pred_data = max_depth - pred_data
            chm_data = chm_data - chm_data.min()

            # Compute metrics
            metrics = compute_metrics(pred_data, chm_data, pseudo_gt_data)

            # Store results
            result = {
                'coordinate': f"{x}_{y}",
                'scale': 1,
                **metrics
            }
            results.append(result)

        # Create DataFrame and save to CSV
        df = pd.DataFrame(results)

        # Sort by coordinate
        df = df.sort_values(['coordinate'])

        # Save results
        output_file = os.path.join(out_dir, f'abalation_{s}.csv')
        df.to_csv(output_file, index=False)

        print(f"Analysis complete! Results saved to: {output_file}")
        print(f"Total samples analyzed: {len(results)}")
        print(f"Scale 1 samples: {len(df[df['scale'] == 1])}")

        # Print summary statistics
        print("\nSummary Statistics:")
        print("=" * 50)
        scale_data = df[df['scale'] == 1]
        if len(scale_data) > 0:
            print(f"\nScale 1 ({len(scale_data)} samples):")
            print(f"  R² CHM - Mean: {scale_data['r2_chm'].mean():.4f}, Std: {scale_data['r2_chm'].std():.4f}")
            print(f"  R² Pseudo - Mean: {scale_data['r2_pseudo'].mean():.4f}, Std: {scale_data['r2_pseudo'].std():.4f}")
            print(f"  % Ground - Mean: {scale_data['%ground'].mean():.2f}%, Std: {scale_data['%ground'].std():.2f}%")
            print(f"  CHM Min - Mean: {scale_data['chm_min'].mean():.2f}, Std: {scale_data['chm_min'].std():.2f}")
            print(f"  CHM Max - Mean: {scale_data['chm_max'].mean():.2f}, Std: {scale_data['chm_max'].std():.2f}")
            print(f"  CHM Std - Mean: {scale_data['chm_std'].mean():.2f}, Std: {scale_data['chm_std'].std():.2f}")

def analyze_sample_distribution(df_scale1):
    """
    Analyze the distribution of samples across different %ground thresholds
    
    This function provides detailed statistics about how samples are distributed
    across different ground coverage percentages, including:
    - Individual threshold ranges
    - Cumulative distributions
    - Mean R² scores and other metrics for each range
    
    Args:
        df_scale1 (pd.DataFrame): DataFrame containing scale=1 samples
        
    Returns:
        tuple: (distribution_df, cumulative_df) - Two DataFrames with analysis results
    """
    
    print("\n" + "=" * 80)
    print("SAMPLE DISTRIBUTION ANALYSIS")
    print("=" * 80)
    
    # Define thresholds from 10% to 0.1% ground coverage
    thresholds = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0.5, 0.2, 0.1]

    print(f"Found {len(df_scale1)} scale=1 samples")
    print(f"Mean %ground: {df_scale1['%ground'].mean():.2f}%")
    print(f"Std %ground: {df_scale1['%ground'].std():.2f}%")
    print(f"Min %ground: {df_scale1['%ground'].min():.2f}%")
    print(f"Max %ground: {df_scale1['%ground'].max():.2f}%")
    
    # Analyze sample distribution across threshold ranges
    distribution_data = []
    
    for i, threshold in enumerate(thresholds):
        if i == 0:
            # First threshold: samples > threshold
            samples_in_range = df_scale1[df_scale1['%ground'] > threshold]
            range_label = f"> {threshold}%"
        else:
            # Other thresholds: samples between previous and current threshold
            prev_threshold = thresholds[i-1]
            samples_in_range = df_scale1[(df_scale1['%ground'] > threshold) & (df_scale1['%ground'] <= prev_threshold)]
            range_label = f"{threshold}% - {prev_threshold}%"
        
        count = len(samples_in_range)
        percentage = (count / len(df_scale1)) * 100 if len(df_scale1) > 0 else 0
        
        distribution_data.append({
            'range': range_label,
            'count': count,
            'percentage': percentage,
            'mean_r2': samples_in_range['r2_chm'].mean() if count > 0 else np.nan,
            'mean_ground': samples_in_range['%ground'].mean() if count > 0 else np.nan,
            'mean_chm_std': samples_in_range['chm_std'].mean() if count > 0 else np.nan
        })
    
    # Create distribution DataFrame
    distribution_df = pd.DataFrame(distribution_data)
    
    print("\nSample Distribution by %ground Threshold:")
    print(distribution_df.to_string(index=False, float_format='%.2f'))
    
    # Cumulative analysis - samples above each threshold
    print("\n" + "-" * 60)
    print("CUMULATIVE ANALYSIS")
    print("-" * 60)
    
    cumulative_data = []
    for threshold in thresholds:
        samples_above = df_scale1[df_scale1['%ground'] > threshold]
        count = len(samples_above)
        percentage = (count / len(df_scale1)) * 100 if len(df_scale1) > 0 else 0
        
        cumulative_data.append({
            'threshold': threshold,
            'samples_above': count,
            'percentage_above': percentage,
            'mean_r2': samples_above['r2_chm'].mean() if count > 0 else np.nan,
            'mean_ground': samples_above['%ground'].mean() if count > 0 else np.nan,
            'mean_chm_std': samples_above['chm_std'].mean() if count > 0 else np.nan
        })
    
    cumulative_df = pd.DataFrame(cumulative_data)
    print("Cumulative Sample Distribution:")
    print(cumulative_df.to_string(index=False, float_format='%.2f'))
    
    return distribution_df, cumulative_df

def analyze_ground_thresholds_with_downsampling(df_scale1, chm_path, pred_path,
                                                pseudo_gt_path=None, eval_target='chm',
                                                plot_flag=False, thresholds=None, target_sizes=None):
    """
    Analyze R² scores for different %ground thresholds with prediction downsampling

    When eval_target='chm':
        - Predictions are downsampled from 1000x1000 to target_size
        - CHM (lower resolution) is upsampled to target_size via bilinear interpolation
    When eval_target='pseudo_gt':
        - Predictions are downsampled from 1000x1000 to target_size
        - Pseudo GT (1000x1000, same as pred) is also downsampled to target_size

    Args:
        df_scale1 (pd.DataFrame): DataFrame containing scale=1 samples
        chm_path (str): Path to CHM data directory
        pred_path (str): Path to prediction data directory
        pseudo_gt_path (str): Path to pseudo ground truth data directory (required when eval_target='pseudo_gt')
        eval_target (str): 'chm' or 'pseudo_gt' - which ground truth to evaluate against
        plot_flag (bool): Whether to generate scatter plots

    Returns:
        pd.DataFrame: Results DataFrame with R² scores for each threshold and target size
    """

    if eval_target == 'pseudo_gt' and pseudo_gt_path is None:
        raise ValueError("pseudo_gt_path must be provided when eval_target='pseudo_gt'")

    gt_label = 'CHM' if eval_target == 'chm' else 'Pseudo GT'
    print("\n" + "=" * 80)
    print(f"GROUND THRESHOLD ANALYSIS (PRED DOWNSAMPLE + {gt_label} {'UPSAMPLE' if eval_target == 'chm' else 'DOWNSAMPLE'})")
    print("=" * 80)
    
    if thresholds is None:
        thresholds = [1, 2, 3]
    if target_sizes is None:
        target_sizes = [(50, 50)]
    target_sizes = [tuple(ts) for ts in target_sizes]
    all_results = []
    
    r2_col = f'r2_{eval_target}_combined'

    for target_size in target_sizes:
        print(f"\n{'='*60}")
        print(f"ANALYSIS (target_size = {target_size})")
        print(f"  - Prediction: smart downsample from 1000x1000 to {target_size}")
        if eval_target == 'chm':
            print(f"  - CHM: bilinear upsample to {target_size}")
        else:
            print(f"  - Pseudo GT: smart downsample from 1000x1000 to {target_size}")
        print(f"{'='*60}")

        results = []

        for threshold in tqdm(thresholds, desc=f"Processing thresholds (target_size={target_size})"):
            # Select samples where %ground > threshold
            filtered_samples = df_scale1[df_scale1['%ground'] > threshold]

            if len(filtered_samples) == 0:
                results.append({
                    'target_size': target_size,
                    'threshold': threshold,
                    r2_col: np.nan,
                    'number_of_pixels': 0,
                    'number_of_samples': 0
                })
                continue

            print(f"\nThreshold {threshold}%: {len(filtered_samples)} samples")

            # Collect all prediction and GT arrays
            all_pred_pixels = []
            all_gt_pixels = []

            for _, row in filtered_samples.iterrows():
                x, y = extract_coordinates(f"crop_{row['coordinate']}.npy")
                if x is None or y is None:
                    continue

                # Load prediction file
                pred_file = os.path.join(pred_path, f"crop_{x}_{y}.npy")

                if eval_target == 'chm':
                    gt_file = os.path.join(chm_path, f"crop_{x}_{y}.tif")
                else:
                    gt_file = os.path.join(pseudo_gt_path, f"crop_{x}_{y}.npy")

                if not os.path.exists(pred_file) or not os.path.exists(gt_file):
                    continue

                try:
                    pred = np.load(pred_file)

                    # Convert metric depth to height map (same as in original analysis)
                    max_depth = 40
                    pred = np.array(pred, dtype=np.float32)
                    pred = max_depth - pred # convert to height map

                    # Downsample prediction using smart_downsample
                    # Prediction is at 1000x1000, downsample to target_size
                    pred_filter_size = pred.shape[0] // target_size[0]
                    if pred_filter_size > 1:
                        pred_resized = smart_downsample(pred, pred_filter_size)
                    else:
                        pred_resized = pred

                    if eval_target == 'chm':
                        # CHM: lower resolution TIF, upsample to target_size
                        gt = read_tif_height(gt_file)
                        gt = gt - gt.min()  # normalize to 0
                        gt_resized = upsample_to_size(gt, target_size)
                    else:
                        # Pseudo GT: 1000x1000 npy, downsample to target_size (same as pred)
                        gt = np.load(gt_file).astype(np.float32)
                        # gt = max_depth - gt  # convert metric depth to height map
                        gt_filter_size = gt.shape[0] // target_size[0]
                        if gt_filter_size > 1:
                            gt_resized = smart_downsample(gt, gt_filter_size)
                        else:
                            gt_resized = gt

                    # Flatten and add to collections
                    all_pred_pixels.extend(pred_resized.flatten())
                    all_gt_pixels.extend(gt_resized.flatten())

                except Exception as e:
                    print(f"Error processing {x}_{y}: {e}")
                    continue

            # Convert to numpy arrays
            all_pred_pixels = np.array(all_pred_pixels)
            all_gt_pixels = np.array(all_gt_pixels)


            # Compute R² score for combined pixels
            if len(all_pred_pixels) > 0:
                r2_score_combined = r2_score(all_gt_pixels, all_pred_pixels)
                rmse = np.sqrt(np.mean((all_gt_pixels - all_pred_pixels)**2))
                # scatter plot
                if target_size == (50, 50) and ('pseudo_full' in pred_path or 'full_pseudo' in pred_path) and plot_flag:
                    plt.figure(figsize=(10, 10))

                    # Create density visualization
                    sns.kdeplot(x=all_gt_pixels, y=all_pred_pixels, fill=True, cmap='YlOrBr',
                               thresh=0.05, levels=100, alpha=1)

                    plt.xlim(-3, 40)
                    plt.ylim(-3, 40)
                    # Overlay scatter plot of all points
                    plt.scatter(all_gt_pixels, all_pred_pixels, alpha=0.2, color = 'brown', s=2 , edgecolors='none')

                    # Calculate and plot the line of best fit
                    z = np.polyfit(all_gt_pixels, all_pred_pixels, 1)
                    p = np.poly1d(z)
                    x_range = np.linspace(-3, 40, 100)
                    plt.plot(x_range, p(x_range), color='c', linewidth=2, label='Best Fit')

                    # Plot the y=x line
                    plt.plot([-3, 40], [-3, 40], color='black', linewidth=2, linestyle='--', label='y=x')

                    plt.legend()
                    plt.xlabel(gt_label)
                    plt.ylabel('Prediction')
                    
                    plt.title(f'Our Method vs {gt_label}\n%ground > {threshold}%, R²: {r2_score_combined:.4f}, RMSE: {rmse:.4f}', fontsize=25)
                    plt.grid(True, alpha=0.3)
                    plt.savefig(f'scatter_plot_HF_YlOrBr_{eval_target}_{target_size}_{threshold}_c.png', dpi=300, bbox_inches='tight')
                    plt.close()
            else:
                r2_score_combined = np.nan
                rmse = np.nan

            results.append({
                'target_size': target_size,
                'threshold': threshold,
                r2_col: r2_score_combined,
                'number_of_pixels': len(all_pred_pixels),
                'number_of_samples': len(filtered_samples),
                'rmse': rmse
            })

            print(f"  Pixels: {len(all_pred_pixels):,}, R²: {r2_score_combined:.4f}")

        all_results.extend(results)
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Print summary tables for each target size
    for target_size in target_sizes:
        print(f"\n{'='*60}")
        print(f"SUMMARY RESULTS FOR TARGET_SIZE = {target_size}")
        print(f"{'='*60}")
        target_results = results_df[results_df['target_size'] == target_size]
        print(target_results[['threshold', r2_col, 'number_of_pixels', 'number_of_samples']].to_string(index=False, float_format='%.4f'))

    # Additional analysis and insights
    print("\n" + "=" * 60)
    print("ADDITIONAL INSIGHTS")
    print("=" * 60)

    # Find best threshold for each target size
    for target_size in target_sizes:
        target_results = results_df[results_df['target_size'] == target_size].dropna()
        if len(target_results) > 0:
            best_idx = target_results[r2_col].idxmax()
            best_threshold = target_results.loc[best_idx, 'threshold']
            best_r2 = target_results.loc[best_idx, r2_col]
            best_pixels = target_results.loc[best_idx, 'number_of_pixels']

            print(f"\nTarget size {target_size}:")
            print(f"  Best R² score: {best_r2:.4f} at threshold {best_threshold}%")
            print(f"  Pixels included: {best_pixels:,}")
            print(f"  Samples included: {target_results.loc[best_idx, 'number_of_samples']}")

    # Create pivot table for easy comparison across target sizes
    print(f"\n{'='*60}")
    print("R² SCORES COMPARISON TABLE")
    print(f"{'='*60}")
    pivot_table = results_df.pivot(index='threshold', columns='target_size', values=r2_col)
    print(pivot_table.round(4))

    return results_df

def comprehensive_ground_analysis(variant, chm_path, pseudo_gt_path, pred_root_path,
                                   analysis_dir, eval_target='chm',
                                   thresholds=None, target_sizes=None):
    """
    Main function that runs all ground analysis components

    This function orchestrates the entire analysis pipeline:
    1. Loads the prediction analysis results CSV
    2. Filters for scale=1 samples only (individual tiles)
    3. Runs sample distribution analysis
    4. Runs ground threshold analysis with smart downsampling
    5. Saves all results to CSV files
    6. Provides overall statistics and insights

    Args:
        sub_path (str): Prediction subdirectory name
        eval_target (str): 'chm' or 'pseudo_gt' - which ground truth to evaluate against

    Returns:
        dict: Dictionary containing all analysis results DataFrames
    """

    print("=" * 80)
    print(f"COMPREHENSIVE GROUND ANALYSIS (eval_target={eval_target})")
    print("=" * 80)

    os.makedirs(analysis_dir, exist_ok=True)
    pred_path = os.path.join(pred_root_path, variant)
    csv_file = os.path.join(analysis_dir, f'abalation_{variant}.csv')
    df = pd.read_csv(csv_file)
    df_scale1 = df[df['scale'] == 1].copy()
    print(f"Found {len(df_scale1)} scale=1 samples")

    distribution_df, cumulative_df = analyze_sample_distribution(df_scale1)

    print(f"\nStarting ground threshold analysis with smart downsampling (eval_target={eval_target})...")
    downsampling_results = analyze_ground_thresholds_with_downsampling(
        df_scale1, chm_path, pred_path, pseudo_gt_path=pseudo_gt_path,
        eval_target=eval_target, plot_flag=True,
        thresholds=thresholds, target_sizes=target_sizes)

    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    distribution_file = os.path.join(analysis_dir, f'sample_distribution_analysis_{eval_target}_{variant}.csv')
    distribution_df.to_csv(distribution_file, index=False)
    print(f"Sample distribution results saved to: {distribution_file}")

    cumulative_file = os.path.join(analysis_dir, f'cumulative_distribution_analysis_{eval_target}_{variant}.csv')
    cumulative_df.to_csv(cumulative_file, index=False)
    print(f"Cumulative distribution results saved to: {cumulative_file}")

    downsampling_file = os.path.join(analysis_dir, f'ground_threshold_analysis_with_downsampling_{eval_target}_{variant}.csv')
    downsampling_results.to_csv(downsampling_file, index=False)
    print(f"Downsampling analysis results saved to: {downsampling_file}")
    
    # Print overall statistics
    print(f"\n{'='*60}")
    print("OVERALL STATISTICS")
    print(f"{'='*60}")
    print(f"Total samples: {len(df_scale1)}")
    print(f"Mean %ground: {df_scale1['%ground'].mean():.2f}%")
    print(f"Std %ground: {df_scale1['%ground'].std():.2f}%")
    print(f"Mean R² (individual samples): {df_scale1['r2_chm'].mean():.4f}")
    print(f"Std R² (individual samples): {df_scale1['r2_chm'].std():.4f}")

    # print the summary statistics of the downsampling results with different target sizes and thresholds in a table
    
    
    print(f"\nAnalysis complete! All results have been saved to the output directory.")
    
    return {
        ################################################################
        'distribution': distribution_df,
        'cumulative': cumulative_df,
        'downsampling': downsampling_results
    }

def main():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from depth_chm.config import add_config_arg, load_config

    parser = argparse.ArgumentParser(description='Comprehensive ground analysis for predictions')
    add_config_arg(parser)
    parser.add_argument('--eval_target', type=str, default=None, choices=['chm', 'pseudo_gt'],
                        help='Override analysis.eval_target from config')
    cli = parser.parse_args()

    cfg = load_config(cli.config)
    paths = cfg['paths']
    a = cfg['analysis']

    root_path = paths['tiles_root']          # contains chm/ and pseudo_gt/
    pred_root_path = paths['predictions_dir'] # contains <variant>/ subdirs
    analysis_dir = paths['analysis_dir']
    variants = a['variants']
    eval_target = cli.eval_target or a['eval_target']
    thresholds = a['thresholds']
    target_sizes = a['target_sizes']

    print("################## Analyzing predictions... ####################")
    analyze_predictions(root_path, pred_root_path, variants, analysis_dir=analysis_dir)

    for v in variants:
        print(f"################## Analyzing {v} (eval_target={eval_target})... ####################")
        comprehensive_ground_analysis(
            v,
            chm_path=paths['chm_dir'],
            pseudo_gt_path=paths['pseudo_gt_dir'],
            pred_root_path=pred_root_path,
            analysis_dir=analysis_dir,
            eval_target=eval_target,
            thresholds=thresholds,
            target_sizes=target_sizes,
        )


if __name__ == "__main__":
    main()