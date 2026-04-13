"""Stage 2 — train DepthAnything with SiLog + L1 loss and Optuna HP search.

Config-driven. Override `train.trainable` / `train.gt` via CLI for convenience:
    python scripts/03_pipeline_train.py --trainable head --gt chm
    python scripts/03_pipeline_train.py --test_run    # quick memory check
"""

import argparse
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from depth_chm.config import add_config_arg, load_config
from depth_chm.utils import (
    get_device,
    list_tiles,
    load_model_and_processor,
    read_tif_height,
    resize_prediction,
)


# ============== Loss Functions ==============

class SiLogLoss(nn.Module):
    """Scale-Invariant Logarithmic Loss"""
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred, target, valid_mask):
        valid_mask = valid_mask.detach()
        diff_log = torch.log(target[valid_mask]) - torch.log(pred[valid_mask])
        loss = torch.sqrt(torch.pow(diff_log, 2).mean() -
                          self.lambd * torch.pow(diff_log.mean(), 2))
        return loss


class CombinedLoss(nn.Module):
    """Combined SiLog + L1 Loss"""
    def __init__(self, l1_weight=0.1, lambd=0.5):
        super().__init__()
        self.silog = SiLogLoss(lambd=lambd)
        self.l1_weight = l1_weight

    def forward(self, pred, target, valid_mask):
        silog_loss = self.silog(pred, target, valid_mask)
        l1_loss = F.l1_loss(pred[valid_mask], target[valid_mask])
        return silog_loss + self.l1_weight * l1_loss


# ============== Dataset ==============

class DepthDataset(Dataset):
    """Dataset for depth estimation training

    Supports two ground truth types:
    - pseudo: .npy files containing height values
    - chm: .tif files containing Canopy Height Model data
    """

    def __init__(self, image_files, depth_files, processor, max_depth=40.0, min_depth=0.001,
                 augment=True, gt_type='pseudo'):
        self.image_files = sorted(image_files)
        self.depth_files = sorted(depth_files)
        self.processor = processor
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.augment = augment
        self.gt_type = gt_type

        assert len(self.image_files) == len(self.depth_files), \
            f"Mismatch: {len(self.image_files)} images vs {len(self.depth_files)} depth files"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_files[idx]).convert('RGB')

        # Load depth based on ground truth type
        depth_file = self.depth_files[idx]

        if depth_file.endswith('.npy'):
            # Pseudo GT: stored as height, convert to depth
            height = np.load(depth_file).astype(np.float32)
            depth = self.max_depth - height
        elif depth_file.endswith('.tif'):
            # CHM: read TIF, normalize, and convert to depth
            height = read_tif_height(depth_file)
            # Normalize height (subtract minimum as in depth_chm_optimize.py)
            height = height - height.min()
            depth = self.max_depth - height
        else:
            raise ValueError(f'Unsupported depth file format: {depth_file}')

        # Clip depth to valid range
        depth = np.clip(depth, self.min_depth, self.max_depth)

        # Random horizontal flip (only during training)
        if self.augment and random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            depth = np.fliplr(depth).copy()

        # Process image using HuggingFace processor
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0)

        # Convert depth to tensor
        depth_tensor = torch.from_numpy(depth)

        return {
            'pixel_values': pixel_values,
            'depth': depth_tensor,
        }


def collate_fn(batch):
    """Custom collate function to handle variable size depths"""
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    depths = [item['depth'] for item in batch]

    return {
        'pixel_values': pixel_values,
        'depths': depths,
    }


# ============== Training Functions ==============

def train_epoch(model, dataloader, optimizer, criterion, device, max_depth, min_depth,
                total_iters, current_iter, base_lr, lr_head_mult, trainable='full'):
    """Train for one epoch with polynomial LR decay"""
    model.train()
    total_loss = 0

    for batch in dataloader:
        optimizer.zero_grad()

        pixel_values = batch['pixel_values'].to(device)
        depths = batch['depths']

        # Check for NaN/Inf in input
        if torch.isnan(pixel_values).any() or torch.isinf(pixel_values).any():
            print('Input contains NaN or Inf')
            return float('inf'), current_iter

        # Forward pass
        outputs = model(pixel_values)
        predicted_depth = outputs.predicted_depth

        # Compute loss for each sample
        batch_loss = 0
        valid_samples = 0
        for pred, target in zip(predicted_depth, depths):
            target = target.to(device)

            # Check depth validity
            if not target.min() > 0:
                print(f'Depth min {target.min()} is not > 0')
                return float('inf'), current_iter

            # Resize prediction to match target size
            pred_resized = resize_prediction(pred, target.shape)

            # Check for NaN/Inf in prediction
            if torch.isnan(pred_resized).any() or torch.isinf(pred_resized).any():
                print('Prediction contains NaN or Inf')
                return float('inf'), current_iter

            # Scale prediction by max_depth
            pred_scaled = pred_resized * max_depth

            # Create valid mask
            valid_mask = (target >= min_depth) & (target <= max_depth)

            if valid_mask.sum() > 0:
                batch_loss += criterion(pred_scaled, target, valid_mask)
                valid_samples += 1

        if valid_samples > 0:
            batch_loss = batch_loss / valid_samples
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()

        # Update learning rate (polynomial decay)
        current_iter += 1
        lr_ = base_lr * (1 - current_iter / total_iters) ** 0.9

        # Update LR based on trainable mode
        if trainable == 'head':
            # Single param group for head-only training
            optimizer.param_groups[0]["lr"] = lr_ * lr_head_mult
        else:
            # Two param groups for full training
            optimizer.param_groups[0]["lr"] = lr_
            optimizer.param_groups[1]["lr"] = lr_ * lr_head_mult

    return total_loss / len(dataloader), current_iter


def validate(model, dataloader, criterion, device, max_depth, min_depth):
    """Validate the model"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch['pixel_values'].to(device)
            depths = batch['depths']

            # Check for NaN/Inf
            if torch.isnan(pixel_values).any() or torch.isinf(pixel_values).any():
                return float('inf')

            outputs = model(pixel_values)
            predicted_depth = outputs.predicted_depth

            batch_loss = 0
            valid_samples = 0
            for pred, target in zip(predicted_depth, depths):
                target = target.to(device)

                if not target.min() > 0:
                    return float('inf')

                pred_resized = resize_prediction(pred, target.shape)

                if torch.isnan(pred_resized).any() or torch.isinf(pred_resized).any():
                    return float('inf')

                pred_scaled = pred_resized * max_depth
                valid_mask = (target >= min_depth) & (target <= max_depth)

                if valid_mask.sum() > 0:
                    batch_loss += criterion(pred_scaled, target, valid_mask)
                    valid_samples += 1

            if valid_samples > 0:
                batch_loss = batch_loss / valid_samples
                total_loss += batch_loss.item()

    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arg(parser)
    parser.add_argument('--trainable', type=str, default=None, choices=['full', 'head'],
                        help='Override train.trainable from config')
    parser.add_argument('--gt', type=str, default=None, choices=['pseudo', 'chm'],
                        help='Override train.gt from config')
    parser.add_argument('--test_run', action='store_true',
                        help='Run one batch to check memory usage, then exit')
    cli = parser.parse_args()

    cfg = load_config(cli.config)
    t = cfg['train']
    paths = cfg['paths']

    trainable = cli.trainable or t['trainable']
    gt = cli.gt or t['gt']
    depth_dir = paths['pseudo_gt_dir'] if gt == 'pseudo' else paths['chm_dir']

    args = SimpleNamespace(
        model_path=t['pretrained_model'],
        image_dir=paths['image_dir'],
        depth_dir=depth_dir,
        output_dir=f"{paths['model_dir']}_{trainable}_{gt}",
        batch_size=t['batch_size'],
        val_batch_size=t['val_batch_size'],
        max_depth=t['max_depth'],
        min_depth=t['min_depth'],
        l1_weight=t['l1_weight'],
        lr_head_multiplier=t['lr_head_multiplier'],
        n_trials=t['n_trials'],
        seed=t['seed'],
        trainable=trainable,
        gt=gt,
        test_run=cli.test_run,
    )

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = get_device()
    print(f'Using device: {device}')

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load processor (model is loaded fresh per Optuna trial below)
    print(f'Loading model from {args.model_path}...')
    processor, _ = load_model_and_processor(args.model_path)
    model_path = args.model_path

    # Load dataset
    print('Loading dataset...')
    print(f'Training mode: {args.trainable}, GT type: {args.gt}')
    image_files = list_tiles(args.image_dir, ('.png',))

    # Load depth files based on GT type
    if args.gt == 'pseudo':
        depth_files = list_tiles(args.depth_dir, ('.npy',))
    elif args.gt == 'chm':
        depth_files = list_tiles(args.depth_dir, ('.tif',))

    print(f'Found {len(image_files)} images and {len(depth_files)} depth files')
    assert len(image_files) == len(depth_files), "Number of images and depth files must match"

    # Split dataset: 90% train, 10% validation (following depth_chm_optimize.py)
    num_samples = len(image_files)
    indices = list(range(num_samples))
    random.seed(42)  # Fixed seed for reproducible split
    random.shuffle(indices)
    split = int(np.floor(0.9 * num_samples))
    train_indices, val_indices = indices[:split], indices[split:]

    train_image_files = [image_files[i] for i in train_indices]
    train_depth_files = [depth_files[i] for i in train_indices]
    val_image_files = [image_files[i] for i in val_indices]
    val_depth_files = [depth_files[i] for i in val_indices]

    print(f'Train: {len(train_image_files)}, Val: {len(val_image_files)}')

    # Loss function
    criterion = CombinedLoss(l1_weight=args.l1_weight)

    # ============== Test Run Mode ==============
    if args.test_run:
        print('\n' + '='*60)
        print('TEST RUN MODE - Checking memory usage')
        print('='*60)

        # Load model
        _, model = load_model_and_processor(model_path, device=device)

        # Set trainable parameters
        if args.trainable == 'head':
            for name, param in model.named_parameters():
                if 'backbone' in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            print(f'Mode: HEAD only (backbone frozen)')
        else:
            for param in model.parameters():
                param.requires_grad = True
            print(f'Mode: FULL model')

        # Create test datasets (use first few samples)
        test_size = max(args.batch_size, args.val_batch_size) * 2
        test_image_files = image_files[:test_size]
        test_depth_files = depth_files[:test_size]

        test_dataset = DepthDataset(test_image_files, test_depth_files, processor,
                                     max_depth=args.max_depth, min_depth=args.min_depth,
                                     augment=True, gt_type=args.gt)

        train_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=4, collate_fn=collate_fn, pin_memory=True)
        val_loader = DataLoader(test_dataset, batch_size=args.val_batch_size,
                                shuffle=False, num_workers=4, collate_fn=collate_fn, pin_memory=True)

        # Setup optimizer
        if args.trainable == 'head':
            optimizer = AdamW([
                {'params': [p for n, p in model.named_parameters() if 'backbone' not in n and p.requires_grad],
                 'lr': 1e-5 * args.lr_head_multiplier}
            ], lr=1e-5, betas=(0.9, 0.999), weight_decay=0.01)
        else:
            optimizer = AdamW([
                {'params': [p for n, p in model.named_parameters() if 'backbone' in n], 'lr': 1e-5},
                {'params': [p for n, p in model.named_parameters() if 'backbone' not in n], 'lr': 1e-5 * args.lr_head_multiplier}
            ], lr=1e-5, betas=(0.9, 0.999), weight_decay=0.01)

        # Helper function to get memory stats
        def get_memory_stats():
            if torch.cuda.is_available():
                gpu_allocated = torch.cuda.memory_allocated() / 1024**3
                gpu_reserved = torch.cuda.memory_reserved() / 1024**3
                gpu_max = torch.cuda.max_memory_allocated() / 1024**3
                return gpu_allocated, gpu_reserved, gpu_max
            return 0, 0, 0

        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

        # Test training batch
        print(f'\n--- Testing TRAINING batch (batch_size={args.batch_size}) ---')
        try:
            model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                pixel_values = batch['pixel_values'].to(device)
                depths = batch['depths']

                outputs = model(pixel_values)
                predicted_depth = outputs.predicted_depth

                batch_loss = 0
                valid_samples = 0
                for pred, target in zip(predicted_depth, depths):
                    target = target.to(device)
                    pred_resized = resize_prediction(pred, target.shape)
                    pred_scaled = pred_resized * args.max_depth
                    valid_mask = (target >= args.min_depth) & (target <= args.max_depth)
                    if valid_mask.sum() > 0:
                        batch_loss += criterion(pred_scaled, target, valid_mask)
                        valid_samples += 1

                if valid_samples > 0:
                    batch_loss = batch_loss / valid_samples
                    batch_loss.backward()
                    optimizer.step()

                gpu_alloc, gpu_res, gpu_max = get_memory_stats()
                print(f'  Training batch OK!')
                print(f'  GPU Memory: {gpu_alloc:.2f} GB allocated, {gpu_res:.2f} GB reserved, {gpu_max:.2f} GB peak')
                break  # Only test one batch

        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f'  FAILED: GPU out of memory!')
                print(f'  Try reducing --batch_size (current: {args.batch_size})')
                torch.cuda.empty_cache()
            else:
                raise e

        # Test validation batch
        print(f'\n--- Testing VALIDATION batch (val_batch_size={args.val_batch_size}) ---')
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

        try:
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    pixel_values = batch['pixel_values'].to(device)
                    depths = batch['depths']

                    outputs = model(pixel_values)
                    predicted_depth = outputs.predicted_depth

                    for pred, target in zip(predicted_depth, depths):
                        target = target.to(device)
                        pred_resized = resize_prediction(pred, target.shape)
                        pred_scaled = pred_resized * args.max_depth

                    gpu_alloc, gpu_res, gpu_max = get_memory_stats()
                    print(f'  Validation batch OK!')
                    print(f'  GPU Memory: {gpu_alloc:.2f} GB allocated, {gpu_res:.2f} GB reserved, {gpu_max:.2f} GB peak')
                    break  # Only test one batch

        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f'  FAILED: GPU out of memory!')
                print(f'  Try reducing --val_batch_size (current: {args.val_batch_size})')
                torch.cuda.empty_cache()
            else:
                raise e

        # Print summary
        print('\n' + '='*60)
        print('TEST RUN COMPLETE')
        print('='*60)
        if torch.cuda.is_available():
            total_gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f'GPU: {torch.cuda.get_device_name(0)}')
            print(f'Total GPU Memory: {total_gpu_memory:.2f} GB')
        print(f'\nCurrent settings:')
        print(f'  --batch_size {args.batch_size}')
        print(f'  --val_batch_size {args.val_batch_size}')
        print(f'  --trainable {args.trainable}')
        print(f'  --gt {args.gt}')
        print('\nIf both tests passed, your settings should work for full training.')
        print('='*60)

        # Clean up and exit
        del model, optimizer, train_loader, val_loader, test_dataset
        torch.cuda.empty_cache()
        return

    # ============== Optuna Objective Function ==============
    def objective(trial):
        # Hyperparameters to tune
        lr = trial.suggest_float('lr', 1e-8, 1e-4, log=True)
        epochs = trial.suggest_int('epochs', 10, 100, step=10)

        print(f'\n--- Trial {trial.number}: lr={lr:.2e}, epochs={epochs} ---')

        # Load fresh model for each trial
        _, model = load_model_and_processor(model_path, device=device)

        # Set trainable parameters based on trainable mode
        if args.trainable == 'head':
            # Freeze backbone, only train head (neck + head layers)
            for name, param in model.named_parameters():
                if 'backbone' in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            print(f'Training HEAD only (backbone frozen)')
        else:  # full
            # Train all parameters
            for param in model.parameters():
                param.requires_grad = True
            print(f'Training FULL model')

        # Create datasets with gt_type
        train_dataset = DepthDataset(train_image_files, train_depth_files, processor,
                                      max_depth=args.max_depth, min_depth=args.min_depth,
                                      augment=True, gt_type=args.gt)
        val_dataset = DepthDataset(val_image_files, val_depth_files, processor,
                                    max_depth=args.max_depth, min_depth=args.min_depth,
                                    augment=False, gt_type=args.gt)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=4, collate_fn=collate_fn, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size,
                                shuffle=False, num_workers=4, collate_fn=collate_fn, pin_memory=True)

        # Setup optimizer based on trainable mode
        if args.trainable == 'head':
            # Only optimize non-backbone parameters (neck + head)
            optimizer = AdamW([
                {'params': [p for n, p in model.named_parameters() if 'backbone' not in n and p.requires_grad],
                 'lr': lr * args.lr_head_multiplier}
            ], lr=lr, betas=(0.9, 0.999), weight_decay=0.01)
        else:  # full
            # Setup optimizer with different learning rates for backbone and head
            optimizer = AdamW([
                {'params': [p for n, p in model.named_parameters() if 'backbone' in n],
                 'lr': lr},
                {'params': [p for n, p in model.named_parameters() if 'backbone' not in n],
                 'lr': lr * args.lr_head_multiplier}
            ], lr=lr, betas=(0.9, 0.999), weight_decay=0.01)

        total_iters = epochs * len(train_loader)
        current_iter = 0

        # Training loop
        for epoch in range(epochs):
            train_loss, current_iter = train_epoch(
                model, train_loader, optimizer, criterion, device,
                args.max_depth, args.min_depth, total_iters, current_iter, lr, args.lr_head_multiplier,
                trainable=args.trainable
            )

            if train_loss == float('inf'):
                return float('inf')

        # Validation
        val_loss = validate(model, val_loader, criterion, device, args.max_depth, args.min_depth)
        print(f'Trial {trial.number} - Val Loss: {val_loss:.4f}')

        # Clean up
        del model, optimizer, train_loader, val_loader, train_dataset, val_dataset
        torch.cuda.empty_cache()

        return val_loss

    # ============== Run Optuna Optimization ==============
    print('\n' + '='*60)
    print('Starting Optuna hyperparameter search...')
    print('='*60)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=args.n_trials, n_jobs=1)

    print('\n' + '='*60)
    print('Optuna search complete!')
    print(f'Best trial: {study.best_trial.number}')
    print(f'Best value: {study.best_trial.value:.4f}')
    print(f'Best params: {study.best_trial.params}')
    print('='*60)

    # ============== Final Training with Best Parameters ==============
    best_lr = study.best_trial.params['lr']
    best_epochs = study.best_trial.params['epochs']

    print(f'\nTraining final model with best params: lr={best_lr:.2e}, epochs={best_epochs}')

    # Load fresh model
    _, model = load_model_and_processor(model_path, device=device)

    # Set trainable parameters based on trainable mode
    if args.trainable == 'head':
        for name, param in model.named_parameters():
            if 'backbone' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        print('Final training: HEAD only (backbone frozen)')
    else:
        for param in model.parameters():
            param.requires_grad = True
        print('Final training: FULL model')

    # Train on FULL dataset (not just train split)
    full_dataset = DepthDataset(image_files, depth_files, processor,
                                 max_depth=args.max_depth, min_depth=args.min_depth,
                                 augment=True, gt_type=args.gt)
    full_loader = DataLoader(full_dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=4, collate_fn=collate_fn, pin_memory=True)

    # Setup optimizer based on trainable mode
    if args.trainable == 'head':
        optimizer = AdamW([
            {'params': [p for n, p in model.named_parameters() if 'backbone' not in n and p.requires_grad],
             'lr': best_lr * args.lr_head_multiplier}
        ], lr=best_lr, betas=(0.9, 0.999), weight_decay=0.01)
    else:
        optimizer = AdamW([
            {'params': [p for n, p in model.named_parameters() if 'backbone' in n],
             'lr': best_lr},
            {'params': [p for n, p in model.named_parameters() if 'backbone' not in n],
             'lr': best_lr * args.lr_head_multiplier}
        ], lr=best_lr, betas=(0.9, 0.999), weight_decay=0.01)

    total_iters = best_epochs * len(full_loader)
    current_iter = 0

    pbar = tqdm(total=total_iters, desc='Final Training')
    for epoch in range(best_epochs):
        model.train()
        epoch_loss = 0

        for batch in full_loader:
            optimizer.zero_grad()

            pixel_values = batch['pixel_values'].to(device)
            depths = batch['depths']

            # Random horizontal flip
            if random.random() < 0.5:
                pixel_values = pixel_values.flip(-1)
                depths = [d.flip(-1) for d in depths]

            outputs = model(pixel_values)
            predicted_depth = outputs.predicted_depth

            batch_loss = 0
            valid_samples = 0
            for pred, target in zip(predicted_depth, depths):
                target = target.to(device)

                pred_resized = resize_prediction(pred, target.shape)

                pred_scaled = pred_resized * args.max_depth
                valid_mask = (target >= args.min_depth) & (target <= args.max_depth)

                if valid_mask.sum() > 0:
                    batch_loss += criterion(pred_scaled, target, valid_mask)
                    valid_samples += 1

            if valid_samples > 0:
                batch_loss = batch_loss / valid_samples
                batch_loss.backward()
                optimizer.step()
                epoch_loss += batch_loss.item()

            # Update learning rate
            current_iter += 1
            lr_ = best_lr * (1 - current_iter / total_iters) ** 0.9

            # Update LR based on trainable mode
            if args.trainable == 'head':
                optimizer.param_groups[0]["lr"] = lr_ * args.lr_head_multiplier
            else:
                optimizer.param_groups[0]["lr"] = lr_
                optimizer.param_groups[1]["lr"] = lr_ * args.lr_head_multiplier

            pbar.update(1)
            pbar.set_postfix({'epoch': epoch + 1, 'loss': batch_loss.item(), 'lr': lr_})

    pbar.close()

    # Save final model as safetensors
    print(f'\nSaving model to {args.output_dir}')
    model.save_pretrained(args.output_dir, safe_serialization=True)
    processor.save_pretrained(args.output_dir)

    # Save training info
    info = {
        'best_lr': best_lr,
        'best_epochs': best_epochs,
        'best_val_loss': study.best_trial.value,
        'n_trials': args.n_trials,
        'batch_size': args.batch_size,
        'max_depth': args.max_depth,
        'min_depth': args.min_depth,
        'l1_weight': args.l1_weight,
        'trainable': args.trainable,
        'gt_type': args.gt,
    }
    import json
    with open(os.path.join(args.output_dir, 'training_info.json'), 'w') as f:
        json.dump(info, f, indent=2)

    print('\nTraining complete!')
    print(f'Model saved to: {args.output_dir}')
    print(f'Best params: lr={best_lr:.2e}, epochs={best_epochs}')


if __name__ == '__main__':
    main()
