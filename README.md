# DepthCHM

Predict Canopy Height Models (CHM) from RGB imagery by fine-tuning HuggingFace
`DepthAnythingForDepthEstimation`, supervised by LiDAR-derived CHM and/or a
higher-resolution pseudo-GT fused from a vanilla depth model.

Trained model also avilable in [HuggingFace](https://huggingface.co/Boxiang/depth_chm)

## Pipeline

**Stage 1 — Data preparation** (inputs: one large RGB GeoTIFF + overlapping LAS point cloud)
1. Crop and align the TIF and LAS into paired tiles and generate per-tile CHM rasters.
2. Run a plain (pretrained) depth estimator on the image tiles to produce vanilla depth maps.
3. Fuse vanilla depth with CHM via histogram matching + gradient-aware smoothing to
   get high-resolution pseudo GT.

Outputs: `(image, chm, pseudo_gt)` tile triples.

**Stage 2 — Train / infer / evaluate**
1. Fine-tune DepthAnything (full model or head-only) against CHM or pseudo GT.
2. Run inference on the test images.
3. Evaluate predictions against CHM across `%ground` coverage thresholds.

## Setup

```bash
conda create -n depth_chm python=3.10 -y
conda activate depth_chm
pip install -r requirements.txt
```

Install a CUDA-matched `torch` build from [pytorch.org](https://pytorch.org) if you need GPU acceleration.

## Data layout

Drop your raw inputs under `data/` (the defaults in `configs/default.yaml`):

```
data/martell/
├── orthomosaic.tif    # large RGB GeoTIFF
├── lidar.las          # overlapping LAS point cloud
└── lidar.prj          # LAS CRS in WKT (optional; falls back to LAS header)
```

Tile outputs and trained models land under `data/martell/tiles/` and `outputs/`
— both gitignored. Edit `configs/default.yaml` to point at different locations.

## Configuration

Everything (paths, crop parameters, training hyperparameters, analysis
thresholds) lives in `configs/default.yaml`. Copy and edit that file, or pass
`--config path/to/your.yaml` to any script. Paths are resolved relative to the
repo root, so scripts can be invoked from any working directory.

## End-to-end run

```bash
# Stage 1: crop tiles + compute CHM
python scripts/01_crop_tif_las.py --config configs/default.yaml

# Stage 1: vanilla pretrained DepthAnything on image tiles
python scripts/01b_vanilla_depth.py --config configs/default.yaml

# Stage 1: fuse vanilla depth with CHM -> pseudo GT
python scripts/02_residual_depth_chm.py --config configs/default.yaml

# Stage 2: train (repeat with --trainable / --gt to produce all variants)
python scripts/03_pipeline_train.py --trainable full --gt pseudo
# python scripts/03_pipeline_train.py --trainable head --gt pseudo
# python scripts/03_pipeline_train.py --trainable full --gt chm
# python scripts/03_pipeline_train.py --trainable head --gt chm

# Stage 2: inference over all trained variants
python scripts/04_pipeline_inference.py --config configs/default.yaml

# Stage 2: evaluate
python scripts/05_comprehensive_ground_analysis.py --config configs/default.yaml
```

Quick GPU-memory sanity check before full training:

```bash
python scripts/03_pipeline_train.py --test_run
```

## Project layout

```
RGB2CHM/
├── configs/default.yaml            # all paths + hyperparameters
├── depth_chm/
│   ├── config.py                   # YAML loader (resolves ${paths.*}, anchors at repo root)
│   └── utils.py                    # shared helpers: read_tif_height, get_device,
│                                   # load_model_and_processor, resize_prediction, list_tiles
├── scripts/
│   ├── 01_crop_tif_las.py
│   ├── 01b_vanilla_depth.py
│   ├── 02_residual_depth_chm.py
│   ├── 03_pipeline_train.py
│   ├── 04_pipeline_inference.py
│   └── 05_comprehensive_ground_analysis.py
├── requirements.txt
└── README.md
```

## Reproducibility notes

- `train.seed` (default 42) seeds `random`, `numpy`, and `torch`.
- Dataset split (90/10) uses a fixed independent seed (also 42).
- GPU required for training; `--test_run` reports peak memory for your batch size.
