"""Stage 1 — run the plain pretrained DepthAnything model on image tiles.

Writes float32 .npy depth predictions (meters, scaled by `inference.max_depth`)
into `paths.vanilla_depth_dir`. Those serve as input to 02_residual_depth_chm.py.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from depth_chm.config import add_config_arg, load_config

# Reuse the inference implementation from script 04.
_this_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_this_dir))
import importlib.util
_spec = importlib.util.spec_from_file_location('pipeline_inference',
                                               _this_dir / '04_pipeline_inference.py')
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
run_inference = _mod.run_inference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arg(parser)
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg['paths']
    run_inference(
        model_path=cfg['train']['pretrained_model'],
        image_dir=paths['image_dir'],
        output_dir=paths['vanilla_depth_dir'],
        max_depth=cfg['inference']['max_depth'],
    )


if __name__ == '__main__':
    main()
