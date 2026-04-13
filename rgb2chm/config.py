"""YAML config loader with `${paths.x}` substitution and repo-root anchoring.

Usage:
    from rgb2chm.config import load_config
    cfg = load_config("configs/default.yaml")
    print(cfg["paths"]["image_dir"])   # absolute path

All keys under `paths:` are resolved to absolute paths anchored at the repo
root (the parent of the `rgb2chm/` package), so scripts can be invoked from
any working directory.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

_TOKEN_RE = re.compile(r"\$\{paths\.([a-zA-Z0-9_]+)\}")


def _resolve_tokens(paths: dict[str, Any]) -> dict[str, str]:
    """Repeatedly substitute ${paths.x} tokens until none remain."""
    resolved = {k: str(v) for k, v in paths.items()}
    for _ in range(16):  # bounded fixpoint
        changed = False
        for k, v in resolved.items():
            def repl(m: re.Match) -> str:
                ref = m.group(1)
                if ref not in resolved:
                    raise KeyError(f"paths.{ref} referenced by paths.{k} but not defined")
                return resolved[ref]
            new_v = _TOKEN_RE.sub(repl, v)
            if new_v != v:
                resolved[k] = new_v
                changed = True
        if not changed:
            break
    if any(_TOKEN_RE.search(v) for v in resolved.values()):
        raise ValueError("Unresolved ${paths.*} tokens (circular reference?)")
    return resolved


def _anchor(path_str: str) -> str:
    p = Path(path_str)
    return str(p if p.is_absolute() else (REPO_ROOT / p).resolve())


def load_config(path: str | os.PathLike) -> dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = (REPO_ROOT / cfg_path).resolve()
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    paths = cfg.get("paths", {})
    resolved = _resolve_tokens(paths)
    cfg["paths"] = {k: _anchor(v) for k, v in resolved.items()}
    return cfg


def add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML config (default: configs/default.yaml)",
    )
