#!/usr/bin/env python3
"""Evaluate model perplexity on held-out data."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import yaml

from model import ByteGPT, GPTConfig


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def compute_perplexity(
    model: ByteGPT,
    data: torch.Tensor,
    block_size: int,
    device: str,
) -> float:
    """Compute perplexity on the given data."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i in range(0, len(data) - block_size - 1, block_size):
            xb = data[i : i + block_size].unsqueeze(0).to(device)
            yb = data[i + 1 : i + block_size + 1].unsqueeze(0).to(device)

            _, loss = model(xb, yb)
            if loss is not None:
                total_loss += loss.item() * block_size
                total_tokens += block_size

    model.train()
    return math.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model perplexity")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path")
    parser.add_argument("--config", default="config.yaml", help="Config path")
    parser.add_argument("--split", default="val", choices=["train", "val"], help="Data split")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_config(config_path)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_cfg = checkpoint["config"]["model"]

    device = resolve_device()
    model = ByteGPT(GPTConfig(**model_cfg)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    project_root = Path(__file__).resolve().parents[1]
    data_cfg = config["data"]
    train_cfg = config["training"]

    output_dir = project_root / data_cfg["output_dir"]
    corpus_path = output_dir / data_cfg["corpus_file"]

    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found at {corpus_path}")

    corpus_array = np.memmap(corpus_path, dtype=np.uint8, mode="c")
    data = torch.from_numpy(corpus_array).long()

    split_idx = int(len(data) * train_cfg["train_split"])
    if args.split == "train":
        data = data[:split_idx]
    else:
        data = data[split_idx:]

    block_size = model_cfg["block_size"]

    print(f"Computing perplexity on {args.split} split...")
    print(f"Data size: {len(data)} tokens")
    print(f"Block size: {block_size}")

    ppl = compute_perplexity(model, data, block_size, device)
    print(f"\nPerplexity: {ppl:.4f}")


if __name__ == "__main__":
    main()