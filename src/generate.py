#!/usr/bin/env python3
"""Generate text from a trained ConfidentlyWrog checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from model import ByteGPT, GPTConfig


def resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_checkpoint(checkpoint_path: Path) -> tuple[dict[str, Any], ByteGPT]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_cfg = checkpoint["config"]["model"]
    model = ByteGPT(GPTConfig(**model_cfg))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return checkpoint, model


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate from Sisyphus")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path")
    parser.add_argument("--prompt", default="", help="UTF-8 prompt text")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--use-cache", action="store_true", default=True, help="Use Monarch KV Paging")
    parser.add_argument("--no-cache", action="store_false", dest="use_cache", help="Disable KV cache")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint, model = load_checkpoint(checkpoint_path)
    config = checkpoint["config"]
    generation_cfg = config["generation"]
    device = resolve_device()
    model = model.to(device)

    prompt_bytes = args.prompt.encode("utf-8", errors="ignore")
    if not prompt_bytes:
        prompt_bytes = b"<FILE path=\"seed\">\n"

    idx = torch.tensor([list(prompt_bytes)], dtype=torch.long, device=device)
    out = model.generate(
        idx,
        max_new_tokens=args.max_new_tokens or generation_cfg["max_new_tokens"],
        temperature=args.temperature or generation_cfg["temperature"],
        top_k=args.top_k or generation_cfg["top_k"],
        use_cache=args.use_cache,
    )
    text = bytes(out[0].tolist()).decode("utf-8", errors="ignore")
    print(text)


if __name__ == "__main__":
    main()
