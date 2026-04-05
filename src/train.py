#!/usr/bin/env python3
"""Train a byte-level GPT model from scratch."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.amp import autocast

from model import ByteGPT, GPTConfig


# === Monarch-inspired optimizations ===

def quantize_tensor_to_int4(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize tensor to INT4, return (quantized, scale, zero_point)."""
    orig_shape = tensor.shape
    flat = tensor.flatten()
    scale = flat.abs().max() / 7.0  # INT4 range is 0-7
    quantized = (flat / scale).round().clamp(0, 7).to(torch.uint8)
    return quantized.view(orig_shape), scale, torch.zeros_like(scale)


def dequantize_int4(quantized: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
    """Dequantize INT4 tensor back to fp16/fp32."""
    return quantized.float() * scale


class GradientQuantizer:
    """Quantize gradients to INT4 during backward pass."""
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.original_grads = {}

    def quantize_gradients(self, model: torch.nn.Module) -> None:
        """Quantize all gradients in the model."""
        if not self.enabled:
            return
        for name, param in model.named_parameters():
            if param.grad is not None:
                q, s, z = quantize_tensor_to_int4(param.grad)
                self.original_grads[name] = (param.grad.clone(), s, z)
                param.grad = q.float() * s

    def restore_gradients(self, model: torch.nn.Module) -> None:
        """Restore original gradients after optimizer step."""
        if not self.enabled:
            return
        for name, (orig_grad, scale, zero_point) in self.original_grads.items():
            for param in model.parameters():
                if param.grad is not None:
                    param.grad = orig_grad
        self.original_grads.clear()


class ActivationCompressor:
    """Compress activations using polar compression for memory savings."""
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.compressed_activations = {}
        self.centroids = {}

    def compress(self, tensor: torch.Tensor, layer_id: int) -> torch.Tensor:
        """Polar compress activation tensor."""
        if not self.enabled:
            return tensor
        flat = tensor.flatten(2)
        if layer_id not in self.centroids:
            self.centroids[layer_id] = flat.mean(dim=-1, keepdim=True)
        centroid = self.centroids[layer_id]
        diff = flat - centroid
        magnitude = diff.norm(dim=1, keepdim=True)
        angle = diff / (magnitude + 1e-8)
        compressed = torch.cat([magnitude, angle], dim=1)
        return compressed

    def decompress(self, compressed: torch.Tensor, layer_id: int, target_shape: torch.Size) -> torch.Tensor:
        """Decompress polar representation back to original shape."""
        if not self.enabled:
            return compressed
        magnitude = compressed[:, :compressed.shape[1]//2, :]
        angle = compressed[:, compressed.shape[1]//2:, :]
        centroid = self.centroids.get(layer_id, torch.zeros_like(magnitude))
        return (angle * magnitude + centroid).view(target_shape)


class SelectiveBackprop:
    """Selectively compute gradients for high-importance layers/tokens."""
    def __init__(self, enabled: bool = False, importance_threshold: float = 0.5):
        self.enabled = enabled
        self.importance_threshold = importance_threshold
        self.layer_importance = {}
        self.hook_handles = []

    def compute_layer_importance(self, model: torch.nn.Module) -> dict[str, float]:
        """Compute importance scores based on gradient magnitudes."""
        importance = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                importance[name] = param.grad.abs().mean().item()
        return importance

    def should_compute_grad(self, param_name: str) -> bool:
        """Check if gradient should be computed for this parameter."""
        if not self.enabled:
            return True
        score = self.layer_importance.get(param_name, 1.0)
        return score > self.importance_threshold


class StickyParameters:
    """Track parameters with high gradient magnitude and keep them in VRAM."""
    def __init__(self, enabled: bool = False, sticky_threshold: int = 3):
        self.enabled = enabled
        self.sticky_threshold = sticky_threshold
        self.promotion_count = {}
        self.sticky_params = set()
        self.sticky_names = ["lm_head", "token_embedding"]  # Always keep these

    def update(self, model: torch.nn.Module) -> None:
        """Update sticky parameter tracking."""
        if not self.enabled:
            return
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_mag = param.grad.abs().mean().item()
                if grad_mag > 1e-4:
                    self.promotion_count[name] = self.promotion_count.get(name, 0) + 1
                    if self.promotion_count[name] >= self.sticky_threshold:
                        self.sticky_params.add(name)

    def is_sticky(self, param_name: str) -> bool:
        """Check if parameter should stay in VRAM."""
        if not self.enabled:
            return False
        return param_name in self.sticky_params or any(s in param_name for s in self.sticky_names)


class GradientPager:
    """Offload low-magnitude gradients to CPU, page back when needed."""
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.gradients_cpu = {}
        self.page_threshold = 1e-6

    def page_out(self, model: torch.nn.Module) -> None:
        """Offload low-magnitude gradients to CPU."""
        if not self.enabled:
            return
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_mag = param.grad.abs().mean().item()
                if grad_mag < self.page_threshold:
                    self.gradients_cpu[name] = param.grad.cpu()
                    param.grad = None

    def page_in(self, model: torch.nn.Module) -> None:
        """Page gradients back from CPU for optimizer step."""
        if not self.enabled:
            return
        for name, cached_grad in self.gradients_cpu.items():
            for param in model.parameters():
                if param.grad is None:
                    param.grad = cached_grad.to(param.device)
                    break
        self.gradients_cpu.clear()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_device(value: str) -> str:
    if value != "auto":
        return value
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_batch(
    data: torch.Tensor, batch_size: int, block_size: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in starts])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])
    return (
        x.to(device=device, dtype=torch.long),
        y.to(device=device, dtype=torch.long),
    )


def load_corpus(corpus_path: Path) -> tuple[torch.Tensor, int]:
    """Memory-map the corpus to avoid materializing large Python integer lists."""
    corpus_size = corpus_path.stat().st_size
    corpus_array = np.memmap(corpus_path, dtype=np.uint8, mode="c")
    return torch.from_numpy(corpus_array), corpus_size


def format_bytes(size: float) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(size)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TiB"


def get_total_system_memory() -> int | None:
    if hasattr(os, "sysconf") and "SC_PAGE_SIZE" in os.sysconf_names and "SC_PHYS_PAGES" in os.sysconf_names:
        page_size = os.sysconf("SC_PAGE_SIZE")
        phys_pages = os.sysconf("SC_PHYS_PAGES")
        if isinstance(page_size, int) and isinstance(phys_pages, int):
            return page_size * phys_pages
    return None


def estimate_training_memory(
    model: ByteGPT,
    batch_size: int,
    block_size: int,
    use_amp: bool,
) -> dict[str, int]:
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    grad_bytes = param_bytes
    optimizer_bytes = param_bytes * 2
    bytes_per_activation = 2 if use_amp else 4
    activation_bytes = batch_size * block_size * model.config.n_embd * model.config.n_layer * bytes_per_activation * 6
    return {
        "params": param_bytes,
        "grads": grad_bytes,
        "optimizer": optimizer_bytes,
        "activations": activation_bytes,
        "total": param_bytes + grad_bytes + optimizer_bytes + activation_bytes,
    }


def get_cuda_memory_snapshot() -> dict[str, int] | None:
    if not torch.cuda.is_available():
        return None
    free_vram, total_vram = torch.cuda.mem_get_info()
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    return {
        "free": free_vram,
        "total": total_vram,
        "allocated": allocated,
        "reserved": reserved,
    }


def validate_guardrail_config(train_cfg: dict[str, Any], model_cfg: dict[str, Any]) -> None:
    batch_size = train_cfg["batch_size"]
    block_size = model_cfg["block_size"]
    accum_steps = train_cfg.get("gradient_accumulation_steps", 1)
    if batch_size < 1:
        raise ValueError("training.batch_size must be >= 1")
    if accum_steps < 1:
        raise ValueError("training.gradient_accumulation_steps must be >= 1")

    max_batch_tokens = train_cfg.get("max_batch_tokens")
    if max_batch_tokens is not None and batch_size * block_size > max_batch_tokens:
        raise ValueError(
            "Configured batch exceeds training.max_batch_tokens. "
            f"Current: {batch_size * block_size}, limit: {max_batch_tokens}."
        )

    max_eval_batch_tokens = train_cfg.get("max_eval_batch_tokens")
    eval_batch_size = train_cfg.get("eval_batch_size", batch_size)
    if max_eval_batch_tokens is not None and eval_batch_size * block_size > max_eval_batch_tokens:
        raise ValueError(
            "Configured eval batch exceeds training.max_eval_batch_tokens. "
            f"Current: {eval_batch_size * block_size}, limit: {max_eval_batch_tokens}."
        )


def enforce_memory_guardrails(
    train_cfg: dict[str, Any],
    estimated_memory: dict[str, int],
    device: str,
) -> None:
    max_ram_utilization = train_cfg.get("max_ram_utilization")
    total_system_memory = get_total_system_memory()
    if (
        total_system_memory is not None
        and max_ram_utilization is not None
        and estimated_memory["total"] > total_system_memory * float(max_ram_utilization)
    ):
        raise RuntimeError(
            "Estimated training memory exceeds training.max_ram_utilization. "
            "Reduce batch_size, block_size, or model size."
        )

    if device != "cuda" or not torch.cuda.is_available():
        return

    snapshot = get_cuda_memory_snapshot()
    if snapshot is None:
        return

    max_vram_utilization = train_cfg.get("max_vram_utilization")
    if (
        max_vram_utilization is not None
        and estimated_memory["total"] > snapshot["total"] * float(max_vram_utilization)
    ):
        raise RuntimeError(
            "Estimated training memory exceeds training.max_vram_utilization. "
            "Reduce batch_size, block_size, or model size."
        )

    min_free_vram_bytes = train_cfg.get("min_free_vram_bytes")
    if min_free_vram_bytes is not None and snapshot["free"] < int(min_free_vram_bytes):
        raise RuntimeError(
            "Free VRAM is below training.min_free_vram_bytes before training starts. "
            "Close other GPU workloads or reduce memory usage."
        )


def check_runtime_cuda_guardrails(
    train_cfg: dict[str, Any],
    phase: str,
) -> None:
    if not torch.cuda.is_available():
        return

    snapshot = get_cuda_memory_snapshot()
    if snapshot is None:
        return

    min_free_vram_bytes = train_cfg.get("min_free_vram_bytes")
    if min_free_vram_bytes is not None and snapshot["free"] < int(min_free_vram_bytes):
        raise RuntimeError(
            f"Aborting during {phase}: free VRAM dropped below training.min_free_vram_bytes "
            f"({format_bytes(snapshot['free'])} free)."
        )

    max_reserved_vram_bytes = train_cfg.get("max_reserved_vram_bytes")
    if max_reserved_vram_bytes is not None and snapshot["reserved"] > int(max_reserved_vram_bytes):
        raise RuntimeError(
            f"Aborting during {phase}: CUDA reserved memory exceeded "
            "training.max_reserved_vram_bytes."
        )


def print_training_preflight(
    corpus_size: int,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    model: ByteGPT,
    device: str,
    use_amp: bool,
    batch_size: int,
    block_size: int,
) -> dict[str, int]:
    memory = estimate_training_memory(model, batch_size, block_size, use_amp)
    print(f"Corpus bytes: {corpus_size}")
    print(f"Corpus storage: memmap uint8 ({format_bytes(corpus_size)})")
    print(f"Train bytes: {len(train_data):,} | val bytes: {len(val_data):,}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(
        "Estimated training memory: "
        f"params {format_bytes(memory['params'])}, "
        f"grads {format_bytes(memory['grads'])}, "
        f"optimizer {format_bytes(memory['optimizer'])}, "
        f"activations ~{format_bytes(memory['activations'])}, "
        f"total ~{format_bytes(memory['total'])}"
    )
    total_system_memory = get_total_system_memory()
    if total_system_memory is not None:
        print(f"System RAM: {format_bytes(total_system_memory)}")
        if memory["total"] > total_system_memory * 0.7:
            print(
                "Warning: estimated training memory exceeds 70% of system RAM. "
                "Reduce batch_size, block_size, or model size before a long run."
            )
    if device == "cuda" and torch.cuda.is_available():
        snapshot = get_cuda_memory_snapshot()
        assert snapshot is not None
        print(
            "CUDA VRAM: "
            f"{format_bytes(snapshot['free'])} free / {format_bytes(snapshot['total'])} total"
        )
        print(
            "CUDA allocator: "
            f"{format_bytes(snapshot['allocated'])} allocated / "
            f"{format_bytes(snapshot['reserved'])} reserved"
        )
        if memory["total"] > snapshot["free"]:
            print(
                "Warning: estimated training memory exceeds currently free VRAM. "
                "Reduce batch_size, block_size, or close other GPU workloads."
            )
    return memory


def is_oom_error(error: BaseException) -> bool:
    message = str(error).lower()
    return "out of memory" in message or "cuda error: out of memory" in message


@torch.no_grad()
def estimate_loss(
    model: ByteGPT,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    batch_size: int,
    block_size: int,
    eval_batches: int,
    device: str,
    train_cfg: dict[str, Any] | None = None,
) -> dict[str, float]:
    model.eval()
    losses: dict[str, float] = {}
    for split, data in (("train", train_data), ("val", val_data)):
        split_losses = torch.zeros(eval_batches)
        for step in range(eval_batches):
            if device == "cuda" and train_cfg is not None:
                check_runtime_cuda_guardrails(train_cfg, f"evaluation/{split}")
            xb, yb = get_batch(data, batch_size, block_size, device)
            _, loss = model(xb, yb)
            split_losses[step] = loss.item()
        losses[split] = split_losses.mean().item()
    model.train()
    return losses


def save_checkpoint(
    checkpoint_path: Path,
    model: ByteGPT,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: dict[str, Any],
    metrics: dict[str, float],
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "metrics": metrics,
        },
        checkpoint_path,
    )


def derive_snapshot_path(checkpoint_path: Path, suffix: str) -> Path:
    return checkpoint_path.with_name(f"{checkpoint_path.stem}.{suffix}{checkpoint_path.suffix}")


def cosine_lr(step: int, max_steps: int, base_lr: float, warmup_steps: int) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return 0.1 * base_lr + 0.9 * base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Sisyphus from scratch")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(config_path)
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    validate_guardrail_config(train_cfg, model_cfg)

    set_seed(train_cfg["seed"])
    device = resolve_device(train_cfg["device"])

    output_dir_setting = Path(data_cfg["output_dir"])
    output_dir = (
        output_dir_setting
        if output_dir_setting.is_absolute()
        else project_root / output_dir_setting
    )
    corpus_path = output_dir / data_cfg["corpus_file"]
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Corpus not found at {corpus_path}. Run src/build_corpus.py first."
        )

    data, corpus_size = load_corpus(corpus_path)
    if corpus_size <= model_cfg["block_size"] + 1:
        raise ValueError("Corpus is too small for the configured block size")

    split_idx = int(len(data) * train_cfg["train_split"])
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    if len(val_data) <= model_cfg["block_size"] + 1:
        raise ValueError("Validation split is too small; adjust train_split or corpus size")

    gpt_config = GPTConfig(**model_cfg)
    model = ByteGPT(gpt_config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        betas=tuple(train_cfg["betas"]),
        weight_decay=train_cfg["weight_decay"],
    )

    checkpoint_path = (
        project_root / train_cfg["checkpoint_dir"] / train_cfg["checkpoint_name"]
    )
    last_checkpoint_path = derive_snapshot_path(checkpoint_path, "last")

    best_val = float("inf")
    best_step = -1
    use_amp = device in ("cuda", "mps")
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # Initialize Monarch optimizations
    gradient_quantizer = GradientQuantizer(enabled=train_cfg.get("gradient_quantization", False))
    activation_compressor = ActivationCompressor(enabled=train_cfg.get("activation_compression", False))
    selective_backprop = SelectiveBackprop(enabled=train_cfg.get("selective_backprop", False))
    sticky_params = StickyParameters(enabled=train_cfg.get("sticky_params", False))
    gradient_pager = GradientPager(enabled=train_cfg.get("gradient_paging", False))

    print(f"Training on {device}")
    print(f"Mixed precision: {use_amp}")
    print(f"Gradient quantization: {gradient_quantizer.enabled}")
    print(f"Activation compression: {activation_compressor.enabled}")
    print(f"Selective backprop: {selective_backprop.enabled}")
    print(f"Sticky parameters: {sticky_params.enabled}")
    print(f"Gradient paging: {gradient_pager.enabled}")
    estimated_memory = print_training_preflight(
        corpus_size=corpus_size,
        train_data=train_data,
        val_data=val_data,
        model=model,
        device=device,
        use_amp=use_amp,
        batch_size=train_cfg["batch_size"],
        block_size=model_cfg["block_size"],
    )
    enforce_memory_guardrails(train_cfg, estimated_memory, device)

    optimizer.zero_grad(set_to_none=True)
    for step in range(train_cfg["max_steps"]):
        lr = cosine_lr(
            step=step,
            max_steps=train_cfg["max_steps"],
            base_lr=train_cfg["learning_rate"],
            warmup_steps=train_cfg["warmup_steps"],
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        xb, yb = get_batch(
            train_data,
            train_cfg["batch_size"],
            model_cfg["block_size"],
            device,
        )
        accum_steps = train_cfg.get("gradient_accumulation_steps", 1)

        try:
            if device == "cuda":
                check_runtime_cuda_guardrails(train_cfg, "training/forward")
            if use_amp:
                with autocast(device_type="cuda" if device == "cuda" else "mps"):
                    _, loss = model(xb, yb)
                loss = loss / accum_steps
                scaler.scale(loss).backward()
            else:
                _, loss = model(xb, yb)
                loss = loss / accum_steps
                loss.backward()
        except RuntimeError as error:
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            if is_oom_error(error):
                raise RuntimeError(
                    "Training ran out of memory. Reduce training.batch_size, "
                    "model.block_size, or model size, or switch training.device to 'cpu'."
                ) from error
            raise

        # Apply Monarch optimizations before optimizer step
        if (step + 1) % accum_steps == 0 or step == train_cfg["max_steps"] - 1:
            # Selective backprop: compute layer importance
            if selective_backprop.enabled:
                selective_backprop.layer_importance = selective_backprop.compute_layer_importance(model)

            # Gradient quantization (compress before clip)
            gradient_quantizer.quantize_gradients(model)

            # Gradient paging (offload low-magnitude grads)
            gradient_pager.page_out(model)

            if use_amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])

            # Restore quantized gradients before optimizer
            gradient_quantizer.restore_gradients(model)

            # Page back gradients for optimizer
            gradient_pager.page_in(model)

            # Update sticky params tracking
            sticky_params.update(model)

            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if step % train_cfg["eval_interval"] == 0 or step == train_cfg["max_steps"] - 1:
            losses = estimate_loss(
                model,
                train_data,
                val_data,
                train_cfg.get("eval_batch_size", train_cfg["batch_size"]),
                model_cfg["block_size"],
                train_cfg["eval_batches"],
                device,
                train_cfg=train_cfg,
            )
            print(
                f"step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} | lr {lr:.6f}"
            )
            if losses["val"] < best_val:
                best_val = losses["val"]
                best_step = step
                save_checkpoint(checkpoint_path, model, optimizer, step, config, losses)

        if step and step % train_cfg["save_interval"] == 0:
            save_checkpoint(
                last_checkpoint_path,
                model,
                optimizer,
                step,
                config,
                {"loss": loss.item()},
            )

    final_metrics = {"best_val": best_val, "best_step": best_step}
    save_checkpoint(
        last_checkpoint_path,
        model,
        optimizer,
        train_cfg["max_steps"],
        config,
        final_metrics,
    )
    metrics_path = checkpoint_path.with_suffix(".metrics.json")
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(final_metrics, handle, indent=2)

    print(f"Best checkpoint written to {checkpoint_path}")
    print(f"Last checkpoint written to {last_checkpoint_path}")


if __name__ == "__main__":
    main()
