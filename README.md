# Sisyphus

`Sisyphus` is a local from-scratch language model project.

It builds a Rust-focused corpus from this repo, official Rust documentation repositories, and an optional bounded set of Rust code repositories, tokenizes at the byte level, and trains a small causal Transformer from random initialization with PyTorch.

This is not LoRA, not adapter tuning, and not base-model fine-tuning.

## What It Does

- walks this repo for local Rust-adjacent text
- can ingest official Rust docs cloned from the web
- can ingest a bounded set of Rust code repositories cloned from the web
- can optionally ingest a bounded streamed slice of `HuggingFaceFW/fineweb-edu`
- keeps source and documentation text
- skips generated, vendor, cache, git, and binary files
- deduplicates exact normalized file content and caps oversized files
- writes a single corpus file plus metadata
- trains a byte-level GPT-style model from scratch
- saves checkpoints you can use for local text generation

## Techniques

- Byte-level language modeling with a fixed `256`-token vocabulary, so no separate tokenizer training step is needed.
- GPT-style causal Transformer training from random initialization with next-token prediction.
- Random fixed-length training windows sampled from the corpus rather than document-by-document epoch training.
- AdamW optimization with warmup plus cosine learning-rate decay.
- Automatic mixed precision on `cuda` and `mps` when available.
- Periodic train/validation loss estimation during training, with best-checkpoint and last-checkpoint saves.
- Corpus assembly from multiple sources: local Rust project text, official Rust web docs, curated Rust code repositories, and optional bounded streamed `fineweb-edu`.
- Exact-content deduplication after normalization to reduce repeated files across the project tree.
- Per-file truncation caps to improve corpus quality without increasing VRAM requirements.
- File-type and directory filtering to skip binary, generated, vendor, cache, and git content.
- Per-source character caps to keep docs, code, and local text in a controlled mix.

Experimental techniques implemented in code and enabled by default in config:

- Gradient quantization
- Activation compression
- Selective backprop
- Sticky parameters
- Gradient paging
- KV-cache and fractal-attention hooks in the model code

These are advanced memory optimizations. Disable them by setting the flags to `false` in `config.yaml` if you experience issues.

## Quick Start

Build the corpus:

```bash
python3 src/fetch_rust_web_corpus.py --config config.yaml
python3 src/fetch_rust_code_corpus.py --config config.yaml
# optional:
# python3 src/fetch_fineweb_edu.py --config config.yaml
python3 src/build_corpus.py --config config.yaml
```

Train from scratch:

```bash
python3 src/train.py --config config.yaml
```

Resume training from a checkpoint:

```bash
python3 src/train.py --config config.yaml --resume checkpoints/sisyphus.last.pt
```

Generate text from a checkpoint:

```bash
python3 src/generate.py --checkpoint checkpoints/sisyphus.pt --prompt "fn main"
```

Or use the wrapper:

```bash
bash train_from_scratch.sh
```

## Hardware Requirements

- **Minimum RAM**: 8 GB system RAM
- **GPU (recommended)**: NVIDIA CUDA-capable GPU with at least 4 GB VRAM, or Apple Silicon Mac with MPS support
- **CPU fallback**: Training is very slow (~1 hour per 100 steps) without a GPU; not recommended for full 3000-step training

## Checkpoint Selection

Two checkpoint files are saved during training:

- **`sisyphus.pt`** — Best checkpoint (lowest validation loss). Use this for generation and evaluation.
- **`sisyphus.last.pt`** — Most recent checkpoint. Use this to resume training if interrupted.

The best checkpoint path and metrics are logged to `sisyphus.metrics.json`.

## Training Notes

- Default training runs for 3000 steps with a batch size of 8
- Each training step processes `batch_size * block_size = 8 * 256 = 2048` tokens
- Evaluation runs every 200 steps
- Training resumes correctly from a checkpoint with `--resume`, restoring optimizer state and the best loss seen so far

## Notes

- The model uses raw bytes as tokens, so it does not need a separate tokenizer-training step.
- The Rust documentation corpus is limited to official sources: `rust-lang/book`, `rust-lang/rust-by-example`, `rust-lang/reference`, `rust-lang/edition-guide`, and `rust-lang/rustc-dev-guide`.
- The Rust code corpus is a bounded curated set of repositories configured under `web_corpus.rust_code` in `config.yaml`.
- `fineweb-edu` is streamed from `HuggingFaceFW/fineweb-edu` and capped by character and document limits before it is added to the local corpus, so dataset quality can improve without changing VRAM requirements.
- The default model is intentionally small enough to be realistic on a local machine.
- Training from scratch on this corpus will produce a niche local model, not a general-purpose assistant.
- Better results usually require more training time, larger context, and careful exclusion rules for noisy files.
- The corpus builder now supports per-source character caps, so you can bias the final mix toward Rust code or documentation without changing the trainer.
- `max_chars_per_file`, `deduplicate_exact`, and the per-source `max_total_characters` caps are the main knobs for improving corpus quality without increasing GPU memory pressure.

## FineWeb-Edu

Use the bounded fetcher before rebuilding the corpus:

```bash
python3 src/fetch_fineweb_edu.py --config config.yaml
python3 src/build_corpus.py --config config.yaml
```

Tune these config keys to control how much crawl data you add without changing the model or batch size:

- `web_corpus.fineweb_edu.max_total_characters`
- `web_corpus.fineweb_edu.max_documents`
- `web_corpus.fineweb_edu.max_document_characters`
