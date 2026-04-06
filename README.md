# Sisyphus

Sisyphus is a small Rust-focused language model project trained from scratch.

It builds a corpus out of local project text, official Rust docs, a bounded set of Rust code repositories, and optionally FineWeb-Edu. It then trains a byte-level decoder model in PyTorch and writes checkpoints for generation and evaluation.

This repo is for training experiments, not product polish. Some files reflect older runs or older ideas, so the code in `src/` and the concrete config/log files are the things to trust first.

## What the project is trying to do

The basic goal is simple:

- keep the stack small enough to run on ordinary hardware
- stay close to the data and training code instead of hiding everything behind tooling
- see how far a small Rust-specific model can go with a better corpus and a more locality-biased attention block

This is not LoRA, not instruction tuning, and not base-model finetuning. The model is trained from random initialization.

## Current shape

The main pieces are:

- `src/build_corpus.py`: walks source roots, filters files, deduplicates normalized content, writes `data/processed/corpus.txt` plus metadata
- `src/fetch_rust_web_corpus.py`: clones Rust docs repos
- `src/fetch_rust_code_corpus.py`: clones the configured Rust code repos
- `src/fetch_top_crates.py`: fetches and clones top crates from crates.io
- `src/fetch_fineweb_edu.py`: optional bounded FineWeb-Edu ingest
- `src/model.py`: byte-level model definition
- `src/train.py`: training loop, checkpointing, preflight memory checks, resume support
- `src/generate.py`: text generation from a checkpoint
- `src/eval.py`: held-out perplexity evaluation

## Model

The current model is not just a stock causal transformer.

`src/model.py` defines a byte-level decoder with:

- vocab size `256`
- learned positional embeddings
- tied token embedding / LM head weights
- residual blocks with MLPs
- `HybridAttention` in each block

`HybridAttention` mixes two paths:

- local causal window attention
- a GRU-like recurrent state path

The local path is there because most code dependencies are nearby. The recurrent path is there to carry some compressed longer-range state without paying for full dense attention everywhere.

There is also older cache/paging-oriented code in the repo. Some of that is experimental or leftover from previous directions. If you want to know what the current training path really is, read `ByteGPT`, `Block`, and `HybridAttention` first.

## Corpus building

The corpus builder is deliberately simple.

It walks configured roots, keeps only allowed extensions, skips binary/generated/vendor/cache trees, caps file size, optionally truncates very large text files, normalizes content for exact deduplication, and writes everything into one byte-level training file with lightweight `<FILE ...>` wrappers.

By default the allowed extensions are:

- `.md`
- `.txt`
- `.rst`
- `.toml`
- `.rs`

The main quality levers are:

- per-source `max_total_characters`
- `max_chars_per_file`
- `min_chars`
- `deduplicate_exact`
- which roots you choose to include at all

The project has several config files because the corpus and training setup have changed over time. `config.yaml` is the broad current config. `config.20m.yaml` and `config.20m.optimized.yaml` document specific 20M-class runs.

## A concrete run

The most interesting logged run in this repo is the one in `config.20m.optimized.yaml` and `logs/train.20m.optimized.log`.

That run used:

- 25,613,312 parameters
- block size `512`
- batch size `12`
- learning rate `2e-4`
- 30,000 steps
- mixed precision on CUDA

The final log line for that run was:

- train loss `0.5834`
- val loss `0.8217`

Best validation loss in the log appears earlier, at step `18500`, with val loss `0.7757`.

The expanded corpus run also records:

- corpus bytes: `177151242`
- train bytes: `159436117`
- val bytes: `17715125`

The top-crates expansion path increased the corpus substantially and seems to have helped more than small architecture polish did.

## Training

Typical workflow:

```bash
python3 src/fetch_rust_web_corpus.py --config config.yaml
python3 src/fetch_rust_code_corpus.py --config config.yaml
python3 src/fetch_top_crates.py --count 500
# optional:
# python3 src/fetch_fineweb_edu.py --config config.yaml
python3 src/build_corpus.py --config config.yaml
python3 src/train.py --config config.yaml
```

Resume from the last checkpoint:

```bash
python3 src/train.py --config config.yaml --resume checkpoints/sisyphus.last.pt
```

Or use the wrapper:

```bash
bash train_from_scratch.sh
```

`train_from_scratch.sh` skips FineWeb-Edu. If you want FineWeb in the corpus, run the fetcher before rebuilding the corpus.

## Generation

Generate from a checkpoint:

```bash
python3 src/generate.py \
  --checkpoint checkpoints/sisyphus.pt \
  --prompt "fn main"
```

Generation is byte-level, so there is no tokenizer training step in this project. The prompt is UTF-8 encoded directly to byte IDs.

## Evaluation

Held-out perplexity:

```bash
python3 src/eval.py \
  --checkpoint checkpoints/sisyphus.pt \
  --config config.yaml \
  --split val
```

Perplexity is useful for tracking training, but it is not enough on its own for code quality. For this repo, syntax validity and repetition behavior are still obvious missing evals.

## Hardware notes

You do not need a huge machine, but you do need to be realistic.

- CUDA is the intended path
- CPU training works in theory and is a bad use of time in practice
- the 4060 Ti runs are viable because the model is small and the trainer has guardrails around batch size, block size, and memory

The training script does a memory preflight and has config guardrails like:

- `max_batch_tokens`
- `max_eval_batch_tokens`
- `max_ram_utilization`
- `max_vram_utilization`
- `min_free_vram_bytes`

The point is not to be clever. The point is to fail early instead of dying 20 minutes into a run.

## FineWeb-Edu

FineWeb-Edu is optional and bounded on purpose.

Fetch it like this:

```bash
python3 src/fetch_fineweb_edu.py --config config.yaml
python3 src/build_corpus.py --config config.yaml
```

Important details:

- the configured dataset is `HuggingFaceFW/fineweb-edu`
- the valid split here is `train`
- first-time fetch requires outbound access to Hugging Face
- if network/DNS is broken, this step can hang and then fail during dataset resolution

If that happens, skip FineWeb and rebuild the corpus from the local and Rust-source inputs.

## Notes

- This repo is intentionally opinionated and a bit rough around the edges.
- Some comments and docs describe experiments that were later disabled, replaced, or only partially kept.
- If a doc and the code disagree, trust the code and the logs.
- If you want to understand the current training path, start with `src/model.py`, `src/train.py`, `config.20m.optimized.yaml`, and `logs/train.20m.optimized.log`.
