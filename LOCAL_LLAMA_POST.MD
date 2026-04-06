# Attention Is All You Need, But All You Can't Afford

I’ve been building a small Rust-focused language model from scratch in PyTorch and wanted to share the current state.

This is not a finetune. It is a byte-level model trained from random init on a Rust-heavy corpus assembled in this repo.

The run I’m talking about used:

- 25.6M parameters
- 512 context length
- 173.5M-byte corpus file
- 30k training steps
- single RTX 4060 Ti 8GB

Final logged metrics from `logs/train.20m.optimized.log`:

- train loss: 0.5834
- val loss: 0.8217
- perplexity: 2.15

For context: I'm an autistic systems programmer. I've been writing code since 2008/2009, started in C, and I tend to approach ML projects like systems projects first: understand the data path, understand the memory behavior, keep the stack small, and only add complexity when I can justify it. That is basically the shape of this repo.

## What is actually in the repo

After going back through the folder, the current project shape is roughly:

- corpus assembly in `src/build_corpus.py` plus the fetchers under `src/`
- model in `src/model.py`
- training loop in `src/train.py`
- eval script in `src/eval.py`
- the 25.6M / 512-context run configured in `config.20m.optimized.yaml`
- results summarized in `logs/train.20m.optimized.log` and `EXPANSION_RESULTS.md`

One important note: the current branch does **not** appear to contain Triton or `torch.library` custom op code. The real implementation in this folder is a pure PyTorch model with a custom hybrid attention block.

## Model

The model in `src/model.py` is a byte-level GPT-style decoder with:

- vocab size 256
- 8 layers
- 8 heads
- 512 embedding dim
- learned positional embeddings
- tied token embedding / LM head weights

The attention block is not standard full attention. Each block uses `HybridAttention`, which combines:

1. local windowed causal attention
2. a GRU-like recurrent state path
3. a learned gate that mixes the two

Very roughly:

```python
local_out = windowed_attention(q, k, v)
rnn_out = recurrent_state_scan(k, v, h_prev)
alpha = sigmoid(gate_proj(x))
y = alpha * local_out + (1 - alpha) * rnn_out
```

The local path is doing the normal short-range syntax work. The recurrent path is meant to carry some compressed long-range state without paying full quadratic attention everywhere.

The gate bias is initialized to ones, so early training starts biased toward the local path.

## Corpus

There are a few different corpus configs in the repo, and that matters because the default `config.yaml` is not the same thing as the run that produced the 25.6M / 512-context numbers.

The run in `config.20m.optimized.yaml` uses:

- official Rust docs repos
- Rust source/compiler/library/tests
- cargo
- rust-analyzer
- tokio / serde / ripgrep / clap / axum

The current `config.yaml` also includes:

- `data/external/top-crates`
- optional FineWeb-Edu shards

The logs and `EXPANSION_RESULTS.md` indicate the corpus was expanded from about 31M chars/tokens to 173.5M by adding the top crates fetch step:

```bash
python3 src/fetch_top_crates.py --count 500
```

Recorded result:

- 461 successful clones out of 492 attempts
- about 14GB on disk
- corpus file size 177,151,242 bytes

The broad conclusion from the repo history looks straightforward: corpus expansion helped more than model cleverness.

## Training setup

From `config.20m.optimized.yaml`:

- block size: 512
- batch size: 12
- eval batch size: 6
- learning rate: 2e-4
- weight decay: 0.1
- betas: (0.9, 0.95)
- max steps: 30000
- warmup: 1000
- eval interval: 500

The optional memory tricks in `src/train.py` exist, but for this run they were disabled:

- gradient quantization: false
- activation compression: false
- selective backprop: false
- sticky params: false
- gradient paging: false

So the interesting result here is not "a pile of exotic tricks made it fit." The interesting result is that a pretty small custom architecture plus mixed precision plus a better corpus was enough to get a plausible training run on a 4060 Ti.

The training log reports:

- estimated training memory ~678.8 MiB
- CUDA VRAM available 7.6 GiB total
- throughput around 96k tokens/sec in `EXPANSION_RESULTS.md`

## Loss curve

From `logs/train.20m.optimized.log`:

| Step | Train Loss | Val Loss |
|------|-----------|----------|
| 0 | 5.5555 | 5.5897 |
| 1000 | 2.4295 | 2.6365 |
| 5000 | 0.9051 | 1.0060 |
| 10000 | 0.8065 | 0.8723 |
| 18500 | 0.6902 | 0.7757 |
| 29999 | 0.5834 | 0.8217 |

Best val loss seems to have happened earlier than the final checkpoint, around step 18.5k.

So the project does seem to have a real training signal, but it also looks like it starts overfitting or at least stops improving cleanly late in the run.

## Generation quality

The samples in `EXPANSION_RESULTS.md` and the HN draft line up with what I’d expect from this scale:

```rust
fn span_path_segment(&self) -> &'static Segment {
    self.span_path_segment()
}
```

and:

```rust
use std::io::AsyncRead;
use std::pin::Pin;
use crate::crypto::cipher::Pin;
use crate::errors::{Error, ErrorKind, Result};
```

So:

- surface Rust syntax looks decent
- imports / signatures / derives can look plausible
- semantics are still weak
- repetition and recursive nonsense are still common

That seems like the honest read of the current state.

## What seems most interesting

To me, the most interesting part of this repo is not "look at my tiny Rust model."

It is that there are a few distinct experiments mixed together here:

1. A byte-level Rust-only pretraining setup.
2. A hybrid local-attention + recurrent block instead of standard full attention in each layer.
3. A corpus expansion from Rust docs/core repos to a broader crate ecosystem.

Of those, the clearest win in the repo history looks like the corpus expansion.

## What I would test next

If I keep pushing this project, the obvious next experiments are:

1. Ablate `HybridAttention` against plain causal self-attention at the same parameter count.
2. Check whether the best checkpoint at 18.5k is consistently better for generation than the final one.
3. Add syntax-validity evals for generated Rust, not just loss/perplexity.
4. Separate the effect of 512 context from the effect of the larger corpus.
5. Decide whether byte-level is still the right tradeoff once the corpus is large enough.

## Questions for LocalLLaMA people

1. For small code models, what evals have actually been useful for you beyond perplexity?
2. Have any of you seen hybrid local + recurrent attention work well for code generation, or does it usually lose to just scaling a plain transformer a bit?
3. If you had this exact setup, would you spend the next bit of compute on more tokens, longer context, or a cleaner baseline/ablation run first?
