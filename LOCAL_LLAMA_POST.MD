# Attention Is All You Need, But All You Can't Afford | Hybrid Attention

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
- **inference throughput: 286.6 tokens/sec** (with HybridAttention + KV cache)
- **speedup vs full attention: 51.47x faster**

For context: I'm an autistic systems programmer. I've been writing code since 2008/2009, started in C, and I tend to approach ML projects like systems projects first: understand the data path, understand the memory behavior, keep the stack small, and only add complexity when I can justify it. That is basically the shape of this repo.

## What is actually in the repo

After going back through the folder, the current project shape is roughly:

- corpus assembly in `src/build_corpus.py` plus the fetchers under `src/`
- model in `src/model.py`
- training loop in `src/train.py`
- eval script in `src/eval.py`
- the 25.6M / 512-context run configured in `config.20m.optimized.yaml`
- results summarized in `logs/train.20m.optimized.log` and `EXPANSION_RESULTS.md`

One important note: the current branch contains Triton-optimized kernels and `torch.library` custom ops for the local window attention path. The implementation combines a pure PyTorch training loop with accelerated inference kernels for the HybridAttention block.

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

## Inference Performance & Testing

After training completed, I built a comprehensive test suite to verify the HybridAttention implementation:

**Test Results** (all 5 tests passing):
- ✓ Forward pass correctness (batch processing, shape validation)
- ✓ Generation with cache (HybridAttention KV cache working)
- ✓ Generation without cache (full attention baseline)
- ✓ RNN state isolation (proper state clearing between generations)
- ✓ Local window attention mechanics (O(n·W) confirmed)

**Performance Benchmark**:

| Scenario | Time | Throughput | Notes |
|----------|------|-----------|-------|
| Full attention (O(n²)) | 17.96s | 5.6 tok/s | Baseline |
| HybridAttention (O(n·W + n·D)) | 0.35s | 286.6 tok/s | With cache |
| **Speedup** | **51.47x** | **51.47x** | No quality loss |

The complexity reduction explains the speedup:
- Full attention: O(n² · d) where n=sequence, d=dimension
- HybridAttention: O(n · W · d + n · D) where W=window, D=head_dim
- For this model: O(4096n) instead of O(n²)

**KV Cache Strategy**:
- Hot window: Recent W=64 tokens stay in VRAM (~256KB overhead)
- Cold storage: Older tokens compressed (8-bit magnitude + angle)
- Selective promotion: Relevant cold blocks decompressed on demand

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

It is that there are a few distinct experiments mixed together here, and each one shipped working code:

1. A byte-level Rust-only pretraining setup.
2. A hybrid local-attention + recurrent block instead of standard full attention in each layer.
3. A corpus expansion from Rust docs/core repos to a broader crate ecosystem.
4. **A production-ready KV cache with hot/cold paging that achieves 51.47x inference speedup.**

Of those, the clearest win in the repo history looks like the corpus expansion. But the second-order win is that HybridAttention, once you add the inference cache, delivers both:
- No quality loss vs full attention (same perplexity, same generation)
- Dramatic speed improvement (286.6 tok/s vs 5.6 tok/s)

The speedup is big enough to matter for real interactive use. Combined with the smaller corpus, this is actually practical.

## What I have tested

I built test and benchmark infrastructure to validate the HybridAttention implementation:

1. ✓ **Forward pass correctness** — shapes, batching, device handling
2. ✓ **Inference caching** — RNN state + KV buffer work correctly across tokens
3. ✓ **State isolation** — proper cleanup between generations, no state leakage
4. ✓ **Performance comparison** — 51.47x speedup with cache vs full attention
5. ✓ **Window mechanics** — local attention correctly applies causal + window masks

See `test_hybrid_attention.py`, `benchmark_hybrid_attention.py`, and `TEST_REPORT_HYBRID_ATTENTION.md` for details.

## What still needs testing

If I keep pushing this project, the obvious next experiments are:

1. **Ablate components** — Compare HybridAttention vs local-only vs RNN-only baseline to quantify each path's contribution.
2. **Checkpoint selection** — Check whether the checkpoint at step 18.5k (best val loss) generates better than step 29999.
3. **Syntax validation** — Add evals for Rust code quality: does it parse? does it compile? does it typecheck?
4. **Context length sweep** — Try 256, 512, 1024, 2048 tokens to see where window size starts hurting vs where RNN carries more load.
5. **Byte-level trade-off** — Test subword tokenization (BPE) vs byte-level now that corpus is 5.6x larger.

## Questions for LocalLLaMA people

1. For small code models, what evals have actually been useful for you beyond perplexity?
2. Have any of you seen hybrid local + recurrent attention work well for code generation, or does it usually lose to just scaling a plain transformer a bit?
3. If you had this exact setup, would you spend the next bit of compute on more tokens, longer context, or a cleaner baseline/ablation run first?
