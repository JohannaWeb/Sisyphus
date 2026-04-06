# Attention Is All You Need, But All You Can't Afford

I trained a 25.6M-parameter language model on Rust code on a single RTX 4060 Ti (8GB VRAM).

The two practical ideas were:

- expand the corpus from 31M to 173.5M tokens by adding popular crates from crates.io
- replace full attention with a hybrid local-window + recurrent path to keep training cheap

Current numbers:

- 25.6M parameters
- 173.5M training tokens
- 512-token context
- validation loss 0.82
- perplexity 2.15

Code is on the `change-transformer-arquitecture` branch.

For context: I'm an autistic systems programmer. I've been writing code since 2008/2009, started in C, and most of my instincts still come from low-level work, memory limits, and wanting to see the actual training code instead of stacking abstractions on top of it. That bias is part of why this project exists.

## Corpus

The initial corpus was:

- Rust book
- stdlib
- rustc-dev-guide

That was about 31M tokens, which looked too small.

I then pulled in the top 500 crates by download count:

```bash
python3 src/fetch_top_crates.py --count 500
```

That produced:

- 461 successful clones out of 492 attempts
- about 14GB on disk
- 173.5M total tokens

This helped more than any later architecture change.

## Model

The model uses a local attention path plus a recurrent path.

Local attention handles nearby syntax:

```python
scores = Q @ K^T[causal window]
```

The recurrent path carries compressed long-range state:

```python
r_t = sigmoid(Wr @ h_{t-1} + Ur @ k_t)
z_t = sigmoid(Wz @ h_{t-1} + k_t)
h_t = (1 - z_t) * h_prev + z_t * (tanh(candidate) * v_t)
```

The outputs are mixed with a learned gate:

```python
alpha = sigmoid(gate_proj(x))
y = alpha * local_out + (1 - alpha) * rnn_out
```

I biased the gate toward the local path at initialization because that was easier to train.

## Implementation

I implemented the custom ops with `torch.library` + Triton instead of writing a C++/CUDA extension.

That made it easier to:

- iterate on kernels
- keep autograd in Python
- work with `torch.compile`

For a project this size, that was simpler than maintaining a PyTorch fork or extension build.

## Training

Training ran for 30k steps over about 184M tokens.

| Step | Train Loss | Val Loss |
|------|-----------|----------|
| 0 | 5.56 | 5.59 |
| 1k | 2.43 | 2.64 |
| 5k | 0.91 | 0.97 |
| 10k | 0.80 | 0.84 |
| 20k | 0.62 | 0.84 |
| 30k | 0.58 | 0.82 |

The model gets Rust syntax and local structure reasonably well. It is much weaker on semantic consistency.

Example:

```rust
fn span_path_segment(&self) -> &'static Segment {
    self.span_path_segment()
}
```

This is syntactically plausible Rust, but semantically not very useful.

Another sample:

```rust
use std::io::AsyncRead;
use std::pin::Pin;
use crate::crypto::cipher::Pin;
use crate::errors::{Error, ErrorKind, Result};
```

Again, structurally plausible, but not something I would trust.

## What Helped

The main takeaways from this run:

1. Corpus size and corpus choice mattered more than model changes.
2. 512-token context was noticeably better than 256 for Rust code.
3. `torch.library` + Triton was a practical way to build custom ops without much tooling overhead.

## What Still Needs Work

The obvious next steps are:

1. Run an ablation against a local-only baseline.
2. Try longer context lengths.
3. Add better evaluation than loss/perplexity.
4. Test whether the current corpus is enough to justify a larger model.

If people are interested, I can post more detail on the corpus pipeline or the Triton kernels.
