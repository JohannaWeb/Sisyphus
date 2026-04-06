# Crate Expansion & Retraining Plan

## Status

**In Progress**: Fetching top 500 crates from crates.io
- Command: `python3 src/fetch_top_crates.py --count 500`
- Output directory: `data/external/top-crates/`
- ETA: 1-2 hours (includes API queries with rate limiting + git cloning)

---

## Next Steps (After Crate Download)

### 1. Rebuild Corpus with Expanded Data
```bash
python3 src/build_corpus.py --config config.yaml
```

**Expected outcome:**
- Current corpus: ~31.3M tokens (2.18 epochs, 0.7:1 token:param ratio)
- New corpus: ~200M+ tokens (potentially Chinchilla-compliant 20:1 ratio)
- Size: 100-150MB corpus file

### 2. Prepare Longer-Sequence Config

Current config uses `block_size: 256` (256 tokens/sequence), which undersells learning opportunity. 
Proposed: `block_size: 512` for better Rust syntax/semantics context.

Already created: `config.20m.optimized.yaml` with:
- `block_size: 512`
- `batch_size: 12` (reduced to fit VRAM with longer sequences)
- `max_steps: 30000` (still ~4.2x compute budget in tokens)
- Matches 4060 Ti 8GB VRAM constraints

### 3. Train New Model
```bash
# Option A: From scratch with longer sequences
python3 src/train.py --config config.20m.optimized.yaml

# Option B: Resume from best checkpoint with new corpus
python3 src/train.py --config config.20m.optimized.yaml --resume checkpoints/sisyphus.pt
```

**Training metrics:**
- Previous: ~97.5k tok/s throughput = 23.8 steps/sec (3000 steps in ~2 min)
- New: Same throughput, longer sequences (512 tokens) = ~16 tokens/step instead of 8
- Compute budget: 30000 steps × 512 tokens × 16 batch tokens = 245.8M token-epoch
- With 200M corpus: ~1.2 epochs minimum training

### 4. Validate Quality Improvement

Generate samples and check for:
1. **Syntax correctness**: Rust functions compile without obvious errors
2. **Semantic coherence**: Function bodies match their signatures  
3. **Context awareness**: References to related types/traits in scope
4. **Longer patterns**: Multi-line impl blocks, trait definitions

Example generation:
```bash
python3 src/generate.py \
  --checkpoint checkpoints/sisyphus.pt \
  --prompt "fn new(" \
  --max-new-tokens 500 \
  --temperature 0.7
```

---

## Corpus Composition (Estimated)

| Source | Tokens | % | 
|--------|--------|---|
| Rust web book (official) | 15M | 7% |
| Rust compiler/stdlib | 80M | 40% |
| Top 500 crates (ecosystem) | 100M | 50% |
| Other (tokio, serde, etc.) | 5M | 3% |
| **Total** | **200M** | **100%** |

---

## Memory & Time Estimates

### Corpus Building (Single-threaded)
- Input: 500 repos + existing data
- Time: 30-60 minutes
- Peak RAM: ~8-12GB (to hold deduplicated content)
- Output: 100-150MB corpus file

### Model Training (4060 Ti 8GB)
- Config: `block_size=512`, `batch_size=12`, `max_steps=30000`
- Tokens per step: 6144 (12 batch × 512 seq)
- Throughput: ~97.5k tok/s = 15.9 steps/sec
- Total tokens: ~184M (30000 × 6144)
- Time: ~31 minutes per epoch, ~40-50 minutes total (assuming 1.2-1.5 epochs)

### Evaluation & Generation
- Perplexity eval: ~5-10 minutes
- Sample generation: ~30 seconds

---

## Known Constraints

1. **VRAM**: 8GB on 4060 Ti
   - Larger batch or longer sequences requires tradeoff
   - `block_size=768` would need `batch_size=6-8` 
   - Gradient checkpointing available if needed

2. **API Rate Limiting**: crates.io respects ~1 req/sec
   - Fetching 500 crates: ~500 metadata queries at 0.1s each = 50+ seconds
   - 500 git clones at ~10-30s each = 1.5-2.5 hours total

3. **Sequence Length Learning**: 
   - Longer sequences extract more signal from same corpus
   - But diminishing returns: 512→768 is probably 10-15% improvement, not 2x
   - Sweet spot: 512-768 for Rust code patterns

---

## Validation Checklist

- [ ] Fetch completes with 500+ crates cloned
- [ ] Corpus rebuilds without errors (~30-60 min)
- [ ] Corpus size is 150M+ tokens
- [ ] Model trains with new config (no OOM)
- [ ] Training reaches similar loss curve (not worse)
- [ ] Generated code shows improved syntax
- [ ] Sample diversity is higher

---

## Fallback Options

If corpus is too large or training is slow:
1. **Reduce crate count**: Use `--count 250` instead of 500
2. **Reduce sequence length**: Use `block_size=384` as compromise
3. **Reduce max_steps**: Train for 10000 steps instead of 30000
4. **Resume from checkpoint**: Use existing model + new corpus instead of from scratch

---

## Files Modified

- `config.yaml`: Added `data/external/top-crates` to extra_roots
- (Pending) `config.20m.optimized.yaml`: Template for new training run

