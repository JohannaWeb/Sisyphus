# Agents

This is a standalone ML training project. No special agent configurations needed, but there are a few project-specific workflow details that matter when touching corpus generation and training.

For context on how this fits into the broader Bastion stack, see the parent memory files.

## Project shape

- Main training code lives in `src/`.
- Corpus inputs are assembled under `data/`.
- Training config is in `config.yaml`.
- Checkpoints are written under `checkpoints/`.

## Primary workflow

Build corpus inputs:

```bash
python3 src/fetch_rust_web_corpus.py --config config.yaml
python3 src/fetch_rust_code_corpus.py --config config.yaml
python3 src/fetch_fineweb_edu.py --config config.yaml   # optional, requires network
python3 src/build_corpus.py --config config.yaml
```

Alternatively, use the wrapper script (skips FineWeb):

```bash
bash train_from_scratch.sh
```

Train (from checkpoint or from scratch):

```bash
python3 src/train.py --config config.yaml
python3 src/train.py --config config.yaml --resume checkpoints/sisyphus.last.pt  # Resume training
```

Generate:

```bash
python3 src/generate.py --checkpoint checkpoints/sisyphus.pt --prompt "fn main"
```

## FineWeb-Edu notes

- FineWeb fetch logic lives in `src/fetch_fineweb_edu.py`.
- It depends on the Hugging Face `datasets` package from `requirements.txt`.
- It streams `HuggingFaceFW/fineweb-edu` using the config block at `web_corpus.fineweb_edu` in `config.yaml`.
- The valid split for the current dataset wiring is `train`. `sample-10BT` is not accepted by the current `datasets` loader path used here.
- It writes shard files to `data/external/fineweb-edu/shards/` and metadata to `data/external/fineweb-edu/metadata.json`.
- Re-running the fetcher deletes existing `fineweb_edu_*.txt` shard files before writing new ones.
- `train_from_scratch.sh` does not call the FineWeb fetcher. FineWeb is only included if `src/fetch_fineweb_edu.py` was run beforehand.

## Known FineWeb failure mode

- FineWeb fetch requires outbound access to Hugging Face on first resolution/fetch.
- If the config uses an invalid split, `datasets` raises `ValueError: Bad split ... Available splits: ['train']`.
- In restricted or offline environments, `python3 src/fetch_fineweb_edu.py --config config.yaml` can appear to hang and then fail during dataset resolution.
- A reproduced failure in this repo looked like:ANAL

```text
[Errno -2] Name or service not known
...
RuntimeError: Cannot send a request, as the client has been closed.
```

- If FineWeb "dies", check network/DNS access first before changing training code.
- If network is unavailable, skip the FineWeb step and rebuild the corpus from local and Rust web sources only.
