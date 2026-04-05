# Sisyphus

ML project for training a language model from scratch on Rust web documentation.

## Project Structure

- `src/` — Python: `train.py`, `model.py`, `generate.py`, corpus builders
- `data/` — Training data (Rust web book)
- `checkpoints/` — Model checkpoints
- `config.yaml` — Training configuration
- `train_from_scratch.sh` — Training script

## Key Commands

```bash
# Train the model
bash train_from_scratch.sh

# Generate text
python src/generate.py
```

## Notes

- Uses PyTorch for the transformer model
- Corpus built from Rust web book via `fetch_rust_web_corpus.py`