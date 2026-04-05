# Sisyphus Run Commands

## Install dependencies

```bash
pip install -r requirements.txt
```

## Build the corpus

```bash
python3 src/fetch_rust_web_corpus.py --config config.yaml
python3 src/fetch_fineweb_edu.py --config config.yaml
python3 src/build_corpus.py --config config.yaml
```

## Train

```bash
python3 src/train.py --config config.yaml
```

## Generate text

```bash
python3 src/generate.py --checkpoint checkpoints/sisyphus.pt --prompt "fn main"
```

## Run the wrapper script

```bash
bash train_from_scratch.sh
```

## Evaluate

```bash
python3 src/eval.py --config config.yaml --checkpoint checkpoints/sisyphus.pt
```
