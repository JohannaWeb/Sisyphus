#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
python3 src/fetch_rust_web_corpus.py --config config.yaml
python3 src/fetch_rust_code_corpus.py --config config.yaml
python3 src/build_corpus.py --config config.yaml
python3 src/train.py --config config.yaml
