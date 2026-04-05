#!/usr/bin/env python3
"""Stream a bounded FineWeb-Edu sample into local text shards."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import yaml


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines).strip()


def write_shard(
    shard_dir: Path,
    shard_index: int,
    documents: list[str],
    character_count: int,
) -> Path:
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shard_dir / f"fineweb_edu_{shard_index:05d}.txt"
    with shard_path.open("w", encoding="utf-8") as handle:
        for doc_index, text in enumerate(documents):
            handle.write(f"<FILE source=\"fineweb-edu\" path=\"doc-{shard_index:05d}-{doc_index:04d}\">\n")
            handle.write(text)
            handle.write("\n</FILE>\n\n")
    print(f"Wrote {shard_path} ({character_count} chars, {len(documents)} docs)")
    return shard_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch bounded FineWeb-Edu shards")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: install the 'datasets' package to fetch FineWeb-Edu."
        ) from exc

    config_path = Path(args.config).resolve()
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(config_path)
    fw_cfg = config["web_corpus"]["fineweb_edu"]

    output_dir = Path(fw_cfg["output_dir"])
    if not output_dir.is_absolute():
        output_dir = (project_root / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = fw_cfg["dataset"]
    split_name = fw_cfg["split"]
    streaming = bool(fw_cfg.get("streaming", True))
    max_documents = int(fw_cfg["max_documents"])
    max_total_characters = int(fw_cfg["max_total_characters"])
    min_document_characters = int(fw_cfg["min_document_characters"])
    max_document_characters = int(fw_cfg["max_document_characters"])
    shard_max_characters = int(fw_cfg["shard_max_characters"])

    try:
        dataset = load_dataset(dataset_name, split=split_name, streaming=streaming)
    except ValueError as exc:
        message = str(exc)
        if "Bad split:" in message and "Available splits:" in message:
            raise SystemExit(
                "Invalid FineWeb-Edu split in config.yaml: "
                f"{split_name!r}. {message}"
            ) from exc
        raise

    for existing in output_dir.glob("fineweb_edu_*.txt"):
        existing.unlink()

    shard_documents: list[str] = []
    shard_characters = 0
    shard_index = 0

    kept_documents = 0
    skipped_short = 0
    total_characters = 0
    written_shards = 0

    for record in dataset:
        text = normalize_text(record.get("text", ""))
        if len(text) < min_document_characters:
            skipped_short += 1
            continue
        if len(text) > max_document_characters:
            text = text[:max_document_characters].rstrip()

        if not text:
            continue

        if total_characters + len(text) > max_total_characters:
            remaining = max_total_characters - total_characters
            if remaining < min_document_characters:
                break
            text = text[:remaining].rstrip()
            if len(text) < min_document_characters:
                break

        projected_shard_size = shard_characters + len(text)
        if shard_documents and projected_shard_size > shard_max_characters:
            write_shard(output_dir, shard_index, shard_documents, shard_characters)
            written_shards += 1
            shard_documents = []
            shard_characters = 0
            shard_index += 1

        shard_documents.append(text)
        shard_characters += len(text)
        total_characters += len(text)
        kept_documents += 1

        if kept_documents >= max_documents or total_characters >= max_total_characters:
            break

    if shard_documents:
        write_shard(output_dir, shard_index, shard_documents, shard_characters)
        written_shards += 1

    metadata = {
        "dataset": dataset_name,
        "split": split_name,
        "streaming": streaming,
        "max_documents": max_documents,
        "max_total_characters": max_total_characters,
        "min_document_characters": min_document_characters,
        "max_document_characters": max_document_characters,
        "shard_max_characters": shard_max_characters,
        "documents_kept": kept_documents,
        "documents_skipped_too_short": skipped_short,
        "total_characters_written": total_characters,
        "shards_written": written_shards,
    }
    metadata_path = output_dir.parent / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Metadata written to {metadata_path}")
    print(
        f"Kept {kept_documents} FineWeb-Edu documents "
        f"for {total_characters} characters across {written_shards} shards"
    )


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except SystemExit as exc:
        code = exc.code
        exit_code = code if isinstance(code, int) else 1
        raise
    except Exception:
        exit_code = 1
        raise
    finally:
        if exit_code == 0:
            # Work around a native teardown crash observed after successful
            # streaming fetches on Python 3.12 with the current datasets stack.
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)
