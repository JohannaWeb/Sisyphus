#!/usr/bin/env python3
"""Build a text corpus from local project files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def is_binary_bytes(sample: bytes) -> bool:
    if not sample:
        return False
    if b"\x00" in sample:
        return True
    non_printable = sum(
        1 for byte in sample if byte < 9 or (13 < byte < 32) or byte == 127
    )
    return (non_printable / len(sample)) > 0.30


def should_skip(path: Path, data_cfg: dict[str, Any]) -> bool:
    excluded_dirs = set(data_cfg["excluded_dir_names"])
    excluded_suffixes = tuple(data_cfg["excluded_file_suffixes"])
    allowed_extensions = set(data_cfg["allowed_extensions"])

    if any(part in excluded_dirs for part in path.parts):
        return True
    if path.suffix.lower() in excluded_suffixes:
        return True
    if path.suffix and path.suffix.lower() not in allowed_extensions:
        return True
    return False


def resolve_optional_paths(
    path_values: list[str], project_root: Path
) -> list[Path]:
    resolved: list[Path] = []
    for value in path_values:
        path = Path(value)
        if path.is_absolute():
            resolved.append(path.resolve())
        else:
            resolved.append((project_root / path).resolve())
    return resolved


def is_under_any(path: Path, roots: list[Path]) -> bool:
    for root in roots:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def collect_files(
    root_path: Path, data_cfg: dict[str, Any], excluded_paths: list[Path]
) -> list[Path]:
    files: list[Path] = []
    excluded_dirs = set(data_cfg["excluded_dir_names"])

    for path in root_path.rglob("*"):
        if not path.is_file():
            continue
        if is_under_any(path.resolve(), excluded_paths):
            continue
        if any(part in excluded_dirs for part in path.parts):
            continue
        if should_skip(path, data_cfg):
            continue
        files.append(path)

    return sorted(files)


def resolve_root(path_value: str, project_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()
    return (project_root / path).resolve()


def build_source_specs(data_cfg: dict[str, Any], project_root: Path) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    root_path_raw = data_cfg["root_path"]
    if root_path_raw == "auto":
        root_path = project_root
    else:
        root_path = resolve_root(root_path_raw, project_root)
    specs.append(
        {
            "label": data_cfg.get("root_label", "local-projects"),
            "path": root_path,
            "max_total_characters": data_cfg.get("root_max_total_characters"),
            "max_files": data_cfg.get("root_max_files"),
        }
    )
    for extra_root in data_cfg.get("extra_roots", []):
        specs.append(
            {
                "label": extra_root["label"],
                "path": resolve_root(extra_root["path"], project_root),
                "max_total_characters": extra_root.get("max_total_characters"),
                "max_files": extra_root.get("max_files"),
            }
        )
    return specs


def read_text(path: Path, max_file_bytes: int) -> str | None:
    try:
        size = path.stat().st_size
        if size == 0 or size > max_file_bytes:
            return None

        raw = path.read_bytes()
        if is_binary_bytes(raw[:4096]):
            return None

        return raw.decode("utf-8", errors="ignore").replace("\r\n", "\n").replace("\r", "\n").strip()
    except OSError:
        return None


def normalize_for_dedupe(text: str) -> str:
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def build_corpus(config_path: Path) -> None:
    config = load_config(config_path)
    data_cfg = config["data"]
    project_root = Path(__file__).resolve().parents[1]
    excluded_paths = resolve_optional_paths(data_cfg.get("excluded_paths", []), project_root)

    source_specs = build_source_specs(data_cfg, project_root)

    output_dir_setting = Path(data_cfg["output_dir"])
    output_dir = (
        output_dir_setting
        if output_dir_setting.is_absolute()
        else (project_root / output_dir_setting).resolve()
    )
    corpus_path = output_dir / data_cfg["corpus_file"]
    metadata_path = output_dir / data_cfg["metadata_file"]
    output_dir.mkdir(parents=True, exist_ok=True)

    included = 0
    skipped_short = 0
    skipped_unreadable = 0
    skipped_duplicate = 0
    truncated_files = 0
    total_chars = 0
    total_scanned = 0
    extension_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    seen_hashes: set[str] = set()
    max_chars_per_file = int(data_cfg.get("max_chars_per_file", 0) or 0)
    deduplicate_exact = bool(data_cfg.get("deduplicate_exact", False))

    with corpus_path.open("w", encoding="utf-8") as corpus_handle:
        for source_spec in source_specs:
            source_label = source_spec["label"]
            source_root = source_spec["path"]
            if not source_root.exists():
                continue

            active_excluded_paths = (
                []
                if is_under_any(source_root.resolve(), excluded_paths)
                else excluded_paths
            )
            files = collect_files(source_root, data_cfg, active_excluded_paths)
            total_scanned += len(files)
            source_char_limit = source_spec.get("max_total_characters")
            source_file_limit = source_spec.get("max_files")
            source_chars = 0
            source_files = 0
            for path in files:
                if source_file_limit is not None and source_files >= int(source_file_limit):
                    break
                text = read_text(path, data_cfg["max_file_bytes"])
                if text is None:
                    skipped_unreadable += 1
                    continue
                if len(text) < data_cfg["min_chars"]:
                    skipped_short += 1
                    continue
                if max_chars_per_file and len(text) > max_chars_per_file:
                    text = text[:max_chars_per_file].rstrip()
                    truncated_files += 1
                if source_char_limit is not None:
                    remaining = int(source_char_limit) - source_chars
                    if remaining < data_cfg["min_chars"]:
                        break
                    if len(text) > remaining:
                        text = text[:remaining].rstrip()
                        if len(text) < data_cfg["min_chars"]:
                            break
                if deduplicate_exact:
                    fingerprint = hashlib.sha256(normalize_for_dedupe(text).encode("utf-8")).hexdigest()
                    if fingerprint in seen_hashes:
                        skipped_duplicate += 1
                        continue
                    seen_hashes.add(fingerprint)

                relative = path.relative_to(source_root)
                header = f"<FILE source=\"{source_label}\" path=\"{relative}\">\n"
                footer = "\n</FILE>\n\n"
                corpus_handle.write(header)
                corpus_handle.write(text)
                corpus_handle.write(footer)

                included += 1
                total_chars += len(text)
                source_chars += len(text)
                source_files += 1
                suffix = path.suffix.lower() or "<no_ext>"
                extension_counts[suffix] = extension_counts.get(suffix, 0) + 1
                source_counts[source_label] = source_counts.get(source_label, 0) + 1

    metadata = {
        "root_path": str(source_specs[0]["path"]) if source_specs else "",
        "source_roots": {
            spec["label"]: str(spec["path"])
            for spec in source_specs
            if spec["path"].exists()
        },
        "corpus_path": str(corpus_path),
        "files_scanned": total_scanned,
        "files_included": included,
        "files_skipped_unreadable_or_large": skipped_unreadable,
        "files_skipped_too_short": skipped_short,
        "files_skipped_duplicate": skipped_duplicate,
        "files_truncated": truncated_files,
        "total_characters": total_chars,
        "extension_counts": dict(sorted(extension_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "excluded_paths": [str(path) for path in excluded_paths],
        "deduplicate_exact": deduplicate_exact,
        "max_chars_per_file": max_chars_per_file,
    }

    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Corpus written to {corpus_path}")
    print(f"Metadata written to {metadata_path}")
    print(f"Included {included} files with {total_chars} characters")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build local training corpus")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config")
    args = parser.parse_args()
    build_corpus(Path(args.config).resolve())


if __name__ == "__main__":
    main()
