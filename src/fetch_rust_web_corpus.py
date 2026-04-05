#!/usr/bin/env python3
"""Fetch official Rust documentation repositories for corpus building."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def run_git(args: list[str], cwd: Path | None = None) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True)


def checkout_repo(base_dir: Path, repo_cfg: dict[str, str]) -> None:
    repo_dir = base_dir / repo_cfg["name"]
    branch = repo_cfg["branch"]
    url = repo_cfg["url"]

    if repo_dir.exists():
        if not (repo_dir / ".git").exists():
            print(f"Removing incomplete checkout at {repo_dir}")
            import shutil
            shutil.rmtree(repo_dir)
        else:
            try:
                run_git(["fetch", "--depth", "1", "origin", branch], cwd=repo_dir)
                run_git(["checkout", branch], cwd=repo_dir)
                run_git(["pull", "--ff-only", "origin", branch], cwd=repo_dir)
                return
            except subprocess.CalledProcessError:
                try:
                    run_git(["fetch", "--depth", "1", "origin", "main"], cwd=repo_dir)
                    run_git(["checkout", "main"], cwd=repo_dir)
                    return
                except subprocess.CalledProcessError:
                    print(f"Could not fetch updates for {repo_cfg['name']}, using existing checkout")
                    return

    try:
        run_git(["clone", "--depth", "1", "--branch", branch, url, str(repo_dir)])
    except subprocess.CalledProcessError:
        run_git(["clone", "--depth", "1", "--branch", "main", url, str(repo_dir)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch official Rust web corpus")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(config_path)

    rust_cfg = config["web_corpus"]["rust"]
    checkout_dir = Path(rust_cfg["checkout_dir"])
    if not checkout_dir.is_absolute():
        checkout_dir = (project_root / checkout_dir).resolve()
    checkout_dir.mkdir(parents=True, exist_ok=True)

    for repo_cfg in rust_cfg["repositories"]:
        checkout_repo(checkout_dir, repo_cfg)
        print(f"Fetched {repo_cfg['name']} into {checkout_dir / repo_cfg['name']}")


if __name__ == "__main__":
    main()
