#!/usr/bin/env python3
"""Fetch top 500 crates by download count from crates.io via API."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    import requests
except ImportError:
    print("requests library required. Install with: pip install requests")
    sys.exit(1)


def get_top_crates_from_api(count: int = 500) -> list[tuple[str, str]]:
    """Get top N crates by download count from crates.io API.

    Returns list of (crate_name, repository_url) tuples.
    """
    crates_with_repos = []
    per_page = 100
    pages_needed = (count + per_page - 1) // per_page

    print(f"Fetching top {count} crates from crates.io API...")

    for page in range(1, pages_needed + 1):
        # Query crates sorted by downloads
        url = f"https://crates.io/api/v1/crates?sort=downloads&per_page={per_page}&page={page}"

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"  Error fetching page {page}: {e}")
            continue

        crates = data.get("crates", [])
        if not crates:
            break

        for crate in crates:
            if len(crates_with_repos) >= count:
                break

            crate_name = crate.get("name")
            # Get full crate info to get repository URL
            crate_info_url = f"https://crates.io/api/v1/crates/{crate_name}"

            try:
                crate_resp = requests.get(crate_info_url, timeout=10)
                crate_resp.raise_for_status()
                crate_data = crate_resp.json()
                crate_full = crate_data.get("crate", {})
                repo_url = crate_full.get("repository")

                if repo_url and repo_url.startswith("https://github.com"):
                    crates_with_repos.append((crate_name, repo_url))
                    print(f"  {len(crates_with_repos):3d}. {crate_name:30s} {crate.get('downloads', 0):10,d} downloads")

                # Rate limit: ~1 req/sec
                time.sleep(0.1)
            except Exception as e:
                # Skip crates we can't fetch
                continue

        print(f"  Page {page} done, {len(crates_with_repos)} crates with repos so far")

    return crates_with_repos[:count]


def checkout_crate(crate_name: str, repo_url: str, checkout_dir: Path) -> bool:
    """Clone a crate repository."""
    crate_path = checkout_dir / crate_name

    if crate_path.exists():
        return True

    print(f"  {crate_name:30s}...", end=" ", flush=True)

    try:
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                repo_url,
                str(crate_path),
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )
        print("✓")
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        print("✗")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch top 500 crates by download count from crates.io"
    )
    parser.add_argument(
        "--count", type=int, default=500, help="Number of top crates to fetch"
    )
    parser.add_argument(
        "--checkout-dir",
        type=Path,
        default=Path("data/external/top-crates"),
        help="Directory to clone crates into",
    )
    args = parser.parse_args()

    checkout_dir = args.checkout_dir
    checkout_dir.mkdir(parents=True, exist_ok=True)

    # Get crates with repository URLs
    top_crates = get_top_crates_from_api(args.count)

    if not top_crates:
        print("Failed to fetch crates from crates.io API")
        sys.exit(1)

    print(f"\nCloning {len(top_crates)} crates...")
    success_count = 0
    for i, (crate_name, repo_url) in enumerate(top_crates, 1):
        if checkout_crate(crate_name, repo_url, checkout_dir):
            success_count += 1
        if i % 50 == 0:
            print(f"Progress: {i}/{len(top_crates)} ({success_count} cloned)\n")

    print(f"\n✓ Completed: {success_count}/{len(top_crates)} crates cloned")
    print(f"✓ Cloned to: {checkout_dir.resolve()}")
    print(f"\nNext step: Add to config.yaml and run build_corpus.py")


if __name__ == "__main__":
    main()
