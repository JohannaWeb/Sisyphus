#!/usr/bin/env python3
"""Tests for corpus building logic."""

import sys
import tempfile
import unittest
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from build_corpus import should_skip, is_binary_bytes, normalize_for_dedupe


class TestCorpusBuilding(unittest.TestCase):
    """Test corpus building utilities."""

    def test_should_skip_excluded_dir(self):
        """Test that files in excluded directories are skipped."""
        data_cfg = {
            "excluded_dir_names": [".git", ".venv", "__pycache__"],
            "excluded_file_suffixes": [".pyc", ".o"],
            "allowed_extensions": [".py", ".md", ".txt"],
        }

        # Should skip files in .git
        path = Path("src/.git/config")
        self.assertTrue(should_skip(path, data_cfg))

        # Should skip files in __pycache__
        path = Path("src/__pycache__/module.pyc")
        self.assertTrue(should_skip(path, data_cfg))

    def test_should_skip_disallowed_extension(self):
        """Test that files with disallowed extensions are skipped."""
        data_cfg = {
            "excluded_dir_names": [],
            "excluded_file_suffixes": [".png", ".jpg", ".lock"],
            "allowed_extensions": [".py", ".md", ".txt"],
        }

        # Should skip .png files
        path = Path("image.png")
        self.assertTrue(should_skip(path, data_cfg))

        # Should skip .lock files
        path = Path("Cargo.lock")
        self.assertTrue(should_skip(path, data_cfg))

        # Should allow .md files
        path = Path("README.md")
        self.assertFalse(should_skip(path, data_cfg))

    def test_binary_detection(self):
        """Test binary file detection."""
        # Null bytes indicate binary
        binary_sample = b"Hello\x00World"
        self.assertTrue(is_binary_bytes(binary_sample))

        # High proportion of non-printable chars
        binary_sample = b"\x00\x01\x02\x03\x04\x05"
        self.assertTrue(is_binary_bytes(binary_sample))

        # Plain text should not be binary
        text_sample = b"Hello world\nLine 2\n"
        self.assertFalse(is_binary_bytes(text_sample))

        # Empty should not be binary
        self.assertFalse(is_binary_bytes(b""))

    def test_normalize_for_dedupe(self):
        """Test text normalization for deduplication."""
        # Single spaces, no blank lines
        text = """Line 1   with    extra spaces

Line 2  with    tabs
        and indentation"""

        normalized = normalize_for_dedupe(text)

        # Should have single spaces, no blank lines
        self.assertNotIn("  ", normalized)  # No double spaces
        self.assertNotIn("\n\n", normalized)  # No blank lines

        # Should be a single continuous string
        lines = normalized.split("\n")
        self.assertGreater(len(lines), 0)
        for line in lines:
            self.assertTrue(len(line) > 0)  # No empty lines

    def test_normalize_idempotent(self):
        """Test that normalize_for_dedupe is idempotent."""
        text = "Hello world"
        norm1 = normalize_for_dedupe(text)
        norm2 = normalize_for_dedupe(norm1)
        self.assertEqual(norm1, norm2)


if __name__ == "__main__":
    unittest.main()
