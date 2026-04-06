#!/usr/bin/env python3
"""Tests for configuration loading and validation."""

import sys
import tempfile
import unittest
from pathlib import Path

import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model import GPTConfig


class TestConfigLoading(unittest.TestCase):
    """Test configuration loading and validation."""

    def test_load_valid_config(self):
        """Test loading the actual project config."""
        config_path = Path(__file__).resolve().parents[1] / "config.yaml"
        self.assertTrue(config_path.exists(), "config.yaml not found")

        with config_path.open("r") as f:
            config = yaml.safe_load(f)

        # Check required sections
        self.assertIn("data", config)
        self.assertIn("model", config)
        self.assertIn("training", config)
        self.assertIn("generation", config)

        # Check data section has required keys
        self.assertIn("root_path", config["data"])
        self.assertIn("output_dir", config["data"])
        self.assertIn("corpus_file", config["data"])

    def test_root_path_auto(self):
        """Test that root_path can be set to 'auto'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test_config.yaml"

            config = {
                "data": {
                    "root_path": "auto",
                    "root_label": "test",
                    "root_max_total_characters": 100000,
                    "output_dir": "data/processed",
                    "corpus_file": "corpus.txt",
                    "metadata_file": "corpus_metadata.json",
                    "max_file_bytes": 262144,
                    "max_chars_per_file": 20000,
                    "min_chars": 40,
                    "deduplicate_exact": True,
                    "allowed_extensions": [".txt", ".md"],
                    "excluded_dir_names": [".git"],
                    "excluded_paths": [],
                    "excluded_file_suffixes": [],
                    "extra_roots": [],
                },
                "model": {
                    "vocab_size": 256,
                    "block_size": 256,
                    "n_layer": 6,
                    "n_head": 6,
                    "n_embd": 384,
                    "dropout": 0.1,
                },
                "training": {
                    "seed": 1337,
                    "device": "cpu",
                    "checkpoint_dir": "checkpoints",
                    "checkpoint_name": "test.pt",
                    "batch_size": 8,
                    "learning_rate": 0.0003,
                    "max_steps": 100,
                },
                "generation": {
                    "max_new_tokens": 200,
                    "temperature": 0.9,
                    "top_k": 50,
                },
            }

            with config_path.open("w") as f:
                yaml.dump(config, f)

            # Load and verify
            with config_path.open("r") as f:
                loaded = yaml.safe_load(f)

            self.assertEqual(loaded["data"]["root_path"], "auto")

    def test_gptconfig_from_model_section(self):
        """Test that GPTConfig can be constructed from config model section."""
        config_path = Path(__file__).resolve().parents[1] / "config.yaml"
        self.assertTrue(config_path.exists())

        with config_path.open("r") as f:
            config = yaml.safe_load(f)

        model_cfg = config["model"]

        # Should not raise
        gpt_config = GPTConfig(**model_cfg)

        # Verify values
        self.assertEqual(gpt_config.vocab_size, 256)
        self.assertEqual(gpt_config.block_size, 256)
        self.assertEqual(gpt_config.n_layer, 6)
        self.assertEqual(gpt_config.n_head, 6)
        self.assertEqual(gpt_config.n_embd, 384)

    def test_gptconfig_with_extra_keys(self):
        """Test that GPTConfig ignores extra keys gracefully or raises."""
        # This should work - extra keys in model_cfg beyond GPTConfig fields
        model_cfg = {
            "vocab_size": 256,
            "block_size": 256,
            "n_layer": 6,
            "n_head": 6,
            "n_embd": 384,
            "dropout": 0.1,
            # These are extra and should cause TypeError from dataclass
        }
        config = GPTConfig(**model_cfg)
        self.assertEqual(config.vocab_size, 256)

        # But unknown fields should raise TypeError
        with self.assertRaises(TypeError):
            GPTConfig(vocab_size=256, unknown_field=True)


if __name__ == "__main__":
    unittest.main()
