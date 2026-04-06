#!/usr/bin/env python3
"""Tests for model architecture and checkpoint functionality."""

import sys
import tempfile
import unittest
from pathlib import Path

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model import ByteGPT, GPTConfig


class TestModelArchitecture(unittest.TestCase):
    """Test ByteGPT model architecture."""

    def test_forward_pass(self):
        """Test model forward pass with valid input."""
        config = GPTConfig(
            vocab_size=256,
            block_size=32,
            n_layer=2,
            n_head=2,
            n_embd=64,
            dropout=0.1,
        )
        model = ByteGPT(config)

        # Random input: (batch_size=2, seq_len=32)
        batch_size, seq_len = 2, 32
        x = torch.randint(0, 256, (batch_size, seq_len))
        y = torch.randint(0, 256, (batch_size, seq_len))

        logits, loss = model(x, y)

        # Check shapes
        self.assertEqual(logits.shape, (batch_size, seq_len, 256))
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)

    def test_forward_no_targets(self):
        """Test model forward pass without targets (inference mode)."""
        config = GPTConfig(
            vocab_size=256,
            block_size=32,
            n_layer=2,
            n_head=2,
            n_embd=64,
        )
        model = ByteGPT(config)

        x = torch.randint(0, 256, (1, 32))
        logits, loss = model(x)

        self.assertEqual(logits.shape, (1, 32, 256))
        self.assertIsNone(loss)

    def test_checkpoint_roundtrip(self):
        """Test checkpoint save and load."""
        torch.manual_seed(42)
        config = GPTConfig(
            vocab_size=256,
            block_size=32,
            n_layer=2,
            n_head=2,
            n_embd=64,
        )
        model1 = ByteGPT(config)

        # Create dummy training data to test save/load
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "test.pt"

            # Save checkpoint
            torch.save(
                {
                    "step": 100,
                    "model_state_dict": model1.state_dict(),
                    "optimizer_state_dict": {},
                    "config": {"model": {
                        "vocab_size": 256,
                        "block_size": 32,
                        "n_layer": 2,
                        "n_head": 2,
                        "n_embd": 64,
                        "dropout": 0.1,
                    }},
                    "metrics": {"loss": 5.0},
                },
                checkpoint_path,
            )

            # Load checkpoint and reconstruct model
            checkpoint = torch.load(checkpoint_path)
            model2 = ByteGPT(GPTConfig(**checkpoint["config"]["model"]))
            model2.load_state_dict(checkpoint["model_state_dict"])

            # Verify both models produce same output (proves state_dict was loaded correctly)
            # Use eval mode to disable dropout
            model1.eval()
            model2.eval()
            x = torch.randint(0, 256, (1, 32))
            with torch.no_grad():
                logits1, _ = model1(x)
                logits2, _ = model2(x)
            self.assertTrue(torch.allclose(logits1, logits2, atol=1e-4))

    def test_generate(self):
        """Test text generation."""
        config = GPTConfig(
            vocab_size=256,
            block_size=32,
            n_layer=2,
            n_head=2,
            n_embd=64,
        )
        model = ByteGPT(config)
        model.eval()

        prompt = torch.tensor([[102, 110, 32]])  # "fn " in bytes
        max_new_tokens = 50

        with torch.no_grad():
            output = model.generate(prompt, max_new_tokens=max_new_tokens)

        # Check output shape
        expected_length = prompt.shape[1] + max_new_tokens
        self.assertEqual(output.shape[1], expected_length)

        # Check output is valid token IDs
        self.assertTrue(torch.all(output >= 0))
        self.assertTrue(torch.all(output < 256))


if __name__ == "__main__":
    unittest.main()
