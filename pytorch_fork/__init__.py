"""PyTorch fork: custom aten ops for Sisyphus via torch.library."""

# Import to trigger registration
from . import local_window_attn  # noqa: F401
from . import gated_rnn_update  # noqa: F401

# All ops are now available as torch.ops.sisyphus.*
