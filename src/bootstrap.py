"""Process settings that must exist before machine-learning libraries import."""

from __future__ import annotations

import os


def configure_environment() -> None:
    """Turn off unused frameworks and anonymous telemetry.

    TensorFlow is installed in this environment, and importing it from
    Transformers adds a long startup with no benefit for these PyTorch models.
    """

    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
