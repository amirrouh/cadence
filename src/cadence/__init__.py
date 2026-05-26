"""
Cadence: Neural model for next clinical event prediction from EHR sequences.

Cadence is a 5.86M-parameter residual MLP that combines:
- Hand-crafted Narrative Velocity features
- Per-event embedding mean (any sentence encoder: PubMedBERT, BERT, etc.)
- Per-event embedding last-event signal

Joint classification (next clinical event cluster) + regression (days to next event).
Achieves 34.18% top-1 accuracy and 36.95 days MAE on MIMIC-IV (paper checkpoint).

Public training API (v1.1.0): supply your own JSONL data and per-event embeddings.
"""

__version__ = "1.1.0"

from cadence.model import NVCClean, load_checkpoint, main
from cadence.features import build_feature_matrix, build_population_prior
from cadence.data import load_embeddings, validate_jsonl
from cadence.train import train
from cadence.inference import predict

__all__ = [
    "NVCClean",
    "load_checkpoint",
    "main",
    "build_feature_matrix",
    "build_population_prior",
    "load_embeddings",
    "validate_jsonl",
    "train",
    "predict",
    "__version__",
]
