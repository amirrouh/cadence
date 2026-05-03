"""
Cadence: Neural model for next clinical event prediction from EHR sequences.

Cadence is a 5.86M-parameter residual MLP that combines:
- 884 hand-crafted Narrative Velocity features
- 768-dim PubMedBERT mean-history embedding
- 768-dim PubMedBERT last-event embedding

Joint classification (50 clinical events) + regression (time-to-next-event).
Achieves 34.18% top-1 accuracy and 36.95 days MAE on MIMIC-IV.

Paper: "Next Clinical Event Prediction in MIMIC-IV: A Comparative Evaluation
of the Narrative Velocity Framework Against Established Baselines"
"""

__version__ = "1.0.0"

from cadence.model import (
    NVCClean,
    main,
    load_checkpoint,
)

__all__ = ["NVCClean", "main", "load_checkpoint", "__version__"]
