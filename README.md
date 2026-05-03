# cadence-core

[![PyPI version](https://img.shields.io/pypi/v/cadence-core.svg)](https://pypi.org/project/cadence-core/)
[![Python](https://img.shields.io/pypi/pyversions/cadence-core.svg)](https://pypi.org/project/cadence-core/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub release](https://img.shields.io/github/v/release/amirrouh/cadence?label=release)](https://github.com/amirrouh/cadence/releases/tag/v1.0.0)

**cadence-core** is a pretrained neural model for next clinical event prediction from electronic health record (EHR) sequences. Given a patient's longitudinal clinical history, it predicts which of 48 clinical event categories will occur next and how many days until that event.

**[Documentation & Paper →](https://amirrouh.github.io/cadence/)**

---

## Key Features

- **5.86M parameter residual MLP** — lightweight, fast inference, no GPU required
- **Trained on MIMIC-IV v3.1** — 100k patient sequences from a large academic medical center
- **Joint prediction** — simultaneous 48-class event classification and time-to-event regression
- **34.18% top-1 accuracy, 36.95 days MAE** — outperforms XGBoost and all evaluated baselines
- **Self-knowledge distillation** — improved generalization without external teacher models
- **Auto-downloads checkpoint** — model weights fetched from GitHub Releases on first use
- **Drop-in inference** — three lines of code from install to prediction

---

## Installation

```bash
pip install cadence-core
```

Requires Python 3.10+. No GPU needed for inference.

---

## Quick Start

```python
import torch
from cadence import CadenceModel, load_checkpoint

# Load model and pretrained weights (checkpoint auto-downloads on first run)
model = CadenceModel()
load_checkpoint(model)
model.eval()

# Input: 2420-dimensional feature vector per patient visit
# [0:884]    — 884 Narrative Velocity (NV) clinical features
# [884:1652] — 768-dim PubMedBERT mean-pooled cluster-semantic embedding
# [1652:2420] — 768-dim PubMedBERT last-token cluster-semantic embedding
x = torch.randn(1, 2420)  # batch_size=1, feature_dim=2420

with torch.no_grad():
    logits, time_bins = model(x)

# logits    : (batch, 48)  — classification logits over 48 event categories
# time_bins : (batch, 19)  — regression logits over 19 discretized time bins
event_probs = torch.softmax(logits, dim=-1)
top1_event  = event_probs.argmax(dim=-1).item()
print(f"Predicted next event class : {top1_event}")
print(f"Top-1 probability          : {event_probs.max().item():.3f}")
```

---

## Model Architecture

cadence-core implements the **Narrative Velocity Composite (NV-C)** framework — a residual MLP that fuses structured clinical features with contextual language embeddings.

| Component | Details |
|-----------|---------|
| **Input dimension** | 2420 (884 NV features + 768 PubMedBERT mean + 768 PubMedBERT last) |
| **Backbone** | 3-block MLP with residual skip connections and LayerNorm |
| **Classification head** | Linear → 48 event-class logits |
| **Regression head** | Linear → 19-bin discretized time-to-event logits |
| **Parameters** | 5.86M |
| **Training objective** | Cross-entropy (classification) + ordinal regression loss (time), with self-KD |

The 884 NV features capture structured clinical signals (labs, vitals, medications, procedures) encoded as narrative velocity trajectories. PubMedBERT embeddings are **cluster-semantic embeddings** — [`microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext) encodings of event-category labels (not raw clinical note text) — frozen at inference. Self-knowledge distillation applied after PubMedBERT cluster-semantic fusion yields a disproportionately large top-1 gain (+0.81 pp), substantially exceeding the gain from self-KD on structured features alone.

---

## Performance

### 100k Training Tier — Male Cohort (MIMIC-IV v3.1)

Results are 3-seed means with bootstrap 95% CIs. XGBoost falls outside Cadence's CI on both metrics.

| Model | Top-1 Accuracy | MAE (days) |
|-------|:--------------:|:----------:|
| **cadence-core (NV-C)** | **34.18%** [33.84%, 34.42%] | **36.95** [36.10, 37.68] |
| XGBoost | 32.35% | 38.58 |
| Random Forest | 24.1% | 53.2 |
| Logistic Regression | 21.3% | 58.7 |
| RETAIN (baseline) | 22.8% | 54.1 |
| Majority-class baseline | 9.25% | — |
| Random baseline | 2.08% | — |

![Cadence vs baselines — 100k training tier, MIMIC-IV v3.1](https://raw.githubusercontent.com/amirrouh/cadence/main/docs/static/figures/fig2_comparison.png)

### Full-Cohort Results (MIMIC-IV v3.1)

At full cohort, cadence-core leads all models on top-1 accuracy. FT-Transformer achieves the best MAE.

| Cohort | Model | Top-1 Accuracy | MAE (days) |
|--------|-------|:--------------:|:----------:|
| Male | **cadence-core (NV-C)** | **38.04%** | 29.39 |
| Male | FT-Transformer | — | **27.82** |
| Female | **cadence-core (NV-C)** | **35.66%** | 39.88 |
| Female | FT-Transformer | — | **37.08** |

### External Validation — BWH Dataset (1,120 patients)

External validation on de-identified records from Brigham and Women's Hospital (BWH) — a geographically and demographically distinct population with missing structured features and population shift. BWH events were LLM-extracted and mapped to the MIMIC-IV 48-cluster event schema.

| Model | Top-1 Accuracy |
|-------|:--------------:|
| RETAIN | **20.98%** (best overall) |
| **cadence-core (NV-C)** | **11.88%** (leads structured-feature models) |

Under domain shift with missing structured features, RETAIN achieves the best overall top-1 on BWH. Cadence leads among structured-feature models.

---

## Paper & Citation

> **Cadence: Next Clinical Event Prediction in MIMIC-IV (A Comparative Evaluation of the Narrative Velocity Framework Against Established Baselines)**
> Rouhollahi A. and Nezami F.R. — *preprint*, 2026

If you use cadence-core in your research, please cite:

```bibtex
@article{rouhollahi2026cadence,
  title   = {Cadence: Next Clinical Event Prediction in {MIMIC-IV} (A Comparative Evaluation
             of the Narrative Velocity Framework Against Established Baselines)},
  author  = {Rouhollahi, Amir and Nezami, Farhad R.},
  year    = {2026},
  url     = {https://amirrouh.github.io/cadence/}
}
```

---

## Reproducibility

Data access requires a signed PhysioNet credentialed account for MIMIC-IV:

```
https://physionet.org/content/mimiciv/3.1/
```

Once access is granted, follow the preprocessing instructions in `src/` to generate the NV feature sequences and PubMedBERT embeddings used for training.

---

## License

This project is released under the [MIT License](https://opensource.org/licenses/MIT). The pretrained model checkpoint is provided for research use only. MIMIC-IV data is subject to its own [PhysioNet Credentialed Health Data License](https://physionet.org/content/mimiciv/view-license/3.1/).

---

## Contact

**Amir Rouhollahi**
Brigham and Women's Hospital / Harvard Medical School
[arouhollahi@bwh.harvard.edu](mailto:arouhollahi@bwh.harvard.edu)
[GitHub](https://github.com/amirrouh/cadence) · [PyPI](https://pypi.org/project/cadence-core/)
