---
license: mit
language:
- en
tags:
- clinical-ai
- ehr
- medical
- next-event-prediction
- mimic-iv
- knowledge-distillation
- pytorch
library_name: pytorch
pipeline_tag: text-classification
---

# Cadence — Next Clinical Event Prediction

**cadence-core** is a pretrained neural model for next clinical event prediction from electronic health record (EHR) sequences. Given a patient's longitudinal clinical history, it predicts which of 48 clinical event categories will occur next and how many days until that event.

[![PyPI](https://img.shields.io/pypi/v/cadence-core.svg)](https://pypi.org/project/cadence-core/)
[![GitHub](https://img.shields.io/badge/GitHub-amirrouh%2Fcadence-blue)](https://github.com/amirrouh/cadence)
[![Paper](https://img.shields.io/badge/Paper-preprint-green)](https://amirrouh.github.io/cadence/)

## Model Description

Cadence implements the **Narrative Velocity Composite (NV-C)** framework — a 5.86M parameter residual MLP that fuses structured EHR features with PubMedBERT cluster-semantic embeddings under self-knowledge distillation.

| Component | Details |
|-----------|---------|
| **Architecture** | 3-block residual MLP with LayerNorm |
| **Input dimension** | 2,420 (884 NV features + 768 PubMedBERT mean + 768 PubMedBERT last) |
| **Classification head** | Linear → 48 event-class logits |
| **Regression head** | Linear → 19-bin discretized time-to-event |
| **Parameters** | 5.86M |
| **Training** | Cross-entropy + ordinal regression + self-knowledge distillation |

## Performance

Trained and evaluated on MIMIC-IV v3.1 (100k training tier, male cohort):

| Model | Top-1 Accuracy | 95% CI | MAE (days) | 95% CI |
|-------|:--------------:|--------|:----------:|--------|
| **Cadence (NV-C)** | **34.18%** | [33.84%, 34.42%] | **36.95** | [36.10, 37.68] |
| XGBoost-884 | 32.35% | — | 38.58 | — |
| Majority-class | 9.25% | — | — | — |
| Random | 2.08% | — | — | — |

Bootstrap 95% CI: N=105,968 test instances, 2,000 resamples. XGBoost falls outside Cadence's CI on both metrics.

**Key finding:** Self-knowledge distillation yields a disproportionately large top-1 gain (+0.81 pp) when applied after PubMedBERT cluster-semantic fusion — compared to ~0 pp gain from self-KD on structured features alone — suggesting a genuine interaction between frozen semantic embeddings and knowledge distillation.

## Installation & Usage

```bash
pip install cadence-core
```

```python
import torch
from cadence import CadenceModel, load_checkpoint

model = CadenceModel()
load_checkpoint(model)
model.eval()

# Input: 2420-dimensional feature vector
# [0:884]    — 884 Narrative Velocity (NV) clinical features
# [884:1652] — 768-dim PubMedBERT mean-pooled cluster-semantic embedding
# [1652:2420] — 768-dim PubMedBERT last-token cluster-semantic embedding
x = torch.randn(1, 2420)

with torch.no_grad():
    logits, time_bins = model(x)

event_probs = torch.softmax(logits, dim=-1)
top1_event  = event_probs.argmax(dim=-1).item()
```

## Training Data

Trained on MIMIC-IV v3.1 (PhysioNet credentialed access required):
- 100k patient sequences, male cohort
- 48 event categories derived from ICD-10 diagnosis and procedure codes
- External validation: 1,120 BWH patients (JSD=0.27 domain shift)

## Intended Use & Limitations

**Intended use:** Research on clinical event prediction, EHR sequence modelling, benchmarking.

**Not intended for:** Direct clinical decision-making without prospective validation.

**Limitations:**
- Trained on MIMIC-IV (US academic medical center); performance under distribution shift varies
- Structured features (labs, vitals, medications) required at inference; performance degrades when unavailable
- 48-class coarse event vocabulary; within-cluster distinctions are discarded
- Calibration analysis pending

## Citation

```bibtex
@article{rouhollahi2026cadence,
  title   = {Cadence: A Benchmark Evaluation of the Narrative Velocity Framework for Next Clinical Event Prediction in {MIMIC-IV}},
  author  = {Rouhollahi, Amir and Nezami, Farhad R.},
  year    = {2026},
  url     = {https://amirrouh.github.io/cadence/}
}
```

## License

MIT License. The pretrained checkpoint is provided for research use only.
MIMIC-IV data is subject to the [PhysioNet Credentialed Health Data License](https://physionet.org/content/mimiciv/view-license/3.1/).
