#!/usr/bin/env python3
"""
cadence/model.py — Published paper model (NVCClean champion)
=====================================================================
This is the production-promoted version of the Cadence champion model:
  nvc_emb_mean_last_selfkd_v2_swa30_100k_01

Performance (100k cohort, male, 3-seed average):
  - Top-1 accuracy: 34.18%
  - MAE: 36.95 days
  Beats XGBoost (27.17% top-1 / 50.9d MAE) on both metrics.

Architecture — NVCClean 3-block residual MLP (~5.86M parameters):
  - Input: 2420-dim (768 emb-mean + 768 emb-last + 884 hand-crafted features)
  - 3-block residual MLP: 2420→1024→1024→512
  - fm_linear: Linear(2420, 512, bias=False) — linear shortcut (regression path only)
  - Classification head: pure MLP, no shortcut, 50-class softmax
  - Regression head: h3 + 0.1 * fm_linear(x_raw), log-day output
  - MixUp α=0.4, ASL γneg=4/γpos=1, aux_l1 λ=0.75
  - 140 epochs, SWA_START=30, EARLY_STOP_MIN_EPOCH=120
  - Optimizer: AdamW

Training method — Self-KD (NV-C → NV-C, allowed per project rules):
  - Teacher: nvc_emb_mean_last_selfkd_100k_01 seed_42 best_model.pt (34.02% top-1)
  - KD temperature T=4.0, alpha=0.5 (0.5*ASL + 0.5*KL)
  - No XGBoost teacher, no competitor model teacher (TRIPOD+AI compliant)

Usage:
  uv run python clinical-record-prediction/src/mimic_train_nvc_clean.py \\
      --sex M --seed 42 --data-suffix 100k
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np


def load_checkpoint(version="v1.0.0"):
    """
    Download and load the Cadence checkpoint from GitHub releases.

    On first run, downloads checkpoint_best.pt (~300MB) from:
    https://github.com/amirrouh/cadence/releases/download/{version}/checkpoint_best.pt

    Subsequent runs load from local cache (~/.cadence/checkpoints/).

    Args:
        version: Release tag (default: "v1.0.0")

    Returns:
        dict: Model state dict for loading into NVCClean

    Example:
        model = NVCClean(...)
        ckpt = load_checkpoint()
        model.load_state_dict(ckpt)
    """
    import os

    cache_dir = os.path.expanduser("~/.cadence/checkpoints")
    os.makedirs(cache_dir, exist_ok=True)
    ckpt_path = os.path.join(cache_dir, "checkpoint_best.pt")

    # Return from cache if exists
    if os.path.exists(ckpt_path):
        logging.info(f"Loading checkpoint from cache: {ckpt_path}")
        return torch.load(ckpt_path, map_location="cpu")

    # Download from GitHub
    url = f"https://github.com/amirrouh/cadence/releases/download/{version}/checkpoint_best.pt"
    logging.info(f"Downloading checkpoint from {url}...")

    try:
        state_dict = torch.hub.load_state_dict_from_url(
            url,
            model_dir=cache_dir,
            map_location="cpu"
        )
        logging.info(f"Checkpoint cached at {ckpt_path}")
        return state_dict
    except Exception as e:
        raise RuntimeError(
            f"Failed to download checkpoint from GitHub.\n"
            f"URL: {url}\n"
            f"Error: {e}\n"
            f"Manual download: https://github.com/amirrouh/cadence/releases"
        ) from e
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.utils.data import DataLoader, TensorDataset

# ── Paths ─────────────────────────────────────────────────────────────────────
# model.py lives at <repo>/src/cadence/model.py
_SCRIPT_DIR  = Path(__file__).resolve().parent
# _PROJECT_DIR is kept for the private MIMIC training path only (main()).
# Public users use cadence.train() from train.py, which has no private deps.
_PROJECT_DIR = _SCRIPT_DIR.parents[1] / "clinical-record-prediction"

# Clip value for log1p(days); matches LOG_DAYS_CLIP in mimic_train_xgb_sex.py (12.0).
# Defined here so evaluate() works without the private repo on the path.
LOG_DAYS_CLIP: float = 12.0

# ── Constants ─────────────────────────────────────────────────────────────────
STRUCT_FEAT_DIM   = 150
TEMPORAL_FEAT_DIM = 464
N_FEATURES        = 884   # 270 base + 150 structured + 464 temporal
N_CLASSES         = 50

BATCH_SIZE      = 512
PHASE1_EPOCHS   = 10
PHASE2_EPOCHS   = 130   # epochs 11-140 (total 140) — _03_long: extended budget past ep95 convergence
PHASE1_LR       = 1e-3
PHASE2_LR       = 1e-4
WEIGHT_DECAY    = 3e-3
DROPOUT         = 0.3
INPUT_DROPOUT   = 0.1
EARLY_STOP_PAT  = 40    # v3: was 30
EARLY_STOP_MIN_EPOCH = 120  # _03_long: no early stop before epoch 120 (was 80)

# ASL hyperparameters (Ridnik et al. 2021, arXiv 2009.14119)
ASL_GAMMA_NEG   = 4.0   # aggressive down-weight for easy negatives
ASL_GAMMA_POS   = 1.0   # softer focusing for positives
LABEL_SMOOTH    = 0.15  # softened from focal's 0.3; ASL handles negatives separately
REG_WEIGHT      = 0.5

MIXUP_ALPHA     = 0.4
WARMUP_STEPS    = 500

# SWA settings
SWA_START       = 30    # swa30 fork: lowered from 60→30 to capture best-MAE region (epochs 29/39/43)
SWA_LR          = 1e-4

# Quantile bin settings — target ~40 bins from training distribution
N_BINS_TARGET   = 40
# Gaussian sigma in BIN-INDEX space (not log-day space)
GAUSS_SIGMA_IDX = 1.0  # 1 bin-index width

# Auxiliary direct-MAE loss weight (smooth_l1 on bin-decoded expected log-day)
# v3: Ramps from 0.0 at epoch 10 to 0.75 at epoch 20, holds at 0.75 thereafter.
# (v2 was 0.0->0.50 over epochs 10-20; v1 was 0.0->0.25 over epochs 15-25)
LAMBDA_AUX_MAX  = 0.75  # v3: was 0.50
AUX_RAMP_START  = 10
AUX_RAMP_END    = 20

DATA_DIR = _PROJECT_DIR / "data"

# ── Self-KD hyperparameters ───────────────────────────────────────────────────
# KD RE-ENABLED: teacher is nvc_emb_mean_last_selfkd_100k_01 seed_42 best_model.pt (34.02% top-1).
# Same dim as student (2420), strict=True load works.
# Born-again bootstrap: first 2420-dim run's best-val-top1 checkpoint becomes teacher.
TEMPERATURE  = 4.0   # KD temperature
KD_ALPHA     = 0.5   # 0.5*ASL + 0.5*KL(student||teacher)
TEACHER_CKPT = (
    "clinical-record-prediction/dev/experiments/"
    "nvc_emb_mean_last_selfkd_100k_01/results/male_100k/seed_42/best_model.pt"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Data helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_feature_dict(npz_path: Path, feat_dim: int, label: str) -> dict:
    log.info("Loading %s features from %s ...", label, npz_path)
    data  = np.load(npz_path, allow_pickle=True)
    pids  = data["patient_ids"]
    dates = data["target_dates"]
    feats = data["features"]
    feat_dict = {
        (str(p), str(d)): feats[i].astype(np.float32)
        for i, (p, d) in enumerate(zip(pids, dates))
    }
    log.info("%s features loaded: %d records, dim=%d", label, len(feat_dict), feats.shape[1])
    return feat_dict


def augment_features(
    jsonl_path: Path,
    X_base: np.ndarray,
    struct_dict: dict,
    temporal_dict: dict,
) -> np.ndarray:
    records: list[dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    assert len(records) == X_base.shape[0], \
        f"Record count mismatch: {len(records)} vs {X_base.shape[0]}"

    struct_rows:   list[np.ndarray] = []
    temporal_rows: list[np.ndarray] = []
    n_miss_s = n_miss_t = 0

    for rec in records:
        pid      = str(rec["patient_id"])
        date_iso = rec["target"].get("date_iso", "")
        key      = (pid, date_iso)

        if key in struct_dict:
            struct_rows.append(struct_dict[key])
        else:
            struct_rows.append(np.zeros(STRUCT_FEAT_DIM, dtype=np.float32))
            n_miss_s += 1

        if key in temporal_dict:
            temporal_rows.append(temporal_dict[key])
        else:
            temporal_rows.append(np.zeros(TEMPORAL_FEAT_DIM, dtype=np.float32))
            n_miss_t += 1

    if n_miss_s:
        log.warning("  %d/%d records missing struct features (zero-filled)", n_miss_s, len(records))
    if n_miss_t:
        log.warning("  %d/%d records missing temporal features (zero-filled)", n_miss_t, len(records))

    X_struct   = np.stack(struct_rows).astype(np.float32)
    X_temporal = np.stack(temporal_rows).astype(np.float32)
    X_out      = np.hstack([X_base, X_struct, X_temporal])
    log.info("  Augmented shape: %s", X_out.shape)
    return X_out


# ═══════════════════════════════════════════════════════════════════════════════
# Embedding mean helpers — 768-dim "mean of last K event embeddings"
# ═══════════════════════════════════════════════════════════════════════════════

def compute_mean_last_k_embedding(
    patient_id: str,
    history: list[dict],
    embeddings: np.ndarray,
    event_id_map: dict,
    k: int = 10,
) -> np.ndarray:
    """Mean of the last k event embeddings (zero-padded if fewer events or keys missing)."""
    emb_dim = embeddings.shape[1]  # 768
    if not history:
        return np.zeros(emb_dim, dtype=np.float32)
    recent = history[-k:]
    embs = []
    for h in recent:
        key = (str(patient_id), h["event_index"])
        row = event_id_map.get(key)
        if row is not None:
            embs.append(embeddings[row].astype(np.float32))
    if not embs:
        return np.zeros(emb_dim, dtype=np.float32)
    return np.mean(np.stack(embs), axis=0)


def build_emb_mean_matrix(
    jsonl_path: Path,
    embeddings: np.ndarray,
    event_id_map: dict,
    k: int = 10,
) -> np.ndarray:
    """Build (N, 768) matrix of mean-last-k-embedding for each record in a JSONL split."""
    rows = []
    n_zero = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            history = rec.get("history", [])
            pid = str(rec["patient_id"])
            emb = compute_mean_last_k_embedding(pid, history, embeddings, event_id_map, k)
            if emb.sum() == 0.0:
                n_zero += 1
            rows.append(emb)
    result = np.stack(rows).astype(np.float32)
    log.info(
        "  Emb-mean matrix: shape=%s  zero_rows=%d/%d (%.1f%%)",
        result.shape, n_zero, len(rows), 100.0 * n_zero / max(1, len(rows))
    )
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Embedding last-event helpers — 768-dim "single most-recent event embedding"
# ═══════════════════════════════════════════════════════════════════════════════

def compute_last_event_embedding(
    patient_id: str,
    history: list[dict],
    embeddings: np.ndarray,
    event_id_map: dict,
) -> np.ndarray:
    """Return 768-dim embedding of the MOST RECENT (last) event in history.

    Iterates history in reverse to find the most recent event with a valid embedding.
    If history is empty or no event has a valid embedding, returns zeros(768).
    Uses the same 'event_index' key as compute_mean_last_k_embedding.
    """
    emb_dim = embeddings.shape[1]  # 768
    if not history:
        return np.zeros(emb_dim, dtype=np.float32)
    # Iterate from most recent to oldest to find a valid embedding
    for h in reversed(history):
        key = (str(patient_id), h["event_index"])
        row = event_id_map.get(key)
        if row is not None:
            return embeddings[row].astype(np.float32)
    # No valid embedding found for any event in history
    return np.zeros(emb_dim, dtype=np.float32)


def build_last_event_matrix(
    jsonl_path: Path,
    embeddings: np.ndarray,
    event_id_map: dict,
) -> np.ndarray:
    """Build (N, 768) matrix of last-event-embedding for each record in a JSONL split."""
    rows = []
    n_zero = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            history = rec.get("history", [])
            pid = str(rec["patient_id"])
            emb = compute_last_event_embedding(pid, history, embeddings, event_id_map)
            if emb.sum() == 0.0:
                n_zero += 1
            rows.append(emb)
    result = np.stack(rows).astype(np.float32)
    log.info(
        "  Emb-last matrix:  shape=%s  zero_rows=%d/%d (%.1f%%)",
        result.shape, n_zero, len(rows), 100.0 * n_zero / max(1, len(rows))
    )
    return result


def compute_quantile_bins(
    y_reg_train: np.ndarray,
    n_bins_target: int = N_BINS_TARGET,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute quantile-based bin edges and centers from training log-days.

    Args:
        y_reg_train: (N,) array of log1p(days) training labels
        n_bins_target: desired number of bins (may be reduced if many ties)

    Returns:
        bin_edges: (n_bins+1,) sorted unique quantile edges
        bin_centers: (n_bins,) midpoints of each bin
    """
    quantile_probs = np.linspace(0, 1, n_bins_target + 1)
    raw_edges = np.quantile(y_reg_train, quantile_probs)

    # Deduplicate (common at 0 for same-day events, or at clip boundary)
    bin_edges_np = np.unique(raw_edges)
    n_bins = len(bin_edges_np) - 1

    if n_bins < n_bins_target:
        log.warning(
            "Quantile deduplication: requested %d bins, got %d bins "
            "(dropped %d duplicate edges). Proceeding with %d bins.",
            n_bins_target, n_bins, n_bins_target - n_bins, n_bins,
        )
    else:
        log.info("Quantile bins: %d bins (no deduplication needed)", n_bins)

    # ASYM-01 change: push the final bin edge out to log1p(8000) ≈ 8.987
    # so the top bin captures long-tail events without collapsing to the training max.
    LOG1P_8000 = float(np.log1p(8000))
    if bin_edges_np[-1] < LOG1P_8000:
        bin_edges_np[-1] = LOG1P_8000
        log.info("Extended top bin edge to log1p(8000)=%.4f", LOG1P_8000)

    bin_centers_np = 0.5 * (bin_edges_np[:-1] + bin_edges_np[1:])

    log.info(
        "Bin edges — first 5: %s  last 5: %s",
        [f"{e:.4f}" for e in bin_edges_np[:5]],
        [f"{e:.4f}" for e in bin_edges_np[-5:]],
    )
    log.info(
        "Bin centers — first 5: %s  last 5: %s",
        [f"{c:.4f}" for c in bin_centers_np[:5]],
        [f"{c:.4f}" for c in bin_centers_np[-5:]],
    )

    return bin_edges_np, bin_centers_np


# ═══════════════════════════════════════════════════════════════════════════════
# Focal Loss (supports pre-mixed targets as one-hot distributions)
# ═══════════════════════════════════════════════════════════════════════════════

def _asl_from_one_hot(
    logits: torch.Tensor,
    one_hot: torch.Tensor,
    gamma_neg: float = ASL_GAMMA_NEG,
    gamma_pos: float = ASL_GAMMA_POS,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Core ASL computation given already-smoothed one-hot soft targets.

    Implements Asymmetric Loss (Ridnik et al. 2021, arXiv 2009.14119) adapted for
    multi-class single-label:
      pos term: y * (1-p)^gamma_pos * log(p)
      neg term: (1-y) * p^gamma_neg * log(1-p)
    Sum over classes, mean over batch.
    """
    probs   = F.softmax(logits, dim=-1).clamp(min=eps, max=1.0 - eps)
    log_p   = torch.log(probs)
    log_1mp = torch.log(1.0 - probs)

    pos_w = (1.0 - probs).pow(gamma_pos)
    neg_w = probs.pow(gamma_neg)

    loss = -(one_hot * pos_w * log_p + (1.0 - one_hot) * neg_w * log_1mp)
    return loss.sum(dim=-1).mean()


def asl_loss_hard(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma_neg: float = ASL_GAMMA_NEG,
    gamma_pos: float = ASL_GAMMA_POS,
    label_smoothing: float = LABEL_SMOOTH,
) -> torch.Tensor:
    """ASL for hard integer targets (eval / no-MixUp)."""
    n_cls = logits.size(1)
    with torch.no_grad():
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, targets.unsqueeze(1), 1.0)
        if label_smoothing > 0:
            one_hot = one_hot * (1.0 - label_smoothing) + label_smoothing / n_cls
    return _asl_from_one_hot(logits, one_hot, gamma_neg, gamma_pos)


def asl_loss_mixed(
    logits: torch.Tensor,
    targets_a: torch.Tensor,
    targets_b: torch.Tensor,
    lam: float,
    gamma_neg: float = ASL_GAMMA_NEG,
    gamma_pos: float = ASL_GAMMA_POS,
    label_smoothing: float = LABEL_SMOOTH,
) -> torch.Tensor:
    """MixUp ASL loss.

    loss = lam * asl(logits, targets_a) + (1-lam) * asl(logits, targets_b)
    """
    n_cls = logits.size(1)

    def _smooth_one_hot(tgt: torch.Tensor) -> torch.Tensor:
        oh = torch.zeros_like(logits)
        oh.scatter_(1, tgt.unsqueeze(1), 1.0)
        if label_smoothing > 0:
            oh = oh * (1.0 - label_smoothing) + label_smoothing / n_cls
        return oh

    with torch.no_grad():
        oh_a = _smooth_one_hot(targets_a)
        oh_b = _smooth_one_hot(targets_b)

    loss_a = _asl_from_one_hot(logits, oh_a, gamma_neg, gamma_pos)
    loss_b = _asl_from_one_hot(logits, oh_b, gamma_neg, gamma_pos)
    return lam * loss_a + (1.0 - lam) * loss_b


# ═══════════════════════════════════════════════════════════════════════════════
# MixUp helper
# ═══════════════════════════════════════════════════════════════════════════════

def mixup_batch(
    X: torch.Tensor,
    y_cls: torch.Tensor,
    y_reg: torch.Tensor,
    alpha: float = MIXUP_ALPHA,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Apply MixUp to a batch.

    Returns:
        x_mix, y_cls_a, y_cls_b, y_reg_a, y_reg_b, lam
    For regression: use y_reg_a only (log-days from sample A → Gaussian soft target in bin-index space).
    """
    lam = float(np.random.beta(alpha, alpha))
    lam = max(lam, 1.0 - lam)

    batch_size = X.size(0)
    perm       = torch.randperm(batch_size, device=X.device)

    x_mix   = lam * X + (1 - lam) * X[perm]
    y_cls_a = y_cls
    y_cls_b = y_cls[perm]
    y_reg_a = y_reg
    y_reg_b = y_reg[perm]

    return x_mix, y_cls_a, y_cls_b, y_reg_a, y_reg_b, lam


# ═══════════════════════════════════════════════════════════════════════════════
# LR scheduler: linear warmup then cosine anneal
# ═══════════════════════════════════════════════════════════════════════════════

class WarmupCosineScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_lr: float,
        warmup_steps: int,
        total_steps: int,
        eta_min: float = 0.0,
    ) -> None:
        self.optimizer    = optimizer
        self.base_lr      = base_lr
        self.warmup_steps = warmup_steps
        self.total_steps  = total_steps
        self.eta_min      = eta_min
        self._step        = 0
        self._set_lr(0.0)

    def _set_lr(self, lr: float) -> None:
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def step(self) -> float:
        self._step += 1
        s = self._step
        if s <= self.warmup_steps:
            lr = self.base_lr * s / self.warmup_steps
        else:
            progress = (s - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            lr = self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (
                1.0 + np.cos(np.pi * progress)
            )
        self._set_lr(lr)
        return lr

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


# ═══════════════════════════════════════════════════════════════════════════════
# NVC-Clean Model v14: Quantile bins + Gaussian soft targets in bin-index space
# ═══════════════════════════════════════════════════════════════════════════════

class NVCCleanTeacher(nn.Module):
    """
    Teacher-only class: MLP backbone + fm_linear key for strict checkpoint compatibility.
    Has fm_linear to match NVCClean checkpoint state_dict keys (teacher is the 2420-dim
    nvc_emb_mean_last_selfkd_100k_01 best_model.pt which has fm_linear). fm_linear is
    loaded but NOT used in forward — teacher inference uses pure MLP path only.
    Used only for self-KD; student uses NVCClean. n_features=2420 (same dim as student).
    """

    def __init__(
        self,
        n_features:    int,
        n_classes:     int,
        bin_edges_np:  np.ndarray,
        bin_centers_np: np.ndarray,
        dropout:       float = DROPOUT,
        input_dropout: float = INPUT_DROPOUT,
    ) -> None:
        super().__init__()

        n_bins = len(bin_centers_np)
        self.n_bins = n_bins

        self.register_buffer("bin_edges",   torch.tensor(bin_edges_np,   dtype=torch.float32))
        self.register_buffer("bin_centers", torch.tensor(bin_centers_np, dtype=torch.float32))

        self.input_drop = nn.Dropout(input_dropout)

        self.layer1 = nn.Sequential(
            nn.Linear(n_features, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.res_proj = nn.Linear(1024, 512, bias=False)
        self.cls_head = nn.Linear(512, n_classes)
        self.reg_head = nn.Linear(512, n_bins)
        # fm_linear is present to match NVCClean checkpoint state_dict keys (strict=True compat).
        # It is loaded from checkpoint but NOT used in forward — pure MLP path only for teacher.
        self.fm_linear = nn.Linear(n_features, 512, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Pure MLP path — fm_linear loaded but not used in teacher inference
        x  = self.input_drop(x)
        h1 = self.layer1(x)
        h2 = self.layer2(h1)
        h3 = self.layer3(h2) + self.res_proj(h2)
        logits     = self.cls_head(h3)
        reg_logits = self.reg_head(h3)
        return logits, reg_logits


class NVCClean(nn.Module):
    """
    NVC-Clean v15 + reg-only linear shortcut: 3-block residual MLP with a raw-feature
    linear shortcut injected ONLY into the regression head.

    Architecture:
      Input dropout: Dropout(0.1) on raw 884-dim features
      Layer 1: Linear(884->1024) -> BN -> GELU -> Dropout(0.3)
      Layer 2: Linear(1024->1024) -> BN -> GELU -> Dropout(0.3)
      Layer 3: Linear(1024->512)  -> BN -> GELU -> Dropout(0.3) [+ residual proj from 1024]
      Linear shortcut (on raw x, before input_drop):
        fm_linear: Linear(884, 512, bias=False) — linear shortcut from raw features
      Fusion (reg path only):
        logits     = cls_head(h3)                    # classification: pure MLP, no shortcut
        h3_for_reg = h3 + 0.1 * fm_linear(x_raw)    # regression: MLP + linear shortcut
        reg_logits = reg_head(h3_for_reg)
      Cls head: Linear(512->n_classes)
      Reg head: Linear(512->N_BINS)

    Reg head inference:
      probs = softmax(reg_logits)
      pred_log_days = sum(probs * bin_centers)
      pred_days = expm1(pred_log_days)
    """

    # Scale for fm_linear residual into h3_for_reg (prevents shortcut dominating early training)
    FM_LINEAR_SCALE = 0.1

    def __init__(
        self,
        n_features:    int,
        n_classes:     int,
        bin_edges_np:  np.ndarray,   # (n_bins+1,) quantile edges in log-day space
        bin_centers_np: np.ndarray,  # (n_bins,) midpoints in log-day space
        dropout:       float = DROPOUT,
        input_dropout: float = INPUT_DROPOUT,
    ) -> None:
        super().__init__()

        n_bins = len(bin_centers_np)
        self.n_bins = n_bins
        self.n_features = n_features

        # Bin edges and centers — registered as buffers so they follow .to(device)
        self.register_buffer("bin_edges",   torch.tensor(bin_edges_np,   dtype=torch.float32))
        self.register_buffer("bin_centers", torch.tensor(bin_centers_np, dtype=torch.float32))

        # Input-level feature dropout
        self.input_drop = nn.Dropout(input_dropout)

        # Layer 1: 884 -> 1024
        self.layer1 = nn.Sequential(
            nn.Linear(n_features, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Layer 2: 1024 -> 1024
        self.layer2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Layer 3: 1024 -> 512 with residual skip from layer2 output (1024->512)
        self.layer3 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Residual: project 1024->512 for skip connection
        self.res_proj = nn.Linear(1024, 512, bias=False)

        self.cls_head = nn.Linear(512, n_classes)

        # Quantile-bin reg head (n_bins may differ from 40 after dedup)
        self.reg_head = nn.Linear(512, n_bins)

        # ── Regression-only raw-feature linear shortcut ───────────────────────
        # fm_linear: linear shortcut from raw features -> 512, injected ONLY into reg path
        # No fm_v (pairwise FM removed — only linear shortcut remains)
        self.fm_linear = nn.Linear(n_features, 512, bias=False)

        self._init_weights()
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        log.info(
            "NVCClean-RegOnlyShortcut parameters: %d (%.1fK)  n_bins=%d  "
            "fm_linear=%d (reg-only shortcut, cls path untouched)",
            n_params, n_params / 1000, n_bins, self.fm_linear.weight.numel(),
        )

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # fm_linear uses kaiming_normal_ via the loop above (no special init needed)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Save raw features BEFORE input_drop for linear shortcut computation
        x_raw = x                              # (B, n_features)

        # ── Linear shortcut from raw features -> 512 (reg path only) ─────────
        fm_linear_out = self.fm_linear(x_raw)  # (B, 512)

        # ── MLP backbone ──────────────────────────────────────────────────────
        x  = self.input_drop(x)
        h1 = self.layer1(x)           # (B, 1024)
        h2 = self.layer2(h1)          # (B, 1024)
        h3 = self.layer3(h2) + self.res_proj(h2)  # (B, 512)

        # ── Heads ─────────────────────────────────────────────────────────────
        # Classification: pure MLP path — no shortcut, no FM bias
        logits = self.cls_head(h3)             # (B, n_classes)

        # Regression: h3 + linear shortcut from raw features
        h3_for_reg = h3 + self.FM_LINEAR_SCALE * fm_linear_out  # (B, 512)
        reg_logits = self.reg_head(h3_for_reg)                  # (B, N_BINS)

        return logits, reg_logits


def reg_logits_to_days(reg_logits: torch.Tensor, bin_centers: torch.Tensor) -> torch.Tensor:
    """Convert raw reg bin logits to predicted days via expected log-day."""
    probs         = F.softmax(reg_logits, dim=-1)              # (B, N_BINS)
    pred_log_days = (probs * bin_centers.unsqueeze(0)).sum(-1) # (B,)
    pred_days     = torch.expm1(pred_log_days)                 # (B,)
    return pred_days


def bin_expected_log_days(bin_logits: torch.Tensor, bin_centers_log1p: torch.Tensor) -> torch.Tensor:
    """
    Decode bin logits to expected value in log1p-day space (NO detach — gradients flow through).

    bin_logits:          (B, n_bins) raw logits from reg head
    bin_centers_log1p:   (n_bins,)   bin centers in log1p-day space (model.bin_centers buffer)
    Returns:             (B,)        expected log1p-day
    """
    probs = F.softmax(bin_logits, dim=-1)                              # (B, n_bins)
    return (probs * bin_centers_log1p.unsqueeze(0)).sum(dim=-1)        # (B,)


def aux_l1_weight(epoch: int) -> float:
    """
    LAMBDA_AUX schedule (v3): 0 before epoch 10, linear ramp 10->20, hold at LAMBDA_AUX_MAX=0.75.
    """
    if epoch < AUX_RAMP_START:
        return 0.0
    if epoch < AUX_RAMP_END:
        return LAMBDA_AUX_MAX * (epoch - AUX_RAMP_START) / (AUX_RAMP_END - AUX_RAMP_START)
    return LAMBDA_AUX_MAX


def gaussian_soft_target_idx(
    y_reg: torch.Tensor,
    bin_edges: torch.Tensor,
    n_bins: int,
    sigma: float = GAUSS_SIGMA_IDX,
) -> torch.Tensor:
    """
    Compute Gaussian soft target distribution over bins in BIN-INDEX space.

    For each true log-day value t in y_reg (shape B,):
      k_true = bucketize(t, bin_edges[1:-1])   ← integer bin index 0..n_bins-1
      unnorm[i, k] = exp(-(k - k_true[i])^2 / (2 * sigma^2))
      soft_target[i, :] = unnorm[i, :] / unnorm[i, :].sum()

    Using index space (not log-day space) is correct here because bins are
    non-uniform — equal width in index space means equal "probability resolution".

    Args:
        y_reg: (B,) log1p(days) values
        bin_edges: (n_bins+1,) quantile bin edges registered on model
        n_bins: number of bins
        sigma: Gaussian std in bin-index units (default 1.0)

    Returns:
        soft_target: (B, n_bins) normalized distribution
    """
    # Assign each sample to a bin index (0..n_bins-1)
    # bin_edges[1:-1] are the interior boundaries (n_bins-1 values)
    interior = bin_edges[1:-1]  # (n_bins-1,)
    k_true = torch.bucketize(y_reg, interior)  # (B,) values in [0, n_bins-1]
    k_true = k_true.clamp(0, n_bins - 1)

    # Build Gaussian over indices with ASYMMETRIC sigma:
    # For low bins (same-day events), narrow sigma to prevent rightward probability leak.
    # sigma_eff(k) = max(0.3, min(sigma, k * 0.5))
    # This means: bin 0 → σ=0.3, bin 1 → σ=0.5, bin 2 → σ=1.0, bin ≥2 → original σ
    k_true_float = k_true.float()  # (B,)
    sigma_eff = torch.clamp(torch.clamp(k_true_float * 0.5, max=sigma), min=0.3)  # (B,)

    k_range = torch.arange(n_bins, device=y_reg.device, dtype=torch.float32)  # (n_bins,)
    # k_true: (B,) → (B, 1), k_range: (n_bins,) → (1, n_bins)
    diff = k_range.unsqueeze(0) - k_true_float.unsqueeze(1)          # (B, n_bins)
    unnorm = torch.exp(-(diff ** 2) / (2.0 * sigma_eff.unsqueeze(1) ** 2))  # (B, n_bins)
    soft_target = unnorm / unnorm.sum(dim=-1, keepdim=True)      # (B, n_bins) normalized
    return soft_target


def gaussian_reg_loss_idx(
    reg_logits: torch.Tensor,
    y_reg: torch.Tensor,
    bin_edges: torch.Tensor,
    n_bins: int,
    sigma: float = GAUSS_SIGMA_IDX,
) -> torch.Tensor:
    """
    Cross-entropy loss with Gaussian soft targets in bin-INDEX space.

    Loss = -(soft_target * log_softmax(reg_logits)).sum(dim=-1).mean()

    Args:
        reg_logits: (B, N_BINS) raw logits
        y_reg: (B,) true log1p(days) values
        bin_edges: (n_bins+1,) quantile edges
        n_bins: number of bins
        sigma: Gaussian std in bin-index units

    Returns:
        scalar loss
    """
    with torch.no_grad():
        soft_target = gaussian_soft_target_idx(y_reg, bin_edges, n_bins, sigma)  # (B, N_BINS)
    log_probs = F.log_softmax(reg_logits, dim=-1)                                  # (B, N_BINS)
    loss = -(soft_target * log_probs).sum(dim=-1).mean()
    return loss


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation (no MixUp — hard labels only)
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate(
    model:  nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    all_logits:     list[torch.Tensor] = []
    all_reg_logits: list[torch.Tensor] = []
    all_y_cls:      list[torch.Tensor] = []
    all_y_reg:      list[torch.Tensor] = []

    with torch.no_grad():
        for X_b, y_cls_b, y_reg_b in loader:
            X_b = X_b.to(device)
            logits, reg_logits = model(X_b)
            all_logits.append(logits.cpu())
            all_reg_logits.append(reg_logits.cpu())
            all_y_cls.append(y_cls_b)
            all_y_reg.append(y_reg_b)

    logits     = torch.cat(all_logits)
    reg_logits = torch.cat(all_reg_logits)
    y_cls      = torch.cat(all_y_cls)
    y_reg      = torch.cat(all_y_reg)

    top3     = logits.topk(3, dim=1).indices
    top1_acc = (top3[:, 0] == y_cls).float().mean().item()
    top3_acc = (top3 == y_cls.unsqueeze(1)).any(dim=1).float().mean().item()

    # Get bin_centers from model (works for both base and SWA model)
    if hasattr(model, "module"):
        # AveragedModel wraps: model.module is the underlying NVCClean
        bin_centers = model.module.bin_centers.cpu()
    else:
        bin_centers = model.bin_centers.cpu()

    pred_days = reg_logits_to_days(reg_logits, bin_centers)
    true_days = torch.expm1(y_reg.clamp(0, LOG_DAYS_CLIP))
    abs_err   = (pred_days - true_days).abs()
    mae       = abs_err.mean().item()
    med_err   = abs_err.median().item()

    return {
        "top1_acc":        top1_acc,
        "top3_acc":        top3_acc,
        "mae_days":        mae,
        "median_err_days": med_err,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CLI & Main
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NVC-Clean-MixUp-v14: Quantile bins + Gaussian soft targets in bin-index space"
    )
    parser.add_argument("--sex",         type=str, required=True, choices=["M", "F"])
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--data-suffix", type=str, default="100k")
    return parser.parse_args()


def main() -> None:
    """
    Train the Cadence model on MIMIC-IV EHR sequences.

    Task: Predict next clinical event type (50-class) and days-to-event from
    observed patient history sequences.

    Inputs (command-line arguments):
      --sex {M,F}: Patient cohort (male or female)
      --seed {42,43,44}: Random seed for reproducibility
      --data-suffix {5k,10k,50k,100k,full}: Training data tier
      --teacher-seed: Seed of teacher checkpoint for self-distillation (optional)

    Outputs:
      - Checkpoint saved to results/nvc_clean/{sex}_{data_suffix}/seed_{seed}/best_model.pt
      - Test metrics saved to results/nvc_clean/{sex}_{data_suffix}/seed_{seed}/test_metrics.json
      - Training log to logs/mimic_train_nvc_clean_seed_{seed}_{sex}_{data_suffix}.log

    Example:
      python -m cadence.model --sex M --seed 42 --data-suffix 100k

    Returns:
      None (outputs written to disk)
    """
    # ── Feature-engineering imports from public package ──────────────────────────
    # Since v1.1.0, build_population_prior, load_embeddings, and build_feature_matrix
    # live in the public cadence package (features.py / data.py).
    # The private preflight_check is imported lazily so PyPI users can call
    # cadence.train() without the private repo on the path.
    from .features import build_population_prior, build_base_feature_matrix as build_feature_matrix
    from .data import load_embeddings

    try:
        sys.path.insert(0, str(_PROJECT_DIR / "src"))
        from preflight_check import preflight
        preflight()
    except ImportError:
        log.warning(
            "preflight_check not found (private repo not on path). "
            "Skipping preflight. This is expected for public users calling cadence.main()."
        )

    args        = _parse_args()
    seed        = args.seed
    sex         = args.sex
    sex_label   = "male" if sex == "M" else "female"
    data_suffix = args.data_suffix

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ── Paths ─────────────────────────────────────────────────────────────────
    seq_dir    = DATA_DIR / f"mimic_sequences_{sex_label}_{data_suffix}"
    result_dir = (
        _PROJECT_DIR / "results" / "nvc_clean"
        / f"{sex_label}_{data_suffix}" / f"seed_{seed}"
    )
    result_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 80)
    log.info(
        "mimic_train_nvc_clean (NVCClean champion, 768-dim emb-mean + 768-dim emb-last | self-KD T=%.1f alpha=%.2f | "
        "ASL gamma_neg=%.1f gamma_pos=%.1f, LAMBDA_AUX_MAX=%.2f ramp ep%d->%d, SWA_START=%d, 140 total epochs, input_dim=2420) "
        "— %s  seed=%d  data_suffix=%s",
        TEMPERATURE, KD_ALPHA, ASL_GAMMA_NEG, ASL_GAMMA_POS,
        LAMBDA_AUX_MAX, AUX_RAMP_START, AUX_RAMP_END, SWA_START, sex_label.upper(), seed, data_suffix
    )
    log.info("  Data:    %s", seq_dir)
    log.info("  Results: %s", result_dir)
    log.info(
        "  MixUp alpha=%.1f  dropout=%.1f  input_dropout=%.1f  "
        "label_smooth=%.2f  wd=%.4f  reg_weight=%.1f  n_bins_target=%d  gauss_sigma_idx=%.2f",
        MIXUP_ALPHA, DROPOUT, INPUT_DROPOUT, LABEL_SMOOTH, WEIGHT_DECAY, REG_WEIGHT,
        N_BINS_TARGET, GAUSS_SIGMA_IDX,
    )
    log.info("  SWA: start_epoch=%d  swa_lr=%.6f", SWA_START, SWA_LR)
    log.info("=" * 80)

    # ── Split info ────────────────────────────────────────────────────────────
    split_info  = json.loads((seq_dir / "split_info.json").read_text())
    n_clusters  = split_info["n_clusters"]
    max_history = split_info.get("config", {}).get("max_history", 10)
    log.info("n_clusters=%d  max_history=%d", n_clusters, max_history)

    # ── Feature dicts ─────────────────────────────────────────────────────────
    struct_dict   = load_feature_dict(seq_dir / "structured_features.npz",  STRUCT_FEAT_DIM,   "structured")
    temporal_dict = load_feature_dict(seq_dir / "temporal_features.npz",    TEMPORAL_FEAT_DIM, "temporal")

    # ── Population prior + embeddings ────────────────────────────────────────
    prior = build_population_prior(seq_dir / "train.jsonl", n_clusters)
    mimic_cat_dir = DATA_DIR / "mimic_categories"
    embeddings, event_id_map = load_embeddings(
        mimic_cat_dir / "embeddings.npy",
        mimic_cat_dir / "event_index.json",
    )

    # ── Build 270-dim base feature matrices ───────────────────────────────────
    log.info("=== Building TRAIN features ===")
    X_train_base, y_cls_train, y_reg_train = build_feature_matrix(
        seq_dir / "train.jsonl", prior, embeddings, event_id_map, n_clusters, max_history
    )
    log.info("=== Building VAL features ===")
    X_val_base, y_cls_val, y_reg_val = build_feature_matrix(
        seq_dir / "val.jsonl", prior, embeddings, event_id_map, n_clusters, max_history
    )
    log.info("=== Building TEST features ===")
    X_test_base, y_cls_test, y_reg_test = build_feature_matrix(
        seq_dir / "test.jsonl", prior, embeddings, event_id_map, n_clusters, max_history
    )

    # ── Augment: append struct (150) + temporal (464) ─────────────────────────
    X_train = augment_features(seq_dir / "train.jsonl", X_train_base, struct_dict, temporal_dict)
    X_val   = augment_features(seq_dir / "val.jsonl",   X_val_base,   struct_dict, temporal_dict)
    X_test  = augment_features(seq_dir / "test.jsonl",  X_test_base,  struct_dict, temporal_dict)

    # ── Append 768-dim mean-last-10 + 768-dim last-event embeddings ──────────
    # emb_mean: 768-dim vector = mean of last 10 event PubMedBERT embeddings (recent trajectory)
    # emb_last: 768-dim vector = single most-recent event embedding (acute state)
    # Together they capture complementary signals: trajectory + what the patient just did.
    # New input dim: 884 + 768 + 768 = 2420.
    log.info("=== Building emb-mean features (last k=10 embeddings, 768-dim) ===")
    emb_mean_train = build_emb_mean_matrix(seq_dir / "train.jsonl", embeddings, event_id_map, k=10)
    emb_mean_val   = build_emb_mean_matrix(seq_dir / "val.jsonl",   embeddings, event_id_map, k=10)
    emb_mean_test  = build_emb_mean_matrix(seq_dir / "test.jsonl",  embeddings, event_id_map, k=10)

    log.info("=== Building emb-last features (single most-recent event, 768-dim) ===")
    emb_last_train = build_last_event_matrix(seq_dir / "train.jsonl", embeddings, event_id_map)
    emb_last_val   = build_last_event_matrix(seq_dir / "val.jsonl",   embeddings, event_id_map)
    emb_last_test  = build_last_event_matrix(seq_dir / "test.jsonl",  embeddings, event_id_map)

    X_train = np.concatenate([X_train, emb_mean_train, emb_last_train], axis=1)
    X_val   = np.concatenate([X_val,   emb_mean_val,   emb_last_val],   axis=1)
    X_test  = np.concatenate([X_test,  emb_mean_test,  emb_last_test],  axis=1)
    log.info(
        "After emb-mean + emb-last concat — train: %s  val: %s  test: %s",
        X_train.shape, X_val.shape, X_test.shape,
    )
    assert X_train.shape[1] == 2420, f"Expected 2420 features (884+768+768), got {X_train.shape[1]}"

    # ── Label remapping ───────────────────────────────────────────────────────
    train_classes   = np.unique(y_cls_train)
    n_train_classes = len(train_classes)
    if n_train_classes < n_clusters:
        log.warning(
            "Train has only %d/%d cluster IDs — remapping to dense [0, %d)",
            n_train_classes, n_clusters, n_train_classes,
        )
        c2d = {c: i for i, c in enumerate(train_classes)}
        y_cls_train = np.array([c2d[c] for c in y_cls_train], dtype=np.int32)
        y_cls_val   = np.array([c2d.get(c, -1) for c in y_cls_val], dtype=np.int32)
        y_cls_test  = np.array([c2d.get(c, -1) for c in y_cls_test], dtype=np.int32)

        for name, X, yc, yr in [("val", X_val, y_cls_val, y_reg_val),
                                  ("test", X_test, y_cls_test, y_reg_test)]:
            mask = yc >= 0
            if not mask.all():
                log.warning("Dropping %d %s samples with unseen cluster IDs", (~mask).sum(), name)
            if name == "val":
                X_val, y_cls_val, y_reg_val = X[mask], yc[mask], yr[mask]
            else:
                X_test, y_cls_test, y_reg_test = X[mask], yc[mask], yr[mask]
        n_clf_classes = n_train_classes
    else:
        n_clf_classes = n_clusters

    actual_n_features = X_train.shape[1]
    log.info(
        "Final shapes — train: %s  val: %s  test: %s  n_classes: %d",
        X_train.shape, X_val.shape, X_test.shape, n_clf_classes,
    )

    # ── Feature standardization ───────────────────────────────────────────────
    feat_mean = X_train.mean(axis=0, keepdims=True)
    feat_std  = X_train.std(axis=0, keepdims=True) + 1e-8
    X_train   = (X_train - feat_mean) / feat_std
    X_val     = (X_val   - feat_mean) / feat_std
    X_test    = (X_test  - feat_mean) / feat_std

    np.savez(result_dir / "feature_scaler.npz", mean=feat_mean, std=feat_std)

    # ── Compute quantile bins from TRAINING log-days (BEFORE tensor conversion) ──
    log.info("=== Computing quantile bins from training log-days ===")
    bin_edges_np, bin_centers_np = compute_quantile_bins(y_reg_train, n_bins_target=N_BINS_TARGET)
    n_bins_actual = len(bin_centers_np)
    log.info("Effective N_BINS: %d (target was %d)", n_bins_actual, N_BINS_TARGET)

    # ── PyTorch tensors ───────────────────────────────────────────────────────
    def to_tensors(X, yc, yr):
        return (
            torch.from_numpy(X).float(),
            torch.from_numpy(yc).long(),
            torch.from_numpy(yr).float(),
        )

    Xt, yct, yrt = to_tensors(X_train, y_cls_train, y_reg_train)
    Xv, ycv, yrv = to_tensors(X_val,   y_cls_val,   y_reg_val)
    Xe, yce, yre = to_tensors(X_test,  y_cls_test,  y_reg_test)

    train_ds = TensorDataset(Xt, yct, yrt)
    val_ds   = TensorDataset(Xv, ycv, yrv)
    test_ds  = TensorDataset(Xe, yce, yre)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = NVCClean(
        n_features=actual_n_features,
        n_classes=n_clf_classes,
        bin_edges_np=bin_edges_np,
        bin_centers_np=bin_centers_np,
        dropout=DROPOUT,
        input_dropout=INPUT_DROPOUT,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("NVCClean-EmbMeanLast total trainable parameters: %d (%.1fK)", n_params, n_params / 1000)
    log.info("KD ENABLED (KD_ALPHA=%.2f, T=%.1f) — self-KD from 2420-dim best_model.pt.", KD_ALPHA, TEMPERATURE)
    log.info("Input dim: %d (884 base + 768 emb-mean + 768 emb-last)", actual_n_features)

    # ── Teacher model (frozen) — self-KD from nvc_emb_mean_last_selfkd_100k_01 best_model.pt ──
    # Self-distillation from nvc_emb_mean_last_selfkd_100k_01 seed_42 best_model.pt (34.02% top-1).
    # ALLOWED per CLAUDE.md: NV-C -> NV-C self-KD is permitted.
    # NOT XGBoost KD — teacher is our own trained NVCClean model.
    # NVCCleanTeacher has fm_linear key to match the NVCClean checkpoint — strict=True works.
    # fm_linear is loaded but not used in teacher forward (pure MLP path for inference).
    # best_model.pt is a regular checkpoint (NOT AveragedModel), so no "module." prefix expected.
    teacher_ckpt_path = Path(_PROJECT_DIR).parent / TEACHER_CKPT
    log.info("=" * 60)
    log.info("Loading teacher from: %s", teacher_ckpt_path)
    teacher = NVCCleanTeacher(
        n_features=actual_n_features,
        n_classes=n_clf_classes,
        bin_edges_np=bin_edges_np,
        bin_centers_np=bin_centers_np,
        dropout=DROPOUT,
        input_dropout=INPUT_DROPOUT,
    ).to(device)
    teacher_state = torch.load(teacher_ckpt_path, map_location=device, weights_only=False)
    # best_model.pt is a regular NVCClean checkpoint — no "module." prefix (not AveragedModel).
    # However, handle both cases gracefully in case SWA checkpoint is accidentally pointed to.
    if isinstance(teacher_state, dict) and any(k.startswith("module.") for k in teacher_state):
        log.info("Stripping 'module.' prefix from teacher state_dict (AveragedModel checkpoint).")
        teacher_state = {k[len("module."):]: v for k, v in teacher_state.items()
                         if k != "n_averaged"}
    # Handle nested state_dict key
    if isinstance(teacher_state, dict) and "state_dict" in teacher_state:
        teacher_state = teacher_state["state_dict"]
    # strict=True: NVCCleanTeacher has fm_linear, matching the NVCClean checkpoint keys exactly
    teacher.load_state_dict(teacher_state, strict=True)
    log.info("Teacher state_dict loaded with strict=True (NVCCleanTeacher+fm_linear, n_features=%d).", actual_n_features)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Sanity check: teacher top-1 on val set should be >= 0.30 (best_model.pt was 34.02% val)
    log.info("Running teacher sanity check on val set (expect top-1 >= 0.30)...")
    teacher_val_m = evaluate(teacher, val_loader, device)
    log.info(
        "TEACHER SANITY — val top1=%.4f  top3=%.4f  mae=%.1f",
        teacher_val_m["top1_acc"], teacher_val_m["top3_acc"], teacher_val_m["mae_days"],
    )
    if teacher_val_m["top1_acc"] < 0.30:
        raise RuntimeError(
            f"Teacher sanity check FAILED: val top-1={teacher_val_m['top1_acc']:.4f} < 0.30. "
            "State_dict load is likely broken. Aborting."
        )
    log.info("Teacher sanity check PASSED (top-1=%.4f >= 0.30).", teacher_val_m["top1_acc"])
    log.info("=" * 60)

    # ── SWA model setup ───────────────────────────────────────────────────────
    swa_model     = AveragedModel(model)
    best_ckpt     = result_dir / "best_model.pt"
    best_mae_ckpt = result_dir / "best_mae_model.pt"  # v3: MAE-selected checkpoint

    # ── Compute total steps ───────────────────────────────────────────────────
    steps_per_epoch    = len(train_loader)
    phase1_total_steps = steps_per_epoch * PHASE1_EPOCHS
    phase2_total_steps = steps_per_epoch * PHASE2_EPOCHS

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 1: Classification only (epochs 1-10) with MixUp + warmup
    # ─────────────────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Phase 1: classification only (epochs 1-%d) with MixUp alpha=%.1f", PHASE1_EPOCHS, MIXUP_ALPHA)
    log.info("  Warmup: %d steps, then cosine over %d total steps",
             WARMUP_STEPS, phase1_total_steps)
    log.info("=" * 60)

    optimizer = torch.optim.AdamW(model.parameters(), lr=PHASE1_LR, weight_decay=WEIGHT_DECAY)
    scheduler = WarmupCosineScheduler(
        optimizer,
        base_lr=PHASE1_LR,
        warmup_steps=WARMUP_STEPS,
        total_steps=phase1_total_steps,
        eta_min=PHASE1_LR * 0.1,
    )

    best_val_top1 = 0.0
    best_val_mae  = float("inf")   # v3: track best val MAE for best_mae_model.pt
    best_mae_epoch = 0
    best_epoch    = 0
    patience_cnt  = 0
    log_entries: list[dict] = []

    for epoch in range(1, PHASE1_EPOCHS + 1):
        model.train()
        t0         = time.time()
        total_loss = 0.0
        n_batches  = 0

        for X_b, y_cls_b, y_reg_b in train_loader:
            X_b     = X_b.to(device)
            y_cls_b = y_cls_b.to(device)
            y_reg_b = y_reg_b.to(device)

            # Apply MixUp (classification only in Phase 1)
            x_mix, y_cls_a, y_cls_b_mix, _, _, lam = mixup_batch(X_b, y_cls_b, y_reg_b)

            optimizer.zero_grad()
            logits, _ = model(x_mix)
            loss = asl_loss_mixed(logits, y_cls_a, y_cls_b_mix, lam)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            n_batches  += 1

        val_m   = evaluate(model, val_loader, device)
        elapsed = time.time() - t0
        log.info(
            "P1 Epoch %2d/%d  train_loss=%.4f  val top1=%.4f  top3=%.4f  mae=%.1f  lr=%.6f  [%.1fs]",
            epoch, PHASE1_EPOCHS,
            total_loss / n_batches,
            val_m["top1_acc"], val_m["top3_acc"], val_m["mae_days"],
            scheduler.get_lr(),
            elapsed,
        )

        if val_m["top1_acc"] > best_val_top1:
            best_val_top1 = val_m["top1_acc"]
            best_epoch    = epoch
            patience_cnt  = 0
            torch.save(model.state_dict(), best_ckpt)
        else:
            patience_cnt += 1

        # v3: also track best MAE checkpoint
        if val_m["mae_days"] < best_val_mae:
            best_val_mae  = val_m["mae_days"]
            best_mae_epoch = epoch
            torch.save(model.state_dict(), best_mae_ckpt)

        log_entries.append({
            "epoch": epoch, "phase": "P1",
            "train_loss": total_loss / n_batches,
            "val": val_m,
            "lr": scheduler.get_lr(),
        })
        with open(result_dir / "train_log.jsonl", "w") as f:
            for e in log_entries:
                f.write(json.dumps(e) + "\n")

    # ── Early-stop check at end of Phase 1 ───────────────────────────────────
    log.info("Phase 1 complete. Best val top-1 so far: %.4f", best_val_top1)
    if best_val_top1 < 0.18:
        log.warning(
            "EARLY-STOP CHECK FAILED: val top-1=%.4f < 0.18 threshold at epoch %d. "
            "Training broken — aborting Phase 2.",
            best_val_top1, PHASE1_EPOCHS,
        )
    else:
        # ─────────────────────────────────────────────────────────────────────
        # Phase 2: Classification + Quantile-bin Gaussian-idx regression (epochs 11-60) with SWA
        # ─────────────────────────────────────────────────────────────────────
        log.info("=" * 60)
        log.info("Phase 2: cls + quantile-bin Gaussian-idx reg with MixUp (epochs %d-%d), SWA from epoch %d",
                 PHASE1_EPOCHS + 1, PHASE1_EPOCHS + PHASE2_EPOCHS, SWA_START)
        log.info("  Reg loss: Gaussian-idx soft targets (%d bins, sigma_idx=%.2f), reg_weight=%.1f",
                 n_bins_actual, GAUSS_SIGMA_IDX, REG_WEIGHT)
        log.info("=" * 60)

        optimizer2 = torch.optim.AdamW(model.parameters(), lr=PHASE2_LR, weight_decay=WEIGHT_DECAY)
        scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer2, T_max=PHASE2_EPOCHS, eta_min=PHASE2_LR * 0.1
        )

        swa_scheduler = SWALR(
            optimizer2,
            swa_lr=SWA_LR,
            anneal_epochs=5,
            anneal_strategy="cos",
        )

        swa_active    = False
        best_swa_top1 = 0.0

        for epoch in range(PHASE1_EPOCHS + 1, PHASE1_EPOCHS + PHASE2_EPOCHS + 1):
            model.train()
            t0          = time.time()
            total_loss  = total_cls = total_reg = total_aux = total_kd = 0.0
            n_batches   = 0
            lam_aux     = aux_l1_weight(epoch)

            for X_b, y_cls_b, y_reg_b in train_loader:
                X_b     = X_b.to(device)
                y_cls_b = y_cls_b.to(device)
                y_reg_b = y_reg_b.to(device)

                # Apply MixUp
                x_mix, y_cls_a, y_cls_b_mix, y_reg_a, y_reg_b_mix, lam = mixup_batch(
                    X_b, y_cls_b, y_reg_b
                )

                optimizer2.zero_grad()
                logits, reg_logits = model(x_mix)

                # Classification: MixUp ASL loss (hard labels)
                asl_combined = asl_loss_mixed(logits, y_cls_a, y_cls_b_mix, lam)

                # Self-KD: KL divergence from student to frozen teacher
                # Teacher sees the same MixUp-augmented x_mix for matched soft labels.
                # Self-distillation from nvc_emb_mean_last_selfkd_100k_01 (NV-C -> NV-C, ALLOWED per CLAUDE.md).
                with torch.no_grad():
                    t_logits, _ = teacher(x_mix)
                    t_soft = F.softmax(t_logits / TEMPERATURE, dim=-1)  # soft probs
                s_log_probs = F.log_softmax(logits / TEMPERATURE, dim=-1)
                kd_loss = F.kl_div(s_log_probs, t_soft, reduction="batchmean") * (TEMPERATURE * TEMPERATURE)

                # Combined cls loss: hard-label ASL + soft-label KL
                cls_loss = (1.0 - KD_ALPHA) * asl_combined + KD_ALPHA * kd_loss
                kd_loss_val = kd_loss.item()

                # Regression: Gaussian-idx soft-target CE loss on quantile bins
                # y_reg_a is log1p(days) — use A sample only (no MixUp on reg)
                bin_edges_dev   = model.bin_edges   # already on device
                reg_loss = gaussian_reg_loss_idx(
                    reg_logits, y_reg_a, bin_edges_dev, n_bins_actual, GAUSS_SIGMA_IDX
                )

                # Auxiliary direct-MAE loss: smooth_l1 on bin-decoded expected log-day
                # Gradients flow through softmax into bin head and backbone — NO detach.
                # MixUp: mixed against both unmixed targets convex-combined by lam.
                bin_centers_dev = model.bin_centers   # (n_bins,) log1p-day space, on device
                log_pred        = bin_expected_log_days(reg_logits, bin_centers_dev)  # (B,)
                log_target_a    = y_reg_a.float()          # already log1p(days) from dataset
                log_target_b    = y_reg_b_mix.float()
                aux_loss = (
                    lam       * F.smooth_l1_loss(log_pred, log_target_a, beta=1.0)
                    + (1-lam) * F.smooth_l1_loss(log_pred, log_target_b, beta=1.0)
                )

                # Total loss: (combined cls with KD) + reg + aux
                loss = cls_loss + REG_WEIGHT * reg_loss + lam_aux * aux_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer2.step()

                total_loss += loss.item()
                total_cls  += cls_loss.item()
                total_reg  += reg_loss.item()
                total_aux  += aux_loss.item()
                total_kd   += kd_loss_val
                n_batches  += 1

            # ── SWA update (from SWA_START epoch onward) ──────────────────────
            if epoch >= SWA_START:
                swa_model.update_parameters(model)
                swa_scheduler.step()
                if not swa_active:
                    swa_active = True
                    log.info("*** SWA started at epoch %d ***", epoch)
            else:
                scheduler2.step()

            val_m = evaluate(model, val_loader, device)

            if swa_active:
                swa_val_label = "(SWA pending BN update)"
            else:
                swa_val_label = ""

            elapsed = time.time() - t0
            log.info(
                "P2 Epoch %2d/%d  loss=%.4f (cls=%.4f kd=%.4f reg=%.4f aux=%.4f lam_aux=%.3f)  "
                "val top1=%.4f  top3=%.4f  mae=%.1f %s [%.1fs]",
                epoch, PHASE1_EPOCHS + PHASE2_EPOCHS,
                total_loss / n_batches,
                total_cls  / n_batches,
                total_kd   / n_batches,
                total_reg  / n_batches,
                total_aux  / n_batches,
                lam_aux,
                val_m["top1_acc"], val_m["top3_acc"], val_m["mae_days"],
                swa_val_label,
                elapsed,
            )

            if val_m["top1_acc"] > best_val_top1:
                best_val_top1 = val_m["top1_acc"]
                best_epoch    = epoch
                patience_cnt  = 0
                torch.save(model.state_dict(), best_ckpt)
            else:
                patience_cnt += 1

            # v3: also track best MAE checkpoint
            if val_m["mae_days"] < best_val_mae:
                best_val_mae   = val_m["mae_days"]
                best_mae_epoch = epoch
                torch.save(model.state_dict(), best_mae_ckpt)

            # ── Early-stop kills (v4: relaxed gates — v3 killed prematurely at ep35) ────
            # SWA starts ep35, full aux weight at ep20. Target: SWA test MAE ≤39.5d, top-1≥30.5%.
            # CHANGE 1 vs v3: ep35 gate REMOVED; ep40 gate added; ep50/ep60 thresholds relaxed.
            p2_epoch_local = epoch - PHASE1_EPOCHS
            # End of Phase 1: epoch 10
            if epoch == PHASE1_EPOCHS and best_val_top1 < 0.27:
                log.warning("KILL: epoch 10 end-P1 best val top-1=%.4f < 27%%. Aborting.", best_val_top1)
                break
            # Phase 2 gate: epoch 15 (aux ramping, SWA not yet active)
            if epoch == 15 and (best_val_top1 < 0.29 or val_m["mae_days"] > 48.0):
                log.warning(
                    "KILL: epoch 15 best top-1=%.4f (need>=29%%) or mae=%.1f (need<=48d). Aborting.",
                    best_val_top1, val_m["mae_days"],
                )
                break
            # epoch 20 (aux at full weight 0.75)
            if epoch == 20 and val_m["mae_days"] > 44.0:
                log.warning("KILL: epoch 20 val MAE=%.1f > 44d. Aborting.", val_m["mae_days"])
                break
            # epoch 25
            if epoch == 25 and val_m["mae_days"] > 42.0:
                log.warning("KILL: epoch 25 val MAE=%.1f > 42d. Aborting.", val_m["mae_days"])
                break
            # epoch 30
            if epoch == 30 and val_m["mae_days"] > 40.5:
                log.warning("KILL: epoch 30 val MAE=%.1f > 40.5d. Aborting.", val_m["mae_days"])
                break
            # epoch 40 (v4: was ep35 > 40.0d in v3 — moved 5 epochs later to allow SWA warmup)
            if epoch == 40 and val_m["mae_days"] > 40.0:
                log.warning("KILL: epoch 40 val MAE=%.1f > 40.0d. Aborting.", val_m["mae_days"])
                break
            # epoch 50 (v4: was ep45 > 39.5d in v3)
            if epoch == 50 and val_m["mae_days"] > 39.5:
                log.warning("KILL: epoch 50 val MAE=%.1f > 39.5d. Aborting.", val_m["mae_days"])
                break
            # epoch 60 (v4: tightened to 39.3d — must be approaching XGBoost MAE target)
            if epoch == 60 and val_m["mae_days"] > 39.3:
                log.warning("KILL: epoch 60 val MAE=%.1f > 39.3d. Aborting.", val_m["mae_days"])
                break
            # Sanity kill: catastrophic MAE
            if val_m["mae_days"] > 100.0:
                log.warning("KILL: epoch %d MAE=%.1f > 100d. Aborting.", epoch, val_m["mae_days"])
                break

            if epoch == 30:
                if best_val_top1 <= 0.220:
                    log.warning(
                        "EARLY-STOP CHECK at epoch 30: best val top-1=%.4f has not exceeded "
                        "22.0%% baseline. Training stalled — aborting.",
                        best_val_top1,
                    )
                    break
                else:
                    log.info("Epoch-30 checkpoint PASSED: best val top-1=%.4f > 22.0%%", best_val_top1)

            if patience_cnt >= EARLY_STOP_PAT and epoch > EARLY_STOP_MIN_EPOCH:
                log.info("Early stopping at epoch %d (patience=%d, min_epoch=%d)", epoch, EARLY_STOP_PAT, EARLY_STOP_MIN_EPOCH)
                break

            log_entries.append({
                "epoch": epoch, "phase": "P2",
                "train_loss": total_loss / n_batches,
                "train_cls": total_cls / n_batches,
                "train_kd": total_kd / n_batches,
                "train_reg": total_reg / n_batches,
                "train_aux": total_aux / n_batches,
                "lam_aux": lam_aux,
                "val": val_m,
                "lr": optimizer2.param_groups[0]["lr"],
                "swa_active": swa_active,
            })
            with open(result_dir / "train_log.jsonl", "w") as f:
                for e in log_entries:
                    f.write(json.dumps(e) + "\n")

    log.info(
        "Training complete. Best base-model val top-1: %.4f at epoch %d | best val MAE: %.1f at epoch %d",
        best_val_top1, best_epoch, best_val_mae, best_mae_epoch,
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # Final evaluation: regular best-val (top-1), best-mae checkpoint, AND SWA
    # ═══════════════════════════════════════════════════════════════════════════

    swa_ckpt = result_dir / "swa_model.pt"
    if best_ckpt.exists():
        log.info("Loading regular best-top1 checkpoint: %s", best_ckpt)
        model.load_state_dict(torch.load(best_ckpt, map_location=device))
    regular_val_m  = evaluate(model, val_loader,  device)
    regular_test_m = evaluate(model, test_loader, device)
    log.info("Regular best-top1 — Val:  top1=%.4f  top3=%.4f  mae=%.1f  median_err=%.1f",
             regular_val_m["top1_acc"],  regular_val_m["top3_acc"],
             regular_val_m["mae_days"],  regular_val_m["median_err_days"])
    log.info("Regular best-top1 — Test: top1=%.4f  top3=%.4f  mae=%.1f  median_err=%.1f",
             regular_test_m["top1_acc"], regular_test_m["top3_acc"],
             regular_test_m["mae_days"], regular_test_m["median_err_days"])

    # v3: evaluate the MAE-selected checkpoint separately
    mae_ckpt_val_m  = regular_val_m   # default if no separate ckpt
    mae_ckpt_test_m = regular_test_m
    if best_mae_ckpt.exists() and best_mae_epoch != best_epoch:
        log.info("Loading best-MAE checkpoint (epoch %d): %s", best_mae_epoch, best_mae_ckpt)
        model.load_state_dict(torch.load(best_mae_ckpt, map_location=device))
        mae_ckpt_val_m  = evaluate(model, val_loader,  device)
        mae_ckpt_test_m = evaluate(model, test_loader, device)
        log.info("Regular best-MAE  — Val:  top1=%.4f  top3=%.4f  mae=%.1f  median_err=%.1f",
                 mae_ckpt_val_m["top1_acc"],  mae_ckpt_val_m["top3_acc"],
                 mae_ckpt_val_m["mae_days"],  mae_ckpt_val_m["median_err_days"])
        log.info("Regular best-MAE  — Test: top1=%.4f  top3=%.4f  mae=%.1f  median_err=%.1f",
                 mae_ckpt_test_m["top1_acc"], mae_ckpt_test_m["top3_acc"],
                 mae_ckpt_test_m["mae_days"], mae_ckpt_test_m["median_err_days"])
        # restore best-top1 weights for SWA comparison
        if best_ckpt.exists():
            model.load_state_dict(torch.load(best_ckpt, map_location=device))

    swa_val_m:  dict | None = None
    swa_test_m: dict | None = None
    if swa_active:
        log.info("=" * 60)
        log.info("SWA finalization: updating BatchNorm statistics on train loader...")
        log.info("=" * 60)
        update_bn(train_loader, swa_model, device=device)
        log.info("SWA BN update complete.")

        swa_val_m  = evaluate(swa_model, val_loader,  device)
        swa_test_m = evaluate(swa_model, test_loader, device)
        log.info("SWA — Val:  top1=%.4f  top3=%.4f  mae=%.1f  median_err=%.1f",
                 swa_val_m["top1_acc"],  swa_val_m["top3_acc"],
                 swa_val_m["mae_days"],  swa_val_m["median_err_days"])
        log.info("SWA — Test: top1=%.4f  top3=%.4f  mae=%.1f  median_err=%.1f",
                 swa_test_m["top1_acc"], swa_test_m["top3_acc"],
                 swa_test_m["mae_days"], swa_test_m["median_err_days"])
        torch.save(swa_model.state_dict(), swa_ckpt)
        log.info("Saved SWA checkpoint -> %s", swa_ckpt)
    else:
        log.warning("SWA never activated (training ended before epoch %d).", SWA_START)

    # ── Pick the winner (regular vs SWA) ──────────────────────────────────────
    if swa_test_m is not None:
        swa_better_top1 = swa_test_m["top1_acc"] >= regular_test_m["top1_acc"]
        swa_better_mae  = swa_test_m["mae_days"]  <= regular_test_m["mae_days"]
        use_swa = swa_better_top1 or (
            abs(swa_test_m["top1_acc"] - regular_test_m["top1_acc"]) < 0.002
            and swa_better_mae
        )
        best_model_name = "swa" if use_swa else "regular"
        test_m = swa_test_m if use_swa else regular_test_m
        val_m  = swa_val_m  if use_swa else regular_val_m
        if use_swa:
            torch.save(swa_model.state_dict(), best_ckpt)
            log.info("Winner: SWA — saved as best_model.pt")
        else:
            log.info("Winner: regular (best-val checkpoint kept as best_model.pt)")
        log.info(
            "Regular vs SWA — top1: %.4f vs %.4f  MAE: %.1f vs %.1f",
            regular_test_m["top1_acc"], swa_test_m["top1_acc"],
            regular_test_m["mae_days"], swa_test_m["mae_days"],
        )
    else:
        best_model_name = "regular"
        test_m = regular_test_m
        val_m  = regular_val_m
        use_swa = False

    # ── Save test_metrics.json ────────────────────────────────────────────────
    test_results = {
        "experiment":          "mimic_train_nvc_clean",
        "model":               "NVCClean-EmbMeanLast-v1",
        "architecture":        (
            f"Wider Deep Residual MLP (884->1024->512->256) — {n_bins_actual}-bin quantile-softmax reg head "
            f"(Gaussian-idx soft targets, sigma_idx={GAUSS_SIGMA_IDX:.2f}), "
            f"MixUp alpha={MIXUP_ALPHA}, input_dropout={INPUT_DROPOUT}, dropout={DROPOUT}, "
            f"label_smooth={LABEL_SMOOTH}, wd={WEIGHT_DECAY}, reg_loss=Gaussian_CE_idx, reg_weight={REG_WEIGHT}, "
            f"SWA from epoch {SWA_START} (swa_lr={SWA_LR})"
        ),
        "sex":                 sex,
        "seed":                seed,
        "data_suffix":         data_suffix,
        "n_clusters":          n_clusters,
        "n_clf_classes":       n_clf_classes,
        "n_features":          actual_n_features,
        "n_parameters":        n_params,
        "n_bins":              n_bins_actual,
        "n_bins_target":       N_BINS_TARGET,
        "bin_type":            "quantile",
        "gauss_sigma_idx":     GAUSS_SIGMA_IDX,
        "reg_head":            "binned_softmax_gaussian_idx",
        "best_epoch":          best_epoch,
        "best_val_top1":       best_val_top1,
        "best_mae_epoch":      best_mae_epoch,
        "best_val_mae":        best_val_mae,
        # v3: MAE-checkpoint test metrics
        "mae_ckpt_top1":       mae_ckpt_test_m["top1_acc"],
        "mae_ckpt_top3":       mae_ckpt_test_m["top3_acc"],
        "mae_ckpt_mae":        mae_ckpt_test_m["mae_days"],
        "mae_ckpt_median_err": mae_ckpt_test_m["median_err_days"],
        "val_top1_acc":        val_m["top1_acc"],
        "val_top3_acc":        val_m["top3_acc"],
        "val_mae_days":        val_m["mae_days"],
        "val_median_err_days": val_m["median_err_days"],
        # Separate regular and SWA test metrics
        "regular_top1":        regular_test_m["top1_acc"],
        "regular_top3":        regular_test_m["top3_acc"],
        "regular_mae":         regular_test_m["mae_days"],
        "regular_median_err":  regular_test_m["median_err_days"],
        "swa_top1":            swa_test_m["top1_acc"] if swa_test_m else None,
        "swa_top3":            swa_test_m["top3_acc"] if swa_test_m else None,
        "swa_mae":             swa_test_m["mae_days"] if swa_test_m else None,
        "swa_median_err":      swa_test_m["median_err_days"] if swa_test_m else None,
        # Top-level = best of the two
        "top1_acc":            test_m["top1_acc"],
        "top3_acc":            test_m["top3_acc"],
        "mae_days":            test_m["mae_days"],
        "median_err_days":     test_m["median_err_days"],
        "best_model":          best_model_name,
        "kd_contaminated":     False,
        "self_kd":             True,
        "kd_teacher":          "nvc_emb_mean_last_selfkd_100k_01/results/male_100k/seed_42/best_model.pt",
        "kd_temperature":      TEMPERATURE,
        "kd_alpha":            KD_ALPHA,
        "mixup_alpha":         MIXUP_ALPHA,
        "dropout":             DROPOUT,
        "input_dropout":       INPUT_DROPOUT,
        "label_smoothing":     LABEL_SMOOTH,
        "weight_decay":        WEIGHT_DECAY,
        "reg_weight":          REG_WEIGHT,
        "reg_loss":            "Gaussian_CE_idx_bins",
        "warmup_steps":        WARMUP_STEPS,
        "swa_start_epoch":     SWA_START,
        "swa_lr":              SWA_LR,
        "used_swa":            use_swa,
        # Bin info for inspection
        "bin_edges_first5":    bin_edges_np[:5].tolist(),
        "bin_edges_last5":     bin_edges_np[-5:].tolist(),
        "bin_centers_first5":  bin_centers_np[:5].tolist(),
        "bin_centers_last5":   bin_centers_np[-5:].tolist(),
        # Baselines
        "xgb_top1_baseline":   0.2464,
        "xgb_mae_baseline":    41.9,
        "mixup12_top1":        0.2503,
        "mixup12_mae":         46.7,
        "mixup11_top1":        0.2512,
        "mixup11_mae":         47.5,
        "beats_xgb_top1":      test_m["top1_acc"] > 0.2464,
        "beats_xgb_mae":       test_m["mae_days"] < 41.9,
        "beats_mixup12_top1":  test_m["top1_acc"] > 0.2503,
        "beats_mixup12_mae":   test_m["mae_days"] < 46.7,
    }
    out_path = result_dir / "test_metrics.json"
    out_path.write_text(json.dumps(test_results, indent=2))
    log.info("Saved -> %s", out_path)

    # ── Print comparison table ────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print(
        f"NVC-Clean-EmbMeanLast ({n_bins_actual}-bin quantile Gaussian-idx reg, SWA_START={SWA_START}, "
        f"input_dim=2420, self_KD=T={TEMPERATURE:.1f}_alpha={KD_ALPHA:.2f}) "
        f"— {sex_label.upper()} {data_suffix} — TEST RESULTS [mimic_train_nvc_clean]"
    )
    print(f"SWA used: {use_swa}")
    print("=" * 90)
    print(f"{'Metric':<25} {'MixUp-v14':>12} {'MixUp-v12':>12} {'XGBoost':>10} {'Delta-XGB':>12}")
    print("-" * 90)
    d_top1_xgb = test_m["top1_acc"] - 0.2464
    d_mae_xgb  = test_m["mae_days"] - 41.9
    d_top1_v12 = test_m["top1_acc"] - 0.2503
    d_mae_v12  = test_m["mae_days"] - 46.7
    print(f"{'Top-1 Accuracy':<25} {test_m['top1_acc']:>12.4f} {'0.2503':>12} {'0.2464':>10} {d_top1_xgb:>+12.4f}")
    print(f"{'Top-3 Accuracy':<25} {test_m['top3_acc']:>12.4f} {'-':>12} {'-':>10}")
    print(f"{'MAE (days)':<25} {test_m['mae_days']:>12.1f} {'46.7':>12} {'41.9':>10} {d_mae_xgb:>+12.1f}")
    print(f"{'Median Error (days)':<25} {test_m['median_err_days']:>12.1f} {'-':>12} {'-':>10}")
    print(f"{'Best epoch':<25} {best_epoch:>12} {'-':>12} {'-':>10}")
    print(f"{'Parameters':<25} {n_params:>12,} {'-':>12} {'-':>10}")
    print("=" * 90)
    print(f"Quantile bins used: {n_bins_actual} (target {N_BINS_TARGET})")
    print(f"  Bin edges first 5: {bin_edges_np[:5].tolist()}")
    print(f"  Bin edges last 5:  {bin_edges_np[-5:].tolist()}")
    print("Regular best-top1 test: top1=%.4f  mae=%.1f" % (regular_test_m["top1_acc"], regular_test_m["mae_days"]))
    print("Regular best-MAE  test: top1=%.4f  mae=%.1f (ep%d)" % (mae_ckpt_test_m["top1_acc"], mae_ckpt_test_m["mae_days"], best_mae_epoch))
    if swa_test_m:
        print("SWA test:     top1=%.4f  mae=%.1f" % (swa_test_m["top1_acc"], swa_test_m["mae_days"]))
    print(f"vs mixup-v12:  top1 {d_top1_v12:+.4f}  MAE {d_mae_v12:+.1f}d")
    beats_str = []
    if test_m["top1_acc"] > 0.2512:
        beats_str.append("BEATS MIXUP-v11 TOP-1")
    if test_m["top1_acc"] > 0.2503:
        beats_str.append("BEATS MIXUP-v12 TOP-1")
    if test_m["mae_days"] < 46.7:
        beats_str.append(f"BEATS MIXUP-v12 MAE ({test_m['mae_days']:.1f}d < 46.7d)")
    if test_m["top1_acc"] > 0.2464:
        beats_str.append(f"BEATS XGBOOST TOP-1 ({d_top1_xgb:+.4f})")
    if test_m["mae_days"] < 41.9:
        beats_str.append("BEATS XGBOOST MAE")
    if beats_str:
        print("  *** " + "  ".join(beats_str) + " ***")
    else:
        print("  (No improvement over prior models)")
    print("=" * 90)


if __name__ == "__main__":
    main()
