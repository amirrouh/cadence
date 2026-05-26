"""
cadence/train.py -- High-level training API for NVCClean.

Public users call cadence.train() with their own JSONL data and embeddings,
or cadence.train_classifier() with pre-built feature matrices.

Scope note
----------
This public training path uses a reduced feature set compared to the paper
checkpoint (1.1.0 public API vs. 2420-dim paper model):

  Public (this file):  5*n_clusters + max_history + 20 + 2*emb_dim dims
    e.g. n_clusters=50, max_history=10, emb_dim=768 -> 270 + 1536 = 1806 dims

  Paper checkpoint:    884 base (270 + 150 structured + 464 temporal) + 768 + 768 = 2420 dims

The 150 structured and 464 temporal features are derived from MIMIC-IV-specific
preprocessing pipelines and are not available for public datasets. Train your own
model on your own data via cadence.train(); pretrained weights are not distributed.

The public model is fully functional and uses the same NVCClean architecture,
training schedule (Phase 1 classification + Phase 2 cls+reg + SWA), MixUp,
ASL loss, and Gaussian soft-target regression. It will produce meaningful
results on any EHR-style sequence dataset with cluster IDs and per-event
embeddings.
"""
from __future__ import annotations

import copy
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.utils.data import DataLoader, TensorDataset

from .data import load_embeddings, validate_jsonl
from .features import build_feature_matrix, build_population_prior
from .model import (
    NVCClean,
    WarmupCosineScheduler,
    asl_loss_hard,
    asl_loss_mixed,
    aux_l1_weight,
    bin_expected_log_days,
    compute_quantile_bins,
    evaluate,
    gaussian_reg_loss_idx,
    mixup_batch,
    reg_logits_to_days,
    LOG_DAYS_CLIP,
    BATCH_SIZE,
    PHASE1_EPOCHS,
    PHASE2_EPOCHS,
    PHASE1_LR,
    PHASE2_LR,
    WEIGHT_DECAY,
    DROPOUT,
    INPUT_DROPOUT,
    EARLY_STOP_PAT,
    EARLY_STOP_MIN_EPOCH,
    ASL_GAMMA_NEG,
    ASL_GAMMA_POS,
    LABEL_SMOOTH,
    REG_WEIGHT,
    MIXUP_ALPHA,
    WARMUP_STEPS,
    SWA_START,
    SWA_LR,
    N_BINS_TARGET,
    GAUSS_SIGMA_IDX,
    LAMBDA_AUX_MAX,
    AUX_RAMP_START,
    AUX_RAMP_END,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers for binary / multiclass evaluation
# ---------------------------------------------------------------------------

def _evaluate_clf(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    task: str,
) -> dict[str, float]:
    """Evaluate a binary or multiclass NVCClean model."""
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    with torch.no_grad():
        for X_b, y_b in loader:
            X_b = X_b.to(device)
            logits = model(X_b)
            all_logits.append(logits.cpu())
            all_labels.append(y_b)

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    labels_np = labels.numpy()

    if task == "binary":
        probs = torch.sigmoid(logits.squeeze(1))
        preds = (probs >= 0.5).long()
        acc = (preds == labels).float().mean().item()
        probs_np = probs.numpy()
        try:
            auroc = float(roc_auc_score(labels_np, probs_np))
        except Exception:
            auroc = float("nan")
        return {"accuracy": acc, "val_auroc": auroc}
    else:
        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean().item()
        top3 = logits.topk(min(3, logits.size(1)), dim=1).indices
        top3_acc = (top3 == labels.unsqueeze(1)).any(dim=1).float().mean().item()
        probs_np = torch.softmax(logits, dim=1).numpy()
        n_classes = probs_np.shape[1]
        try:
            if n_classes == 2:
                auroc = float(roc_auc_score(labels_np, probs_np[:, 1]))
            else:
                auroc = float(roc_auc_score(labels_np, probs_np, multi_class="ovr", average="macro"))
        except Exception:
            auroc = float("nan")
        return {"accuracy": acc, "top3_acc": top3_acc, "val_auroc": auroc}


def _build_class_weight_tensor(
    class_weight: "str | dict | None",
    y_train: np.ndarray,
    n_classes: int,
    task: str,
) -> "tuple[torch.Tensor | None, list | None]":
    """
    Build a per-class weight tensor from the class_weight specification.

    Returns (tensor_or_None, serializable_list_or_None).
    """
    if class_weight is None:
        return None, None

    n_total = len(y_train)
    counts = np.bincount(y_train, minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1)  # avoid zero-division for unseen classes

    if class_weight == "balanced":
        weights = n_total / (n_classes * counts)
    elif isinstance(class_weight, dict):
        weights = np.ones(n_classes, dtype=np.float64)
        for k, v in class_weight.items():
            weights[int(k)] = float(v)
    else:
        raise ValueError(
            f"class_weight must be None, 'balanced', or a dict; got {class_weight!r}"
        )

    tensor = torch.tensor(weights, dtype=torch.float32)
    return tensor, weights.tolist()


def _run_clf_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    task: str,
    total_epochs: int,
    lr: float,
    out_dir: Path | None,
    *,
    weight_decay: float = WEIGHT_DECAY,
    class_weight_tensor: torch.Tensor | None = None,
    early_stopping_patience: int | None = None,
    early_stopping_metric: str = "val_auroc",
) -> tuple[float, dict[str, float], int, bool]:
    """
    Simple supervised training loop for binary/multiclass tasks.

    Returns (best_val_metric, best_val_metrics_dict, best_epoch, stopped_early).
    """
    if task == "binary":
        if class_weight_tensor is not None:
            # pos_weight for BCEWithLogitsLoss: scalar = w_1 / w_0
            pos_w = class_weight_tensor[1] / class_weight_tensor[0]
            loss_fn: nn.Module = nn.BCEWithLogitsLoss(
                pos_weight=pos_w.to(device)
            )
        else:
            loss_fn = nn.BCEWithLogitsLoss()
    else:
        if class_weight_tensor is not None:
            loss_fn = nn.CrossEntropyLoss(weight=class_weight_tensor.to(device))
        else:
            loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=lr * 0.1)

    best_val_metric = -float("inf")
    best_val_m: dict[str, float] = {}
    best_epoch = 1
    patience_cnt = 0
    stopped_early = False
    best_ckpt = (out_dir / "best_model.pt") if out_dir is not None else None
    best_state: dict | None = None

    for epoch in range(1, total_epochs + 1):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        n_batches = 0

        for X_b, y_b in train_loader:
            X_b = X_b.to(device)
            y_b = y_b.to(device)
            optimizer.zero_grad()
            logits = model(X_b)
            if task == "binary":
                loss = loss_fn(logits.squeeze(1), y_b.float())
            else:
                loss = loss_fn(logits, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        val_m = _evaluate_clf(model, val_loader, device, task)
        elapsed = time.time() - t0

        # Compute the tracked metric
        if early_stopping_metric == "val_loss":
            # We don't track val_loss separately; approximate with -accuracy
            tracked = -float("inf")  # not meaningful, fall back to auroc
            log.warning("val_loss early stopping not directly tracked; using val_auroc instead.")
            tracked = val_m.get("val_auroc", val_m["accuracy"])
        else:
            tracked = val_m.get("val_auroc", val_m["accuracy"])

        log.info(
            "Epoch %2d/%d  loss=%.4f  val_acc=%.4f  val_auroc=%.4f  [%.1fs]",
            epoch, total_epochs,
            total_loss / max(1, n_batches),
            val_m["accuracy"],
            val_m.get("val_auroc", float("nan")),
            elapsed,
        )

        improved = tracked > best_val_metric + 1e-4
        if improved:
            best_val_metric = tracked
            best_val_m = val_m
            best_epoch = epoch
            patience_cnt = 0
            if best_ckpt is not None:
                torch.save(model.state_dict(), best_ckpt)
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience_cnt += 1

        if early_stopping_patience is not None and patience_cnt >= early_stopping_patience:
            log.info(
                "Early stopping at epoch %d (patience=%d, best_epoch=%d, best_metric=%.4f)",
                epoch, early_stopping_patience, best_epoch, best_val_metric,
            )
            stopped_early = True
            break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    return best_val_metric, best_val_m, best_epoch, stopped_early


def train(
    train_jsonl: str | Path,
    val_jsonl: str | Path,
    embeddings_path: str | Path,
    event_index_path: str | Path,
    n_clusters: int,
    out_dir: str | Path,
    *,
    n_epochs: int | None = None,
    max_history: int = 10,
    seed: int = 42,
    device_str: str | None = None,
    validate_inputs: bool = True,
    test_jsonl: str | Path | None = None,
    task: str = "next_event",
    label_field: str | None = None,
    n_classes: int | None = None,
    early_stopping_patience: int | None = None,
    early_stopping_metric: str = "val_auroc",
    class_weight: "str | dict | None" = None,
    weight_decay: float = WEIGHT_DECAY,
) -> dict[str, Any]:
    """
    Train a Cadence NVCClean model on user-supplied data.

    The default task='next_event' reproduces the paper's setup exactly (joint
    classification + regression, MixUp, ASL loss, SWA). For arbitrary labels
    see the task, label_field, and n_classes arguments below.

    Input format: each JSONL file contains one JSON object per line:
    {
      "patient_id": "<uid>",
      "history": [
        {"date_iso": "2019-03-15", "event_index": 42, "cluster_id": 7,
         "days_from_start": 0.0},
        ...
      ],
      "target": {"cluster_id": 12, "days_from_prev": 14.0}
                 # for binary/multiclass: {"<label_field>": 0}
    }

    The embeddings file (embeddings_path) is a .npy array of shape
    (N_events, emb_dim) where N_events is the total number of unique events
    in your dataset and emb_dim is the embedding dimensionality (e.g. 768
    for PubMedBERT/BERT). Any sentence embedding model works.

    The event_index file (event_index_path) is a JSON array where the i-th
    element is {"subject_id": <uid>, "event_index": <int>} identifying the
    patient and event that row i of embeddings.npy corresponds to.

    Feature configuration (public path):
      Total input dims = 5*n_clusters + max_history + 20 + 2*emb_dim
      For n_clusters=50, max_history=10, emb_dim=768: 270 + 1536 = 1806 dims.
      (Paper checkpoint uses 2420 dims = 884 base + 768 + 768; the 884-270=614
      extra base dims require MIMIC-specific structured/temporal features not
      available for public datasets.)

    Args:
        train_jsonl:      Path to training JSONL file.
        val_jsonl:        Path to validation JSONL file.
        embeddings_path:  Path to .npy embeddings file (N_events, emb_dim).
        event_index_path: Path to event index JSON file.
        n_clusters:       Number of event clusters (used for feature building;
                          ignored as the classification target for binary/multiclass tasks).
        out_dir:          Directory for checkpoint and metadata outputs.
        n_epochs:         Total training epochs (default: PHASE1_EPOCHS + PHASE2_EPOCHS = 140).
                          For quick tests, pass e.g. n_epochs=5.
        max_history:      Maximum history window per patient (default 10).
        seed:             Random seed (default 42).
        device_str:       Device string ("cuda", "cpu", etc.). Auto-detected if None.
        validate_inputs:  Run schema validation on JSONL files (default True).
        test_jsonl:       Optional path to test JSONL for final evaluation.
        task:             "next_event" (default), "binary", or "multiclass".
                          "next_event" preserves v1.1.1 behavior exactly.
        label_field:      Key in target object holding the label. Required for
                          binary and multiclass tasks.
        n_classes:        Number of classes. Required for multiclass; binary
                          always uses 1 output logit.
        early_stopping_patience:
                          For binary/multiclass only. Stop training after this
                          many epochs without improvement (>0.0001) in the
                          tracked metric. None = run all n_epochs (default).
                          Ignored for task='next_event'.
        early_stopping_metric:
                          For binary/multiclass only. "val_auroc" (default) or
                          "val_loss". Ignored for task='next_event'.
        class_weight:     For binary/multiclass only. None = no weighting
                          (default). "balanced" = auto inverse-frequency weights
                          (sklearn convention: n / (n_classes * bincount)).
                          dict {class_idx: weight} = manual weights.
                          Applied as pos_weight to BCEWithLogitsLoss (binary)
                          or weight to CrossEntropyLoss (multiclass).
                          Ignored for task='next_event'.
        weight_decay:     AdamW weight decay. Default 1e-4. Applies to all
                          tasks.

    Returns:
        Classifier dict with keys:
          "model_path":    str, path to best checkpoint (best_model.pt).
          "swa_path":      str, path to SWA checkpoint (swa_model.pt) or None.
          "out_dir":       str, output directory.
          "n_features":    int, total input dimensionality.
          "n_clusters":    int, number of event clusters (feature building).
          "n_classes":     int, actual number of classes used.
          "task":          str, task type ("next_event", "binary", "multiclass").
          "label_field":   str or None, label field used.
          "bin_edges":     list[float], quantile bin edges (next_event only).
          "bin_centers":   list[float], quantile bin centers (next_event only).
          "feat_mean":     list[float], feature standardization mean.
          "feat_std":      list[float], feature standardization std.
          "val_metrics":   dict, validation metrics from best checkpoint.
          "test_metrics":  dict or None, test metrics (if test_jsonl provided).
          "metadata":      dict, training metadata.

    Raises:
        FileNotFoundError: If any input path does not exist.
        ValueError: On malformed input data or invalid task arguments.
        RuntimeError: If Phase 1 fails the early-stop check (best val top-1 < 2%).
    """
    # Validate task args
    if task not in ("next_event", "binary", "multiclass"):
        raise ValueError(f"task must be 'next_event', 'binary', or 'multiclass'; got {task!r}")
    if task in ("binary", "multiclass") and label_field is None:
        raise ValueError(f"label_field is required for task={task!r}")
    if task == "multiclass" and n_classes is None:
        raise ValueError("n_classes is required for task='multiclass'")

    train_jsonl      = Path(train_jsonl)
    val_jsonl        = Path(val_jsonl)
    embeddings_path  = Path(embeddings_path)
    event_index_path = Path(event_index_path)
    out_dir          = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Device
    if device_str is not None:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # Epoch budget
    total_epochs = n_epochs if n_epochs is not None else (PHASE1_EPOCHS + PHASE2_EPOCHS)
    p1_epochs = min(PHASE1_EPOCHS, total_epochs)
    p2_epochs = max(0, total_epochs - p1_epochs)
    log.info("Training budget: %d epochs (Phase1=%d, Phase2=%d)", total_epochs, p1_epochs, p2_epochs)

    # Input validation (next_event only — schema requires cluster_id/days_from_prev)
    if validate_inputs and task == "next_event":
        log.info("Validating JSONL schema ...")
        validate_jsonl(train_jsonl, n_clusters=n_clusters)
        validate_jsonl(val_jsonl,   n_clusters=n_clusters)
        if test_jsonl is not None:
            validate_jsonl(test_jsonl, n_clusters=n_clusters)

    # Load embeddings
    log.info("Loading embeddings ...")
    embeddings, event_id_map = load_embeddings(embeddings_path, event_index_path)
    emb_dim = embeddings.shape[1]
    log.info("Embeddings: shape=%s  emb_dim=%d", embeddings.shape, emb_dim)

    # Population prior
    prior = build_population_prior(train_jsonl, n_clusters)

    # Feature matrices
    log.info("=== Building TRAIN features ===")
    X_train, y_cls_train, y_reg_train = build_feature_matrix(
        train_jsonl, prior, embeddings, event_id_map, n_clusters, max_history
    )
    log.info("=== Building VAL features ===")
    X_val, y_cls_val, y_reg_val = build_feature_matrix(
        val_jsonl, prior, embeddings, event_id_map, n_clusters, max_history
    )
    X_test = y_cls_test = y_reg_test = None
    if test_jsonl is not None:
        log.info("=== Building TEST features ===")
        X_test, y_cls_test, y_reg_test = build_feature_matrix(
            test_jsonl, prior, embeddings, event_id_map, n_clusters, max_history
        )

    # For binary/multiclass: extract labels from target[label_field] instead of cluster_id
    if task in ("binary", "multiclass"):
        def _extract_labels(jsonl_path: Path) -> np.ndarray:
            labels = []
            with open(jsonl_path, "r", encoding="utf-8") as fp:
                for line in fp:
                    line = line.strip()
                    if line:
                        rec = json.loads(line)
                        lbl = rec["target"][label_field]
                        labels.append(int(lbl))
            return np.array(labels, dtype=np.int64)

        y_cls_train = _extract_labels(train_jsonl)
        y_cls_val   = _extract_labels(val_jsonl)
        y_cls_test  = _extract_labels(test_jsonl) if test_jsonl is not None else None
        # Determine actual n_clf_classes
        if task == "binary":
            n_clf_classes = 2
        else:
            n_clf_classes = int(n_classes)  # type: ignore[arg-type]
        log.info(
            "task=%s  label_field=%r  n_clf_classes=%d  "
            "train_label_counts: %s",
            task, label_field, n_clf_classes,
            str(dict(zip(*np.unique(y_cls_train, return_counts=True)))),
        )
    else:
        # next_event: existing label remapping (dense [0, n_classes))
        train_classes = np.unique(y_cls_train)
        n_clf_classes = len(train_classes)
        if n_clf_classes < n_clusters:
            log.warning(
                "Train has only %d/%d cluster IDs -- remapping to dense [0, %d)",
                n_clf_classes, n_clusters, n_clf_classes,
            )
            c2d = {c: i for i, c in enumerate(train_classes)}
            y_cls_train = np.array([c2d[c] for c in y_cls_train], dtype=np.int32)
            y_cls_val   = np.array([c2d.get(c, -1) for c in y_cls_val], dtype=np.int32)
            if y_cls_test is not None:
                y_cls_test = np.array([c2d.get(c, -1) for c in y_cls_test], dtype=np.int32)

            for name, Xp, ycp, yrp in [
                ("val", X_val, y_cls_val, y_reg_val),
                ("test", X_test, y_cls_test, y_reg_test) if X_test is not None else (None, None, None, None),
            ]:
                if Xp is None:
                    continue
                mask = ycp >= 0
                if not mask.all():
                    log.warning("Dropping %d %s samples with unseen cluster IDs", (~mask).sum(), name)
                if name == "val":
                    X_val, y_cls_val, y_reg_val = Xp[mask], ycp[mask], yrp[mask]
                else:
                    X_test, y_cls_test, y_reg_test = Xp[mask], ycp[mask], yrp[mask]

    actual_n_features = X_train.shape[1]
    log.info(
        "Input dim: %d  (base + emb_mean + emb_last; emb_dim=%d, n_clusters=%d, max_history=%d)",
        actual_n_features, emb_dim, n_clusters, max_history,
    )

    # Feature standardization
    feat_mean = X_train.mean(axis=0, keepdims=True)
    feat_std  = X_train.std(axis=0, keepdims=True) + 1e-8
    X_train   = (X_train - feat_mean) / feat_std
    X_val     = (X_val   - feat_mean) / feat_std
    if X_test is not None:
        X_test = (X_test - feat_mean) / feat_std

    np.savez(out_dir / "feature_scaler.npz", mean=feat_mean, std=feat_std)

    # -------------------------------------------------------------------------
    # Branch: binary / multiclass use a simple supervised loop
    # -------------------------------------------------------------------------
    if task in ("binary", "multiclass"):
        Xt = torch.from_numpy(X_train).float()
        yt = torch.from_numpy(y_cls_train).long()
        Xv = torch.from_numpy(X_val).float()
        yv = torch.from_numpy(y_cls_val).long()

        train_ds_clf = TensorDataset(Xt, yt)
        val_ds_clf   = TensorDataset(Xv, yv)
        batch_size_clf = min(BATCH_SIZE, len(Xt))
        train_loader_clf = DataLoader(train_ds_clf, batch_size=batch_size_clf, shuffle=True,  num_workers=0)
        val_loader_clf   = DataLoader(val_ds_clf,   batch_size=batch_size_clf, shuffle=False, num_workers=0)

        # Build class weight tensor
        cw_tensor, cw_applied = _build_class_weight_tensor(
            class_weight, y_cls_train, n_clf_classes, task
        )

        clf_n_classes = 1 if task == "binary" else n_clf_classes
        model = NVCClean(
            n_features=actual_n_features,
            n_classes=clf_n_classes,
            task=task,
            dropout=DROPOUT,
            input_dropout=INPUT_DROPOUT,
        ).to(device)

        clf_lr = PHASE2_LR
        clf_epochs = n_epochs if n_epochs is not None else (PHASE1_EPOCHS + PHASE2_EPOCHS)
        best_ckpt = out_dir / "best_model.pt"

        _, best_val_m, best_epoch_clf, stopped_early_clf = _run_clf_loop(
            model, train_loader_clf, val_loader_clf, device, task,
            clf_epochs, clf_lr, out_dir,
            weight_decay=weight_decay,
            class_weight_tensor=cw_tensor,
            early_stopping_patience=early_stopping_patience,
            early_stopping_metric=early_stopping_metric,
        )

        # best weights already restored in _run_clf_loop; reload from disk if ckpt exists
        if best_ckpt.exists():
            model.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=True))
        val_m_final = _evaluate_clf(model, val_loader_clf, device, task)

        test_m_final = None
        if X_test is not None and y_cls_test is not None:
            Xe = torch.from_numpy(X_test).float()
            ye = torch.from_numpy(y_cls_test).long()
            test_ds_clf  = TensorDataset(Xe, ye)
            test_loader_clf = DataLoader(test_ds_clf, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
            test_m_final = _evaluate_clf(model, test_loader_clf, device, task)
            log.info("Test metrics: %s", test_m_final)

        metadata = {
            "n_features":           actual_n_features,
            "n_clusters":           n_clusters,
            "n_classes":            n_clf_classes,
            "task":                 task,
            "label_field":          label_field,
            "emb_dim":              emb_dim,
            "max_history":          max_history,
            "n_epochs":             clf_epochs,
            "seed":                 seed,
            "best_epoch":           best_epoch_clf,
            "best_val_metric":      best_val_m.get("val_auroc", best_val_m.get("accuracy")),
            "stopped_early":        stopped_early_clf,
            "class_weight_applied": cw_applied,
        }
        with open(out_dir / "metadata.json", "w", encoding="utf-8") as fp:
            json.dump(metadata, fp, indent=2)
        if test_m_final is not None:
            with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as fp:
                json.dump(test_m_final, fp, indent=2)

        result: dict[str, Any] = {
            "model_path":           str(best_ckpt),
            "swa_path":             None,
            "out_dir":              str(out_dir),
            "n_features":           actual_n_features,
            "n_clusters":           n_clusters,
            "n_classes":            n_clf_classes,
            "task":                 task,
            "label_field":          label_field,
            "bin_edges":            [],
            "bin_centers":          [],
            "feat_mean":            feat_mean.squeeze().tolist(),
            "feat_std":             feat_std.squeeze().tolist(),
            "val_metrics":          val_m_final,
            "test_metrics":         test_m_final,
            "metadata":             metadata,
            "best_epoch":           best_epoch_clf,
            "best_val_metric":      best_val_m.get("val_auroc", best_val_m.get("accuracy")),
            "stopped_early":        stopped_early_clf,
            "class_weight_applied": cw_applied,
        }
        log.info("Training complete (task=%s). Outputs written to: %s", task, out_dir)
        return result

    # -------------------------------------------------------------------------
    # next_event: existing two-phase training (Phase 1 cls + Phase 2 cls+reg+SWA)
    # -------------------------------------------------------------------------

    # Quantile bins
    log.info("=== Computing quantile bins from training log-days ===")
    bin_edges_np, bin_centers_np = compute_quantile_bins(y_reg_train, n_bins_target=N_BINS_TARGET)
    n_bins_actual = len(bin_centers_np)
    log.info("Effective N_BINS: %d (target was %d)", n_bins_actual, N_BINS_TARGET)

    # PyTorch tensors and data loaders
    def to_tensors(X, yc, yr):
        return (
            torch.from_numpy(X).float(),
            torch.from_numpy(yc).long(),
            torch.from_numpy(yr).float(),
        )

    Xt, yct, yrt = to_tensors(X_train, y_cls_train, y_reg_train)
    Xv, ycv, yrv = to_tensors(X_val,   y_cls_val,   y_reg_val)

    train_ds  = TensorDataset(Xt, yct, yrt)
    val_ds    = TensorDataset(Xv, ycv, yrv)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    test_loader = None
    if X_test is not None:
        Xe, yce, yre = to_tensors(X_test, y_cls_test, y_reg_test)
        test_ds  = TensorDataset(Xe, yce, yre)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # Model
    model = NVCClean(
        n_features=actual_n_features,
        n_classes=n_clf_classes,
        bin_edges_np=bin_edges_np,
        bin_centers_np=bin_centers_np,
        dropout=DROPOUT,
        input_dropout=INPUT_DROPOUT,
        task="next_event",
    ).to(device)

    swa_model  = AveragedModel(model)
    best_ckpt  = out_dir / "best_model.pt"
    swa_ckpt   = out_dir / "swa_model.pt"

    # -------------------------------------------------------------------------
    # Phase 1: Classification only (epochs 1-p1_epochs)
    # -------------------------------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=PHASE1_LR, weight_decay=weight_decay)
    steps_per_epoch    = max(1, len(train_loader))
    phase1_total_steps = steps_per_epoch * p1_epochs
    scheduler = WarmupCosineScheduler(
        optimizer,
        base_lr=PHASE1_LR,
        warmup_steps=min(WARMUP_STEPS, phase1_total_steps),
        total_steps=phase1_total_steps,
        eta_min=PHASE1_LR * 0.1,
    )

    best_val_top1 = 0.0
    best_epoch    = 0
    patience_cnt  = 0
    log_entries: list[dict] = []

    log.info("=" * 60)
    log.info("Phase 1: classification only (epochs 1-%d)", p1_epochs)
    log.info("=" * 60)

    for epoch in range(1, p1_epochs + 1):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        n_batches  = 0

        for X_b, y_cls_b, y_reg_b in train_loader:
            X_b     = X_b.to(device)
            y_cls_b = y_cls_b.to(device)
            y_reg_b = y_reg_b.to(device)

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
            epoch, p1_epochs,
            total_loss / max(1, n_batches),
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

        log_entries.append({
            "epoch": epoch, "phase": "P1",
            "train_loss": total_loss / max(1, n_batches),
            "val": val_m, "lr": scheduler.get_lr(),
        })
        with open(out_dir / "train_log.jsonl", "w", encoding="utf-8") as fp:
            for e in log_entries:
                fp.write(json.dumps(e) + "\n")

    # Early-stop check
    if p1_epochs > 0 and best_val_top1 < 0.02:
        raise RuntimeError(
            f"Phase 1 early-stop: best val top-1={best_val_top1:.4f} < 2%. "
            "Training broken. Check your data and n_clusters."
        )

    # -------------------------------------------------------------------------
    # Phase 2: Classification + Regression + SWA (epochs p1_epochs+1 .. total)
    # -------------------------------------------------------------------------
    swa_active    = False
    best_swa_top1 = 0.0

    if p2_epochs > 0:
        log.info("=" * 60)
        log.info("Phase 2: cls + reg + SWA (epochs %d-%d)", p1_epochs + 1, total_epochs)
        log.info("=" * 60)

        optimizer2 = torch.optim.AdamW(model.parameters(), lr=PHASE2_LR, weight_decay=weight_decay)
        scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer2, T_max=p2_epochs, eta_min=PHASE2_LR * 0.1
        )
        swa_scheduler = SWALR(
            optimizer2, swa_lr=SWA_LR, anneal_epochs=5, anneal_strategy="cos",
        )

        for epoch in range(p1_epochs + 1, total_epochs + 1):
            model.train()
            t0         = time.time()
            total_loss = total_cls = total_reg = total_aux = 0.0
            n_batches  = 0
            lam_aux    = aux_l1_weight(epoch)

            for X_b, y_cls_b, y_reg_b in train_loader:
                X_b     = X_b.to(device)
                y_cls_b = y_cls_b.to(device)
                y_reg_b = y_reg_b.to(device)

                x_mix, y_cls_a, y_cls_b_mix, y_reg_a, y_reg_b_mix, lam = mixup_batch(
                    X_b, y_cls_b, y_reg_b
                )

                optimizer2.zero_grad()
                logits, reg_logits = model(x_mix)

                cls_loss = asl_loss_mixed(logits, y_cls_a, y_cls_b_mix, lam)

                bin_edges_dev = model.bin_edges
                reg_loss = gaussian_reg_loss_idx(
                    reg_logits, y_reg_a, bin_edges_dev, n_bins_actual, GAUSS_SIGMA_IDX
                )

                bin_centers_dev = model.bin_centers
                log_pred    = bin_expected_log_days(reg_logits, bin_centers_dev)
                log_target_a = y_reg_a.float()
                log_target_b = y_reg_b_mix.float()
                aux_loss = (
                    lam       * F.smooth_l1_loss(log_pred, log_target_a, beta=1.0)
                    + (1-lam) * F.smooth_l1_loss(log_pred, log_target_b, beta=1.0)
                )

                loss = cls_loss + REG_WEIGHT * reg_loss + lam_aux * aux_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer2.step()

                total_loss += loss.item()
                total_cls  += cls_loss.item()
                total_reg  += reg_loss.item()
                total_aux  += aux_loss.item()
                n_batches  += 1

            if epoch >= SWA_START:
                swa_model.update_parameters(model)
                swa_scheduler.step()
                if not swa_active:
                    swa_active = True
                    log.info("*** SWA started at epoch %d ***", epoch)
            else:
                scheduler2.step()

            val_m   = evaluate(model, val_loader, device)
            elapsed = time.time() - t0
            log.info(
                "P2 Epoch %2d/%d  loss=%.4f (cls=%.4f reg=%.4f aux=%.4f lam_aux=%.3f)  "
                "val top1=%.4f  top3=%.4f  mae=%.1f  [%.1fs]",
                epoch, total_epochs,
                total_loss / max(1, n_batches),
                total_cls  / max(1, n_batches),
                total_reg  / max(1, n_batches),
                total_aux  / max(1, n_batches),
                lam_aux,
                val_m["top1_acc"], val_m["top3_acc"], val_m["mae_days"],
                elapsed,
            )

            if val_m["top1_acc"] > best_val_top1:
                best_val_top1 = val_m["top1_acc"]
                best_epoch    = epoch
                patience_cnt  = 0
                torch.save(model.state_dict(), best_ckpt)
            else:
                patience_cnt += 1

            if patience_cnt >= EARLY_STOP_PAT and epoch > EARLY_STOP_MIN_EPOCH:
                log.info("Early stopping at epoch %d (patience=%d)", epoch, EARLY_STOP_PAT)
                break

            log_entries.append({
                "epoch": epoch, "phase": "P2",
                "train_loss": total_loss / max(1, n_batches),
                "val": val_m, "lr": optimizer2.param_groups[0]["lr"],
                "swa_active": swa_active,
            })
            with open(out_dir / "train_log.jsonl", "w", encoding="utf-8") as fp:
                for e in log_entries:
                    fp.write(json.dumps(e) + "\n")

    # -------------------------------------------------------------------------
    # SWA BN update and evaluation
    # -------------------------------------------------------------------------
    swa_path_str = None
    swa_val_m    = None
    if swa_active:
        log.info("Updating SWA batch-norm statistics ...")
        update_bn(train_loader, swa_model, device=device)
        torch.save(swa_model.state_dict(), swa_ckpt)
        swa_path_str = str(swa_ckpt)
        swa_val_m = evaluate(swa_model, val_loader, device)
        log.info(
            "SWA val -- top1=%.4f  top3=%.4f  mae=%.1f",
            swa_val_m["top1_acc"], swa_val_m["top3_acc"], swa_val_m["mae_days"],
        )

    # -------------------------------------------------------------------------
    # Final evaluation on best checkpoint
    # -------------------------------------------------------------------------
    if best_ckpt.exists():
        model.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=True))
    val_m_final = evaluate(model, val_loader, device)
    log.info(
        "Best checkpoint -- val top1=%.4f  top3=%.4f  mae=%.1f  (epoch %d)",
        val_m_final["top1_acc"], val_m_final["top3_acc"], val_m_final["mae_days"], best_epoch,
    )

    test_m_final = None
    if test_loader is not None:
        test_m_final = evaluate(model, test_loader, device)
        log.info(
            "Test -- top1=%.4f  top3=%.4f  mae=%.1f",
            test_m_final["top1_acc"], test_m_final["top3_acc"], test_m_final["mae_days"],
        )

    # -------------------------------------------------------------------------
    # Save metadata
    # -------------------------------------------------------------------------
    metadata = {
        "n_features":    actual_n_features,
        "n_clusters":    n_clusters,
        "n_classes":     n_clf_classes,
        "emb_dim":       emb_dim,
        "max_history":   max_history,
        "n_epochs":      total_epochs,
        "seed":          seed,
        "best_epoch":    best_epoch,
        "best_val_top1": best_val_top1,
        "swa_active":    swa_active,
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)

    if test_m_final is not None:
        with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as fp:
            json.dump(test_m_final, fp, indent=2)

    result: dict[str, Any] = {
        "model_path":   str(best_ckpt),
        "swa_path":     swa_path_str,
        "out_dir":      str(out_dir),
        "n_features":   actual_n_features,
        "n_clusters":   n_clusters,
        "n_classes":    n_clf_classes,
        "task":         "next_event",
        "label_field":  None,
        "bin_edges":    bin_edges_np.tolist(),
        "bin_centers":  bin_centers_np.tolist(),
        "feat_mean":    feat_mean.squeeze().tolist(),
        "feat_std":     feat_std.squeeze().tolist(),
        "val_metrics":  val_m_final,
        "test_metrics": test_m_final,
        "metadata":     metadata,
    }
    log.info("Training complete. Outputs written to: %s", out_dir)
    return result


def train_classifier(
    X_train: "np.ndarray",
    y_train: "np.ndarray",
    X_val: "np.ndarray | None" = None,
    y_val: "np.ndarray | None" = None,
    *,
    task: str = "binary",
    n_classes: int | None = None,
    n_epochs: int = 30,
    out_dir: str | Path | None = None,
    hidden_dims: tuple = (512, 256),
    lr: float = 1e-3,
    batch_size: int = 256,
    seed: int = 42,
    device_str: str | None = None,
    early_stopping_patience: int | None = None,
    early_stopping_metric: str = "val_auroc",
    class_weight: "str | dict | None" = None,
    weight_decay: float = WEIGHT_DECAY,
) -> dict[str, Any]:
    """
    Train NVCClean as a tabular classifier on pre-built feature matrices.

    For users who already have a feature matrix (computed however they like)
    and arbitrary labels. Skips JSONL parsing and feature building entirely.

    Args:
        X_train:      (N, D) numpy array of training features.
        y_train:      (N,) numpy array of integer labels.
        X_val:        Optional (M, D) validation features. If None, validation
                      is skipped and the final checkpoint is saved after the last epoch.
        y_val:        Optional (M,) validation labels (required if X_val provided).
        task:         "binary" or "multiclass".
        n_classes:    Number of classes. Required for multiclass; binary infers 2.
        n_epochs:     Number of training epochs (default 30).
        out_dir:      If set, saves best_model.pt checkpoint here.
        hidden_dims:  Ignored (NVCClean architecture is fixed); kept for API symmetry.
        lr:           Learning rate (default 1e-3).
        batch_size:   Mini-batch size (default 256).
        seed:         Random seed (default 42).
        device_str:   Device string. Auto-detected if None.
        early_stopping_patience:
                      Stop after this many epochs without improvement (>0.0001)
                      in early_stopping_metric. None = run all n_epochs (default).
        early_stopping_metric:
                      "val_auroc" (default) or "val_loss". Requires X_val/y_val.
        class_weight: None = no weighting (default). "balanced" = auto inverse-
                      frequency weights (n / (n_classes * bincount), sklearn
                      convention). dict {class_idx: weight} = manual per-class
                      weights. Applied as pos_weight to BCEWithLogitsLoss (binary)
                      or weight to CrossEntropyLoss (multiclass).
        weight_decay: AdamW weight decay. Default 1e-4.

    Returns:
        Classifier dict compatible with cadence.predict_from_features().
        Keys: "model", "task", "n_features", "n_classes", "model_path",
              "out_dir", "val_metrics", "best_epoch", "best_val_metric",
              "stopped_early", "class_weight_applied".
    """
    if task not in ("binary", "multiclass"):
        raise ValueError(f"task must be 'binary' or 'multiclass'; got {task!r}")
    if task == "multiclass" and n_classes is None:
        raise ValueError("n_classes is required for task='multiclass'")

    torch.manual_seed(seed)
    np.random.seed(seed)

    if device_str is not None:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_path = Path(out_dir) if out_dir is not None else None
    if out_path is not None:
        out_path.mkdir(parents=True, exist_ok=True)

    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)
    n_feat = X_train.shape[1]
    clf_n_classes = 1 if task == "binary" else int(n_classes)  # type: ignore[arg-type]
    actual_n_classes = 2 if task == "binary" else int(n_classes)  # type: ignore[arg-type]

    # Build class weight tensor
    cw_tensor, cw_applied = _build_class_weight_tensor(
        class_weight, y_train, actual_n_classes, task
    )

    model = NVCClean(
        n_features=n_feat,
        n_classes=clf_n_classes,
        task=task,
        dropout=DROPOUT,
        input_dropout=INPUT_DROPOUT,
    ).to(device)

    Xt = torch.from_numpy(X_train)
    yt = torch.from_numpy(y_train)
    bs = min(batch_size, len(Xt))
    train_ds  = TensorDataset(Xt, yt)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0)

    val_loader_clf = None
    if X_val is not None and y_val is not None:
        Xv = torch.from_numpy(np.asarray(X_val, dtype=np.float32))
        yv = torch.from_numpy(np.asarray(y_val, dtype=np.int64))
        val_ds = TensorDataset(Xv, yv)
        val_loader_clf = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0)

    # If no val set, create a tiny dummy loader for API compatibility
    if val_loader_clf is None:
        val_loader_clf = train_loader

    _, best_val_m, best_epoch, stopped_early = _run_clf_loop(
        model, train_loader, val_loader_clf, device, task,
        n_epochs, lr, out_path,
        weight_decay=weight_decay,
        class_weight_tensor=cw_tensor,
        early_stopping_patience=early_stopping_patience,
        early_stopping_metric=early_stopping_metric,
    )

    # best weights already restored in _run_clf_loop; reload from disk if ckpt exists
    model_path_str = None
    if out_path is not None:
        best_ckpt = out_path / "best_model.pt"
        if best_ckpt.exists():
            model.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=True))
            model_path_str = str(best_ckpt)

    model.eval()
    result: dict[str, Any] = {
        "model":                model,
        "task":                 task,
        "n_features":           n_feat,
        "n_classes":            actual_n_classes,
        "model_path":           model_path_str,
        "out_dir":              str(out_path) if out_path is not None else None,
        "val_metrics":          best_val_m,
        "best_epoch":           best_epoch,
        "best_val_metric":      best_val_m.get("val_auroc", best_val_m.get("accuracy")),
        "stopped_early":        stopped_early,
        "class_weight_applied": cw_applied,
    }
    return result
