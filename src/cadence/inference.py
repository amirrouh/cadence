"""
cadence/inference.py -- Inference API for NVCClean.

Public users call cadence.predict() to run inference on a trained checkpoint
(from cadence.train()).
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .data import load_embeddings
from .features import build_feature_matrix, build_population_prior

log = logging.getLogger(__name__)


def predict(
    classifier: dict[str, Any] | str | Path,
    jsonl_path: str | Path,
    *,
    embeddings_path: str | Path | None = None,
    event_index_path: str | Path | None = None,
    device_str: str | None = None,
    use_swa: bool = False,
) -> list[dict[str, Any]]:
    """
    Run inference with a trained Cadence NVCClean classifier.

    Args:
        classifier:       Either the dict returned by cadence.train(), or a path
                          to a metadata.json file written by cadence.train(), or
                          a path to a checkpoint .pt file (in which case
                          embeddings_path and event_index_path must be provided
                          along with a separate metadata dict -- not recommended;
                          use the train() dict form instead).
        jsonl_path:       Path to JSONL file to run predictions on.
        embeddings_path:  Path to embeddings .npy file. If classifier is a dict,
                          this is inferred from the paths used at train time (but
                          embeddings must still be passed explicitly since the
                          array is not serialized in the metadata).
        event_index_path: Path to event index JSON file. Same note as above.
        device_str:       Device string. Auto-detected if None.
        use_swa:          If True and classifier has a swa_path, load the SWA
                          checkpoint instead of the regular best checkpoint.

    Returns:
        List of prediction dicts, one per record in jsonl_path:
          {
            "patient_id":        <str>,
            "top_3_clusters":    [int, int, int],    # descending probability
            "top_3_probs":       [float, float, float],
            "days_until_next":   float,
          }

    Raises:
        FileNotFoundError: If any required path does not exist.
        ValueError: On incompatible metadata / missing required arguments.
    """
    from .model import NVCClean

    # -------------------------------------------------------------------------
    # Resolve classifier metadata
    # -------------------------------------------------------------------------
    if isinstance(classifier, (str, Path)):
        classifier = Path(classifier)
        if classifier.suffix == ".json":
            meta_path = classifier
            with open(meta_path, "r", encoding="utf-8") as fp:
                metadata = json.load(fp)
            out_dir = meta_path.parent
        else:
            raise ValueError(
                "When classifier is a path, it must point to a metadata.json file "
                "produced by cadence.train(). Got: " + str(classifier)
            )
        bin_edges   = None
        bin_centers = None
        feat_mean   = None
        feat_std    = None
        ckpt_path   = out_dir / "swa_model.pt" if use_swa else out_dir / "best_model.pt"
        if use_swa and not ckpt_path.exists():
            log.warning("swa_model.pt not found; falling back to best_model.pt")
            ckpt_path = out_dir / "best_model.pt"
        scaler_path = out_dir / "feature_scaler.npz"
        if scaler_path.exists():
            sc = np.load(scaler_path)
            feat_mean = sc["mean"]
            feat_std  = sc["std"]
        # Bin edges/centers must be in out_dir/metadata.json (train() saves them)
        raise ValueError(
            "Use the dict returned by cadence.train() as the classifier argument. "
            "Loading from a metadata.json path directly is not yet supported."
        )
    else:
        # Dict form (returned by cadence.train())
        metadata     = classifier.get("metadata", {})
        n_features   = classifier["n_features"]
        n_classes    = classifier["n_classes"]
        task         = classifier.get("task", "next_event")
        bin_edges_np = np.array(classifier.get("bin_edges", []),   dtype=np.float32)
        bin_centers_np = np.array(classifier.get("bin_centers", []), dtype=np.float32)
        feat_mean    = np.array(classifier["feat_mean"],   dtype=np.float32).reshape(1, -1)
        feat_std     = np.array(classifier["feat_std"],    dtype=np.float32).reshape(1, -1)
        n_clusters   = classifier["n_clusters"]
        max_history  = metadata.get("max_history", 10)

        ckpt_key = "swa_path" if (use_swa and classifier.get("swa_path")) else "model_path"
        ckpt_path = Path(classifier[ckpt_key])
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # -------------------------------------------------------------------------
    # Require embeddings
    # -------------------------------------------------------------------------
    if embeddings_path is None or event_index_path is None:
        raise ValueError(
            "embeddings_path and event_index_path must be provided to cadence.predict()."
        )

    # -------------------------------------------------------------------------
    # Device
    # -------------------------------------------------------------------------
    if device_str is not None:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------------------------
    # Load embeddings
    # -------------------------------------------------------------------------
    embeddings, event_id_map = load_embeddings(embeddings_path, event_index_path)

    # -------------------------------------------------------------------------
    # Build population prior and feature matrix
    # -------------------------------------------------------------------------
    prior = build_population_prior(jsonl_path, n_clusters)
    X, _, _ = build_feature_matrix(
        jsonl_path, prior, embeddings, event_id_map, n_clusters, max_history
    )

    # Standardize
    X = (X - feat_mean) / feat_std

    # -------------------------------------------------------------------------
    # Load model
    # -------------------------------------------------------------------------
    if task == "next_event":
        model = NVCClean(
            n_features=n_features,
            n_classes=n_classes,
            bin_edges_np=bin_edges_np,
            bin_centers_np=bin_centers_np,
            task="next_event",
        ).to(device)
    else:
        # binary: n_classes stored in dict is 2, but NVCClean uses cls_out=1
        clf_out = 1 if task == "binary" else n_classes
        model = NVCClean(
            n_features=n_features,
            n_classes=clf_out,
            task=task,
        ).to(device)

    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    # Handle AveragedModel wrapper (swa_model.pt has 'module.' prefix)
    if any(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items() if k != "n_averaged"}
    model.load_state_dict(state, strict=True)
    model.eval()

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------
    X_t = torch.from_numpy(X).float().to(device)

    with open(jsonl_path, "r", encoding="utf-8") as f:
        record_lines = [l.strip() for l in f if l.strip()]

    CHUNK = 512
    results: list[dict[str, Any]] = []

    if task == "next_event":
        bin_centers_t = model.bin_centers
        all_top3_clusters = []
        all_top3_probs    = []
        all_days          = []

        with torch.no_grad():
            for start in range(0, len(X_t), CHUNK):
                X_b     = X_t[start : start + CHUNK]
                logits, reg_logits = model(X_b)

                probs    = F.softmax(logits, dim=-1)
                top3     = probs.topk(min(3, probs.size(1)), dim=1)
                top3_cls = top3.indices.cpu().tolist()
                top3_p   = top3.values.cpu().tolist()

                reg_probs    = F.softmax(reg_logits, dim=-1)
                pred_log_days = (reg_probs * bin_centers_t.unsqueeze(0)).sum(-1)
                pred_days     = torch.expm1(pred_log_days).cpu().tolist()

                all_top3_clusters.extend(top3_cls)
                all_top3_probs.extend(top3_p)
                all_days.extend(pred_days)

        for i, raw_line in enumerate(record_lines):
            rec = json.loads(raw_line)
            top3 = all_top3_clusters[i]
            # Pad to 3 if n_classes < 3
            while len(top3) < 3:
                top3 = top3 + [top3[-1]]
            top3p = all_top3_probs[i]
            while len(top3p) < 3:
                top3p = top3p + [0.0]
            results.append({
                "patient_id":      str(rec["patient_id"]),
                "top_3_clusters":  top3[:3],
                "top_3_probs":     [round(p, 6) for p in top3p[:3]],
                "days_until_next": round(float(max(0.0, all_days[i])), 2),
            })

    else:
        # binary / multiclass: return probabilities per patient
        all_probs: list = []

        with torch.no_grad():
            for start in range(0, len(X_t), CHUNK):
                X_b    = X_t[start : start + CHUNK]
                logits = model(X_b)
                if task == "binary":
                    probs_b = torch.sigmoid(logits.squeeze(1)).cpu().tolist()   # list of float
                    all_probs.extend(probs_b)
                else:
                    probs_b = F.softmax(logits, dim=-1).cpu().tolist()           # list of list
                    all_probs.extend(probs_b)

        for i, raw_line in enumerate(record_lines):
            rec = json.loads(raw_line)
            results.append({
                "patient_id":   str(rec["patient_id"]),
                "probabilities": round(float(all_probs[i]), 6) if task == "binary"
                                 else [round(float(p), 6) for p in all_probs[i]],
            })

    log.info("Inference complete: %d predictions", len(results))
    return results


def predict_from_features(
    classifier: dict[str, Any],
    X: "np.ndarray",
) -> "np.ndarray":
    """
    Run inference with a classifier trained via cadence.train_classifier().

    Args:
        classifier:   Dict returned by cadence.train_classifier().
        X:            (N, D) numpy array of features (same dimensionality as training).

    Returns:
        For task="binary":     (N,) numpy array of probabilities in [0, 1].
        For task="multiclass": (N, K) numpy array of class probabilities.
    """
    import torch
    import torch.nn.functional as F
    import numpy as np

    task = classifier.get("task", "binary")
    model = classifier.get("model")

    if model is None:
        raise ValueError(
            "classifier dict has no 'model' key. Use the dict returned by "
            "cadence.train_classifier(), not cadence.train()."
        )

    device = next(model.parameters()).device
    X_t = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(device)

    model.eval()
    all_out = []
    CHUNK = 512
    with torch.no_grad():
        for start in range(0, len(X_t), CHUNK):
            X_b = X_t[start : start + CHUNK]
            logits = model(X_b)
            if task == "binary":
                probs = torch.sigmoid(logits.squeeze(1))  # (B,)
            else:
                probs = F.softmax(logits, dim=-1)          # (B, K)
            all_out.append(probs.cpu())

    result = torch.cat(all_out, dim=0).numpy()
    log.info("predict_from_features: %d samples, task=%s", len(result), task)
    return result
