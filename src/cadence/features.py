"""
cadence/features.py -- Feature engineering for the Cadence model.

All functions accept explicit path arguments; no hardcoded MIMIC paths exist here.
Public users supply their own embeddings and JSONL data.

Feature layout (public configuration, 270 base + 2*emb_dim):
  Group A: sequence structure (12 dims)
  Group B: cluster bag-of-words (n_clusters dims)
  Group C: last-event one-hot (n_clusters dims)
  Group D: population anomaly features (n_clusters*3 + max_history + 3 dims)
  Group E: NV velocity scalars (5 dims)
  Embedding mean: emb_dim dims (mean of last k=10 event embeddings)
  Embedding last: emb_dim dims (most-recent event embedding)

Total for n_clusters=50, max_history=10, emb_dim=768:
  12 + 50 + 50 + (50+50+50+10+3) + 5 = 270 base
  + 768 emb-mean + 768 emb-last = 1806 dims

The paper's 2420-dim model includes additional structured (150) and temporal (464)
features derived from MIMIC-IV-specific preprocessing; those are not available in
the public training path. A model trained here will use 270+2*emb_dim input dims.
"""
from __future__ import annotations

import json
import logging
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

LOG_DAYS_CLIP: float = 12.0
EPS: float = 1e-8


# ===============================================================================
# Population prior
# ===============================================================================

def build_population_prior(train_jsonl: Path, n_clusters: int) -> dict:
    """
    Compute population-level statistics from training sequences.

    Reads every record in train_jsonl and accumulates:
      - Per-cluster frequencies
      - Pairwise transition frequencies (from_cluster -> to_cluster)
      - Per-transition gap medians and MADs
      - Global gap median and MAD

    Args:
        train_jsonl: Path to training JSONL file.
        n_clusters:  Number of event clusters (must match the event_index cluster IDs).

    Returns:
        dict with keys: cluster_freq, common_cluster_threshold,
            transition_gap_median, transition_gap_mad, global_gap_median,
            global_gap_mad, transition_freq, transition_common_threshold,
            n_clusters.
    """
    import statistics

    log.info("Building population prior from %s ...", train_jsonl)
    cluster_counts: Counter = Counter()
    transition_gaps: dict[str, list[float]] = defaultdict(list)
    all_gaps: list[float] = []
    transition_counts: list[list[int]] = [[0] * n_clusters for _ in range(n_clusters)]

    with open(train_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            history = rec["history"]
            target = rec["target"]
            cluster_counts[target["cluster_id"]] += 1
            if history:
                from_c = history[-1]["cluster_id"]
                to_c = target["cluster_id"]
                gap = float(target["days_from_prev"])
                transition_gaps[f"{from_c}->{to_c}"].append(gap)
                all_gaps.append(gap)
                if 0 <= from_c < n_clusters and 0 <= to_c < n_clusters:
                    transition_counts[from_c][to_c] += 1

    total_targets = sum(cluster_counts.values())
    cluster_freq = [
        (cluster_counts.get(c, 0) + 1) / (total_targets + n_clusters)
        for c in range(n_clusters)
    ]
    common_cluster_threshold = 2.0 / n_clusters

    transition_gap_median: dict[str, float] = {}
    transition_gap_mad: dict[str, float] = {}
    for key, gaps in transition_gaps.items():
        med = statistics.median(gaps)
        mad = statistics.median([abs(g - med) for g in gaps]) if len(gaps) > 1 else 1.0
        transition_gap_median[key] = med
        transition_gap_mad[key] = max(mad, 1.0)

    global_gap_median = statistics.median(all_gaps) if all_gaps else 7.0
    global_gap_mad = statistics.median(
        [abs(g - global_gap_median) for g in all_gaps]
    ) if len(all_gaps) > 1 else 1.0
    global_gap_mad = max(global_gap_mad, 1.0)

    transition_freq: list[list[float]] = []
    for i in range(n_clusters):
        row_total = sum(transition_counts[i]) + n_clusters
        transition_freq.append([
            (transition_counts[i][j] + 1) / row_total for j in range(n_clusters)
        ])

    transition_common_threshold = 2.0 / n_clusters

    log.info("Prior built: %d clusters, global median gap=%.1f days", n_clusters, global_gap_median)
    return {
        "cluster_freq": cluster_freq,
        "common_cluster_threshold": common_cluster_threshold,
        "transition_gap_median": transition_gap_median,
        "transition_gap_mad": transition_gap_mad,
        "global_gap_median": global_gap_median,
        "global_gap_mad": global_gap_mad,
        "transition_freq": transition_freq,
        "transition_common_threshold": transition_common_threshold,
        "n_clusters": n_clusters,
    }


# ===============================================================================
# Population anomaly features (Group D, 3*n_clusters + max_history + 3 dims)
# ===============================================================================

def compute_population_anomaly_features(
    history: list[dict],
    prior: dict,
    max_history: int,
) -> dict:
    """
    Compute population-relative anomaly signals for a single patient history.

    Returns:
        dict with keys: missing_mask (n_clusters), gap_anomalies (max_history),
            kl_divergence (scalar), transition_next_probs (n_clusters),
            last_event_missing_mask (n_clusters).
    """
    n_clusters = prior["n_clusters"]
    cluster_freq = prior["cluster_freq"]
    threshold = prior["common_cluster_threshold"]
    transition_gap_median = prior["transition_gap_median"]
    transition_gap_mad = prior["transition_gap_mad"]
    global_gap_median = prior["global_gap_median"]
    global_gap_mad = prior["global_gap_mad"]
    transition_freq = prior.get("transition_freq")
    transition_common_threshold = prior.get("transition_common_threshold", 2.0 / n_clusters)

    if len(history) > max_history:
        history = history[-max_history:]
    hist_len = len(history)

    seen_clusters = set(h["cluster_id"] for h in history)
    missing_mask = [
        1 if (cluster_freq[c] > threshold and c not in seen_clusters) else 0
        for c in range(n_clusters)
    ]

    gap_anomalies = [0.0] * max_history
    for i in range(1, hist_len):
        from_c = history[i - 1]["cluster_id"]
        to_c = history[i]["cluster_id"]
        gap = max(history[i]["days_from_start"] - history[i - 1]["days_from_start"], 0.5)
        key = f"{from_c}->{to_c}"
        med = transition_gap_median.get(key, global_gap_median)
        mad = transition_gap_mad.get(key, global_gap_mad)
        z = max(-5.0, min(5.0, (gap - med) / (1.4826 * mad)))
        gap_anomalies[max_history - hist_len + i] = float(z)

    hist_counter = Counter(h["cluster_id"] for h in history)
    total_hist = sum(hist_counter.values())
    kl = 0.0
    if total_hist > 0:
        for c in range(n_clusters):
            p = hist_counter.get(c, 0) / total_hist
            q = cluster_freq[c]
            if p > 0:
                kl += p * math.log((p + 1e-9) / (q + 1e-9))
        kl = max(0.0, kl)

    if hist_len > 0 and transition_freq is not None:
        last_c = history[-1]["cluster_id"]
        transition_next_probs = transition_freq[last_c] if 0 <= last_c < n_clusters else cluster_freq
    else:
        transition_next_probs = cluster_freq

    if hist_len > 0 and transition_freq is not None:
        last_c = history[-1]["cluster_id"]
        if 0 <= last_c < n_clusters:
            row = transition_freq[last_c]
            last_event_missing_mask = [
                1 if (row[c] > transition_common_threshold and c not in seen_clusters) else 0
                for c in range(n_clusters)
            ]
        else:
            last_event_missing_mask = [0] * n_clusters
    else:
        last_event_missing_mask = [0] * n_clusters

    return {
        "missing_mask": missing_mask,
        "gap_anomalies": gap_anomalies,
        "kl_divergence": kl,
        "transition_next_probs": transition_next_probs,
        "last_event_missing_mask": last_event_missing_mask,
    }


# ===============================================================================
# NV velocity scalars (Group E, 5 dims)
# ===============================================================================

def compute_nv_scalars(
    patient_id: str,
    history: list[dict],
    embeddings: np.ndarray,
    event_id_map: dict,
    max_history: int,
) -> dict:
    """
    Compute Narrative Velocity scalars for a single patient's history.

    These 5 scalars characterize the *dynamics* of the embedding trajectory:
      velocity_mean, velocity_std, velocity_trend, turbulence_onset,
      semantic_viscosity.

    Args:
        patient_id:  Patient identifier (str).
        history:     List of history dicts (each has event_index, days_from_start).
        embeddings:  (N_events, emb_dim) array.
        event_id_map: Maps (patient_id_str, event_index) -> row index in embeddings.
        max_history: Maximum history length to consider.

    Returns:
        dict with keys: velocity_mean, velocity_std, velocity_trend,
            turbulence_onset, semantic_viscosity.
    """
    if len(history) > max_history:
        history = history[-max_history:]

    hist_len = len(history)
    emb_dim = embeddings.shape[1]

    embs: list[np.ndarray] = []
    for h in history:
        key = (str(patient_id), h["event_index"])
        row = event_id_map.get(key)
        if row is not None:
            embs.append(embeddings[row].astype(np.float32))
        else:
            embs.append(np.zeros(emb_dim, dtype=np.float32))

    velocity_real: list[float] = []
    for i in range(hist_len):
        if i == 0:
            velocity_real.append(0.0)
        else:
            dist = float(np.linalg.norm(embs[i] - embs[i - 1]))
            gap = max(history[i]["days_from_start"] - history[i - 1]["days_from_start"], 0.5)
            velocity_real.append(dist / gap)

    real_v = [v for v in velocity_real if not math.isnan(v)]
    if len(real_v) == 0:
        return {
            "velocity_mean": 0.0, "velocity_std": 0.0, "velocity_trend": 0.0,
            "turbulence_onset": 0.0, "semantic_viscosity": 1.0,
        }

    velocity_mean = float(np.mean(real_v))
    velocity_std = float(np.std(real_v)) if len(real_v) > 1 else 0.0

    if len(real_v) >= 2:
        xs = np.arange(len(real_v), dtype=np.float32)
        ys = np.array(real_v, dtype=np.float32)
        xs_c = xs - xs.mean()
        denom = float(np.dot(xs_c, xs_c))
        velocity_trend = float(np.dot(xs_c, ys) / denom) if denom > EPS else 0.0
    else:
        velocity_trend = 0.0

    turbulence_onset = float(max(real_v)) / (velocity_mean + EPS)

    if hist_len < 2:
        semantic_viscosity = 1.0
    else:
        norms = np.linalg.norm(np.stack(embs), axis=1, keepdims=True)
        norms = np.maximum(norms, EPS)
        embs_normed = np.stack(embs) / norms
        cos_sim = embs_normed @ embs_normed.T
        upper_tri = cos_sim[np.triu_indices(hist_len, k=1)]
        semantic_viscosity = float(upper_tri.mean()) if len(upper_tri) > 0 else 1.0

    return {
        "velocity_mean": velocity_mean,
        "velocity_std": velocity_std,
        "velocity_trend": velocity_trend,
        "turbulence_onset": turbulence_onset,
        "semantic_viscosity": semantic_viscosity,
    }


# ===============================================================================
# Single-record feature extraction (270 base dims)
# ===============================================================================

def extract_features(
    record: dict,
    prior: dict,
    embeddings: np.ndarray,
    event_id_map: dict,
    n_clusters: int,
    max_history: int,
) -> np.ndarray:
    """
    Extract 270-dim base feature vector for a single patient record.

    Feature groups:
      A: sequence structure (12)
      B: cluster bag-of-words (n_clusters)
      C: last-event one-hot (n_clusters)
      D: population anomaly (3*n_clusters + max_history + 3)
      E: NV velocity scalars (5)

    For n_clusters=50, max_history=10: 12+50+50+163+5 = 280 dims.
    For n_clusters=4, max_history=5:  12+4+4+17+5   = 42 dims.
    The exact dim = 12 + 2*n_clusters + (3*n_clusters + max_history + 3) + 5
                  = 5*n_clusters + max_history + 20.

    Args:
        record:      Single JSONL record dict.
        prior:       Population prior from build_population_prior().
        embeddings:  (N_events, emb_dim) float32 array.
        event_id_map: Maps (patient_id_str, event_index) -> row index.
        n_clusters:  Number of event clusters.
        max_history: Maximum history window.

    Returns:
        (D,) float32 feature vector.
    """
    history = record["history"][-max_history:] if len(record["history"]) > max_history else record["history"]
    hist_len = len(history)
    feats: list[float] = []

    # Group A: sequence structure (12)
    feats.append(float(hist_len))
    for i in range(1, 6):
        if hist_len >= i:
            feats.append(float(history[-i]["cluster_id"]))
        else:
            feats.append(0.0)

    if hist_len >= 2:
        gaps = [
            max(history[i]["days_from_start"] - history[i - 1]["days_from_start"], 0.5)
            for i in range(1, hist_len)
        ]
        last_gap = gaps[-1]
    else:
        gaps = []
        last_gap = 0.5

    feats.append(math.log1p(last_gap) if hist_len > 0 else 0.0)

    if gaps:
        log_gaps = [math.log1p(g) for g in gaps]
        feats.extend([
            float(np.mean(log_gaps)),
            float(np.std(log_gaps)) if len(log_gaps) >= 2 else 0.0,
            float(np.min(log_gaps)),
            float(np.max(log_gaps)),
        ])
    else:
        feats.extend([0.0, 0.0, 0.0, 0.0])

    feats.append(float(history[-1]["days_from_start"]) / 3650.0 if hist_len > 0 else 0.0)

    # Group B: cluster bag-of-words (n_clusters)
    bow = [0.0] * n_clusters
    for h in history:
        cid = h["cluster_id"]
        if 0 <= cid < n_clusters:
            bow[cid] += 1.0
    feats.extend(bow)

    # Group C: last-event one-hot (n_clusters)
    onehot = [0.0] * n_clusters
    if hist_len > 0:
        last_cid = history[-1]["cluster_id"]
        if 0 <= last_cid < n_clusters:
            onehot[last_cid] = 1.0
    feats.extend(onehot)

    # Group D: population anomaly (3*n_clusters + max_history + 3)
    pop = compute_population_anomaly_features(history, prior, max_history)
    feats.append(float(pop["kl_divergence"]))
    feats.extend([float(p) for p in pop["transition_next_probs"]])
    feats.extend([float(v) for v in pop["missing_mask"]])
    feats.extend([float(v) for v in pop["last_event_missing_mask"]])

    gap_anom = [abs(float(g)) for g in pop["gap_anomalies"]]
    feats.append(float(np.mean(gap_anom)))
    feats.append(float(np.max(gap_anom)) if gap_anom else 0.0)

    # Group E: NV velocity scalars (5)
    nv = compute_nv_scalars(
        str(record["patient_id"]), history, embeddings, event_id_map, max_history
    )
    feats.append(nv["velocity_mean"])
    feats.append(nv["velocity_std"])
    feats.append(nv["velocity_trend"])
    feats.append(nv["turbulence_onset"])
    feats.append(nv["semantic_viscosity"])

    return np.array(feats, dtype=np.float32)


# ===============================================================================
# Embedding helpers
# ===============================================================================

def compute_mean_last_k_embedding(
    patient_id: str,
    history: list[dict],
    embeddings: np.ndarray,
    event_id_map: dict,
    k: int = 10,
) -> np.ndarray:
    """Mean of the last k event embeddings; zeros if no valid keys found."""
    emb_dim = embeddings.shape[1]
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


def compute_last_event_embedding(
    patient_id: str,
    history: list[dict],
    embeddings: np.ndarray,
    event_id_map: dict,
) -> np.ndarray:
    """768-dim embedding of the most recent event; zeros if history is empty."""
    emb_dim = embeddings.shape[1]
    if not history:
        return np.zeros(emb_dim, dtype=np.float32)
    for h in reversed(history):
        key = (str(patient_id), h["event_index"])
        row = event_id_map.get(key)
        if row is not None:
            return embeddings[row].astype(np.float32)
    return np.zeros(emb_dim, dtype=np.float32)


# ===============================================================================
# Feature matrix builders
# ===============================================================================

def build_base_feature_matrix(
    jsonl_path: Path,
    prior: dict,
    embeddings: np.ndarray,
    event_id_map: dict,
    n_clusters: int,
    max_history: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the 270-dim BASE feature matrix only (no embedding columns appended).

    Used internally by model.py's private MIMIC training path, which appends
    structured (150) + temporal (464) features separately before concatenating
    embedding columns. Public users should call build_feature_matrix() instead.

    Returns:
        X:          (N, base_dim) float32 base features only.
        y_cluster:  (N,) int32 target cluster IDs.
        y_log_days: (N,) float32 log1p(days_to_next_event), clipped.
    """
    log.info("Building base features from %s ...", jsonl_path)
    records: list[dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    log.info("  Loaded %d records", len(records))

    rows: list[np.ndarray] = []
    y_cluster_list: list[int] = []
    y_log_days_list: list[float] = []

    for i, rec in enumerate(records):
        if i % 10000 == 0 and i > 0:
            log.info("  Features: %d/%d", i, len(records))
        rows.append(extract_features(rec, prior, embeddings, event_id_map, n_clusters, max_history))
        y_cluster_list.append(rec["target"]["cluster_id"])
        days = max(float(rec["target"]["days_from_prev"]), 0.5)
        y_log_days_list.append(min(math.log1p(days), LOG_DAYS_CLIP))

    X          = np.stack(rows).astype(np.float32)
    y_cluster  = np.array(y_cluster_list, dtype=np.int32)
    y_log_days = np.array(y_log_days_list, dtype=np.float32)
    log.info("  Base feature matrix: %s", X.shape)
    return X, y_cluster, y_log_days


def build_feature_matrix(
    jsonl_path: Path,
    prior: dict,
    embeddings: np.ndarray,
    event_id_map: dict,
    n_clusters: int,
    max_history: int,
    k_emb: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the full feature matrix for a JSONL split (public API).

    Concatenates base features with emb-mean (emb_dim) and emb-last (emb_dim).
    Total input dim: 5*n_clusters + max_history + 20 + 2*emb_dim.

    For n_clusters=50, max_history=10, emb_dim=768: 270 + 1536 = 1806 dims.

    Args:
        jsonl_path:  Path to JSONL file (train/val/test).
        prior:       Population prior dict from build_population_prior().
        embeddings:  (N_events, emb_dim) float32 array.
        event_id_map: Maps (patient_id_str, event_index) -> row index.
        n_clusters:  Number of event clusters.
        max_history: Maximum history window.
        k_emb:       Number of recent events for emb-mean (default 10).

    Returns:
        X:          (N, D) float32 feature matrix.
        y_cluster:  (N,) int32 target cluster IDs.
        y_log_days: (N,) float32 log1p(days_to_next_event), clipped to LOG_DAYS_CLIP.
    """
    log.info("Building features from %s ...", jsonl_path)
    records: list[dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    log.info("  Loaded %d records", len(records))

    base_rows: list[np.ndarray] = []
    emb_mean_rows: list[np.ndarray] = []
    emb_last_rows: list[np.ndarray] = []
    y_cluster_list: list[int] = []
    y_log_days_list: list[float] = []

    for i, rec in enumerate(records):
        if i % 5000 == 0 and i > 0:
            log.info("  Features: %d/%d", i, len(records))

        pid = str(rec["patient_id"])
        history = rec.get("history", [])

        base_rows.append(extract_features(rec, prior, embeddings, event_id_map, n_clusters, max_history))
        emb_mean_rows.append(compute_mean_last_k_embedding(pid, history, embeddings, event_id_map, k=k_emb))
        emb_last_rows.append(compute_last_event_embedding(pid, history, embeddings, event_id_map))

        y_cluster_list.append(rec["target"]["cluster_id"])
        days = max(float(rec["target"]["days_from_prev"]), 0.5)
        y_log_days_list.append(min(math.log1p(days), LOG_DAYS_CLIP))

    X_base     = np.stack(base_rows).astype(np.float32)
    X_emb_mean = np.stack(emb_mean_rows).astype(np.float32)
    X_emb_last = np.stack(emb_last_rows).astype(np.float32)
    X = np.concatenate([X_base, X_emb_mean, X_emb_last], axis=1)

    y_cluster  = np.array(y_cluster_list, dtype=np.int32)
    y_log_days = np.array(y_log_days_list, dtype=np.float32)
    log.info("  Feature matrix: %s  (base=%d + emb_mean=%d + emb_last=%d)",
             X.shape, X_base.shape[1], X_emb_mean.shape[1], X_emb_last.shape[1])
    return X, y_cluster, y_log_days
