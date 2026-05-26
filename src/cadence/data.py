"""
cadence/data.py -- JSONL schema, validation, and embedding loader.

Public JSONL schema for cadence.train():

  Each line in the JSONL file is one patient prediction record:

  {
    "patient_id": "<any unique string or int>",
    "history": [
      {
        "date_iso":    "<ISO 8601 date string, e.g. 2019-03-15>",
        "event_index": <int, row index into the embeddings file>,
        "cluster_id":  <int, 0..n_clusters-1>,
        "days_from_start": <float, days since first event in this history>
      },
      ...
    ],
    "target": {
      "cluster_id":    <int, 0..n_clusters-1>,
      "days_from_prev": <float, days between last history event and target event>
    }
  }

Notes:
  - history may be empty (cold-start patients); the model zero-fills features.
  - event_index must be a valid row index into your embeddings.npy (0-based).
  - cluster_id values must be in [0, n_clusters-1].
  - days_from_start is relative to the first event in the history window; it is
    used only for gap-anomaly features. If you do not have absolute timestamps,
    set days_from_start = cumulative sum of inter-event gaps.
  - days_from_prev is the regression target (time to next event in days).

Embeddings file (embeddings_path):
  A NumPy .npy file of shape (N_events, emb_dim) where emb_dim is the
  dimensionality of your per-event embeddings (e.g. 768 for PubMedBERT).
  Any embedding model works -- BERT, PubMedBERT, domain-specific, etc.
  The emb_dim must be consistent with the n_features passed to Cadence.

Event index file (event_index_path):
  A JSON file: a list of objects, each with:
    { "subject_id": <str or int>, "event_index": <int> }
  The position of each object in the list is its row index in embeddings.npy.
  load_embeddings() builds a (subject_id_str, event_index) -> row_int dict from it.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema definition (for documentation and validation)
# ---------------------------------------------------------------------------

PATIENT_RECORD_SCHEMA = {
    "patient_id": str,
    "history": [
        {
            "date_iso":       str,
            "event_index":    int,
            "cluster_id":     int,
            "days_from_start": float,
        }
    ],
    "target": {
        "cluster_id":    int,
        "days_from_prev": float,
    },
}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_jsonl(path: Path | str, n_clusters: int | None = None) -> list[dict]:
    """
    Load and validate a JSONL file against the Cadence patient record schema.

    Args:
        path:       Path to the JSONL file.
        n_clusters: If provided, validates that all cluster_id values are in
                    [0, n_clusters-1]. Pass None to skip cluster range check.

    Returns:
        List of validated record dicts.

    Raises:
        ValueError: On the first malformed record (with line number and reason).
        FileNotFoundError: If the path does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: JSON parse error: {e}") from e

            # Required top-level keys
            for key in ("patient_id", "history", "target"):
                if key not in rec:
                    raise ValueError(f"{path}:{lineno}: missing required key '{key}'")

            if not isinstance(rec["history"], list):
                raise ValueError(f"{path}:{lineno}: 'history' must be a list")

            # Validate each history event
            for ei, ev in enumerate(rec["history"]):
                for hkey in ("event_index", "cluster_id", "days_from_start"):
                    if hkey not in ev:
                        raise ValueError(
                            f"{path}:{lineno}: history[{ei}] missing key '{hkey}'"
                        )
                if not isinstance(ev["event_index"], int):
                    raise ValueError(
                        f"{path}:{lineno}: history[{ei}]['event_index'] must be int, "
                        f"got {type(ev['event_index']).__name__}"
                    )
                if n_clusters is not None:
                    cid = ev["cluster_id"]
                    if not (0 <= cid < n_clusters):
                        raise ValueError(
                            f"{path}:{lineno}: history[{ei}]['cluster_id']={cid} "
                            f"not in [0, {n_clusters-1}]"
                        )

            # Validate target
            target = rec["target"]
            for tkey in ("cluster_id", "days_from_prev"):
                if tkey not in target:
                    raise ValueError(f"{path}:{lineno}: target missing key '{tkey}'")
            if n_clusters is not None:
                tcid = target["cluster_id"]
                if not (0 <= tcid < n_clusters):
                    raise ValueError(
                        f"{path}:{lineno}: target['cluster_id']={tcid} "
                        f"not in [0, {n_clusters-1}]"
                    )

            records.append(rec)

    log.info("Validated %d records from %s", len(records), path)
    return records


# ---------------------------------------------------------------------------
# Embedding loader
# ---------------------------------------------------------------------------

def load_embeddings(
    embeddings_path: Path | str,
    event_index_path: Path | str,
) -> tuple[np.ndarray, dict[tuple[str, int], int]]:
    """
    Load per-event embeddings and build a lookup map.

    Args:
        embeddings_path:  Path to a .npy file of shape (N_events, emb_dim).
        event_index_path: Path to a JSON file (list of objects with
                          'subject_id' and 'event_index' keys).
                          The i-th object corresponds to row i of embeddings.

    Returns:
        embeddings:  (N_events, emb_dim) float32 NumPy array.
        event_id_map: dict mapping (subject_id_str, event_index_int) -> row_int.

    Raises:
        FileNotFoundError: If either path does not exist.
        ValueError: If event_index.json is malformed or has wrong length.
    """
    embeddings_path  = Path(embeddings_path)
    event_index_path = Path(event_index_path)

    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    if not event_index_path.exists():
        raise FileNotFoundError(f"Event index file not found: {event_index_path}")

    embeddings = np.load(embeddings_path).astype(np.float32)
    categories = json.loads(event_index_path.read_text(encoding="utf-8"))

    if not isinstance(categories, list):
        raise ValueError(
            f"event_index.json must be a JSON array (list), got {type(categories).__name__}"
        )
    if len(categories) != len(embeddings):
        raise ValueError(
            f"event_index.json has {len(categories)} entries but "
            f"embeddings.npy has {len(embeddings)} rows"
        )

    event_id_map: dict[tuple[str, int], int] = {}
    for i, c in enumerate(categories):
        if "subject_id" not in c or "event_index" not in c:
            raise ValueError(
                f"event_index.json[{i}] missing 'subject_id' or 'event_index' key"
            )
        event_id_map[(str(c["subject_id"]), int(c["event_index"]))] = i

    log.info(
        "Loaded embeddings %s, mapped %d events", embeddings.shape, len(event_id_map)
    )
    return embeddings, event_id_map
