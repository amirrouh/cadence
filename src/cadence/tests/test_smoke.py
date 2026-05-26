"""
Smoke test for the cadence public training + inference API.

Runs cadence.train() for 2 epochs on toy data, then cadence.predict() on the
test split. Verifies output structure without asserting specific metric values.

Runs in under 30 seconds on CPU with the toy fixture (50 events, 10 patients
per split, emb_dim=32, n_clusters=4).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np


TOY_DIR = Path(__file__).parent / "toy_data"
TOY_BINARY = Path(__file__).parent / "toy_data_binary"
TOY_MULTICLASS = Path(__file__).parent / "toy_data_multiclass"


def test_smoke() -> None:
    import cadence

    assert hasattr(cadence, "__version__"), "cadence.__version__ missing"
    assert hasattr(cadence, "train"),       "cadence.train missing"
    assert hasattr(cadence, "predict"),     "cadence.predict missing"
    assert hasattr(cadence, "NVCClean"),    "cadence.NVCClean missing"

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "smoke_run"

        # ------------------------------------------------------------------ #
        # Training
        # ------------------------------------------------------------------ #
        result = cadence.train(
            train_jsonl=TOY_DIR / "train.jsonl",
            val_jsonl=TOY_DIR / "val.jsonl",
            embeddings_path=TOY_DIR / "embeddings.npy",
            event_index_path=TOY_DIR / "event_index.json",
            n_clusters=4,
            out_dir=out_dir,
            n_epochs=2,
        )

        # Check returned dict keys
        required_keys = {
            "model_path", "out_dir", "n_features", "n_clusters", "n_classes",
            "bin_edges", "bin_centers", "feat_mean", "feat_std",
            "val_metrics", "test_metrics", "metadata",
        }
        missing = required_keys - set(result.keys())
        assert not missing, f"cadence.train() result missing keys: {missing}"

        assert result["n_clusters"] == 4,      f"n_clusters mismatch: {result['n_clusters']}"
        assert result["n_features"] > 0,       f"n_features must be positive"
        assert result["n_classes"] > 0,        f"n_classes must be positive"
        assert len(result["bin_edges"]) > 1,   "bin_edges must have at least 2 entries"
        assert len(result["bin_centers"]) > 0, "bin_centers must be non-empty"

        val_m = result["val_metrics"]
        assert "top1_acc" in val_m,  "val_metrics missing top1_acc"
        assert "mae_days"  in val_m, "val_metrics missing mae_days"

        model_path = Path(result["model_path"])
        assert model_path.exists(), f"best_model.pt not written: {model_path}"

        # ------------------------------------------------------------------ #
        # Inference
        # ------------------------------------------------------------------ #
        preds = cadence.predict(
            result,
            TOY_DIR / "test.jsonl",
            embeddings_path=TOY_DIR / "embeddings.npy",
            event_index_path=TOY_DIR / "event_index.json",
        )

        assert isinstance(preds, list), "cadence.predict() must return a list"
        assert len(preds) == 10,        f"Expected 10 predictions, got {len(preds)}"

        for i, p in enumerate(preds):
            assert "patient_id"      in p, f"pred[{i}] missing patient_id"
            assert "top_3_clusters"  in p, f"pred[{i}] missing top_3_clusters"
            assert "top_3_probs"     in p, f"pred[{i}] missing top_3_probs"
            assert "days_until_next" in p, f"pred[{i}] missing days_until_next"
            assert len(p["top_3_clusters"]) == 3, \
                f"pred[{i}] top_3_clusters should have 3 entries"
            assert len(p["top_3_probs"]) == 3, \
                f"pred[{i}] top_3_probs should have 3 entries"
            assert p["days_until_next"] >= 0.0, \
                f"pred[{i}] days_until_next must be non-negative"

    # If we get here, the smoke test passed
    print("cadence smoke test PASS")
    print(f"  n_features={result['n_features']}  n_classes={result['n_classes']}")
    print(f"  val top1={val_m['top1_acc']:.4f}  val mae={val_m['mae_days']:.1f}d")
    print(f"  {len(preds)} predictions returned, sample: {preds[0]}")


def test_smoke_binary() -> None:
    import cadence

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "smoke_binary"

        result = cadence.train(
            train_jsonl=TOY_BINARY / "train.jsonl",
            val_jsonl=TOY_BINARY / "val.jsonl",
            embeddings_path=TOY_BINARY / "embeddings.npy",
            event_index_path=TOY_BINARY / "event_index.json",
            n_clusters=4,
            out_dir=out_dir,
            n_epochs=2,
            task="binary",
            label_field="label",
        )

        assert result["task"] == "binary", f"task mismatch: {result['task']}"
        assert "model_path" in result
        assert "val_metrics" in result
        assert "accuracy" in result["val_metrics"], f"val_metrics missing accuracy: {result['val_metrics']}"

        preds = cadence.predict(
            result,
            TOY_BINARY / "test.jsonl",
            embeddings_path=TOY_BINARY / "embeddings.npy",
            event_index_path=TOY_BINARY / "event_index.json",
        )

        assert isinstance(preds, list), "predict must return a list"
        assert len(preds) == 10, f"Expected 10 predictions, got {len(preds)}"
        for i, p in enumerate(preds):
            assert "patient_id" in p, f"pred[{i}] missing patient_id"
            assert "probabilities" in p, f"pred[{i}] missing probabilities"
            prob = p["probabilities"]
            assert isinstance(prob, float), f"pred[{i}] probabilities should be float for binary, got {type(prob)}"
            assert 0.0 <= prob <= 1.0, f"pred[{i}] probability out of [0,1]: {prob}"

    print("cadence binary smoke test PASS")
    print(f"  val_metrics={result['val_metrics']}")
    print(f"  sample pred: {preds[0]}")


def test_smoke_multiclass() -> None:
    import cadence

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "smoke_multiclass"

        result = cadence.train(
            train_jsonl=TOY_MULTICLASS / "train.jsonl",
            val_jsonl=TOY_MULTICLASS / "val.jsonl",
            embeddings_path=TOY_MULTICLASS / "embeddings.npy",
            event_index_path=TOY_MULTICLASS / "event_index.json",
            n_clusters=4,
            out_dir=out_dir,
            n_epochs=2,
            task="multiclass",
            label_field="label",
            n_classes=4,
        )

        assert result["task"] == "multiclass", f"task mismatch: {result['task']}"
        assert "model_path" in result
        assert "accuracy" in result["val_metrics"], f"val_metrics missing accuracy: {result['val_metrics']}"

        preds = cadence.predict(
            result,
            TOY_MULTICLASS / "test.jsonl",
            embeddings_path=TOY_MULTICLASS / "embeddings.npy",
            event_index_path=TOY_MULTICLASS / "event_index.json",
        )

        assert isinstance(preds, list), "predict must return a list"
        assert len(preds) == 10, f"Expected 10 predictions, got {len(preds)}"
        for i, p in enumerate(preds):
            assert "patient_id" in p, f"pred[{i}] missing patient_id"
            assert "probabilities" in p, f"pred[{i}] missing probabilities"
            probs = p["probabilities"]
            assert isinstance(probs, list), f"pred[{i}] probabilities should be list for multiclass"
            assert len(probs) == 4, f"pred[{i}] expected 4 class probs, got {len(probs)}"
            assert all(0.0 <= x <= 1.0 for x in probs), f"pred[{i}] probs out of [0,1]: {probs}"

    print("cadence multiclass smoke test PASS")
    print(f"  val_metrics={result['val_metrics']}")
    print(f"  sample pred: {preds[0]}")


def test_smoke_train_classifier() -> None:
    import cadence

    rng = np.random.RandomState(0)
    X_train = rng.randn(50, 16).astype(np.float32)
    y_train = (rng.rand(50) > 0.5).astype(int)
    X_val   = rng.randn(20, 16).astype(np.float32)
    y_val   = (rng.rand(20) > 0.5).astype(int)
    X_test  = rng.randn(15, 16).astype(np.float32)

    clf = cadence.train_classifier(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        task="binary",
        n_epochs=5,
    )

    assert clf["task"] == "binary", f"task mismatch: {clf['task']}"
    assert clf["n_features"] == 16
    assert clf["n_classes"] == 2
    assert "val_metrics" in clf
    assert "accuracy" in clf["val_metrics"], f"val_metrics missing accuracy: {clf['val_metrics']}"

    probs = cadence.predict_from_features(clf, X_test)
    assert hasattr(probs, "shape"), "predict_from_features should return numpy array"
    assert probs.shape == (15,), f"Expected (15,) for binary, got {probs.shape}"
    assert all(0.0 <= p <= 1.0 for p in probs), "Probabilities out of [0,1]"

    print("cadence train_classifier smoke test PASS")
    print(f"  val_metrics={clf['val_metrics']}")
    print(f"  probs[:3]={probs[:3]}")


if __name__ == "__main__":
    test_smoke()
    test_smoke_binary()
    test_smoke_multiclass()
    test_smoke_train_classifier()
