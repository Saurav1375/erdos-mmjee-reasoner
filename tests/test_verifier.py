"""Learned verifier: answer keys, features, models, CV and best-of-N selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mmjee_reasoner.verifier.features import answer_key, handcrafted_features, valid_format
from mmjee_reasoner.verifier.model import select, selection_accuracy


def test_answer_key_and_format():
    assert answer_key("2.50", "Numerical") == answer_key("2.5", "Numerical")
    assert answer_key("BA", "MCQ-Multiple") == "AB"
    assert valid_format("ABD", "MCQ-Multiple") and not valid_format("AB", "MCQ-Single")
    assert valid_format("-3.2", "Numerical") and not valid_format("x", "Numerical")


def _pool():
    return pd.DataFrame({
        "uid": ["q"] * 4, "pred": ["1", "2", "2", "3"], "question_type": ["Numerical"] * 4,
        "correct": [True, False, False, False], "text": ["a \\boxed{1}"] * 4,
        "completion_tokens": [10] * 4, "n_steps": [2] * 4, "n_code_blocks": [1] * 4,
        "n_code_ok": [1] * 4, "n_code_err": [0] * 4, "used_fallback": [False] * 4,
        "subject": ["Physics"] * 4, "language": ["English"] * 4, "requires_image": [False] * 4,
    })


def test_selection_methods():
    pool = _pool()
    assert select(pool, None, "majority")["pred"] == "2"
    assert select(pool, np.array([0.9, 0.1, 0.1, 0.2]), "max_prob")["correct"] == 1.0
    assert select(pool, np.array([0.9, 0.3, 0.3, 0.2]), "weighted_vote")["pred"] == "1"
    assert select(pool, None, "oracle")["correct"] == 1.0
    assert select(pool, None, "random_expected")["correct"] == pytest.approx(0.25)


def test_handcrafted_features_pool_agreement():
    f = handcrafted_features(_pool())
    assert f["agree_frac"].tolist() == pytest.approx([0, 1 / 3, 1 / 3, 0], abs=1e-6)
    assert list(f["is_majority"]) == [0, 1, 1, 0]
    assert not f.isna().any().any()


def test_verifier_learns_separable_signal(cfg):
    from mmjee_reasoner.verifier.model import FeatureMatrix, classification_metrics, make_model

    rng = np.random.default_rng(0)
    rows = []
    for q in range(60):
        for s in range(8):
            ok = rng.random() < 0.3
            rows.append({**{k: v[0] for k, v in _pool().to_dict("list").items()},
                         "uid": f"q{q}", "pred": "1" if ok else str(rng.integers(2, 9)),
                         "correct": ok, "n_code_ok": 1 if ok else 0,
                         "n_code_err": 0 if ok else 1})
    df = pd.DataFrame(rows)
    fm = FeatureMatrix(handcrafted_features(df), rng.normal(size=(len(df), 8)))
    y = df["correct"].astype(int).to_numpy()
    for model in ("logreg", "xgboost"):
        X, h, e = fm.select("all")
        pipe = make_model(model, h, e, 4, (len(y) - y.sum()) / y.sum(), 0).fit(X, y)
        auc = classification_metrics(y, pipe.predict_proba(X)[:, 1])["auc"]
        assert auc > 0.9, model
    acc, _ = selection_accuracy(df, df["correct"].astype(float).to_numpy(), "max_prob")
    assert acc == selection_accuracy(df, None, "oracle")[0]


def test_oof_predictions_with_groupkfold(cfg):
    """GroupKFold yields integer positions; _subset must index rows, not columns."""
    from mmjee_reasoner.verifier.model import FeatureMatrix
    from mmjee_reasoner.verifier.train import oof_predictions

    rng = np.random.default_rng(1)
    base = {k: v[0] for k, v in _pool().to_dict("list").items()}
    rows = [{**base, "uid": f"q{q}", "pred": str(rng.integers(1, 4)),
             "correct": bool(rng.random() < 0.4)} for q in range(30) for _ in range(4)]
    df = pd.DataFrame(rows)
    fm = FeatureMatrix(handcrafted_features(df), rng.normal(size=(len(df), 8)))
    y = df["correct"].astype(int).to_numpy()
    cfg["verifier"]["cv_folds"] = 3
    oof = oof_predictions("logreg", "all", fm, y, df["uid"].to_numpy(), cfg)
    assert oof.shape == (len(df),) and ((oof > 0) & (oof < 1)).all()
