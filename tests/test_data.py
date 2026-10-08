"""Dataset cleaning, twin keys, split leakage checks and the pilot subset."""

from __future__ import annotations

import pytest

from mmjee_reasoner.data.dataset import twin_key, validate_splits


def test_twin_key_pairs_languages():
    en = "2023_P1_English_Mathematics_MCQ-Single_q1_MCQ-Multiple_page1"
    hi = "2023_P1_Hindi_Mathematics_MCQ-Multiple_q1_MCQ-Multiple_page1"
    assert twin_key(en) == twin_key(hi)


def test_clean_metadata_fixes(dataset):
    assert dataset["uid"].is_unique
    assert set(dataset["split"]) == {"train", "test"}
    assert dataset.loc[dataset["question_type"] == "Numerical", "expanded_answer"].any()


def test_validate_splits_detects_leak(cfg, dataset):
    cfg["data"]["expected_counts"] = {}
    validate_splits(dataset, cfg)
    leaky = dataset.copy()
    leaky.loc[leaky.index[0], "split"] = "test"  # a 2019 row in test
    with pytest.raises(AssertionError):
        validate_splits(leaky, cfg)


def test_pilot_subset_is_stratified_and_paired(cfg, dataset):
    from mmjee_reasoner.pipeline.common import select_questions, set_subset, subset_suffix

    cfg["pilot"] = {"test_pairs": 4, "train_pairs": 6}
    set_subset(cfg, pilot=True)
    assert subset_suffix(cfg) == ".pilot"
    test = select_questions(cfg, "test")
    train = select_questions(cfg, "train")
    assert test["twin_key"].nunique() == 4 and len(test) == 8      # both languages kept
    assert train["twin_key"].nunique() == 6 and set(train["split"]) == {"train"}
    assert select_questions(cfg, "test")["uid"].tolist() == test["uid"].tolist()  # fixed


@pytest.mark.dataset
def test_real_pilot_subset(real_cfg):
    from mmjee_reasoner.pipeline.common import select_questions, set_subset

    set_subset(real_cfg, pilot=True)
    test = select_questions(real_cfg, "test")
    train = select_questions(real_cfg, "train")
    assert len(test) == 40 and test["twin_key"].nunique() == 20
    assert set(test["question_type"]) == {"Numerical", "MCQ-Single", "MCQ-Multiple", "Matching"}
    assert 145 <= len(train) <= 155 and train["year"].nunique() == 6
