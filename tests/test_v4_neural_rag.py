"""Phase 4: neural verifier (tiny random encoder, CPU) and RAG method cards (mock)."""

from __future__ import annotations

import numpy as np

from mmjee_reasoner.config import load_experiment
from mmjee_reasoner.data.dataset import load_questions
from mmjee_reasoner.llm import MockEngine, make_engine
from mmjee_reasoner.llm.mock import FakeVLM, golds_for
from mmjee_reasoner.llm.types import GenResult


def _tiny_encoder(path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import ModernBertConfig, ModernBertForSequenceClassification
    from transformers import PreTrainedTokenizerFast

    words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "Type", "Answer", "Solution", "finished",
             "cut", "off", "A", "B", "C", "D", "Step", "boxed"]
    tk = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="[PAD]", unk_token="[UNK]",
                                  cls_token="[CLS]", sep_token="[SEP]")
    model = ModernBertForSequenceClassification(ModernBertConfig(
        vocab_size=len(words), hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=2, num_labels=1, pad_token_id=0, cls_token_id=2, sep_token_id=3,
        bos_token_id=2, eos_token_id=3))
    tok.save_pretrained(path)
    model.save_pretrained(path)


def test_neural_verifier_loyo_resumable(cfg, dataset, tmp_path):
    from mmjee_reasoner.pipeline.correct import run_correct_stage
    from mmjee_reasoner.pipeline.solve import run_solve_stage
    from mmjee_reasoner.rag.tagger import run_transcribe_stage, run_translate_stage
    from mmjee_reasoner.verifier import neural

    qs = load_questions(cfg)
    fake = MockEngine(FakeVLM(golds_for(qs), p_correct=0.5))
    solver = make_engine(cfg, "solver", mock=fake)
    run_solve_stage(cfg, load_experiment("pot_train"), solver)
    run_correct_stage(cfg, load_experiment("train_corr_cross"),
                      mock_engines={"solver": fake, "critic_cross": fake})
    run_transcribe_stage(cfg, qs, solver)
    run_translate_stage(cfg, qs, solver)
    _tiny_encoder(tmp_path / "enc")
    cfg["verifier"]["dir_name"] = "verifier_v4"
    cfg["verifier_neural"] = {"model": str(tmp_path / "enc"), "max_len": 48,
                              "question_tokens": 8, "epochs": 1, "batch_size": 16,
                              "grad_accum": 1}
    neural.train_neural_loyo(cfg, tag="t", final=True)
    df, s = neural.load_oof(cfg, "t", with_extra=False)
    assert len(s) == len(df) and np.all((s >= 0) & (s <= 1))
    assert not df["year"].isin(cfg["data"]["test_years"]).any()     # no held-out year
    out = neural.neural_dir(cfg, "t")
    mtimes = {p.name: p.stat().st_mtime for p in out.glob("oof_*.parquet")}
    assert len(mtimes) == df["year"].nunique()
    neural.train_neural_loyo(cfg, tag="t", final=True)                  # resumes: no refit
    assert {p.name: p.stat().st_mtime for p in out.glob("oof_*.parquet")} == mtimes
    sc = neural.score_pool(cfg, "t", df.head(5))
    assert sc.shape == (5,)

    from mmjee_reasoner.verifier.v4 import VARIANTS, train_verifier_v4

    rep = train_verifier_v4(cfg, neural_tag="t")
    assert set(rep["variants"]) == set(VARIANTS)
    assert rep["variants"]["v3_ref"]["gate_pass"] is False
    assert "v3_16pool" in rep["baselines"] and "v4_pool" not in rep["baselines"]


def test_encode_rows_keeps_solution_tail():
    from mmjee_reasoner.verifier.neural import NeuralConfig, encode_rows

    class Tok:
        cls_token_id, sep_token_id = 1, 2

        def __call__(self, texts, add_special_tokens=False):
            return {"input_ids": [[int(t) for t in x.split()] for x in texts]}

    n = NeuralConfig(max_len=10, question_tokens=2)
    ids = encode_rows(Tok(), ["5 6 7"], ["8"], [" ".join(str(i) for i in range(10, 30))], n)[0]
    assert ids[:4] == [1, 5, 6, 2] and len(ids) == 10 and ids[-2] == 29 and ids[-1] == 2


class CardEngine:
    name = "card"

    def __init__(self):
        self.calls = []

    def generate(self, reqs):
        self.calls += reqs
        return [GenResult("Method: energy conservation\nPlan: a -> b\nThe answer is (B), "
                          "value 42 so \\boxed{42}.", 10, 30, "stop") for _ in reqs]


def test_cards_stage_sanitizes_and_merges(cfg, dataset, monkeypatch):
    from mmjee_reasoner.rag import cards, kb

    qs = load_questions(cfg, "train")
    e = [{"kb_id": f"k{i}", "uid": u, "problem": f"P{i}", "solution": f"S{i}"}
         for i, u in enumerate(qs["uid"].head(3))]
    monkeypatch.setattr(kb, "load_kb", lambda cfg: (e, None, {}))
    monkeypatch.setattr(cards, "_answers", lambda cfg: {e[0]["uid"]: "42"})
    eng = CardEngine()
    assert cards.run_cards_stage(cfg, eng) == 3
    got = cards.load_cards(cfg)
    c0 = got[cards.card_key(e[0])]
    assert "42" not in c0 and "(B)" not in c0 and "\\boxed" not in c0
    assert "Method: energy conservation" in c0
    assert cards.run_cards_stage(cfg, eng) == 0 and len(eng.calls) == 3   # resumable merge


def test_card_exemplar_format():
    from mmjee_reasoner.prompts.templates import format_card_exemplars, format_exemplars

    assert format_card_exemplars([]) is None
    t = format_card_exemplars([{"problem": "P", "card": "Method: x"}])
    assert "Method card:\nMethod: x" in t and "Worked solution" not in t
    assert "Worked solution:\nS" in format_exemplars([{"problem": "P", "solution": "S"}])
