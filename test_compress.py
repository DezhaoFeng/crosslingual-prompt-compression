"""Unit tests for compress.py (Gate G0 and DECISIONS.md B2/B11). Run: pytest test_compress.py"""
import json
import math
import tempfile
from pathlib import Path

import compress as cp


def test_target_budget_exact_ceil():
    assert cp.target_budget(100, 0.33) == 33
    assert cp.target_budget(10, 0.33) == 4  # ceil(3.3) = 4
    assert cp.target_budget(10, 0.5) == 5
    assert cp.target_budget(10, 0.2) == 2
    assert cp.target_budget(0, 0.5) == 0
    assert cp.target_budget(10, 0.0) == 0
    assert cp.target_budget(10, 1.0) == 10


def test_word_scores_mean_over_subwords():
    # word 0 has 2 tokens (0.2, 0.8 -> 0.5), word 1 has 1 token (0.9), word 2 has 2 tokens (0.1, 0.3 -> 0.2)
    word_ids = [0, 0, 1, 2, 2]
    probs = [0.2, 0.8, 0.9, 0.1, 0.3]
    scores = cp.word_scores_from_token_probs(word_ids, probs, 3)
    assert math.isclose(scores[0], 0.5)
    assert math.isclose(scores[1], 0.9)
    assert math.isclose(scores[2], 0.2)


def test_whitespace_only_words_excluded_from_budget_and_never_kept():
    # Spans with an empty span [2, 2] representing a whitespace-only word
    # Give the whitespace-only word score 1.0 (highest); it must still be excluded from n and get label 0
    spans = [[0, 2], [2, 2], [2, 4], [4, 6], [6, 8]]
    scores = [0.1, 1.0, 0.9, 0.8, 0.2]
    labels, k, n = cp.select_top_words(spans, scores, rho=0.5)
    assert n == 4  # 4 content words, not 5
    assert k == 2  # ceil(0.5 * 4) = 2
    assert labels[1] == 0  # whitespace-only word is never kept
    assert labels == [0, 0, 1, 1, 0]


def test_tie_breaking_keeps_earlier_word():
    spans = [[0, 1], [2, 3], [4, 5], [6, 7]]
    scores = [0.5, 0.5, 0.5, 0.5]
    labels, k, n = cp.select_top_words(spans, scores, rho=0.5)
    assert (k, n) == (2, 4)
    assert labels == [1, 1, 0, 0]


def test_b2_and_random_exact_word_retention_on_real_labeled_data():
    labels_dir = Path(__file__).parent / "data" / "labels"
    if not labels_dir.exists():
        return
    for lang in ("en", "zh"):
        path = labels_dir / f"{lang}_dev.jsonl"
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(next(f)) for _ in range(10)]
        for r in rows:
            text, spans, wids = r["text"], r["word_spans"], r["word_ids"]
            # Deterministic synthetic token probs
            t_probs = [((i * 37 + 13) % 101) / 100.0 for i in range(len(wids))]
            w_scores = cp.word_scores_from_token_probs(wids, t_probs, len(spans))
            for rho in (0.5, 0.33, 0.2):
                out_b2 = cp.compress_with_scores(text, spans, w_scores, rho)
                n_c = out_b2["content_words"]
                expected_ret = math.ceil(rho * n_c) / n_c
                assert math.isclose(out_b2["word_retention"], expected_ret, abs_tol=1e-12)
                assert sum(out_b2["labels"]) == math.ceil(rho * n_c)
                # Ensure whitespace-only words have label 0
                for (s, e), lab in zip(spans, out_b2["labels"]):
                    if s == e:
                        assert lab == 0

                out_rnd = cp.compress_random(text, spans, rho, seed=713, lang=lang, passage_sha256=r["text_sha256"])
                assert math.isclose(out_rnd["word_retention"], expected_ret, abs_tol=1e-12)
                assert sum(out_rnd["labels"]) == math.ceil(rho * n_c)
                # Re-running random with same arguments must be identical
                out_rnd2 = cp.compress_random(text, spans, rho, seed=713, lang=lang, passage_sha256=r["text_sha256"])
                assert out_rnd["labels"] == out_rnd2["labels"]
                assert out_rnd["compressed_text"] == out_rnd2["compressed_text"]


def test_score_tokens_xlmr_wraps_bos_eos_and_chunks():
    import torch

    seen_inputs = []

    class DummyOutput:
        def __init__(self, logits):
            self.logits = logits

    class DummyModel:
        def eval(self):
            return self

        def parameters(self):
            yield torch.zeros(1)

        def __call__(self, input_ids):
            seen_inputs.append(input_ids.tolist()[0])
            seq_len = input_ids.shape[1]
            # Make logit for class 1 increase with token id
            logits = torch.zeros((1, seq_len, 2), dtype=torch.float32)
            for pos in range(seq_len):
                logits[0, pos, 1] = float(input_ids[0, pos]) / 100.0
            return DummyOutput(logits)

    body_ids = list(range(10, 530))  # 520 tokens -> 2 chunks (510 + 10)
    probs = cp.score_tokens_xlmr(body_ids, DummyModel(), max_body=510)
    assert len(probs) == 520
    assert len(seen_inputs) == 2
    assert seen_inputs[0][0] == cp.XLMR_BOS_ID and seen_inputs[0][-1] == cp.XLMR_EOS_ID
    assert len(seen_inputs[0]) == 512
    assert seen_inputs[1][0] == cp.XLMR_BOS_ID and seen_inputs[1][-1] == cp.XLMR_EOS_ID
    assert len(seen_inputs[1]) == 12


def test_diagnose_external_compression_and_g0_summary():
    text = "The old library was closed in 2015."
    spans = [[0, 3], [4, 7], [8, 15], [16, 19], [20, 26], [27, 29], [30, 34], [34, 35]]
    # Legal deletion keeping 3/8 words = 0.375 for rho=0.33
    d1 = cp.diagnose_external_compression(text, spans, "library closed 2015", rho=0.33)
    assert d1["align_status"] in ("ok", "ok_relaxed")
    assert math.isclose(d1["word_retention"], 3 / 8)
    assert not d1["anomalies"]["all_dropped"]
    assert not d1["anomalies"]["near_copy"]
    assert not d1["anomalies"]["large_deviation"]

    # Near-copy anomaly (8/8 = 1.0 for rho=0.33)
    d2 = cp.diagnose_external_compression(text, spans, text, rho=0.33)
    assert d2["anomalies"]["near_copy"]
    assert d2["anomalies"]["large_deviation"]

    # Empty output anomaly
    d3 = cp.diagnose_external_compression(text, spans, "", rho=0.33)
    assert d3["anomalies"]["all_dropped"]
    assert d3["word_retention"] == 0.0

    recs = [
        {"passage_id": "p1", "lang": "zh", "method": "m0", "rho": 0.33,
         "word_retention": d1["target_retention"], "target_retention": d1["target_retention"], "anomalies": {}},
        {"passage_id": "p1", "lang": "zh", "method": "m0_orig", "rho": 0.33, **d2},
    ]
    summary = cp.g0_summary(recs)
    assert summary["m0__zh__rho0.33"]["exact_budget_match_rate"] == 1.0
    assert summary["m0_orig__zh__rho0.33"]["anomalies"]["near_copy"] == 1


def test_read_results_repairs_half_line_and_check_resume():
    tmp = Path(tempfile.mkdtemp()) / "c.jsonl"
    cfg = {"method": "random", "rho": 0.33}
    r1 = {"passage_id": "en_p000", "passage_sha256": cp.sha256("hello"), "config": cfg}
    tmp.write_bytes(json.dumps(r1).encode() + b"\n" + b'{"passage_id": "en_p001"')
    recs = cp.read_results(tmp)
    assert len(recs) == 1 and recs[0]["passage_id"] == "en_p000"
    cp.check_resume(recs, cfg, {"en_p000": {"passage_sha256": cp.sha256("hello")}})

    try:
        cp.check_resume(recs, {"method": "m0", "rho": 0.33}, {"en_p000": {"passage_sha256": cp.sha256("hello")}})
    except SystemExit:
        pass
    else:
        raise AssertionError("config mismatch must raise SystemExit")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("all compress tests passed")

