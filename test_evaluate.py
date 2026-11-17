"""Unit tests for evaluate.py (Gate G6 frozen evaluation harness). Run: pytest test_evaluate.py"""
import math
import warnings

import evaluate as ev


def test_compute_u_normal_and_near_zero_denominator_guard():
    # Normal case: full=0.80, none=0.30, rho=0.55 -> U = (0.55 - 0.30) / (0.80 - 0.30) = 0.50
    u = ev.compute_u(0.55, 0.80, 0.30)
    assert math.isclose(u, 0.50)

    # Near-zero denominator (0.32 - 0.30 = 0.02 < 0.05) must warn and return NaN
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        u_small = ev.compute_u(0.31, 0.32, 0.30)
        assert math.isnan(u_small)
        assert len(w) == 1 and issubclass(w[0].category, UserWarning)

    # Negative denominator (full <= none) must also warn and return NaN
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        u_neg = ev.compute_u(0.30, 0.25, 0.30)
        assert math.isnan(u_neg)
        assert len(w) == 1


def test_paired_passage_bootstrap_detects_consistent_gain_despite_between_passage_variance():
    # Construct 40 passages, each with 2 questions.
    # Half of the passages are easy (M0 gets 1/2, M1 gets 2/2) and half are hard (M0 gets 0/2, M1 gets 1/2).
    # Across passages variance is large, but on EVERY passage M1 beats M0 by exactly +0.50!
    # Paired passage bootstrap must give point estimate = +0.50 with 0 variance (CI = [0.50, 0.50]).
    full_recs, none_recs, m0_recs, m1_recs, m4_recs = [], [], [], [], []
    for p_idx in range(40):
        pid = f"zh_p{p_idx:03d}"
        easy = (p_idx % 2 == 0)
        for q_sub in range(2):
            qid = f"zh_q{p_idx * 2 + q_sub:03d}"
            base = {"question_id": qid, "passage_id": pid, "word_retention": 0.33}
            full_recs.append({**base, "correct": True})
            none_recs.append({**base, "correct": False})
            m0_recs.append({**base, "correct": easy and (q_sub == 0)})
            m1_recs.append({**base, "correct": True if easy else (q_sub == 0)})
            m4_recs.append({**base, "correct": easy and (q_sub == 0)})

    cond_map = {
        "full": full_recs,
        "none": none_recs,
        "m0__rho0.33": m0_recs,
        "m1__rho0.33": m1_recs,
        "m4__rho0.33": m4_recs,
    }
    suite = ev.evaluate_language_suite(cond_map, n_boot=500, seed=713)
    assert math.isclose(suite["metrics"]["m0__rho0.33"]["raw_accuracy"]["estimate"], 0.25)
    assert math.isclose(suite["metrics"]["m1__rho0.33"]["raw_accuracy"]["estimate"], 0.75)

    hyps = ev.evaluate_hypotheses({"zh": suite}, rhos=(0.33,), primary_rho=0.33)
    r33 = hyps["by_rho"]["rho0.33"]
    assert r33["is_primary_analysis"] is True
    h1 = r33["H1_zh_M1_minus_M0"]
    assert math.isclose(h1["estimate"], 0.50)
    assert math.isclose(h1["ci_low"], 0.50) and math.isclose(h1["ci_high"], 0.50)
    assert h1["ci_above_zero"] is True

    h2 = r33["H2_zh_double_diff_M1_vs_M4"]
    assert math.isclose(h2["estimate"], 0.50)
    assert h2["ci_above_zero"] is True


def test_crosslingual_gap_flags_false_narrowing_from_english_degradation():
    # Suppose on en: M0=0.80, M1=0.60 (English degraded by -0.20!)
    # On zh: M0=0.50, M1=0.50 (Chinese did not improve: +0.00)
    # The En-Zh gap shrank from 0.30 to 0.10 (-0.20), but ONLY because English got worse!
    def make_lang_suite(lang, m0_acc, m1_acc):
        n_p = 20
        full_r, none_r, m0_r, m1_r = [], [], [], []
        for i in range(n_p):
            pid = f"{lang}_p{i:03d}"
            qid = f"{lang}_q{i:03d}"
            full_r.append({"question_id": qid, "passage_id": pid, "correct": True, "word_retention": 1.0})
            none_r.append({"question_id": qid, "passage_id": pid, "correct": False, "word_retention": 0.0})
            m0_r.append({"question_id": qid, "passage_id": pid, "correct": (i < int(m0_acc * n_p)), "word_retention": 0.33})
            m1_r.append({"question_id": qid, "passage_id": pid, "correct": (i < int(m1_acc * n_p)), "word_retention": 0.33})
        return ev.evaluate_language_suite(
            {"full": full_r, "none": none_r, "m0__rho0.33": m0_r, "m1__rho0.33": m1_r},
            n_boot=200,
            seed=713,
        )

    en_suite = make_lang_suite("en", 0.80, 0.60)
    zh_suite = make_lang_suite("zh", 0.50, 0.50)
    hyps = ev.evaluate_hypotheses({"en": en_suite, "zh": zh_suite}, rhos=(0.33,), primary_rho=0.33)
    gap_change = hyps["by_rho"]["rho0.33"]["gap_change_M1_vs_M0"]
    assert math.isclose(gap_change["estimate"], -0.20)
    assert math.isclose(gap_change["en_delta"], -0.20)
    assert math.isclose(gap_change["zh_delta"], 0.0)
    assert gap_change["driven_by_en_degradation"] is True


def test_word_classification_and_error_analysis():
    assert ev.classify_word("，", "zh") == "punctuation"
    assert ev.classify_word("...", "en") == "punctuation"
    assert ev.classify_word("1938年", "zh") == "number_date"
    assert ev.classify_word("2015", "en") == "number_date"
    assert ev.classify_word("爆发", "zh") == "qualifier_negation"
    assert ev.classify_word("超过", "zh") == "qualifier_negation"
    assert ev.classify_word("may", "en") == "qualifier_negation"
    assert ev.classify_word("not", "en") == "qualifier_negation"
    assert ev.classify_word("的", "zh") == "function_stopword"
    assert ev.classify_word("the", "en") == "function_stopword"
    assert ev.classify_word("图书馆", "zh") == "other_content"
    assert ev.classify_word("library", "en") == "other_content"

    # Error analysis on a sample Chinese sentence: "抗日战争爆发后，清华大学在1938年迁往昆明。"
    # Words: ["抗日战争", "爆发", "后", "，", "清华大学", "在", "1938年", "迁往", "昆明", "。"]
    text = "抗日战争爆发后，清华大学在1938年迁往昆明。"
    spans = [[0, 4], [4, 6], [6, 7], [7, 8], [8, 12], [12, 13], [13, 18], [18, 20], [20, 22], [22, 23]]
    # Drop "爆发" (qualifier), "，" (punct), "在" (function), "。" (punct) -> 4 dropped words (2 punct = 50%)
    labels = [1, 0, 1, 0, 1, 0, 1, 1, 1, 0]
    ea = ev.analyze_deleted_words(
        {"h1": text},
        [{"passage_sha256": "h1", "labels": labels}],
        {"h1": spans},
        lang="zh",
    )
    assert ea["total_content_words"] == 10
    assert ea["total_dropped_words"] == 4
    assert math.isclose(ea["by_category"]["punctuation"]["share_of_all_dropped"], 0.50)
    assert math.isclose(ea["by_category"]["qualifier_negation"]["drop_rate_within_category"], 1.0)
    assert math.isclose(ea["by_category"]["number_date"]["drop_rate_within_category"], 0.0)


def test_end_to_end_cli_generates_json_and_markdown_reports():
    import json
    import tempfile
    from pathlib import Path

    tmp = Path(tempfile.mkdtemp())
    c_dir = tmp / "compress"
    r_dir = tmp / "reader"
    c_dir.mkdir()
    r_dir.mkdir()

    for lang in ("en", "zh"):
        text = "The library was built in 1890." if lang == "en" else "图书馆1890年建造。"
        spans = [[0, 3], [4, 11], [12, 15], [16, 21], [22, 24], [25, 29], [29, 30]] if lang == "en" else [
            [0, 3], [3, 8], [8, 10], [10, 11]
        ]
        labels = [0, 1, 0, 1, 0, 1, 0] if lang == "en" else [1, 1, 1, 0]
        c_rec = {
            "passage_id": f"{lang}_p000",
            "lang": lang,
            "method": "m0",
            "rho": 0.33,
            "passage_sha256": f"h_{lang}",
            "orig_text": text,
            "word_spans": spans,
            "labels": labels,
            "word_retention": 0.5,
            "reader_token_retention": 0.48,
            "cl100k_retention": 0.49,
        }
        (c_dir / f"m0__{lang}__rho0.33.jsonl").write_text(json.dumps(c_rec) + "\n", encoding="utf-8")

        for cond, acc in (("full", 0.9), ("none", 0.2), ("m0__rho0.33", 0.6)):
            fname = f"Qwen2.5-7B-Instruct__{cond}__{lang}.jsonl" if "__rho" not in cond else (
                f"Qwen2.5-7B-Instruct__m0__{lang}__rho0.33.jsonl"
            )
            recs = [
                {
                    "question_id": f"{lang}_q{i:03d}",
                    "passage_id": f"{lang}_p{i // 2:03d}",
                    "correct": i < int(acc * 20),
                    "word_retention": 1.0 if cond == "full" else (0.0 if cond == "none" else 0.33),
                }
                for i in range(20)
            ]
            (r_dir / fname).write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")

    out_json = tmp / "eval.json"
    out_md = tmp / "eval.md"
    ev.main([
        "--reader", "Qwen/Qwen2.5-7B-Instruct",
        "--langs", "en,zh",
        "--rhos", "0.33",
        "--reader-dir", str(r_dir),
        "--compress-dir", str(c_dir),
        "--bootstrap", "100",
        "--out-json", str(out_json),
        "--out-md", str(out_md),
    ])
    rep = json.loads(out_json.read_text(encoding="utf-8"))
    md = out_md.read_text(encoding="utf-8")
    assert rep["harness_version"] == ev.HARNESS_VERSION
    assert "m0__rho0.33" in rep["languages"]["zh"]["conditions"]
    assert "## 3. Error Analysis" in md


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("all evaluate tests passed")
