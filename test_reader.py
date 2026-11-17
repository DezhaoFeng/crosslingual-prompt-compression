"""Unit tests for run_reader.py (Gate G1 and reader execution). Run: pytest test_reader.py"""
import math
from pathlib import Path

import run_reader as rr


SAMPLE_Q = {
    "question_id": "en_q000",
    "question_index": 0,
    "question_number": 1,
    "lang": "en",
    "passage_id": "en_p000",
    "flores_passage": "The old library was built in 1890 and closed in 2015.",
    "passage_sha256": rr.sha256("The old library was built in 1890 and closed in 2015."),
    "question": "When was the library built?",
    "mc_answer1": "1880",
    "mc_answer2": "1890",
    "mc_answer3": "2015",
    "mc_answer4": "1900",
    "correct_answer_num": "2",
    "gold_letter": "B",
    "question_sha256": "qhash1",
}


def test_candidate_reader_order_fixed():
    assert rr.CANDIDATE_READERS[0] == "Qwen/Qwen2.5-7B-Instruct"
    assert rr.CANDIDATE_READERS[1] == "deepseek-flash"


def test_prompt_never_compresses_question_or_options():
    p_comp = rr.build_user_prompt(SAMPLE_Q, "library built 1890 closed 2015.", is_none=False)
    assert "Passage:\nlibrary built 1890 closed 2015." in p_comp
    assert "Question: When was the library built?" in p_comp
    assert "A. 1880\nB. 1890\nC. 2015\nD. 1900" in p_comp

    p_none = rr.build_user_prompt(SAMPLE_Q, "", is_none=True)
    assert "Passage:" not in p_none
    assert "Question: When was the library built?" in p_none
    assert "A. 1880\nB. 1890\nC. 2015\nD. 1900" in p_none


def test_parse_mc_answer():
    assert rr.parse_mc_answer("B") == "B"
    assert rr.parse_mc_answer("  (C) ") == "C"
    assert rr.parse_mc_answer("D. 1900") == "D"
    assert rr.parse_mc_answer("Answer: A") == "A"
    assert rr.parse_mc_answer("") == "invalid"
    assert rr.parse_mc_answer(None) == "invalid"
    assert rr.parse_mc_answer("I cannot tell from the text.") == "invalid"


def test_score_options_local_picks_highest_option_logit():
    import torch

    class DummyTokenizer:
        chat_template = "dummy"

        def encode(self, text, add_special_tokens=False):
            return [{"A": 10, "B": 11, "C": 12, "D": 13}[text]]

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            return messages[-1]["content"]

        def __call__(self, text, return_tensors="pt"):
            return {"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)}

    class DummyOut:
        def __init__(self, logits):
            self.logits = logits

    class DummyModel:
        def eval(self):
            return self

        def parameters(self):
            yield torch.zeros(1)

        def __call__(self, input_ids, attention_mask=None):
            logits = torch.zeros((1, input_ids.shape[1], 20), dtype=torch.float32)
            logits[0, -1, 10] = 1.0  # A
            logits[0, -1, 11] = 2.0  # B
            logits[0, -1, 12] = 5.0  # C (highest)
            logits[0, -1, 13] = 0.5  # D
            return DummyOut(logits)

    tok = DummyTokenizer()
    opt_ids = rr.resolve_option_token_ids(tok)
    res = rr.score_options_local(DummyModel(), tok, opt_ids, "prompt")
    assert res["pred_letter"] == "C"
    assert res["option_logits"]["C"] == 5.0
    assert math.isclose(sum(res["option_probs"].values()), 1.0, abs_tol=1e-4)
    assert res["option_probs"]["C"] > res["option_probs"]["B"] > res["option_probs"]["A"]


def test_g1_eligibility_requires_all_languages_to_pass():
    full_en = [{"correct": i < 75, "pred_letter": "A", "seconds": 0.1, "input_tokens": 200, "output_tokens": 1}
               for i in range(100)]
    none_en = [{"correct": i < 35, "pred_letter": "A", "seconds": 0.1, "input_tokens": 80, "output_tokens": 1}
               for i in range(100)]
    full_zh = [{"correct": i < 70, "pred_letter": "B", "seconds": 0.1, "input_tokens": 180, "output_tokens": 1}
               for i in range(100)]
    none_zh = [{"correct": i < 30, "pred_letter": "B", "seconds": 0.1, "input_tokens": 70, "output_tokens": 1}
               for i in range(100)]

    rep_pass = rr.evaluate_g1_eligibility({"en": full_en, "zh": none_zh and full_zh}, {"en": none_en, "zh": none_zh})
    assert rep_pass["g1_passed"] is True
    assert math.isclose(rep_pass["per_language"]["en"]["gap_full_minus_none"], 0.40)
    assert math.isclose(rep_pass["per_language"]["zh"]["gap_full_minus_none"], 0.40)
    assert rep_pass["extrapolated_full_eval"]["total_queries"] == 900 * 2 * 17

    # If zh gap is only 0.10 (< 0.15), overall G1 must fail even though en passed
    weak_full_zh = [{"correct": i < 40, "pred_letter": "B", "seconds": 0.1, "input_tokens": 180, "output_tokens": 1}
                    for i in range(100)]
    rep_fail = rr.evaluate_g1_eligibility({"en": full_en, "zh": weak_full_zh}, {"en": none_en, "zh": none_zh})
    assert rep_fail["per_language"]["en"]["passed"] is True
    assert rep_fail["per_language"]["zh"]["passed"] is False
    assert rep_fail["g1_passed"] is False


def test_check_reader_resume_detects_tampering():
    cfg = rr.reader_run_config("Qwen/Qwen2.5-7B-Instruct", "m0", "en", 0.33)
    comp_by_hash = {SAMPLE_Q["passage_sha256"]: {"compressed_text": "library built 1890"}}
    rec = {
        "question_id": "en_q000",
        "config": cfg,
        "question_sha256": SAMPLE_Q["question_sha256"],
        "compressed_passage_sha256": rr.sha256("library built 1890"),
    }
    rr.check_reader_resume([rec], cfg, {"en_q000": SAMPLE_Q}, comp_by_hash)

    # Changing compressed passage text must abort resume
    bad_comp = {SAMPLE_Q["passage_sha256"]: {"compressed_text": "different compression"}}
    try:
        rr.check_reader_resume([rec], cfg, {"en_q000": SAMPLE_Q}, bad_comp)
    except SystemExit:
        pass
    else:
        raise AssertionError("changed compressed passage must raise SystemExit")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("all reader tests passed")

