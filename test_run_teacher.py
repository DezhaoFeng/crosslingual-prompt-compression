"""Resume/eligibility checks for run_teacher.py (no GPU or model needed). Run: pytest test_run_teacher.py"""
import json
import tempfile
from argparse import Namespace
from pathlib import Path

import run_teacher as rt

ARGS_V1 = Namespace(model="CohereLabs/tiny-aya-global", max_new_tokens=1024, prompt="v1")
CFG = rt.run_config(ARGS_V1)


def rec(i, text="t", cfg=CFG):
    return {"id": f"x{i}", "text_sha256": rt.sha256(text), "teacher": {"config": cfg}}


def write(lines_bytes):
    p = Path(tempfile.mkdtemp()) / "r.jsonl"
    p.write_bytes(lines_bytes)
    return p


def dumps(r):
    return json.dumps(r).encode() + b"\n"


def test_truncated_generation_not_usable():
    assert rt.is_usable("ok", "eos")
    assert not rt.is_usable("ok", "length")
    assert not rt.is_usable("added", "eos")


def test_half_written_last_line_is_dropped():
    p = write(dumps(rec(1)) + dumps(rec(2)) + b'{"id": "x3", "te')
    assert [r["id"] for r in rt.read_results(p)] == ["x1", "x2"]
    assert p.read_bytes() == dumps(rec(1)) + dumps(rec(2))  # file repaired for appending


def test_lost_final_newline_is_restored():
    p = write(dumps(rec(1)) + json.dumps(rec(2)).encode())
    assert len(rt.read_results(p)) == 2
    assert p.read_bytes().endswith(b"}\n")


def test_corrupt_middle_line_is_an_error():
    p = write(dumps(rec(1)) + b"{broken\n" + dumps(rec(3)))
    try:
        rt.read_results(p)
    except ValueError:
        return
    raise AssertionError("corrupt middle line must raise")


def test_resume_rejects_other_config():
    other = {**CFG, "params": {"do_sample": False, "max_new_tokens": 512}}
    try:
        rt.check_resume([rec(1, cfg=other)], CFG, [{"id": "x1", "text": "t"}])
    except SystemExit:
        return
    raise AssertionError("config mismatch must stop")


def test_resume_rejects_changed_text():
    try:
        rt.check_resume([rec(1, text="old")], CFG, [{"id": "x1", "text": "new"}])
    except SystemExit:
        return
    raise AssertionError("changed input text must stop")


def test_resume_accepts_same_config_and_text():
    rt.check_resume([rec(1, text="t")], CFG, [{"id": "x1", "text": "t"}])


def test_fingerprints_of_existing_results_do_not_drift():
    # adding prompts/options must not change the fingerprint of earlier runs, or their results become unusable
    by_version = {v: k for k, v in rt.PROMPT_VERSIONS.items()}
    for p in (Path(__file__).parent / "results" / "teacher").glob("*.jsonl"):
        stored = rt.read_results(p)[0]["teacher"]["config"]
        args = Namespace(model=stored["model"], max_new_tokens=stored["params"]["max_new_tokens"],
                         prompt=by_version[stored["prompt_version"]],
                         decoding="constrained" if "decoding" in stored else "free")
        now = rt.run_config(args)
        # constraint_sha256 changes whenever constrained.py is edited; that is meant to block resuming
        stored = {k: v for k, v in stored.items() if k != "constraint_sha256"}
        now = {k: v for k, v in now.items() if k != "constraint_sha256"}
        assert stored == now, p.name


def test_v3_is_v2_plus_target_sentence():
    for lang in rt.EXAMPLES:
        v2, v3 = rt.user_prompt("v2", lang, "INPUT"), rt.user_prompt("v3", lang, "INPUT")
        assert v3.replace(rt.TARGET_RULE, "", 1) == v2 and rt.TARGET_RULE in v3
    assert rt.prompt_sha256("v3") != rt.prompt_sha256("v2")


def test_v2_examples_are_legal_deletions():
    from align import align
    from segment import load_tokenizer, segment
    tok = load_tokenizer()
    for lang, (orig, comp) in rt.EXAMPLES.items():
        r = align(orig, segment(orig, lang, tok)["word_spans"], comp)
        assert r["status"] == "ok", (lang, r["reason"])


def test_v2_prompt_uses_example_in_input_language():
    for lang, (orig, comp) in rt.EXAMPLES.items():
        p = rt.user_prompt("v2", lang, "INPUT")
        assert orig in p and comp in p and p.rstrip().endswith("INPUT\nThe compressed text is:")
        assert "Output ONLY the compressed text" in p and "You can ONLY remove unimportant words" in p
        other = [o for l, (o, _) in rt.EXAMPLES.items() if l != lang]
        assert not any(o in p for o in other)


def test_constrained_decoding_has_its_own_fingerprint():
    free = rt.run_config(Namespace(model="CohereLabs/tiny-aya-global", max_new_tokens=1024, prompt="v2"))
    con = rt.run_config(Namespace(model="CohereLabs/tiny-aya-global", max_new_tokens=1024, prompt="v2",
                                  decoding="constrained"))
    assert "decoding" not in free and con["decoding"] == rt.CONSTRAINED_VERSION and con != free


def test_prompt_change_changes_config():
    old = rt.USER
    try:
        rt.USER = old + " "
        assert rt.run_config(ARGS_V1) != CFG
    finally:
        rt.USER = old


def test_seq2seq_config_records_quantisation_and_prompt_format():
    a = Namespace(model="CohereLabs/aya-101", max_new_tokens=1024, prompt="v2", decoding="constrained", quant="int8")
    cfg = rt.run_config(a)
    assert cfg["dtype"] == "bnb-int8+bf16-compute" and "plain" in cfg["prompt_format"]
    assert rt.run_config(Namespace(**{**vars(a), "quant": "nf4"})) != cfg


def test_qwen_config_records_quantisation_and_no_thinking():
    cfg = rt.run_config(Namespace(model="Qwen/Qwen3-14B", max_new_tokens=1024, prompt="v2",
                                  decoding="constrained", quant="int8"))
    assert cfg["dtype"] == "bnb-int8+bf16-compute" and cfg["chat_template_kwargs"] == {"enable_thinking": False}
    assert "prompt_format" not in cfg


def test_v4_examples_are_legal_and_keep_about_a_third():
    from align import align, content_words, teacher_stats
    from segment import load_tokenizer, segment
    tok = load_tokenizer()
    for lang, (orig, _) in rt.EXAMPLES.items():
        comp = rt.EXAMPLES_V4[lang]
        spans = segment(orig, lang, tok)["word_spans"]
        r = align(orig, spans, comp)
        assert r["status"] == "ok", (lang, r["reason"])
        assert 0.28 <= teacher_stats(orig, spans, comp, r["labels"])["word_keep_rate"] <= 0.38, lang


def test_v4_is_v3_with_only_the_example_compression_changed():
    for lang, (_, old) in rt.EXAMPLES.items():
        v3, v4 = rt.user_prompt("v3", lang, "INPUT"), rt.user_prompt("v4", lang, "INPUT")
        assert v4.replace(rt.EXAMPLES_V4[lang], old, 1) == v3
    assert rt.prompt_sha256("v4") != rt.prompt_sha256("v3")


def test_start_skips_development_samples():
    pilot = Path(__file__).parent / "data" / "pilot"
    dev = rt.load_chunks(pilot, ["sw"], "train", 3)
    val = rt.load_chunks(pilot, ["sw"], "train", 7, start=3)
    assert len(val) == 7 and not {r["id"] for r in dev} & {r["id"] for r in val}
