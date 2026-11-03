"""Cases for align_relaxed(): a tolerant tier on top of the strict align(). Run: pytest test_align_relaxed.py

Real outputs come from results/; these are development samples, used to set the tolerance.
"""
import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import pytest

from align import align, align_relaxed
from run_teacher import is_usable
from segment import load_tokenizer, segment
from test_align import EN, TEA, ZH, spans_of

ROOT = Path(__file__).parent
TOK = load_tokenizer()


def relaxed(text, words, teacher):
    return align_relaxed(text, spans_of(text, words), teacher)


def check(text, words, teacher, status, labels=None):
    r = relaxed(text, words, teacher)
    assert r["status"] == status, (text, teacher, r)
    if labels is not None:
        assert r["labels"] == labels, (text, teacher, r)
    if status not in ("ok", "ok_relaxed"):
        assert r["labels"] is None and r["reason"], r
    return r


def long_case(n=200):
    # unique words, so there is exactly one correct label for each kept word
    words = [f"w{i:03d}" for i in range(n)]
    return " ".join(words), words


def keep_even(words):
    kept = words[::2]
    return kept, [int(i % 2 == 0) for i in range(len(words))]


# --- strict results are never changed ---

def test_strict_ok_stays_ok():
    r = check(*EN, "cat sat mat.", "ok", [0, 1, 1, 0, 0, 1, 1])
    assert r["edits"] == 0


def test_strict_reason_is_kept():
    text, words = EN
    strict = align(text, spans_of(text, words), "cat sat mat.")
    r = relaxed(text, words, "cat sat mat.")
    assert (r["status"], r["labels"], r["reason"]) == (strict["status"], strict["labels"], strict["reason"])


def test_failure_reports_the_strict_status():
    text, words = TEA
    strict = align(text, spans_of(text, words), "tea green")
    r = relaxed(text, words, "tea green")
    assert (r["status"], r["reason"]) == (strict["status"], strict["reason"]) and r["labels"] is None


def test_empty_output_still_fails():
    check(*EN, "", "uncertain")
    check(*EN, "   \n", "uncertain")


# --- tolerated: case and character width cost no edits ---

def test_sentence_initial_capital():
    r = check(*EN, "Cat sat mat.", "ok_relaxed", [0, 1, 1, 0, 0, 1, 1])
    assert r["edits"] == 0


def test_all_case_changes():
    check(*EN, "CAT SAT MAT.", "ok_relaxed", [0, 1, 1, 0, 0, 1, 1])


def test_full_width_punctuation():
    text, words = "你好,世界", ["你好", ",", "世界"]
    r = check(text, words, "你好，世界", "ok_relaxed", [1, 1, 1])
    assert r["edits"] == 0


def test_full_width_letters_and_digits():
    check("room 12", ["room", "12"], "ＲＯＯＭ １２", "ok_relaxed", [1, 1])


def test_curly_and_straight_quotes_cost_nothing():
    text, words = "反“围剿”和长征", ["反", "“", "围剿", "”", "和", "长征"]
    r = check(text, words, '反"围剿"和长征', "ok_relaxed", [1] * 6)
    assert r["edits"] == 0
    check("it’s", ["it", "’", "s"], "it's", "ok_relaxed", [1, 1, 1])


def test_corner_brackets_are_not_quotes():
    assert relaxed("「好」", ["「", "好", "」"], '"好"')["status"] not in ("ok", "ok_relaxed")


def test_repeated_word_keeps_earliest():
    check("the the cat", ["the", "the", "cat"], "The cat", "ok_relaxed", [1, 0, 1])


# --- tolerated: a few changed characters, up to max(5, floor(1% of the output's normalized characters));
# outputs under 100 normalized characters get no such tolerance ---

def with_extra_marks(n_marks, n_words=200):
    text, words = long_case(n_words)
    kept, labels = keep_even(words)
    out = list(kept)
    for k in range(n_marks):
        out.insert(10 + 10 * k, "Z")  # a character the original does not contain
    return text, words, " ".join(out), labels


def test_added_characters_at_the_floor():
    text, words, out, labels = with_extra_marks(5)  # 400 + 5 chars -> 1% is 4, floor is 5
    r = check(text, words, out, "ok_relaxed", labels)
    assert r["edits"] == 5


def test_added_characters_over_the_floor():
    text, words, out, _ = with_extra_marks(6)  # 400 + 6 chars -> limit 5
    assert relaxed(text, words, out)["status"] not in ("ok", "ok_relaxed")


def test_added_characters_at_one_percent():
    text, words, out, labels = with_extra_marks(7, n_words=350)  # 700 + 7 chars -> limit 7
    r = check(text, words, out, "ok_relaxed", labels)
    assert r["edits"] == 7


def test_added_characters_over_one_percent():
    text, words, out, _ = with_extra_marks(8, n_words=350)  # 700 + 8 chars -> limit 7
    assert relaxed(text, words, out)["status"] not in ("ok", "ok_relaxed")


def test_no_tolerance_under_100_characters():
    text, words, out, _ = with_extra_marks(1, n_words=48)  # 96 + 1 chars
    assert relaxed(text, words, out)["status"] not in ("ok", "ok_relaxed")


def test_floor_applies_from_100_characters():
    # 24 kept words of 4 characters + 4 marks: exactly 100 characters
    text, words = long_case(48)
    kept, labels = keep_even(words)
    r = check(text, words, " ".join(kept + ["Z"] * 4), "ok_relaxed", labels)
    assert r["edits"] == 4


def test_short_outputs_have_no_tolerance():
    assert relaxed(*TEA, "I like greenish tea")["status"] not in ("ok", "ok_relaxed")
    assert relaxed(*TEA, "tea green")["status"] not in ("ok", "ok_relaxed")  # order is not relaxed
    long_text, long_words = long_case()
    assert relaxed(long_text, long_words, "摘要")["status"] not in ("ok", "ok_relaxed")


def test_word_with_at_least_half_its_characters_is_kept():
    words = [f"w{i:03d}" for i in range(100)] + ["abcdef"] + [f"x{i:03d}" for i in range(100)]
    text = " ".join(words)
    labels = [1] * 100 + [1] + [1] * 100
    out = " ".join(words[:100] + ["abc"] + words[101:])
    r = check(text, words, out, "ok_relaxed", labels)
    assert r["edits"] == 3  # kept word's three missing characters


def test_word_with_under_half_its_characters_is_dropped():
    words = [f"w{i:03d}" for i in range(100)] + ["abcdefg"] + [f"x{i:03d}" for i in range(100)]
    text = " ".join(words)
    labels = [1] * 100 + [0] + [1] * 100
    out = " ".join(words[:100] + ["ab"] + words[101:])
    r = check(text, words, out, "ok_relaxed", labels)
    assert r["edits"] == 2  # two output characters no kept word accounts for


def test_chinese_partial_word_deletion():
    head = "".join(chr(0x4e00 + i) for i in range(400))
    text = head + "呈现为急倾斜"
    words = [head[i:i + 2] for i in range(0, 400, 2)] + ["呈现为", "急", "倾斜"]
    r = check(text, words, head + "呈急倾斜", "ok_relaxed", [1] * 200 + [0, 1, 1])
    assert r["edits"] == 1


# --- never tolerated, however small ---

@pytest.mark.parametrize("text, words, out", [
    ("a nice day", ["a", "nice", "day"], "an ice day"),         # teacher whitespace inside a kept word
    ("A nice day", ["A", "nice", "day"], "an ice day"),
    ("a part", ["a", "part"], "apart"),                         # words merged across original whitespace
    ("A Part", ["A", "Part"], "apart"),
])
def test_word_boundary_changes_are_rejected(text, words, out):
    assert relaxed(text, words, out)["status"] not in ("ok", "ok_relaxed")


def test_whitespace_between_chinese_characters_may_go():
    r = check("第二方面军 政治部", ["第二方面军", "政治部"], "第二方面军政治部", "ok_relaxed", [1, 1])
    assert r["edits"] == 0
    # also when dropped words sit between them
    check("红军 第二 政治部", ["红军", "第二", "政治部"], "红军政治部", "ok_relaxed", [1, 0, 1])


@pytest.mark.parametrize("text, words, out", [
    ("红军 army", ["红军", "army"], "红军army"),      # only one side is Chinese
    ("army 红军", ["army", "红军"], "army红军"),
    ("5 日", ["5", "日"], "5日"),
    ("红军 第二 army", ["红军", "第二", "army"], "红军army"),
    ("army 第二 红军", ["army", "第二", "红军"], "army红军"),
])
def test_merge_rule_still_applies_unless_both_sides_are_chinese(text, words, out):
    assert relaxed(text, words, out)["status"] not in ("ok", "ok_relaxed")


def test_word_boundary_change_is_rejected_in_a_long_output():
    filler, filler_words = long_case()
    text, words = filler + " a part", filler_words + ["a", "part"]
    assert relaxed(text, words, filler + " apart")["status"] not in ("ok", "ok_relaxed")


def test_word_split_is_rejected_in_a_long_output():
    filler, filler_words = long_case()
    text, words = filler + " a nice day", filler_words + ["a", "nice", "day"]
    assert relaxed(text, words, filler + " an ice day")["status"] not in ("ok", "ok_relaxed")


def test_whitespace_inside_a_kept_word_is_rejected_in_a_long_output():
    filler, filler_words = long_case()
    text, words = filler + " nice day", filler_words + ["nice", "day"]
    assert relaxed(text, words, filler + " ni ce day")["status"] not in ("ok", "ok_relaxed")


def test_word_keeping_its_second_half_is_kept():
    words = [f"w{i:03d}" for i in range(100)] + ["abcdef"] + [f"x{i:03d}" for i in range(100)]
    out = " ".join(words[:100] + ["def"] + words[101:])
    r = check(" ".join(words), words, out, "ok_relaxed", [1] * 201)
    assert r["edits"] == 3


def test_many_partial_words_go_over_the_limit():
    text, words = long_case()
    out = " ".join(w[:2] if i < 10 else w for i, w in enumerate(words))  # 20 lost characters, limit 7 (780 chars)
    assert relaxed(text, words, out)["status"] not in ("ok", "ok_relaxed")


def test_one_long_partial_word_over_the_limit_is_rejected():
    words = [f"w{i:03d}" for i in range(100)] + ["abcdefghij" * 4]
    out = " ".join(words[:100] + ["abcdefghij" * 2])  # 20 lost characters, limit 5
    assert relaxed(" ".join(words), words, out)["status"] not in ("ok", "ok_relaxed")


def test_reordered_blocks_are_rejected():
    text, words = long_case()
    out = " ".join(words[100:] + words[:100])
    assert relaxed(text, words, out)["status"] not in ("ok", "ok_relaxed")


def test_rewrite_is_rejected():
    text, words = long_case()
    assert relaxed(text, words, "A short summary written by the teacher.")["status"] not in ("ok", "ok_relaxed")


def test_new_sentence_is_rejected():
    text, words = long_case()
    kept, _ = keep_even(words)
    assert relaxed(text, words, " ".join(kept) + " This sentence is new.")["status"] not in ("ok", "ok_relaxed")


def test_whitespace_only_word_is_never_kept():
    # empty middle span, as segment() produces for a jieba-inserted space; capital forces the relaxed path
    text = "好的A"
    r = align_relaxed(text, [(0, 1), (1, 1), (1, 2), (2, 3)], "好的a")
    assert r["status"] == "ok_relaxed" and r["labels"] == [1, 0, 1, 1]


def test_chinese_strict_cases_are_unchanged():
    check(*ZH, "今天很好", "ok", [1, 0, 1, 1])


# --- real outputs ---

def norm(s):
    s = unicodedata.normalize("NFKC", s).casefold().translate(str.maketrans("“”‘’", "\"\"''"))
    return "".join(c for c in s if not c.isspace())


def edits_against_kept_words(text, spans, labels, out):
    # independent of the implementation: kept words vs output, in order
    a, b = norm("".join(text[s:e] for (s, e), k in zip(spans, labels) if k)), norm(out)
    same = sum(x.size for x in SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks())
    return (len(a) - same) + (len(b) - same), len(b)


def pilot_rows():
    rows = {}
    for lang in ("en", "zh", "sw"):
        for line in open(ROOT / "data" / "pilot" / f"{lang}.jsonl", encoding="utf-8"):
            r = json.loads(line)
            rows[r["id"]] = r
    return rows


def teacher_file(name, folder="teacher"):
    return [json.loads(l) for l in open(ROOT / "results" / folder / name, encoding="utf-8")]


ROWS = pilot_rows()
CHAT = teacher_file("chatgpt_dev_trial.jsonl", "chat_trial")


@pytest.mark.parametrize("rec", CHAT, ids=lambda r: r["id"])
def test_chat_outputs_get_consistent_labels(rec):
    # line-end backslashes come from copying the chat page, not from the model
    out = re.sub(r"\\\s*\n", "\n", rec["raw_output"])
    row = ROWS[rec["id"]]
    spans = segment(row["text"], row["lang"], TOK)["word_spans"]
    r = align_relaxed(row["text"], spans, out)
    assert r["status"] in ("ok", "ok_relaxed"), r["reason"]
    edits, n = edits_against_kept_words(row["text"], spans, r["labels"], out)
    assert edits <= max(5, n // 100), (edits, n)


@pytest.mark.parametrize("rec", teacher_file("claude_dev_trial.jsonl", "chat_trial"), ids=lambda r: r["id"])
def test_opus_chat_outputs_get_consistent_labels(rec):
    row = ROWS[rec["id"]]
    spans = segment(row["text"], row["lang"], TOK)["word_spans"]
    r = align_relaxed(row["text"], spans, rec["raw_output"])
    assert r["status"] in ("ok", "ok_relaxed"), r["reason"]
    edits, n = edits_against_kept_words(row["text"], spans, r["labels"], rec["raw_output"])
    assert edits <= max(5, n // 100), (edits, n)


@pytest.mark.parametrize("rec", teacher_file("tiny-aya-global__v2.jsonl"), ids=lambda r: r["id"])
def test_tiny_aya_rewrites_stay_rejected(rec):
    row = ROWS[rec["id"]]
    spans = segment(row["text"], row["lang"], TOK)["word_spans"]
    assert align_relaxed(row["text"], spans, rec["teacher"]["raw_output"])["status"] not in ("ok", "ok_relaxed")


@pytest.mark.parametrize("rec", teacher_file("Qwen3-14B__v3__constrained.jsonl"), ids=lambda r: r["id"])
def test_constrained_outputs_stay_strict_ok(rec):
    row = ROWS[rec["id"]]
    spans = segment(row["text"], row["lang"], TOK)["word_spans"]
    r = align_relaxed(row["text"], spans, rec["teacher"]["raw_output"])
    assert r["status"] == "ok" and r["labels"] == rec["labels"]


@pytest.mark.parametrize("rec", teacher_file("deepseek_dev_trial.jsonl", "chat_trial"), ids=lambda r: r["id"])
def test_deepseek_chat_outputs_get_consistent_labels(rec):
    row = ROWS[rec["id"]]
    spans = segment(row["text"], row["lang"], TOK)["word_spans"]
    r = align_relaxed(row["text"], spans, rec["raw_output"])
    assert r["status"] in ("ok", "ok_relaxed"), r["reason"]
    edits, n = edits_against_kept_words(row["text"], spans, r["labels"], rec["raw_output"])
    assert edits <= max(5, n // 100), (edits, n)


# --- usable for training ---

@pytest.mark.parametrize("status, finish, want", [
    ("ok", "eos", True),
    ("ok_relaxed", "eos", True),
    ("ok_relaxed", "length", False),
    ("uncertain", "eos", False),
])
def test_is_usable(status, finish, want):
    assert is_usable(status, finish) is want
