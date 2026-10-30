"""Checks for segment(). Run: python test_segment.py  (or pytest)

The pilot-data test runs over data/pilot/*.jsonl if present.
"""
import json
import random
import unicodedata
from pathlib import Path

from llmlingua.utils import get_pure_token, is_begin_of_new_word

from align import align
from segment import M0, XLMR_MAX, SegmentError, expand_labels, jieba_space, load_tokenizer, segment

TOK = load_tokenizer()
PILOT = Path(__file__).parent / "data" / "pilot"


def words_of(text, seg):
    return [text[s:e] for s, e in seg["word_spans"]]


def official_words(seg):
    # what LLMLingua-2 inference (__merge_token_to_word) produces for the same tokens
    words = []
    for tok in TOK.convert_ids_to_tokens(seg["input_ids"]):
        if not words or is_begin_of_new_word(tok, M0, [], {}):
            words.append(get_pure_token(tok, M0))
        else:
            words[-1] += get_pure_token(tok, M0)
    return [w.lstrip("▁") for w in words]


def check_invariants(text, lang):
    seg = segment(text, lang, TOK)
    spans = seg["word_spans"]
    # spans are ordered, inside the text, and contain no whitespace
    assert all(0 <= s <= e <= len(text) for s, e in spans)
    assert all(spans[i][1] <= spans[i + 1][0] for i in range(len(spans) - 1)), "spans overlap"
    assert not any(c.isspace() for w in words_of(text, seg) for c in w)
    # every non-whitespace character is in exactly one word
    covered = [False] * len(text)
    for s, e in spans:
        for p in range(s, e):
            covered[p] = True
    assert all(covered[p] or text[p].isspace() for p in range(len(text))), "character not in any word"
    # removing the inserted spaces gives the original back
    prep, ins = seg["prep_text"], set(seg["inserted_spaces"])
    assert "".join(c for p, c in enumerate(prep) if p not in ins) == text
    assert all(prep[p] == " " for p in ins)
    # one word id per token, non-decreasing, covering all words
    wid = seg["word_ids"]
    assert len(wid) == len(seg["input_ids"])
    assert wid == sorted(wid) and set(wid) == set(range(len(spans)))
    # same words as the official inference merge (up to XLM-R's NFKC normalisation;
    # characters missing from the vocabulary come out as "<unk>" there)
    ours = [unicodedata.normalize("NFKC", w) for w in words_of(text, seg)]
    theirs = [unicodedata.normalize("NFKC", w) for w in official_words(seg)]
    assert len(ours) == len(theirs)
    bad = [(a, b) for a, b in zip(ours, theirs) if a != b and "<unk>" not in b]
    assert not bad, bad[:3]
    return seg


def test_jieba_space_roundtrip():
    prep, ins = jieba_space("今天天气很好，我们去 GPT-4 公园。\n第二行")
    assert "".join(c for p, c in enumerate(prep) if p not in set(ins)) == "今天天气很好，我们去 GPT-4 公园。\n第二行"
    assert "天气 很" in prep or "天气 很好" in prep


def test_zh_words_split_at_jieba_boundaries():
    text = "今天天气很好"
    seg = check_invariants(text, "zh")
    assert len(seg["word_spans"]) >= 3   # without jieba this is one word: "▁今天天气很好" -> 1-2 words
    assert "".join(words_of(text, seg)) == text


def test_en_punctuation_is_separate_word():
    text = "Hello, world!"
    seg = check_invariants(text, "en")
    assert words_of(text, seg) == ["Hello", ",", "world", "!"]


def test_unicode_normalisation_rejected():
    # NFKC inside XLM-R: the combining accent ends up in no word; "㈱" -> "(株)" overlaps
    for text in ["Café noir", "㈱ test"]:
        try:
            segment(text, "en", TOK)
        except SegmentError:
            continue
        raise AssertionError(f"{text!r} should be rejected")


def test_expand_labels():
    seg = segment("Tokenization works.", "en", TOK)
    labels = [i % 2 for i in range(len(seg["word_spans"]))]
    tl = expand_labels(seg["word_ids"], labels)
    assert len(tl) == len(seg["input_ids"])
    assert all(tl[t] == labels[w] for t, w in enumerate(seg["word_ids"]))


def test_pilot_data():
    if not PILOT.exists():
        return
    rng = random.Random(0)
    for lang in ["en", "zh", "sw"]:
        for line in open(PILOT / f"{lang}.jsonl", encoding="utf-8"):
            text = json.loads(line)["text"]
            seg = check_invariants(text, lang)
            assert len(seg["input_ids"]) <= XLMR_MAX, (lang, len(seg["input_ids"]))
            # deletion-only output built from these words must align back exactly
            spans = seg["word_spans"]
            labels = [int(rng.random() < 0.5 and e > s) for s, e in spans]
            teacher = " ".join(text[s:e] for (s, e), l in zip(spans, labels) if l)
            r = align(text, spans, teacher)
            assert r["status"] == "ok", (lang, r["reason"])
            kept = "".join(text[s:e] for (s, e), l in zip(spans, r["labels"]) if l)
            assert kept == teacher.replace(" ", "")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("all segment tests passed")
