"""Checks for the constrained-decoding grammar with a toy byte-level vocabulary (no model needed).
Run: python test_constrained.py  (or pytest)"""
import json
import random
from pathlib import Path

from align import align
from constrained import DeletionConstraint, TokenTable
from segment import load_tokenizer, segment

XLMR = load_tokenizer()
PILOT = Path(__file__).parent / "data" / "pilot"
END, EOS = 0, 1


def toy_table(text):
    # ids 0/1 = END_RESPONSE/EOS (special), then all single bytes, then multi-byte pieces of the text
    pieces = {text[i:i + n].encode() for n in (2, 3, 5) for i in range(0, len(text) - n, 3)}
    pieces |= {b" " + p for p in list(pieces)[:200]}
    pieces |= {text[:1].encode("utf-8")[:2]} if text and len(text[0].encode()) > 1 else set()  # partial UTF-8
    return TokenTable([None, None] + [bytes([b]) for b in range(256)] + sorted(pieces))


def walk(text, lang, rng, stop_p=0.02):
    """Random generation under the constraint; returns (output_text, constraint)."""
    spans = segment(text, lang, XLMR)["word_spans"]
    table = toy_table(text)
    c = DeletionConstraint(text, spans, table, END, EOS)
    out = b""
    for _ in range(5000):
        allowed = c.allowed_tokens()
        assert allowed, "no legal continuation"
        if END in allowed and (rng.random() < stop_p or len(allowed) == 1):
            c.feed(END)
            assert c.allowed_tokens() == [EOS]
            break
        tok = rng.choice([t for t in allowed if t != END])
        c.feed(tok)
        out += table.id_bytes[tok]
    assert not c.no_legal_path
    return out.decode("utf-8"), spans, c


def test_every_constrained_output_is_a_legal_deletion():
    rng = random.Random(0)
    for lang in ["en", "zh", "sw"]:
        for line in list(open(PILOT / f"{lang}.jsonl", encoding="utf-8"))[:4]:
            text = json.loads(line)["text"]
            for _ in range(3):
                out, spans, _ = walk(text, lang, rng)
                r = align(text, spans, out)
                assert r["status"] == "ok", (lang, r["reason"], out[:80])


def accepts(text, lang, output):
    """Feed output byte by byte (single-byte tokens); True if the grammar accepts it and may end."""
    spans = segment(text, lang, XLMR)["word_spans"]
    table = toy_table(text)
    c = DeletionConstraint(text, spans, table, END, EOS)
    for b in output.encode("utf-8"):
        if 2 + b not in c.allowed_tokens():
            return False
        c.feed(2 + b)
    return END in c.allowed_tokens()


def test_grammar_accepts_legal_deletions():
    rng = random.Random(1)
    for lang in ["en", "zh", "sw"]:
        for line in list(open(PILOT / f"{lang}.jsonl", encoding="utf-8"))[:3]:
            text = json.loads(line)["text"]
            spans = [sp for sp in segment(text, lang, XLMR)["word_spans"] if sp[1] > sp[0]]
            kept = [text[s:e] for s, e in spans if rng.random() < 0.4] or [text[spans[0][0]:spans[0][1]]]
            assert accepts(text, lang, " ".join(kept)), lang


def test_rules():
    assert accepts("a part", "en", "a part")
    assert not accepts("a part", "en", "apart")                 # merge across original space
    assert not accepts("He is a nice man", "en", "He is an ice man")  # word split
    assert accepts("the cat the dog", "en", "the dog")          # repeated word
    assert not accepts("the cat", "en", "cat the")              # reorder
    assert not accepts("the cat", "en", "the  cat")             # at most one space
    # jieba gives 今天天气 / 很 / 好
    assert accepts("今天天气很好", "zh", "今天天气好")             # zh needs no spaces
    assert accepts("今天天气很好", "zh", "今天天气 好")
    assert not accepts("今天天气很好", "zh", "今天 天气好")         # space inside a word
    assert not accepts("今天天气很好", "zh", "今天天好")           # partial word
    assert accepts("Hello, world!", "en", "Hello world!")


def test_end_only_after_a_word():
    text = "the cat"
    spans = segment(text, "en", XLMR)["word_spans"]
    c = DeletionConstraint(text, spans, toy_table(text), END, EOS)
    assert END not in c.allowed_tokens()                         # empty output impossible
    for b in b"the":
        c.feed(2 + b)
    assert END in c.allowed_tokens()
    c.feed(END)
    assert c.allowed_tokens() == [EOS]


def sp_table(text, lang):
    """SentencePiece-like toy vocabulary: no single-byte fallback, pieces with/without a leading
    space, some words split in two, and full-width punctuation missing (like mT5's normaliser)."""
    words = [text[s:e] for s, e in segment(text, lang, XLMR)["word_spans"] if e > s]
    pieces = {" "}
    for w in words:
        if any(c in w for c in "，。、（）：；"):
            continue
        pieces |= {w, " " + w}
        if len(w) > 1:
            pieces |= {w[: len(w) // 2], w[len(w) // 2:], " " + w[: len(w) // 2]}
    return TokenTable([None, None] + sorted(p.encode("utf-8") for p in pieces), kind="sentencepiece")


def test_sentencepiece_walk_never_dead_ends_and_stays_legal():
    rng = random.Random(2)
    for lang in ["en", "zh", "sw"]:
        for line in list(open(PILOT / f"{lang}.jsonl", encoding="utf-8"))[:3]:
            text = json.loads(line)["text"]
            spans = segment(text, lang, XLMR)["word_spans"]
            table = sp_table(text, lang)
            for _ in range(3):
                c = DeletionConstraint(text, spans, table, EOS, EOS, allow_leading_space=True)
                out = b""
                for _ in range(5000):
                    allowed = c.allowed_tokens()
                    assert allowed and not c.no_legal_path, "dead end"
                    if EOS in allowed and (rng.random() < 0.03 or allowed == [EOS]):
                        break
                    tok = rng.choice([t for t in allowed if t != EOS])
                    c.feed(tok)
                    out += table.id_bytes[tok]
                r = align(text, spans, out.decode("utf-8"))
                assert r["status"] == "ok", (lang, r["reason"])
            if lang == "zh":
                assert c.unreachable_words > 0  # full-width punctuation cannot be spelled


def test_leading_space_only_when_enabled():
    text = "the cat"
    spans = segment(text, "en", XLMR)["word_spans"]
    table = toy_table(text)
    space = 2 + 0x20
    assert space not in DeletionConstraint(text, spans, table, END, EOS).allowed_tokens()
    assert space in DeletionConstraint(text, spans, table, END, EOS, allow_leading_space=True).allowed_tokens()


def test_real_aya101_sentencepiece_vocab():
    # the real mT5/aya-101 vocabulary: "▁" pieces, no byte fallback, NFKC-normalised (no "，")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("CohereLabs/aya-101", revision="e7dad472b9de8a30a00cf08a05c19003bf59028d")
    table = TokenTable.from_hf(tok)
    assert table.kind == "sentencepiece"
    assert table.id_bytes[tok.convert_tokens_to_ids("▁Hello")] == b" Hello"
    assert table.id_bytes[tok.eos_token_id] is None
    rng = random.Random(3)
    for lang in ["en", "zh", "sw"]:
        text = json.loads(open(PILOT / f"{lang}.jsonl", encoding="utf-8").readline())["text"]
        spans = segment(text, lang, XLMR)["word_spans"]
        for _ in range(2):
            c = DeletionConstraint(text, spans, table, tok.eos_token_id, tok.eos_token_id, allow_leading_space=True)
            gen = []
            for _ in range(3000):
                allowed = c.allowed_tokens()
                assert allowed and not c.no_legal_path, "dead end"
                if tok.eos_token_id in allowed and (rng.random() < 0.02 or allowed == [tok.eos_token_id]):
                    break
                t = rng.choice([x for x in allowed if x != tok.eos_token_id])
                c.feed(t)
                gen.append(t)
            out = tok.decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            r = align(text, spans, out)
            assert r["status"] == "ok", (lang, r["reason"], out[:80])
        if lang == "zh":
            assert c.unreachable_words > 0  # "，" etc. are normalised away by mT5


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("all constrained tests passed")
