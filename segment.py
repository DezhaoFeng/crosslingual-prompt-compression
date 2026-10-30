"""One word segmentation for labels, training and inference (DECISIONS.md B1).

Words are exactly what LLMLingua-2 inference merges: XLM-R tokens grouped by the
official is_begin_of_new_word (force_tokens=[], token_map={}). For zh we first
insert spaces between jieba tokens so XLM-R sees word boundaries.
"""
import jieba
from llmlingua.utils import is_begin_of_new_word

M0 = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
M0_REV = "ebaba9b0e874dadd3003ffcff828e4397e568089"
XLMR_MAX = 510  # 512 minus <s> </s>


def jieba_space(text):
    """Insert one space between consecutive non-whitespace jieba tokens.

    Returns (prep_text, inserted) where inserted are positions in prep_text of added spaces.
    """
    out, inserted, n, prev_ws = [], [], 0, True
    for tok in jieba.cut(text):
        ws = tok.isspace()
        if not ws and not prev_ws:
            inserted.append(n)
            out.append(" ")
            n += 1
        out.append(tok)
        n += len(tok)
        prev_ws = ws
    return "".join(out), inserted


class SegmentError(ValueError):
    """Word spans do not map cleanly back to the text; the sample must be skipped and logged."""


def check_spans(text, spans):
    """None if spans are ordered, disjoint, and cover every non-whitespace character exactly once.

    XLM-R normalises text (NFKC), so offsets can lose characters (e.g. a combining accent in
    "Café") or map several words onto one character (e.g. "㈱" -> "(株)").
    """
    last = 0
    for s, e in spans:
        if s < last:
            return f"overlapping word spans at char {s}: {text[s:e]!r}"
        last = max(last, e)
    covered = bytearray(len(text))
    for s, e in spans:
        covered[s:e] = b"\x01" * (e - s)
    for p, c in enumerate(text):
        if not covered[p] and not c.isspace():
            return f"character {c!r} (U+{ord(c):04X}) at {p} is in no word"
    return None


def segment(text, lang, tokenizer):
    """Returns dict with prep_text, inserted_spaces, input_ids (no special tokens),
    word_ids (one per token) and word_spans ([start, end] in the original text).

    Raises SegmentError if the spans cannot be mapped back cleanly (see check_spans)."""
    prep, inserted = jieba_space(text) if lang == "zh" else (text, [])
    enc = tokenizer(prep, add_special_tokens=False, return_offsets_mapping=True)
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"])

    spans, word_ids = [], []
    for tok, (s, e) in zip(tokens, enc["offset_mapping"]):
        if not spans or is_begin_of_new_word(tok, M0, [], {}):
            spans.append(None)
        # a bare "▁" token stands for whitespace, but the tokenizer reports the offset of the
        # *next* character; it must not add characters to the word
        if tok != "▁":
            cur = spans[-1]
            spans[-1] = [s, e] if cur is None else [min(cur[0], s), max(cur[1], e)]
        word_ids.append(len(spans) - 1)
    # whitespace-only word: empty span placed after the previous word
    for i, sp in enumerate(spans):
        if sp is None:
            p = spans[i - 1][1] if i else 0
            spans[i] = [p, p]

    # prep position -> original position (only zh has inserted spaces)
    ins = set(inserted)
    to_orig, k = [], 0
    for p in range(len(prep) + 1):
        to_orig.append(p - k)
        if p in ins:
            k += 1

    word_spans = []
    for s, e in spans:
        while s < e and prep[s].isspace():
            s += 1
        while e > s and prep[e - 1].isspace():
            e -= 1
        word_spans.append([to_orig[s], to_orig[e]])  # empty span = whitespace-only word
    problem = check_spans(text, word_spans)
    if problem:
        raise SegmentError(problem)
    return {
        "prep_text": prep,
        "inserted_spaces": inserted,
        "input_ids": enc["input_ids"],
        "word_ids": word_ids,
        "word_spans": word_spans,
    }


def expand_labels(word_ids, labels):
    """Word labels -> token labels (special tokens are added by the caller)."""
    return [labels[w] for w in word_ids]


def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(M0, revision=M0_REV)
