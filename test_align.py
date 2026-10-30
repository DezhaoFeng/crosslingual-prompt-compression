"""Hand-written cases for align(). Run: python test_align.py  (or pytest)"""
from align import align, teacher_stats


def spans_of(text, words):
    spans, k = [], 0
    for w in words:
        k = text.index(w, k)
        spans.append((k, k + len(w)))
        k += len(w)
    return spans


def check(text, words, teacher, status, labels=None):
    r = align(text, spans_of(text, words), teacher)
    assert r["status"] == status, (text, teacher, r)
    if labels is not None:
        assert r["labels"] == labels, (text, teacher, r)
    if status != "ok":
        assert r["labels"] is None and r["reason"], r
    return r


EN = ("The cat sat on the mat.", ["The", "cat", "sat", "on", "the", "mat", "."])
TEA = ("I like green tea", ["I", "like", "green", "tea"])
ZH = ("今天天气很好", ["今天", "天气", "很", "好"])


def test_normal_deletion():
    check(*EN, "cat sat mat.", "ok", [0, 1, 1, 0, 0, 1, 1])


def test_keep_all():
    check(*EN, "The cat sat on the mat.", "ok", [1] * 7)


def test_extra_whitespace_and_newlines_allowed():
    check(*EN, "The\n  cat   sat on the mat .", "ok", [1] * 7)


def test_repeated_word_unambiguous():
    check("a b a c", ["a", "b", "a", "c"], "b a", "ok", [0, 1, 1, 0])


def test_repeated_word_ambiguous_keeps_earliest():
    r = check("the cat the dog", ["the", "cat", "the", "dog"], "the dog", "ok", [1, 0, 0, 1])
    assert "ambiguous" in r["reason"]


def test_word_split_rejected():
    # same characters, but "a nice" became "an ice"
    check("He is a nice man", ["He", "is", "a", "nice", "man"], "He is an ice man", "uncertain")


def test_words_merged_rejected():
    check("a part", ["a", "part"], "apart", "uncertain")


def test_finds_later_legal_match():
    # first "a" + "b" would merge across a space; second "a" + "b" is legal
    check("a a,b", ["a", "a", ",", "b"], "ab", "ok", [0, 1, 0, 1])


def test_merge_not_reported_as_reorder():
    r = check("a part", ["a", "part"], "apart", "uncertain")
    assert "merges" in r["reason"]


def test_reordered():
    check(*TEA, "tea like", "reordered")


def test_reorder_respects_word_counts():
    # "bba" splits into b/b/a, but the original has only one standalone "b"
    check("a b ab", ["a", "b", "ab"], "bba", "uncertain")


def test_added_word():
    check(*TEA, "I love tea", "added")


def test_zh_no_spaces():
    check(*ZH, "今天天气好", "ok", [1, 1, 0, 1])


def test_zh_teacher_spaces_at_word_boundaries_allowed():
    check(*ZH, "今天 天气 好", "ok", [1, 1, 0, 1])


def test_zh_space_inside_word_rejected():
    check(*ZH, "今 天天气好", "uncertain")


def test_zh_rewrite_new_char():
    check(*ZH, "今天天气不错", "added")


def test_zh_partial_word_deletion():
    check(*ZH, "今天天好", "uncertain")


def test_zh_en_mixed():
    check("我用GPT-4写代码", ["我", "用", "GPT", "-", "4", "写", "代码"], "用 GPT-4 写代码", "ok", [0, 1, 1, 1, 1, 1, 1])


def test_zh_en_mixed_merge_rejected():
    check("我用 Hello World 写", ["我", "用", "Hello", "World", "写"], "用HelloWorld写", "uncertain")


def test_punctuation_dropped():
    check("Hello, world!", ["Hello", ",", "world", "!"], "Hello world!", "ok", [1, 0, 1, 1])


def test_space_before_punctuation_allowed():
    check("Hello, world!", ["Hello", ",", "world", "!"], "Hello , world !", "ok", [1, 1, 1, 1])


def test_sw_deletion():
    text = "Habari za asubuhi, rafiki yangu."
    words = ["Habari", "za", "asubuhi", ",", "rafiki", "yangu", "."]
    check(text, words, "Habari asubuhi rafiki.", "ok", [1, 0, 1, 0, 1, 0, 1])


def test_empty_output():
    check("The cat", ["The", "cat"], "  \n", "uncertain")


def test_teacher_stats_independent_of_segmentation():
    # official word-level stats would give variation 1 here (["天", "气"] vs re-split ["天气"])
    text = "天气"
    s = teacher_stats(text, spans_of(text, ["天", "气"]), "天气", [1, 1])
    assert s == {"char_variation_rate": 0.0, "char_comp_rate": 1.0, "word_keep_rate": 1.0}


def test_whitespace_only_word_is_not_original_whitespace():
    # zh: an empty word comes from a jieba-inserted space; the original has no whitespace there,
    # so the two neighbours may be joined without a space
    text = "甲乙"
    assert align(text, [(0, 1), (1, 1), (1, 2)], "甲乙")["status"] == "ok"
    # real original whitespace is still enforced
    assert align("a b", [(0, 1), (1, 1), (2, 3)], "ab")["status"] == "uncertain"


def test_word_keep_rate_ignores_whitespace_only_words():
    text = "今天 天气"
    spans = [(0, 2), (2, 2), (3, 5)]  # middle word is whitespace-only (empty span)
    s = teacher_stats(text, spans, "今天", [1, 0, 0])
    assert s["word_keep_rate"] == 0.5


def test_teacher_stats_added_chars():
    text, words = TEA
    s = teacher_stats(text, spans_of(text, words), "I love tea", None)
    assert abs(s["char_variation_rate"] - 2 / 8) < 1e-9   # "o", "v" are not in the original
    assert s["word_keep_rate"] is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("all align tests passed")
