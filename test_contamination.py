"""G3 checks for the contamination filter. Run: pytest test_contamination.py

Uses the real Belebele passages (HF cache or network). Passages are loaded here, not through
contamination.py, so the expected values do not depend on the code under test.
"""
import json
import random
import string
import unicodedata
from functools import lru_cache
from pathlib import Path

import pytest
from datasets import load_dataset

from contamination import build_index, filter_chunks, is_contaminated, load_passages, main, normalize

REV = "7899cdfa4e1e0d733fd77c848e2c273cb1d32be2"
CONFIGS = {"en": "eng_Latn", "zh": "zho_Hans", "sw": "swh_Latn"}
N = {"en": 50, "zh": 20, "sw": 50}
LANGS = list(CONFIGS)

# hand-written, on topics unrelated to Belebele
UNRELATED = {
    "en": "The Kalmar lighthouse keeper recorded tide tables by hand for forty years. His notebooks, kept in "
          "a small municipal archive, describe seasonal changes in the harbour and the arrival of herring boats.",
    "zh": "这座古桥由当地石匠在清代修建，桥身共有七个拱洞。每逢雨季，河水上涨，附近的村民便会聚在桥头观看水势，"
          "并把每年的最高水位刻在岸边的石柱上。",
    "sw": "Mlinzi wa mnara wa taa aliandika majira ya bahari kwa mkono kwa miaka arobaini. Daftari zake zimehifadhiwa "
          "katika jengo dogo la mji, na zinaeleza mabadiliko ya bandari pamoja na kuwasili kwa mashua za wavuvi.",
}


def norm(s):
    # the agreed normalization, restated independently of the code under test
    return "".join(c for c in unicodedata.normalize("NFKC", s).casefold() if unicodedata.category(c)[0] not in "PSZC")


@lru_cache(maxsize=None)
def passages(lang):
    ds = load_dataset("facebook/belebele", CONFIGS[lang], split="test", revision=REV)
    return tuple(sorted(set(ds["flores_passage"])))


@lru_cache(maxsize=None)
def index(lang):
    return build_index(passages(lang), lang)


def halves(passage):
    # cut the raw passage where half of its normalized characters have gone by
    target = len(norm(passage)) // 2
    cut = next(i for i in range(len(passage) + 1) if len(norm(passage[:i])) >= target)
    return passage[:cut], passage[cut:]


def perturb(passage, rng):
    # same words, different punctuation and whitespace
    out = []
    for c in passage:
        if c.isspace():
            out.append(rng.choice([" ", "  ", "\n", "\t", "　"]))
        elif unicodedata.category(c)[0] == "P":
            out.append(rng.choice(["", ",", "。", " - ", '"']))
        else:
            out.append(c)
    return "".join(out)


# --- normalization ---

@pytest.mark.parametrize("raw, want", [
    ("Hello, WORLD!\n", "helloworld"),
    ("你好，世界。 ", "你好世界"),
    ("ＡＢＣ１２３", "abc123"),         # full-width -> ASCII via NFKC
    ("a​b c　d", "abcd"),  # zero-width and non-ASCII spaces
    ("a+b=c $5", "abc5"),                # symbols go, digits stay
    ("ng'ombe", "ngombe"),
    ("Café", "café"),              # combining accent composed and kept
    ("q́", "q́"),              # no composed form: the mark itself stays
    ("", ""),
])
def test_normalize(raw, want):
    assert normalize(raw) == want


# --- G3 required cases, on all real passages ---

@pytest.mark.parametrize("lang", LANGS)
def test_load_passages_are_the_pinned_unique_passages(lang):
    assert sorted(load_passages(lang)) == list(passages(lang))
    assert len(load_passages(lang)) == 488


@pytest.mark.parametrize("lang", LANGS)
def test_full_passage_is_removed(lang):
    missed = [p for p in passages(lang) if not is_contaminated(p, index(lang), lang)]
    assert not missed


@pytest.mark.parametrize("lang", LANGS)
def test_half_passage_spliced_into_other_text_is_removed(lang):
    head, tail = UNRELATED[lang][:60], UNRELATED[lang][60:]
    missed = []
    for p in passages(lang):
        for half in halves(p):
            if not is_contaminated(head + half + tail, index(lang), lang):
                missed.append(half)
    assert not missed


@pytest.mark.parametrize("lang", LANGS)
def test_punctuation_and_whitespace_changes_are_removed(lang):
    rng = random.Random(713)
    missed = [p for p in passages(lang) if not is_contaminated(perturb(p, rng), index(lang), lang)]
    assert not missed


def test_case_change_is_removed():
    p = passages("en")[0]
    assert is_contaminated(p.upper(), index("en"), "en")


@pytest.mark.parametrize("lang", LANGS)
def test_unrelated_text_is_kept(lang):
    assert not is_contaminated(UNRELATED[lang], index(lang), lang)


# --- fixed parameters: exactly one shared n-gram of the agreed size decides ---

@pytest.mark.parametrize("lang", LANGS)
def test_ngram_boundary(lang):
    n = N[lang]
    rng = random.Random(lang)
    if lang == "zh":
        src = "".join(chr(0x4e00 + i) for i in rng.sample(range(5000), 200))
    else:
        src = "".join(rng.choices(string.ascii_lowercase, k=200))
    idx = build_index([src], lang)
    fill = "7" * 100  # digits are never in src, so only the copied run can match
    assert not is_contaminated(fill + src[30:30 + n - 1] + fill, idx, lang)
    assert is_contaminated(fill + src[30:30 + n] + fill, idx, lang)


@pytest.mark.parametrize("lang", LANGS)
def test_ngram_at_the_very_end_or_alone_is_found(lang):
    n = N[lang]
    src = "".join(random.Random(3).choices(string.ascii_lowercase, k=200))
    idx = build_index([src], lang)
    assert is_contaminated("7" * 100 + src[30:30 + n], idx, lang)
    assert is_contaminated(src[30:30 + n], idx, lang)


def test_load_passages_requests_the_pinned_revision(monkeypatch):
    seen = {}

    def fake_load_dataset(*args, **kwargs):
        seen.update(kwargs)
        return [{"flores_passage": "p"}]

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
    load_passages("en")
    assert seen["revision"] == REV


def test_two_separate_fragments_do_not_add_up():
    src = "".join(random.Random(1).choices(string.ascii_lowercase, k=200))
    idx = build_index([src], "en")
    assert not is_contaminated(src[:49] + "7777" + src[100:149], idx, "en")


def test_text_shorter_than_n_is_not_contaminated():
    src = "".join(random.Random(2).choices(string.ascii_lowercase, k=200))
    idx = build_index([src], "en")
    assert not is_contaminated(src[:10], idx, "en")
    assert not is_contaminated("", idx, "en")


@pytest.mark.parametrize("bad", ["", "!!! ...", "too short"])
def test_passage_shorter_than_n_is_rejected(bad):
    # such a passage could never be detected, so it must not be dropped silently
    with pytest.raises(ValueError):
        build_index(["x" * 100, bad], "en")


# --- filter_chunks ---

def row(rid, lang, text):
    return {"id": rid, "lang": lang, "text": text, "split": "train"}


def test_filter_chunks_splits_rows_and_counts_per_language():
    idx = {l: index(l) for l in LANGS}
    rows = [
        row("en_a", "en", UNRELATED["en"]),
        row("en_b", "en", passages("en")[0]),
        row("zh_a", "zh", UNRELATED["zh"]),
        row("en_c", "en", UNRELATED["en"] + " More."),
        row("en_d", "en", passages("en")[1]),
    ]
    snapshot = json.dumps(rows)
    kept, removed, stats = filter_chunks(rows, idx)
    assert [r["id"] for r in kept] == ["en_a", "zh_a", "en_c"]      # input order kept
    assert [r["id"] for r in removed] == ["en_b", "en_d"]
    assert kept[0] == rows[0]                                       # rows pass through unchanged
    assert json.dumps(rows) == snapshot                             # input not mutated
    assert stats == {
        "en": {"total": 4, "removed": 2, "kept": 2, "removed_rate": 0.5},
        "zh": {"total": 1, "removed": 0, "kept": 1, "removed_rate": 0.0},
    }


def test_filter_chunks_uses_the_index_of_the_rows_language():
    # an English passage must not be matched through the zh index
    kept, removed, _ = filter_chunks([row("zh_x", "zh", passages("en")[0])], {"zh": index("zh")})
    assert [r["id"] for r in kept] == ["zh_x"] and not removed


def test_filter_chunks_missing_language_index_fails_loudly():
    with pytest.raises(KeyError):
        filter_chunks([row("sw_a", "sw", UNRELATED["sw"])], {"en": index("en")})


def test_filter_chunks_empty():
    assert filter_chunks([], {}) == ([], [], {})


# --- command line ---

def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def read_jsonl(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]


def test_main_writes_filtered_files_and_report(tmp_path):
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    for lang in LANGS:
        write_jsonl(src / f"{lang}.jsonl", [row(f"{lang}_ok", lang, UNRELATED[lang]),
                                            row(f"{lang}_bad", lang, passages(lang)[0])])
    main(["--in", str(src), "--out", str(out)])
    for lang in LANGS:
        assert [r["id"] for r in read_jsonl(out / f"{lang}.jsonl")] == [f"{lang}_ok"]
    report = json.loads((out / "filter_report.json").read_text(encoding="utf-8"))
    assert report["belebele_revision"] == REV
    assert report["ngram_chars"] == N
    assert report["stats"]["zh"] == {"total": 2, "removed": 1, "kept": 1, "removed_rate": 0.5}
    assert {r["id"] for r in report["removed"]} == {f"{l}_bad" for l in LANGS}
    assert all(r["lang"] in LANGS and r["preview"] for r in report["removed"])


def test_main_refuses_to_overwrite_its_input(tmp_path):
    for lang in LANGS:
        write_jsonl(tmp_path / f"{lang}.jsonl", [row(f"{lang}_ok", lang, UNRELATED[lang])])
    before = (tmp_path / "en.jsonl").read_bytes()
    with pytest.raises(ValueError):
        main(["--in", str(tmp_path), "--out", str(tmp_path)])
    assert (tmp_path / "en.jsonl").read_bytes() == before


# --- fixed real Wikipedia negatives ---

@pytest.mark.parametrize("lang", LANGS)
def test_pilot_chunks_are_kept(lang):
    rows = read_jsonl(Path(__file__).parent / "data" / "pilot" / f"{lang}.jsonl")
    assert len(rows) == 40
    assert not [r["id"] for r in rows if is_contaminated(r["text"], index(lang), lang)]
