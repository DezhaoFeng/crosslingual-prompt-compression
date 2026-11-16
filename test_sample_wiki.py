import json
from pathlib import Path

from sample_wiki import strip_conversion


def test_plain_markup():
    assert strip_conversion("本名九鼎，字-{于}-震，号澄江") == "本名九鼎，字于震，号澄江"
    assert strip_conversion("-{余秉中}-与-{昆}-") == "余秉中与昆"


def test_variant_markup_prefers_simplified():
    assert strip_conversion("-{zh-hans:内存;zh-hant:記憶體}-") == "内存"
    assert strip_conversion("-{zh-tw:滑鼠; zh-cn:鼠标}-") == "鼠标"
    assert strip_conversion("-{A|zh-hans:软件;zh-hant:軟體}-") == "软件"


def test_no_markup_unchanged():
    for s in ["a - {b} - c", "x-{", "}-y", "普通文本：没有标记"]:
        assert strip_conversion(s) == s


def test_pilot_markup_rows_cleaned():
    rows = [json.loads(l) for l in open(Path(__file__).parent / "data/pilot/zh.jsonl", encoding="utf-8")]
    hits = [r for r in rows if "-{" in r["text"]]
    assert len(hits) == 3
    for r in hits:
        out = strip_conversion(r["text"])
        assert "-{" not in out and "}-" not in out
    assert "字于震" in strip_conversion(next(r["text"] for r in hits if r["id"] == "zh_3584185_c0"))
