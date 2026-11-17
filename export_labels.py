"""Training labels for M1 / M4 from the full distillation batches (DECISIONS.md, "M1 / M4 对齐").

python export_labels.py
Reads results/teacher/full/{lang}_{split}_{start}_{end}.jsonl in stream order, keeps usable rows, takes the first
N per language with N = min over languages (same N for en and zh), writes data/labels/{lang}_{split}.jsonl and
data/labels/summary.json.
"""
import json
import statistics as st
import unicodedata
from collections import Counter
from pathlib import Path

LANGS = ["en", "zh"]
SPLITS = ["train", "dev"]
SRC = Path("results/teacher/full")
OUT = Path("data/labels")
KEEP = ["id", "lang", "split", "text", "text_sha256", "prep_text", "inserted_spaces", "input_ids", "word_ids",
        "word_spans", "labels", "status", "stats"]


def batch_rows(lang, split):
    files = sorted(SRC.glob(f"{lang}_{split}_*_*.jsonl"), key=lambda p: int(p.stem.split("_")[2]))
    rows = [json.loads(line) for f in files for line in open(f, encoding="utf-8")]
    pool = [json.loads(line)["id"] for line in open(f"data/full/{lang}.jsonl", encoding="utf-8")]
    pool = [i for i in pool if i.startswith(lang)]
    order = {i: k for k, i in enumerate(pool)}
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)), f"{lang} {split}: duplicate ids"
    assert ids == sorted(ids, key=order.__getitem__), f"{lang} {split}: rows not in stream order"
    return rows


def is_punct(word):
    return all(unicodedata.category(c)[0] in "PSZ" for c in word)


def describe(rows, exported):
    keep = [r["stats"]["word_keep_rate"] for r in exported]
    deleted = punct = words = 0
    for r in exported:
        for (s, e), lab in zip(r["word_spans"], r["labels"]):
            w = r["text"][s:e]
            if not w.strip():
                continue
            words += 1
            if lab == 0:
                deleted += 1
                punct += is_punct(w)
    q = st.quantiles(keep, n=10)
    return {
        "generated": len(rows),
        "usable": sum(r["usable"] for r in rows),
        "status": dict(Counter(r["status"] for r in rows)),
        "truncated": sum(r["teacher"]["finish"] != "eos" for r in rows),
        "exported": len(exported),
        "word_keep_rate": {"mean": round(st.mean(keep), 4), "median": round(st.median(keep), 4),
                           "p10": round(q[0], 4), "p90": round(q[-1], 4)},
        "near_copy_ge_0.95": sum(k >= 0.95 for k in keep),
        "drop_label_share": round(deleted / words, 4),
        "dropped_punct_share": round(punct / deleted, 4),
        "dropped_nonpunct_words": deleted - punct,
        "teacher_tokens": {"input": sum(r["teacher"]["input_tokens"] for r in rows),
                           "output": sum(r["teacher"]["output_tokens"] for r in rows)},
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split in SPLITS:
        rows = {lang: batch_rows(lang, split) for lang in LANGS}
        usable = {lang: [r for r in rows[lang] if r["usable"]] for lang in LANGS}
        n = min(len(u) for u in usable.values())
        for lang in LANGS:
            exported = usable[lang][:n]
            with open(OUT / f"{lang}_{split}.jsonl", "w", encoding="utf-8") as f:
                for r in exported:
                    f.write(json.dumps({**{k: r[k] for k in KEEP}, "teacher_output": r["teacher"]["raw_output"]},
                                       ensure_ascii=False) + "\n")
            summary[f"{lang}_{split}"] = describe(rows[lang], exported)
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    for k, v in summary.items():
        print(k, {x: v[x] for x in ["generated", "usable", "exported", "word_keep_rate", "dropped_punct_share"]})


if __name__ == "__main__":
    assert is_punct("，") and is_punct(" .") and not is_punct("a,")
    main()
