"""Contamination filter (gate G3): remove Wikipedia chunks that contain Belebele passage text.

Usage: python contamination.py --in data/pilot --out data/pilot_clean
Writes {lang}.jsonl (kept rows) and filter_report.json. One shared character n-gram after normalization removes a chunk.
"""
import argparse
import json
import unicodedata
from pathlib import Path

BELEBELE = "facebook/belebele"
BELEBELE_REV = "7899cdfa4e1e0d733fd77c848e2c273cb1d32be2"
BELEBELE_SPLIT = "test"
PASSAGE_COLUMN = "flores_passage"
CONFIGS = {"en": "eng_Latn", "zh": "zho_Hans", "sw": "swh_Latn"}
LANGS = ["en", "zh", "sw"]
NGRAM_CHARS = {"en": 50, "zh": 20, "sw": 50}
DROP_CATEGORIES = "PSZC"  # first letter of the Unicode general category
PREVIEW_CHARS = 200
NORMALIZATION = "NFKC, casefold, drop characters of Unicode category P*, S*, Z*, C*"


def normalize(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    return "".join(c for c in text if unicodedata.category(c)[0] not in DROP_CATEGORIES)


def load_passages(lang):
    from datasets import load_dataset

    ds = load_dataset(BELEBELE, CONFIGS[lang], split=BELEBELE_SPLIT, revision=BELEBELE_REV)
    return sorted({row[PASSAGE_COLUMN] for row in ds})


def build_index(passages, lang):
    n = NGRAM_CHARS[lang]
    index = set()
    for passage in passages:
        s = normalize(passage)
        if len(s) < n:
            raise ValueError(f"{lang} passage normalizes to {len(s)} < {n} characters: {passage!r}")
        index.update(s[i:i + n] for i in range(len(s) - n + 1))
    return index


def is_contaminated(text, index, lang):
    n = NGRAM_CHARS[lang]
    s = normalize(text)
    return any(s[i:i + n] in index for i in range(len(s) - n + 1))


def filter_chunks(rows, indexes):
    kept, removed, stats = [], [], {}
    for row in rows:
        lang = row["lang"]
        hit = is_contaminated(row["text"], indexes[lang], lang)
        (removed if hit else kept).append(row)
        s = stats.setdefault(lang, {"total": 0, "removed": 0, "kept": 0, "removed_rate": 0.0})
        s["total"] += 1
        s["removed" if hit else "kept"] += 1
    for s in stats.values():
        s["removed_rate"] = s["removed"] / s["total"]
    return kept, removed, stats


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    in_dir, out = Path(args.in_dir), Path(args.out)
    if in_dir.resolve() == out.resolve():
        raise ValueError(f"--in and --out are the same directory: {in_dir.resolve()}")

    rows = []
    for lang in LANGS:
        # iterate the file, not splitlines(): ensure_ascii=False leaves U+2028 etc. raw inside rows
        with open(in_dir / f"{lang}.jsonl", encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f)
    indexes = {lang: build_index(load_passages(lang), lang) for lang in LANGS}
    kept, removed, stats = filter_chunks(rows, indexes)

    out.mkdir(parents=True, exist_ok=True)
    for lang in LANGS:
        with open(out / f"{lang}.jsonl", "w", encoding="utf-8") as f:
            for r in kept:
                if r["lang"] == lang:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
    report = {
        "belebele_revision": BELEBELE_REV,
        "ngram_chars": NGRAM_CHARS,
        "normalization": NORMALIZATION,
        "stats": stats,
        "removed": [{"id": r["id"], "lang": r["lang"], "preview": r["text"][:PREVIEW_CHARS]} for r in removed],
    }
    (out / "filter_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for lang, s in stats.items():
        print(lang, s)


if __name__ == "__main__":
    main()
