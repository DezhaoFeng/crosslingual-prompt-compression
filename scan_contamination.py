"""Hit-rate check for the contamination filter on a seeded sample of whole Wikipedia documents.

Usage: python scan_contamination.py [--docs 3000] [--out results/contamination/wiki_scan.json]
A hit means a document shares an n-gram with a Belebele passage; whether it is a false positive needs a manual read.
"""
import argparse
import json
from pathlib import Path

import opencc
from datasets import load_dataset

from contamination import BELEBELE_REV, LANGS, NGRAM_CHARS, build_index, load_passages, normalize
from sample_wiki import WIKI, WIKI_DATE, WIKI_REV

SEED = 713
SHUFFLE_BUFFER = 2000
CONTEXT_CHARS = 80
T2S = opencc.OpenCC("t2s")


def raw_context(text, gram):
    # per-character normalization: positions are approximate if NFKC combines across characters
    ends, s = [], ""
    for c in text:
        s += normalize(c)
        ends.append(len(s))
    at = s.find(gram)
    start = next(i for i, e in enumerate(ends) if e > at)
    stop = next(i for i, e in enumerate(ends) if e >= at + len(gram))
    return text[max(0, start - CONTEXT_CHARS):stop + CONTEXT_CHARS + 1]


def scan_lang(lang, docs):
    n = NGRAM_CHARS[lang]
    passages = load_passages(lang)
    index = build_index(passages, lang)
    norm_passages = [normalize(p) for p in passages]
    ds = load_dataset(WIKI, f"{WIKI_DATE}.{lang}", split="train", streaming=True, revision=WIKI_REV)
    seen, hits = 0, []
    for doc in ds.shuffle(seed=SEED, buffer_size=SHUFFLE_BUFFER):
        seen += 1
        text = T2S.convert(doc["text"]) if lang == "zh" else doc["text"]
        s = normalize(text)
        grams = [s[i:i + n] for i in range(len(s) - n + 1) if s[i:i + n] in index]
        if grams:
            hits.append({
                "doc_id": doc["id"], "title": doc["title"], "url": doc["url"], "n_shared_ngrams": len(grams),
                "first_ngram": grams[0], "doc_context": raw_context(text, grams[0]),
                "passage": next(p for p, q in zip(passages, norm_passages) if grams[0] in q),
            })
        if seen >= docs:
            break
    return {"docs": seen, "docs_with_hit": len(hits), "hit_rate": len(hits) / seen, "hits": hits}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=3000)
    ap.add_argument("--out", default="results/contamination/wiki_scan.json")
    args = ap.parse_args()
    report = {"wiki_revision": WIKI_REV, "belebele_revision": BELEBELE_REV, "seed": SEED,
              "shuffle_buffer": SHUFFLE_BUFFER, "ngram_chars": NGRAM_CHARS,
              "langs": {lang: scan_lang(lang, args.docs) for lang in LANGS}}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
