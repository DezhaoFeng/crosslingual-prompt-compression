"""Pilot sample: Wikipedia chunks per language with a document-level train/dev split.

Usage: python sample_wiki.py [--out data/pilot] [--seed 713]
Writes {lang}.jsonl and manifest.json. Chunking follows DECISIONS.md B5; zh is converted to Simplified first.
"""
import argparse
import hashlib
import importlib.metadata
import json
import re
from pathlib import Path

import datasets
import opencc
import tiktoken
import tokenizers
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer

from segment import jieba_space

WIKI = "wikimedia/wikipedia"
WIKI_REV = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
WIKI_DATE = "20231101"
M0 = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
M0_REV = "ebaba9b0e874dadd3003ffcff828e4397e568089"
LANGS = ["en", "zh", "sw"]
QUOTA = {"train": 30, "dev": 10}
DEV_BUCKET = 4  # doc goes to dev if sha256(...) % 4 == 0, i.e. ~25%
CHUNK_TOKENS = 512
XLMR_MAX = 510  # 512 minus <s> </s>; en often exceeds this at 512 gpt-4 tokens
MIN_DOC_TOKENS = 256
SHUFFLE_BUFFER = 10_000
# end-of-sentence marks; the official compress.py uses only "." and "\n"
SENT_END = re.compile(r"(?<=[.\n。！？])")

enc = tiktoken.encoding_for_model("gpt-4")
T2S_CONFIG = "t2s"
T2S = opencc.OpenCC(T2S_CONFIG)


def ntok(s):
    return len(enc.encode(s))


def split_of(seed, lang, doc_id):
    h = int(hashlib.sha256(f"{seed}:{lang}:{doc_id}".encode()).hexdigest(), 16)
    return "dev" if h % DEV_BUCKET == 0 else "train"


def hard_cut(piece, fits):
    # one sentence too long on its own: cut by characters until each part fits
    parts = []
    while not fits(piece):
        cut = len(piece) * 9 // 10
        while not fits(piece[:cut]):
            cut = cut * 9 // 10
        parts.append(piece[:cut])
        piece = piece[cut:]
    return parts + [piece]


def chunk_text(text, fits):
    """Pack whole sentences into chunks for which fits(chunk) holds. Lossless."""
    pieces = [p for s in SENT_END.split(text) if s for p in hard_cut(s, fits)]
    chunks, cur = [], ""
    for p in pieces:
        if cur and not fits(cur + p):
            chunks.append(cur)
            cur = ""
        cur += p
    if cur:
        chunks.append(cur)
    assert "".join(chunks) == text
    return chunks


def sample_lang(lang, seed, xlmr):
    def nx(s):
        # length as segment() will see it: zh gets jieba spaces first
        s = jieba_space(s)[0] if lang == "zh" else s
        return len(xlmr(s, add_special_tokens=False)["input_ids"])

    def fits(s):
        return ntok(s) <= CHUNK_TOKENS and nx(s) <= XLMR_MAX

    ds = load_dataset(WIKI, f"{WIKI_DATE}.{lang}", split="train", streaming=True, revision=WIKI_REV)
    ds = ds.shuffle(seed=seed, buffer_size=SHUFFLE_BUFFER)
    rows, taken, seen = [], {"train": 0, "dev": 0}, 0
    for doc in ds:
        seen += 1
        split = split_of(seed, lang, doc["id"])
        # zh: Traditional -> Simplified before chunking (Belebele is zho_Hans); positions refer to converted text
        doc_text = T2S.convert(doc["text"]) if lang == "zh" else doc["text"]
        if taken[split] >= QUOTA[split] or ntok(doc_text) < MIN_DOC_TOKENS:
            continue
        chunks = chunk_text(doc_text, fits)
        # pilot: only the first chunk of each doc, for topic diversity
        text = chunks[0]
        row = {
            "id": f"{lang}_{doc['id']}_c0",
            "lang": lang,
            "source_doc_id": doc["id"],
            "chunk_id": 0,
            "split": split,
            "text": text,
            "n_gpt4_tokens": ntok(text),
            "n_xlmr_tokens": nx(text),
        }
        if lang == "zh":
            # original characters of the same span; None if conversion changed the length (no 1:1 mapping)
            row["text_orig"] = doc["text"][:len(text)] if len(doc_text) == len(doc["text"]) else None
        rows.append(row)
        taken[split] += 1
        if taken == QUOTA:
            break
    return rows, seen


def check(rows):
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate ids"
    assert all(r["text"].strip() for r in rows), "empty chunk"
    by_split = {s: {r["source_doc_id"] for r in rows if r["split"] == s} for s in QUOTA}
    assert not by_split["train"] & by_split["dev"], "doc in both train and dev"
    for s, q in QUOTA.items():
        assert sum(r["split"] == s for r in rows) == q, f"{s} count != {q}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/pilot")
    ap.add_argument("--seed", type=int, default=713)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    xlmr = AutoTokenizer.from_pretrained(M0, revision=M0_REV)

    stats = {}
    for lang in LANGS:
        rows, seen = sample_lang(lang, args.seed, xlmr)
        check(rows)
        with open(out / f"{lang}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        x = [r["n_xlmr_tokens"] for r in rows]
        g = [r["n_gpt4_tokens"] for r in rows]
        stats[lang] = {
            "docs_seen": seen,
            "chunks": {s: sum(r["split"] == s for r in rows) for s in QUOTA},
            "gpt4_tokens_min_max": [min(g), max(g)],
            "xlmr_tokens_min_max": [min(x), max(x)],
            "xlmr_over_510": sum(v > 510 for v in x),
        }
        if lang == "zh":
            stats[lang]["t2s_length_changed"] = sum(r["text_orig"] is None for r in rows)
            stats[lang]["t2s_chunks_changed"] = sum(r["text_orig"] != r["text"] for r in rows)
        print(lang, stats[lang])

    manifest = {
        "dataset": WIKI, "revision": WIKI_REV, "configs": [f"{WIKI_DATE}.{l}" for l in LANGS],
        "seed": args.seed,
        "sampling": f"streaming shuffle(seed, buffer_size={SHUFFLE_BUFFER}); skip docs < {MIN_DOC_TOKENS} gpt-4 tokens; "
                    f"split = dev if sha256('seed:lang:doc_id') % {DEV_BUCKET} == 0 else train; take docs in stream "
                    f"order until quota {QUOTA} is filled; first chunk of each doc only",
        "chunking": f"split after . \\n 。！？, pack whole sentences up to {CHUNK_TOKENS} gpt-4 tokens "
                    f"and {XLMR_MAX} XLM-R tokens (zh: after jieba spacing, as segment() sees it); "
                    "overlong sentences cut by characters; lossless (asserted)",
        "chunk_tokenizer": {"name": "tiktoken encoding_for_model('gpt-4')", "tiktoken": tiktoken.__version__},
        "length_check_tokenizer": {"name": M0, "revision": M0_REV, "transformers": transformers.__version__,
                                   "tokenizers": tokenizers.__version__,
                                   "note": "zh counted after jieba spacing (segment.jieba_space)", "jieba": importlib.metadata.version("jieba")},
        "datasets": datasets.__version__,
        "zh_preprocess": {"order": "raw doc -> OpenCC t2s -> chunk; text and positions are Simplified, "
                                   "text_orig keeps the raw characters of the same span",
                          "opencc_config": T2S_CONFIG, "opencc": importlib.metadata.version("opencc")},
        "stats": stats,
        "sha256": {f"{l}.jsonl": hashlib.sha256((out / f"{l}.jsonl").read_bytes()).hexdigest() for l in LANGS},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
