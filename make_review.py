"""Audit sheet for usable teacher samples: original first, label-reconstructed text folded below.

python make_review.py results/teacher/deepseek-flash__v4__new_en_zh.jsonl
writes <stem>_review.md (read) and <stem>_review.csv (fill in), next to the input.
"""
import csv
import json
import sys
from pathlib import Path


def reconstruct(text, spans, labels):
    """Kept words in original order; original separator between adjacent kept words, else ' ' (en) / '' (zh)."""
    out, prev_kept, prev_end = [], False, 0
    for (s, e), lab in zip(spans, labels):
        if lab:
            gap = text[prev_end:s]
            if out:
                out.append(gap if prev_kept else (" " if any(c.isspace() for c in gap) and gap.strip() == "" else ""))
            out.append(text[s:e])
        prev_kept, prev_end = bool(lab), e
    return "".join(out).strip()


def main(path):
    path = Path(path)
    recs = [json.loads(l) for l in open(path, encoding="utf-8")]
    usable = [r for r in recs if r["usable"]]
    md = [f"# 人工审核：{path.name}\n", "先只读原文，在 CSV 里记下 main_facts，再展开重建文本。\n"]
    for r in usable:
        rec = reconstruct(r["text"], r["word_spans"], r["labels"])
        md += [f"## {r['id']}（保留率 {r['stats']['word_keep_rate']:.2f}）\n", r["text"], "",
               "<details><summary>重建文本</summary>\n", rec, "\n</details>\n"]
    stem = path.with_suffix("")
    Path(f"{stem}_review.md").write_text("\n".join(md), encoding="utf-8")
    with open(f"{stem}_review.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "main_facts", "preserved", "failure_reason", "reviewer"])
        w.writerows([r["id"], "", "", "", ""] for r in usable)
    print(f"{len(usable)}/{len(recs)} usable -> {stem}_review.md, {stem}_review.csv")


if __name__ == "__main__":
    # self-check: dropped middle word, en keeps a space, zh joins directly
    assert reconstruct("a b c", [(0, 1), (2, 3), (4, 5)], [1, 0, 1]) == "a c"
    assert reconstruct("甲乙丙", [(0, 1), (1, 2), (2, 3)], [1, 0, 1]) == "甲丙"
    assert reconstruct("a, b", [(0, 1), (1, 2), (3, 4)], [1, 1, 1]) == "a, b"
    main(sys.argv[1])
