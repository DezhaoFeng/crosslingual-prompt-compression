"""Side-by-side review of teacher result files (DECISIONS.md B7 evaluation items).

    python report_teacher.py results/teacher/A.jsonl results/teacher/B.jsonl --target 0.33 [--show]

Per chunk: word keep rate and its deviation from --target, longest run of skipped words (share of
the text), finish reason, status. --show prints the outputs for manual review. Numbers here do not
judge whether important facts survived; that needs reading (and, for sw, a native speaker).
"""
import argparse
import json
from pathlib import Path

from align import content_words
from run_teacher import read_results


def row_stats(r):
    if r["status"] != "ok":
        return None
    idx = content_words(r["word_spans"])
    kept = [k for k, i in enumerate(idx) if r["labels"][i]]
    n = len(idx)
    gaps = [b - a - 1 for a, b in zip([-1] + kept, kept + [n])]
    return {"keep": len(kept) / n, "longest_skip": max(gaps) / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--target", type=float, default=None)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    runs = {Path(f).stem: {r["id"]: r for r in read_results(Path(f))} for f in args.files}
    ids = sorted({i for rs in runs.values() for i in rs}, key=lambda i: (i[:2], i))
    for name, rs in runs.items():
        st = [s for s in (row_stats(r) for r in rs.values()) if s]
        secs = sum(r["teacher"]["seconds"] for r in rs.values())
        toks = sum(r["teacher"]["output_tokens"] for r in rs.values())
        dev = (f" mean|keep-target|={sum(abs(s['keep'] - args.target) for s in st) / len(st):.3f}"
               if args.target is not None and st else "")
        print(f"## {name}: n={len(rs)} ok={len(st)} usable={sum(r['usable'] for r in rs.values())} "
              f"finish={sorted({r['teacher']['finish'] for r in rs.values()})} tok/s={toks / max(secs, 1e-9):.1f}"
              f"{dev} max longest_skip={max((s['longest_skip'] for s in st), default=0):.0%}")
    for i in ids:
        print(f"\n{i}")
        for name, rs in runs.items():
            r = rs.get(i)
            if r is None:
                continue
            s = row_stats(r)
            nums = (f"keep={s['keep']:5.1%} longest_skip={s['longest_skip']:4.0%}" if s else "")
            print(f"  {name:40s} {r['status']:10s} {r['teacher']['finish']:6s} {nums}")
            if args.show and r["teacher"].get("raw_output"):
                print("     " + r["teacher"]["raw_output"][:400].replace("\n", " | "))


if __name__ == "__main__":
    main()
