"""Full distillation in fixed batches (DECISIONS.md, "全量数据池与 teacher 批次规则").

python distill_full.py --lang zh --split train
Runs run_teacher.py on data/full in stream order, 200 chunks per batch, until the usable count reaches the
target or the pool is exhausted. Existing batch files are reused (run_teacher resumes), so a rerun continues.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

TARGET = {"train": 1500, "dev": 300}
START = {"train": 30, "dev": 0}  # train skips the 30 pilot chunks
BATCH = 200
OUT = Path("results/teacher/full")


def pool_size(lang, split):
    with open(f"data/full/{lang}.jsonl", encoding="utf-8") as f:
        return sum(json.loads(line)["split"] == split for line in f)


def batches(start, size):
    # zh train 30-130 was the G5 audit batch; keep its boundary so that file is reused as is
    edges = [start] + ([130] if start == 30 else []) + list(range(start + BATCH + (100 if start == 30 else 0), size, BATCH)) + [size]
    edges = sorted(set(e for e in edges if e <= size))
    return list(zip(edges, edges[1:]))


def usable(path):
    with open(path, encoding="utf-8") as f:
        return sum(json.loads(line)["usable"] for line in f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--split", required=True, choices=list(TARGET))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    total = 0
    for a, b in batches(START[args.split], pool_size(args.lang, args.split)):
        out = OUT / f"{args.lang}_{args.split}_{a:04d}_{b:04d}.jsonl"
        subprocess.run([sys.executable, "run_teacher.py", "--model", "deepseek-flash", "--prompt", "v4",
                        "--langs", args.lang, "--split", args.split, "--start", str(a), "--n", str(b - a),
                        "--max-new-tokens", "1024", "--pilot", "data/full", "--out", str(out)], check=True)
        total += usable(out)
        print(f"{args.lang} {args.split} {a}-{b}: usable so far {total}/{TARGET[args.split]}", flush=True)
        if total >= TARGET[args.split]:
            break
    else:
        print(f"{args.lang} {args.split}: pool exhausted with {total} usable (< {TARGET[args.split]})", flush=True)


if __name__ == "__main__":
    assert batches(30, 3500)[:3] == [(30, 130), (130, 330), (330, 530)] and batches(30, 3500)[-1][1] == 3500
    assert batches(0, 800) == [(0, 200), (200, 400), (400, 600), (600, 800)]
    assert batches(30, 2500)[1] == (130, 330)
    main()
