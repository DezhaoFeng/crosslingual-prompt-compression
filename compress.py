"""Passage compression and Gate G0 diagnostics on Belebele (DECISIONS.md B2, B11; plan.md G0).

Compresses unique Belebele passages (488 per language) only, never questions or options.
Supports:
  - full / none: upper and lower bounds (no model needed)
  - random: budget-matched random word deletion, keeping ceil(rho * n) content words with fixed seed
  - m0 / m1 / m4: our B2 inference (segment -> XLM-R token p(keep) with <s>...</s> -> mean by word_id
    -> top ceil(rho * n) content words -> reconstruct from original spans)
  - m0_orig: official LLMLingua-2 PromptCompressor(rate=rho) for G0 diagnosis only
  - teacher: DeepSeek Flash v4 teacher applied directly to Belebele passages

Usage on CSIL:
    python compress.py --method full --langs en,zh
    python compress.py --method none --langs en,zh
    python compress.py --method random --langs en,zh --rhos 0.5,0.33,0.2 --seed 713
    python compress.py --method m0 --langs en,zh --rhos 0.5,0.33,0.2
    python compress.py --method m0_orig --langs en,zh --rhos 0.5,0.33,0.2
    python compress.py --method m1 --model-path checkpoints/m1_best --langs en,zh --rhos 0.5,0.33,0.2
    python compress.py --method m4 --model-path checkpoints/m4_best --langs en,zh --rhos 0.5,0.33,0.2
    python compress.py --summarize results/compress --g0-report results/compress/g0_report.json
"""
import argparse
import hashlib
import json
import math
import os
import random
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

from align import USABLE_STATUSES, align_relaxed, content_words
from contamination import BELEBELE, BELEBELE_REV, BELEBELE_SPLIT, CONFIGS, PASSAGE_COLUMN
from make_review import reconstruct

M0 = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
M0_REV = "ebaba9b0e874dadd3003ffcff828e4397e568089"
XLMR_MAX = 510  # 512 minus <s> and </s>
XLMR_BOS_ID = 0  # <s> in XLM-RoBERTa vocabulary
XLMR_EOS_ID = 2  # </s> in XLM-RoBERTa vocabulary

DEFAULT_LANGS = ("en", "zh")
DEFAULT_RHOS = (0.5, 0.33, 0.2)
DEFAULT_SEED = 713
METHODS = ("full", "none", "random", "m0", "m1", "m4", "m0_orig", "teacher")
MODEL_METHODS = ("m0", "m1", "m4")
RATE_FREE_METHODS = ("full", "none")

NEAR_COPY_THRESHOLD = 0.95
DEVIATION_THRESHOLD = 0.15


def sha256(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def target_budget(n_content, rho):
    """Exact number of content words to retain under DECISIONS.md B2: ceil(rho * n)."""
    if n_content < 0:
        raise ValueError(f"n_content must be non-negative, got {n_content}")
    if not (0.0 <= rho <= 1.0):
        raise ValueError(f"rho must be in [0, 1], got {rho}")
    if n_content == 0 or rho == 0.0:
        return 0
    return min(n_content, math.ceil(rho * n_content))


def word_scores_from_token_probs(word_ids, token_probs, n_words):
    """Mean token p(keep) for each word (DECISIONS.md B2 step 2)."""
    if len(word_ids) != len(token_probs):
        raise ValueError(f"len(word_ids)={len(word_ids)} != len(token_probs)={len(token_probs)}")
    sums = [0.0] * n_words
    counts = [0] * n_words
    for wid, prob in zip(word_ids, token_probs):
        if wid is None:
            continue
        if not (0 <= wid < n_words):
            raise ValueError(f"word_id {wid} out of range [0, {n_words})")
        sums[wid] += float(prob)
        counts[wid] += 1
    return [sums[w] / counts[w] if counts[w] else 0.0 for w in range(n_words)]


def select_top_words(spans, word_scores, rho):
    """Keep the ceil(rho * n) content words with highest score; ties broken by earlier word index.

    Whitespace-only words (empty span) are never counted in n and always get label 0 (DECISIONS.md B2).
    Returns (labels, k_kept, n_content).
    """
    if len(spans) != len(word_scores):
        raise ValueError(f"len(spans)={len(spans)} != len(word_scores)={len(word_scores)}")
    c_idx = content_words(spans)
    n = len(c_idx)
    k = target_budget(n, rho)
    ranked = sorted(c_idx, key=lambda i: (-float(word_scores[i]), i))
    keep_set = set(ranked[:k])
    labels = [1 if i in keep_set else 0 for i in range(len(spans))]
    return labels, k, n


def select_random_words(spans, rho, seed, lang, passage_sha256):
    """Budget-matched random baseline: keep ceil(rho * n) content words with a deterministic per-passage RNG."""
    c_idx = content_words(spans)
    n = len(c_idx)
    k = target_budget(n, rho)
    sub_seed = hashlib.sha256(f"{seed}:{lang}:{passage_sha256}:{rho}".encode("utf-8")).digest()
    rng = random.Random(sub_seed)
    keep_set = set(rng.sample(c_idx, k)) if k > 0 else set()
    labels = [1 if i in keep_set else 0 for i in range(len(spans))]
    return labels, k, n


def compress_with_scores(text, spans, word_scores, rho):
    """Run B2 selection and reconstruct compressed text from original spans."""
    labels, k, n = select_top_words(spans, word_scores, rho)
    comp = reconstruct(text, spans, labels)
    ret = (k / n) if n > 0 else 0.0
    return {
        "compressed_text": comp,
        "labels": labels,
        "kept_words": k,
        "content_words": n,
        "target_retention": ret,
        "word_retention": ret,
    }


def compress_random(text, spans, rho, seed, lang, passage_sha256):
    """Run budget-matched random selection and reconstruct compressed text from original spans."""
    labels, k, n = select_random_words(spans, rho, seed, lang, passage_sha256)
    comp = reconstruct(text, spans, labels)
    ret = (k / n) if n > 0 else 0.0
    return {
        "compressed_text": comp,
        "labels": labels,
        "kept_words": k,
        "content_words": n,
        "target_retention": ret,
        "word_retention": ret,
    }


def segment_passage(text, lang, tokenizer):
    """Segment passage with segment.segment(); fallback to NFKC if combining characters trigger SegmentError."""
    from segment import SegmentError, segment

    try:
        return text, segment(text, lang, tokenizer), False
    except SegmentError:
        norm_text = unicodedata.normalize("NFKC", text)
        return norm_text, segment(norm_text, lang, tokenizer), True


def score_tokens_xlmr(input_ids, model, bos_id=XLMR_BOS_ID, eos_id=XLMR_EOS_ID, max_body=XLMR_MAX, device=None):
    """Forward XLM-R token classifier with <s>...</s> in chunks of <= 510 body tokens.

    Returns list of p(keep) (class 1 probability) of length len(input_ids).
    """
    import torch

    if not input_ids:
        return []
    if device is None:
        device = next(model.parameters()).device
    probs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(input_ids), max_body):
            chunk = input_ids[start:start + max_body]
            ids = torch.tensor([[bos_id] + list(chunk) + [eos_id]], dtype=torch.long, device=device)
            logits = model(input_ids=ids).logits
            p_keep = torch.softmax(logits, dim=-1)[0, 1:-1, 1].detach().cpu().tolist()
            probs.extend(p_keep)
    assert len(probs) == len(input_ids), f"expected {len(input_ids)} token probs, got {len(probs)}"
    return probs


def diagnose_external_compression(orig_text, orig_spans, comp_text, rho, lang=None, tokenizer=None):
    """Measure word retention and G0 anomaly flags for m0_orig or teacher outputs."""
    c_idx = content_words(orig_spans)
    n = len(c_idx)
    target = target_budget(n, rho) / n if n > 0 else 0.0
    if not comp_text or not comp_text.strip():
        return {
            "compressed_text": "",
            "labels": [0] * len(orig_spans),
            "kept_words": 0,
            "content_words": n,
            "target_retention": target,
            "word_retention": 0.0,
            "align_status": "empty_output",
            "anomalies": {
                "all_dropped": True,
                "near_copy": False,
                "large_deviation": abs(0.0 - rho) > DEVIATION_THRESHOLD,
            },
        }

    al = align_relaxed(orig_text, orig_spans, comp_text)
    if al["labels"] is not None:
        labels = al["labels"]
        kept = sum(labels[i] for i in c_idx)
        ret = kept / n if n > 0 else 0.0
        status = al["status"]
    else:
        labels = None
        status = al["status"]
        if tokenizer is not None and lang is not None:
            _, comp_seg, _ = segment_passage(comp_text, lang, tokenizer)
            kept = len(content_words(comp_seg["word_spans"]))
        else:
            kept = len(comp_text.split())
        ret = kept / n if n > 0 else 0.0

    return {
        "compressed_text": comp_text,
        "labels": labels,
        "kept_words": kept,
        "content_words": n,
        "target_retention": target,
        "word_retention": ret,
        "align_status": status,
        "anomalies": {
            "all_dropped": ret == 0.0,
            "near_copy": ret >= NEAR_COPY_THRESHOLD,
            "large_deviation": abs(ret - rho) > DEVIATION_THRESHOLD,
        },
    }


def read_results(path):
    """Read a JSONL results file, repairing a half-written final line or missing trailing newline."""
    path = Path(path)
    if not path.exists():
        return []
    data = path.read_bytes()
    lines = data.split(b"\n")
    recs, offset = [], 0
    for i, line in enumerate(lines):
        if line.strip():
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                if i == len(lines) - 1:
                    print(f"warning: dropping half-written last line of {path} ({len(line)} bytes)")
                    with open(path, "r+b") as f:
                        f.truncate(offset)
                    break
                raise ValueError(f"{path}: corrupt JSON on line {i + 1}") from None
        offset += len(line) + 1
    else:
        if data and not data.endswith(b"\n"):
            with open(path, "ab") as f:
                f.write(b"\n")
    return recs


def check_resume(recs, cfg, passages_by_id):
    """Refuse to mix compression records from a different configuration or changed passage text."""
    for rec in recs:
        if rec.get("config") != cfg:
            raise SystemExit(
                f"{rec['passage_id']} in output file was made with another config:\n"
                f"  file: {rec.get('config')}\n  now:  {cfg}\nuse a different output path"
            )
        p = passages_by_id.get(rec["passage_id"])
        if p is not None and rec["passage_sha256"] != p["passage_sha256"]:
            raise SystemExit(f"{rec['passage_id']}: passage text changed since it was run")


def load_belebele_passages(lang):
    """Load the 488 unique Belebele passages for lang in deterministic first-appearance order."""
    from datasets import load_dataset

    ds = load_dataset(BELEBELE, CONFIGS[lang], split=BELEBELE_SPLIT, revision=BELEBELE_REV)
    seen = set()
    passages = []
    for row in ds:
        text = row[PASSAGE_COLUMN]
        h = sha256(text)
        if h not in seen:
            seen.add(h)
            pid = f"{lang}_p{len(passages):03d}"
            passages.append({"passage_id": pid, "lang": lang, "text": text, "passage_sha256": h})
    return passages


def percentile(sorted_vals, q):
    """Linearly interpolated percentile for q in [0, 1] on a non-empty sorted list."""
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_vals[lo])
    frac = pos - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


def dist_stats(vals):
    """Distribution summary (mean, median, p10, p90, min, max) for a list of numbers."""
    clean = sorted(float(v) for v in vals if v is not None)
    if not clean:
        return None
    return {
        "n": len(clean),
        "mean": round(sum(clean) / len(clean), 4),
        "median": round(percentile(clean, 0.5), 4),
        "p10": round(percentile(clean, 0.1), 4),
        "p90": round(percentile(clean, 0.9), 4),
        "min": round(clean[0], 4),
        "max": round(clean[-1], 4),
    }


def g0_summary(records):
    """Summarize G0 compression rates and anomalies across records grouped by (method, lang, rho)."""
    grouped = defaultdict(list)
    for r in records:
        grouped[(r["method"], r["lang"], r.get("rho"))].append(r)

    report = {}
    for (method, lang, rho), recs in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1], str(x[0][2]))):
        key = f"{method}__{lang}" if rho is None else f"{method}__{lang}__rho{rho}"
        w_rets = [r["word_retention"] for r in recs]
        exact_matches = [
            abs(r["word_retention"] - r["target_retention"]) < 1e-12
            for r in recs
            if r.get("target_retention") is not None
        ]
        cl_rets = [r.get("cl100k_retention") for r in recs if r.get("cl100k_retention") is not None]
        rd_rets = [r.get("reader_token_retention") for r in recs if r.get("reader_token_retention") is not None]

        anom_counts = {"all_dropped": 0, "near_copy": 0, "large_deviation": 0, "nfkc_fallback": 0}
        for r in recs:
            an = r.get("anomalies") or {}
            if an.get("all_dropped", r["word_retention"] == 0.0 and rho not in (0.0, None)):
                anom_counts["all_dropped"] += 1
            if an.get("near_copy", r["word_retention"] >= NEAR_COPY_THRESHOLD and rho not in (1.0, None)):
                anom_counts["near_copy"] += 1
            if rho is not None and an.get("large_deviation", abs(r["word_retention"] - rho) > DEVIATION_THRESHOLD):
                anom_counts["large_deviation"] += 1
            if r.get("nfkc_fallback"):
                anom_counts["nfkc_fallback"] += 1

        report[key] = {
            "method": method,
            "lang": lang,
            "rho": rho,
            "n_passages": len(recs),
            "word_retention": dist_stats(w_rets),
            "exact_budget_match_rate": round(sum(exact_matches) / len(exact_matches), 4) if exact_matches else None,
            "cl100k_retention": dist_stats(cl_rets),
            "reader_token_retention": dist_stats(rd_rets),
            "anomalies": anom_counts,
        }
    return report


def token_retention_ratios(orig_text, comp_text, cl100k_enc=None, reader_tok=None):
    """Compute optional cl100k and reader-tokenizer retention ratios."""
    out = {}
    if cl100k_enc is not None:
        n_orig = len(cl100k_enc.encode(orig_text))
        n_comp = len(cl100k_enc.encode(comp_text)) if comp_text else 0
        out["cl100k_orig_tokens"] = n_orig
        out["cl100k_comp_tokens"] = n_comp
        out["cl100k_retention"] = (n_comp / n_orig) if n_orig > 0 else 0.0
    if reader_tok is not None:
        n_orig = len(reader_tok.encode(orig_text, add_special_tokens=False))
        n_comp = len(reader_tok.encode(comp_text, add_special_tokens=False)) if comp_text else 0
        out["reader_orig_tokens"] = n_orig
        out["reader_comp_tokens"] = n_comp
        out["reader_token_retention"] = (n_comp / n_orig) if n_orig > 0 else 0.0
    return out


def build_config(args, lang, rho):
    """Configuration fingerprint for a single (method, lang, rho) compression file."""
    cfg = {
        "belebele_dataset": BELEBELE,
        "belebele_revision": BELEBELE_REV,
        "lang": lang,
        "method": args.method,
        "rho": rho,
    }
    if args.method in MODEL_METHODS:
        model_id = args.model_path or M0
        cfg["model"] = model_id
        cfg["revision"] = M0_REV if model_id == M0 else "local_checkpoint"
        cfg["inference"] = "B2-exact-word-budget-v1"
    elif args.method == "random":
        cfg["seed"] = args.seed
        cfg["inference"] = "B2-random-word-budget-v1"
    elif args.method == "m0_orig":
        cfg["model"] = M0
        cfg["revision"] = M0_REV
        cfg["inference"] = "official-PromptCompressor"
    elif args.method == "teacher":
        cfg["teacher_model"] = "deepseek-flash"
        cfg["prompt_version"] = "v4"
    return cfg


def out_path(out_dir, method, lang, rho):
    if rho is None:
        return Path(out_dir) / f"{method}__{lang}.jsonl"
    return Path(out_dir) / f"{method}__{lang}__rho{rho}.jsonl"


def run_compression(args):
    from segment import load_tokenizer

    langs = [l.strip() for l in args.langs.split(",") if l.strip()]
    rhos = [None] if args.method in RATE_FREE_METHODS else [float(x) for x in args.rhos.split(",") if x.strip()]

    xlmr_tok = load_tokenizer()
    cl100k_enc = None
    if not args.no_cl100k:
        try:
            import tiktoken
            cl100k_enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            cl100k_enc = None

    reader_tok = None
    if args.reader_tokenizer:
        from transformers import AutoTokenizer
        reader_tok = AutoTokenizer.from_pretrained(args.reader_tokenizer)

    model = None
    device = None
    if args.method in MODEL_METHODS:
        import torch
        from transformers import AutoModelForTokenClassification

        model_id = args.model_path or M0
        kw = {"revision": M0_REV} if model_id == M0 else {}
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModelForTokenClassification.from_pretrained(model_id, **kw).to(device)
        model.eval()

    orig_compressor = None
    if args.method == "m0_orig":
        import torch
        from llmlingua import PromptCompressor

        device_map = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        orig_compressor = PromptCompressor(
            model_name=M0,
            device_map=device_map,
            use_llmlingua2=True,
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_written_paths = []

    for lang in langs:
        passages = load_belebele_passages(lang)
        if args.n is not None:
            passages = passages[:args.n]
        by_id = {p["passage_id"]: p for p in passages}

        # Precompute segmentation (and model word scores if applicable) once per passage across all rhos
        seg_cache = {}
        score_cache = {}
        for p in passages:
            pid = p["passage_id"]
            used_text, seg, nfkc_fb = segment_passage(p["text"], lang, xlmr_tok)
            seg_cache[pid] = (used_text, seg, nfkc_fb)
            if args.method in MODEL_METHODS:
                t_probs = score_tokens_xlmr(seg["input_ids"], model, device=device)
                score_cache[pid] = word_scores_from_token_probs(
                    seg["word_ids"], t_probs, len(seg["word_spans"])
                )

        for rho in rhos:
            path = out_path(out_dir, args.method, lang, rho)
            all_written_paths.append(path)
            cfg = build_config(args, lang, rho)
            existing = read_results(path)
            check_resume(existing, cfg, by_id)
            done = {r["passage_id"] for r in existing}
            todo = [p for p in passages if p["passage_id"] not in done]
            print(f"[compress] {path.name}: {len(done)}/{len(passages)} done, {len(todo)} to run")

            with open(path, "a", encoding="utf-8") as f:
                for p in todo:
                    t0 = time.time()
                    pid = p["passage_id"]
                    used_text, seg, nfkc_fb = seg_cache[pid]
                    spans = seg["word_spans"]
                    n_content = len(content_words(spans))

                    if args.method == "full":
                        res = {
                            "compressed_text": p["text"],
                            "labels": [1 if e > s else 0 for s, e in spans],
                            "kept_words": n_content,
                            "content_words": n_content,
                            "target_retention": 1.0,
                            "word_retention": 1.0,
                        }
                    elif args.method == "none":
                        res = {
                            "compressed_text": "",
                            "labels": [0] * len(spans),
                            "kept_words": 0,
                            "content_words": n_content,
                            "target_retention": 0.0,
                            "word_retention": 0.0,
                        }
                    elif args.method == "random":
                        res = compress_random(used_text, spans, rho, args.seed, lang, p["passage_sha256"])
                    elif args.method in MODEL_METHODS:
                        res = compress_with_scores(used_text, spans, score_cache[pid], rho)
                    elif args.method == "m0_orig":
                        raw = orig_compressor.compress_prompt([p["text"]], rate=rho)
                        comp_text = raw.get("compressed_prompt", "") if isinstance(raw, dict) else str(raw)
                        res = diagnose_external_compression(used_text, spans, comp_text, rho, lang, xlmr_tok)
                    elif args.method == "teacher":
                        from run_teacher import generate_api, is_usable

                        key = os.environ.get("DEEPSEEK_API_KEY")
                        if not key:
                            raise SystemExit("DEEPSEEK_API_KEY environment variable is required for --method teacher")
                        row = {"id": pid, "lang": lang, "text": used_text}
                        g = generate_api("deepseek-flash", key, "v4", row, max_new_tokens=1024)
                        raw_out = g["raw_output"] or ""
                        res = diagnose_external_compression(used_text, spans, raw_out, rho or 0.33, lang, xlmr_tok)
                        res["teacher_meta"] = {
                            "finish": g["finish"],
                            "usable": is_usable(res.get("align_status", ""), g["finish"]),
                            "input_tokens": g["input_tokens"],
                            "output_tokens": g["output_tokens"],
                            "request_id": g["request_id"],
                        }
                        # If teacher output is usable, reconstruct cleanly from original word spans
                        if res["teacher_meta"]["usable"] and res["labels"] is not None:
                            res["compressed_text"] = reconstruct(used_text, spans, res["labels"])

                    tok_stats = token_retention_ratios(
                        p["text"], res["compressed_text"], cl100k_enc=cl100k_enc, reader_tok=reader_tok
                    )
                    rec = {
                        "passage_id": pid,
                        "lang": lang,
                        "method": args.method,
                        "rho": rho,
                        "passage_sha256": p["passage_sha256"],
                        "orig_text": used_text,
                        "word_spans": spans,
                        "nfkc_fallback": nfkc_fb,
                        **res,
                        **tok_stats,
                        "seconds": round(time.time() - t0, 4),
                        "config": cfg,
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()

    return all_written_paths


def main(argv=None):
    ap = argparse.ArgumentParser(description="Belebele passage compression and Gate G0 diagnostics.")
    ap.add_argument("--method", choices=METHODS, default="m0")
    ap.add_argument("--model-path", default=None, help="Checkpoint dir for m1/m4 (defaults to M0)")
    ap.add_argument("--langs", default=",".join(DEFAULT_LANGS))
    ap.add_argument("--rhos", default=",".join(str(r) for r in DEFAULT_RHOS))
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--n", type=int, default=None, help="Optional limit on unique passages per language")
    ap.add_argument("--device", default=None)
    ap.add_argument("--reader-tokenizer", default="Qwen/Qwen2.5-7B-Instruct",
                    help="Tokenizer for reader_token_retention (pass '' to skip)")
    ap.add_argument("--no-cl100k", action="store_true")
    ap.add_argument("--out-dir", default="results/compress")
    ap.add_argument("--summarize", default=None, help="Directory of compression JSONLs to summarize without running")
    ap.add_argument("--g0-report", default=None, help="Path for G0 JSON summary (defaults to <out-dir>/g0_report.json)")
    args = ap.parse_args(argv)

    target_dir = Path(args.summarize) if args.summarize else Path(args.out_dir)
    if not args.summarize:
        run_compression(args)

    records = []
    for p in sorted(target_dir.glob("*.jsonl")):
        records.extend(read_results(p))

    summary = g0_summary(records)
    rep_path = Path(args.g0_report) if args.g0_report else (target_dir / "g0_report.json")
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    rep_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    for k, v in summary.items():
        wr = v["word_retention"]
        mean_str = f"{wr['mean']:.4f} [{wr['min']:.4f}, {wr['max']:.4f}]" if wr else "N/A"
        print(f"{k}: n={v['n_passages']} word_ret={mean_str} exact_budget={v['exact_budget_match_rate']} "
              f"anomalies={v['anomalies']}")


if __name__ == "__main__":
    main()
