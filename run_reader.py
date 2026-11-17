"""Belebele reading-comprehension evaluation and Gate G1 reader eligibility check (DECISIONS.md B11; plan.md G1).

Candidate Reader Order (fixed before G1 in DECISIONS.md B11):
  1. Qwen/Qwen2.5-7B-Instruct (local GPU on CSIL 4090, BF16, scored deterministically on A/B/C/D option logits)
  2. deepseek-flash (API fallback, temperature=0, thinking disabled)

Usage on CSIL:
    # Gate G1: check reader eligibility on the first 100 questions per language (full vs none >= 15 pp)
    python run_reader.py --reader Qwen/Qwen2.5-7B-Instruct --langs en,zh --g1-check

    # Full Belebele evaluation (900 questions per language) for a compression method
    python run_reader.py --reader Qwen/Qwen2.5-7B-Instruct --method full --langs en,zh
    python run_reader.py --reader Qwen/Qwen2.5-7B-Instruct --method none --langs en,zh
    python run_reader.py --reader Qwen/Qwen2.5-7B-Instruct --method random --langs en,zh --rhos 0.5,0.33,0.2
    python run_reader.py --reader Qwen/Qwen2.5-7B-Instruct --method m0 --langs en,zh --rhos 0.5,0.33,0.2
    python run_reader.py --reader Qwen/Qwen2.5-7B-Instruct --method m1 --langs en,zh --rhos 0.5,0.33,0.2
    python run_reader.py --reader Qwen/Qwen2.5-7B-Instruct --method m4 --langs en,zh --rhos 0.5,0.33,0.2
"""
import argparse
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from compress import (
    DEFAULT_LANGS,
    DEFAULT_RHOS,
    METHODS,
    RATE_FREE_METHODS,
    out_path as compress_out_path,
    read_results,
    sha256,
)
from contamination import BELEBELE, BELEBELE_REV, BELEBELE_SPLIT, CONFIGS, PASSAGE_COLUMN

CANDIDATE_READERS = (
    "Qwen/Qwen2.5-7B-Instruct",
    "deepseek-flash",
)
LOCAL_READER_REVISIONS = {
    "Qwen/Qwen2.5-7B-Instruct": None,  # pinned at runtime via model.config._commit_hash if available
}
API_BASE = "https://api.deepseek.com"
API_READERS = {"deepseek-flash", "deepseek-v4-pro"}

OPTION_LETTERS = ("A", "B", "C", "D")
NUM_TO_LETTER = {"1": "A", "2": "B", "3": "C", "4": "D", 1: "A", 2: "B", 3: "C", 4: "D"}

G1_NUM_QUESTIONS = 100
G1_MIN_GAP = 0.15  # full - none >= 15 percentage points

READER_PROMPT_VERSION = "belebele-mcq-v1"
SYSTEM_PROMPT = (
    "You are a careful reading-comprehension assistant. Select the single best option (A, B, C, or D) "
    "that answers the question. Respond with the option letter only."
)
USER_WITH_PASSAGE = (
    "Passage:\n{passage}\n\n"
    "Question: {question}\n"
    "A. {mc_answer1}\n"
    "B. {mc_answer2}\n"
    "C. {mc_answer3}\n"
    "D. {mc_answer4}\n\n"
    "Answer with the option letter only (A, B, C, or D)."
)
USER_NO_PASSAGE = (
    "Question: {question}\n"
    "A. {mc_answer1}\n"
    "B. {mc_answer2}\n"
    "C. {mc_answer3}\n"
    "D. {mc_answer4}\n\n"
    "Answer with the option letter only (A, B, C, or D)."
)


def reader_slug(reader_name):
    return reader_name.split("/")[-1]


def build_user_prompt(question_row, compressed_passage, is_none=False):
    """Build the multiple-choice prompt. Question and options are NEVER compressed."""
    if is_none:
        return USER_NO_PASSAGE.format(
            question=question_row["question"],
            mc_answer1=question_row["mc_answer1"],
            mc_answer2=question_row["mc_answer2"],
            mc_answer3=question_row["mc_answer3"],
            mc_answer4=question_row["mc_answer4"],
        )
    return USER_WITH_PASSAGE.format(
        passage=compressed_passage,
        question=question_row["question"],
        mc_answer1=question_row["mc_answer1"],
        mc_answer2=question_row["mc_answer2"],
        mc_answer3=question_row["mc_answer3"],
        mc_answer4=question_row["mc_answer4"],
    )


def prompt_sha256():
    return sha256(SYSTEM_PROMPT + "\x00" + USER_WITH_PASSAGE + "\x00" + USER_NO_PASSAGE)


def gold_letter_of(row):
    raw = row["correct_answer_num"]
    if raw not in NUM_TO_LETTER:
        raise ValueError(f"Unexpected correct_answer_num: {raw!r}")
    return NUM_TO_LETTER[raw]


def parse_mc_answer(raw_text):
    """Extract A/B/C/D from a model's text reply. Returns 'invalid' if no unambiguous option is found."""
    if not raw_text:
        return "invalid"
    s = raw_text.strip()
    if s in OPTION_LETTERS:
        return s
    # Match leading option letter like "A", "A.", "(A)", "A:"
    m = re.match(r"^[\(\[\s]*([ABCD])[\)\]\.\:\s]", s, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Match explicit "Answer: B" or standalone single letter
    m2 = re.search(r"\b([ABCD])\b", s)
    if m2:
        return m2.group(1)
    return "invalid"


def resolve_option_token_ids(tokenizer):
    """Return dict {'A': token_id, 'B': token_id, 'C': token_id, 'D': token_id} for option-logit scoring."""
    ids = {}
    for letter in OPTION_LETTERS:
        toks = tokenizer.encode(letter, add_special_tokens=False)
        if not toks:
            raise ValueError(f"Tokenizer returned empty encoding for option {letter!r}")
        ids[letter] = toks[0]
    if len(set(ids.values())) != 4:
        raise ValueError(f"Option token IDs are not distinct: {ids}")
    return ids


def score_options_local(model, tokenizer, option_ids, user_content, device=None):
    """Score option letters A/B/C/D via next-token logits under the model's chat template."""
    import math
    import torch

    if device is None:
        device = next(model.parameters()).device
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt_text = f"{SYSTEM_PROMPT}\n\n{user_content}\nAnswer: "

    enc = tokenizer(prompt_text, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    model.eval()
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        next_logits = out.logits[0, -1]

    raw_logits = {letter: float(next_logits[tid].detach().cpu().item()) for letter, tid in option_ids.items()}
    max_l = max(raw_logits.values())
    exps = {k: math.exp(v - max_l) for k, v in raw_logits.items()}
    total = sum(exps.values())
    probs = {k: round(v / total, 6) for k, v in exps.items()}
    # Deterministic tie-breaking in A, B, C, D order
    pred = max(OPTION_LETTERS, key=lambda l: raw_logits[l])
    return {
        "pred_letter": pred,
        "option_logits": {k: round(v, 6) for k, v in raw_logits.items()},
        "option_probs": probs,
        "input_tokens": int(input_ids.shape[1]),
        "output_tokens": 1,
        "finish": "eos",
    }


def answer_api(reader_model, key, user_content, question_id, max_tokens=16, retries=5, post=requests.post):
    """Call OpenAI-compatible API reader with temperature 0 and thinking disabled."""
    body = {
        "model": reader_model,
        "temperature": 0,
        "max_tokens": max_tokens,
        "thinking": {"type": "disabled"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    }
    failed = []
    for attempt in range(retries):
        sent = datetime.now(timezone.utc).isoformat(timespec="seconds")
        try:
            r = post(
                f"{API_BASE}/chat/completions",
                json=body,
                timeout=120,
                headers={"Authorization": f"Bearer {key}"},
            )
        except requests.RequestException as e:
            failed.append({
                "sent_utc": sent,
                "error": type(e).__name__,
                "cost_unknown": not isinstance(e, requests.ConnectTimeout),
            })
        else:
            if r.status_code == 200:
                break
            if r.status_code not in (429, 500, 502, 503, 504):
                raise SystemExit(f"{question_id}: API error {r.status_code}: {r.text[:500]}")
            failed.append({"sent_utc": sent, "error": f"HTTP {r.status_code}", "cost_unknown": r.status_code >= 500})
        print(f"  {question_id}: {failed[-1]['error']}, retry {attempt + 1}/{retries}")
        time.sleep(2 ** attempt * 3)
    else:
        raise SystemExit(f"{question_id}: API still failing after {retries} tries; rerun to resume.")

    d = r.json()
    choice = d["choices"][0]
    usage = d.get("usage") or {}
    raw_out = choice["message"].get("content")
    pred = parse_mc_answer(raw_out)
    finish_map = {"stop": "eos", "length": "length"}
    return {
        "pred_letter": pred,
        "raw_output": raw_out,
        "finish": finish_map.get(choice.get("finish_reason"), choice.get("finish_reason")),
        "input_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
        "served_model": d.get("model"),
        "system_fingerprint": d.get("system_fingerprint"),
        "request_id": d.get("id"),
        "sent_utc": sent,
        "failed_attempts": failed,
        "cost_unknown": any(a["cost_unknown"] for a in failed),
    }


def load_belebele_questions(lang, n=None):
    """Load Belebele questions in deterministic dataset order."""
    from datasets import load_dataset

    ds = load_dataset(BELEBELE, CONFIGS[lang], split=BELEBELE_SPLIT, revision=BELEBELE_REV)
    questions = []
    seen_passages = {}
    for idx, row in enumerate(ds):
        if n is not None and idx >= n:
            break
        p_text = row[PASSAGE_COLUMN]
        p_hash = sha256(p_text)
        if p_hash not in seen_passages:
            seen_passages[p_hash] = f"{lang}_p{len(seen_passages):03d}"
        pid = seen_passages[p_hash]
        qid = f"{lang}_q{idx:03d}"
        q_payload = json.dumps(
            [row["question"], row["mc_answer1"], row["mc_answer2"], row["mc_answer3"], row["mc_answer4"]],
            ensure_ascii=False,
        )
        questions.append({
            "question_id": qid,
            "question_index": idx,
            "question_number": row.get("question_number", idx + 1),
            "lang": lang,
            "passage_id": pid,
            "flores_passage": p_text,
            "passage_sha256": p_hash,
            "question": row["question"],
            "mc_answer1": row["mc_answer1"],
            "mc_answer2": row["mc_answer2"],
            "mc_answer3": row["mc_answer3"],
            "mc_answer4": row["mc_answer4"],
            "correct_answer_num": row["correct_answer_num"],
            "gold_letter": gold_letter_of(row),
            "question_sha256": sha256(q_payload),
        })
    return questions


def load_compressed_lookup(compress_dir, method, lang, rho, questions):
    """Return dict mapping passage_sha256 -> compressed record."""
    if method == "none":
        return {
            q["passage_sha256"]: {
                "passage_id": q["passage_id"],
                "compressed_text": "",
                "word_retention": 0.0,
                "target_retention": 0.0,
            }
            for q in questions
        }
    c_path = compress_out_path(compress_dir, method, lang, rho)
    if method == "full" and not c_path.exists():
        return {
            q["passage_sha256"]: {
                "passage_id": q["passage_id"],
                "compressed_text": q["flores_passage"],
                "word_retention": 1.0,
                "target_retention": 1.0,
            }
            for q in questions
        }
    if not c_path.exists():
        raise FileNotFoundError(
            f"Compressed passages file {c_path} not found. Run compress.py --method {method} --langs {lang} first."
        )
    recs = read_results(c_path)
    by_hash = {r["passage_sha256"]: r for r in recs}
    missing = [q["passage_id"] for q in questions if q["passage_sha256"] not in by_hash]
    if missing:
        raise ValueError(f"{c_path} is missing {len(missing)} passages required by questions (e.g. {missing[:3]})")
    return by_hash


def reader_run_config(reader_name, method, lang, rho):
    mode = "api-chat-temp0" if reader_name in API_READERS else "local-option-logits-bf16"
    return {
        "belebele_dataset": BELEBELE,
        "belebele_revision": BELEBELE_REV,
        "reader": reader_name,
        "reader_mode": mode,
        "prompt_version": READER_PROMPT_VERSION,
        "prompt_sha256": prompt_sha256(),
        "method": method,
        "lang": lang,
        "rho": rho,
    }


def reader_out_path(out_dir, reader_name, method, lang, rho, suffix=""):
    slug = reader_slug(reader_name)
    tag = f"{slug}__{method}__{lang}" if rho is None else f"{slug}__{method}__{lang}__rho{rho}"
    return Path(out_dir) / f"{tag}{suffix}.jsonl"


def check_reader_resume(recs, cfg, questions_by_id, comp_by_hash):
    """Ensure existing reader records match current config, question text, and compressed passage text."""
    for rec in recs:
        if rec.get("config") != cfg:
            raise SystemExit(
                f"{rec['question_id']} in output file was made with another config:\n"
                f"  file: {rec.get('config')}\n  now:  {cfg}\nuse a different --out-dir"
            )
        q = questions_by_id.get(rec["question_id"])
        if q is not None:
            if rec["question_sha256"] != q["question_sha256"]:
                raise SystemExit(f"{rec['question_id']}: question/options changed since it was run")
            comp_text = comp_by_hash[q["passage_sha256"]]["compressed_text"]
            if rec["compressed_passage_sha256"] != sha256(comp_text):
                raise SystemExit(f"{rec['question_id']}: compressed passage text changed since it was run")


def evaluate_g1_eligibility(full_recs_by_lang, none_recs_by_lang, min_gap=G1_MIN_GAP, total_conditions=17):
    """Compute Gate G1 reader eligibility report across languages and extrapolate full evaluation cost."""
    lang_reports = {}
    all_passed = True
    total_tokens_in = 0
    total_tokens_out = 0
    total_seconds = 0.0
    total_q = 0

    for lang in sorted(full_recs_by_lang):
        f_recs = full_recs_by_lang[lang]
        n_recs = none_recs_by_lang[lang]
        if len(f_recs) != len(n_recs) or not f_recs:
            raise ValueError(f"Mismatched or empty G1 records for {lang}: full={len(f_recs)}, none={len(n_recs)}")

        acc_full = sum(1 for r in f_recs if r["correct"]) / len(f_recs)
        acc_none = sum(1 for r in n_recs if r["correct"]) / len(n_recs)
        gap = acc_full - acc_none
        passed = gap >= min_gap - 1e-12
        all_passed = all_passed and passed

        secs = sum(r.get("seconds", 0.0) for r in f_recs) + sum(r.get("seconds", 0.0) for r in n_recs)
        tok_in = sum(r.get("input_tokens") or 0 for r in f_recs) + sum(r.get("input_tokens") or 0 for r in n_recs)
        tok_out = sum(r.get("output_tokens") or 0 for r in f_recs) + sum(r.get("output_tokens") or 0 for r in n_recs)
        total_seconds += secs
        total_tokens_in += tok_in
        total_tokens_out += tok_out
        total_q += len(f_recs) + len(n_recs)

        lang_reports[lang] = {
            "n_questions": len(f_recs),
            "acc_full": round(acc_full, 4),
            "acc_none": round(acc_none, 4),
            "gap_full_minus_none": round(gap, 4),
            "threshold": min_gap,
            "passed": passed,
            "invalid_full": sum(1 for r in f_recs if r["pred_letter"] == "invalid"),
            "invalid_none": sum(1 for r in n_recs if r["pred_letter"] == "invalid"),
            "seconds": round(secs, 2),
            "input_tokens": tok_in,
            "output_tokens": tok_out,
        }

    # Extrapolate to 900 questions per language across total_conditions (default 17: full, none, + 5 methods * 3 rhos)
    n_langs = len(lang_reports)
    scale_factor = (900 * n_langs * total_conditions) / max(1, total_q)
    return {
        "g1_passed": all_passed,
        "min_gap_required": min_gap,
        "per_language": lang_reports,
        "pilot_totals": {
            "queries": total_q,
            "seconds": round(total_seconds, 2),
            "queries_per_sec": round(total_q / max(1e-9, total_seconds), 2),
            "input_tokens": total_tokens_in,
            "output_tokens": total_tokens_out,
        },
        "extrapolated_full_eval": {
            "conditions": total_conditions,
            "questions_per_lang": 900,
            "total_queries": 900 * n_langs * total_conditions,
            "estimated_minutes": round((total_seconds * scale_factor) / 60.0, 1),
            "estimated_input_tokens": int(round(total_tokens_in * scale_factor)),
            "estimated_output_tokens": int(round(total_tokens_out * scale_factor)),
        },
    }


def load_local_reader(reader_name, device=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    tok = AutoTokenizer.from_pretrained(reader_name)
    model = AutoModelForCausalLM.from_pretrained(reader_name, torch_dtype=dtype).to(device)
    model.eval()
    opt_ids = resolve_option_token_ids(tok)
    return model, tok, opt_ids, device


def run_condition(reader_name, method, lang, rho, n_questions, compress_dir, out_dir, local_bundle=None, suffix=""):
    questions = load_belebele_questions(lang, n=n_questions)
    q_by_id = {q["question_id"]: q for q in questions}
    comp_by_hash = load_compressed_lookup(compress_dir, method, lang, rho, questions)

    out_file = reader_out_path(out_dir, reader_name, method, lang, rho, suffix=suffix)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    cfg = reader_run_config(reader_name, method, lang, rho)

    existing = read_results(out_file)
    check_reader_resume(existing, cfg, q_by_id, comp_by_hash)
    done = {r["question_id"] for r in existing}
    todo = [q for q in questions if q["question_id"] not in done]
    print(f"[reader] {out_file.name}: {len(done)}/{len(questions)} done, {len(todo)} to run")

    api_key = None
    if reader_name in API_READERS and todo:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise SystemExit(f"DEEPSEEK_API_KEY environment variable is required for API reader {reader_name}")

    with open(out_file, "a", encoding="utf-8") as f:
        for q in todo:
            t0 = time.time()
            comp_rec = comp_by_hash[q["passage_sha256"]]
            comp_text = comp_rec["compressed_text"]
            user_text = build_user_prompt(q, comp_text, is_none=(method == "none"))

            if reader_name in API_READERS:
                ans = answer_api(reader_name, api_key, user_text, q["question_id"])
            else:
                model, tok, opt_ids, device = local_bundle
                ans = score_options_local(model, tok, opt_ids, user_text, device=device)

            rec = {
                "question_id": q["question_id"],
                "question_index": q["question_index"],
                "question_number": q["question_number"],
                "lang": lang,
                "method": method,
                "rho": rho,
                "passage_id": q["passage_id"],
                "passage_sha256": q["passage_sha256"],
                "compressed_passage_sha256": sha256(comp_text),
                "question_sha256": q["question_sha256"],
                "word_retention": comp_rec.get("word_retention"),
                "gold_letter": q["gold_letter"],
                "pred_letter": ans["pred_letter"],
                "correct": ans["pred_letter"] == q["gold_letter"],
                **ans,
                "seconds": round(time.time() - t0, 4),
                "config": cfg,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()

    return read_results(out_file)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run reader on Belebele and check Gate G1 eligibility.")
    ap.add_argument("--reader", default=CANDIDATE_READERS[0], help="Reader model (local HF or API)")
    ap.add_argument("--method", choices=METHODS, default="full")
    ap.add_argument("--langs", default=",".join(DEFAULT_LANGS))
    ap.add_argument("--rhos", default=",".join(str(r) for r in DEFAULT_RHOS))
    ap.add_argument("--n", type=int, default=None, help="Limit questions per language (e.g. 100 for G1)")
    ap.add_argument("--g1-check", action="store_true",
                    help="Run Gate G1 eligibility check (first 100 questions, full vs none)")
    ap.add_argument("--compress-dir", default="results/compress")
    ap.add_argument("--out-dir", default="results/reader")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    langs = [l.strip() for l in args.langs.split(",") if l.strip()]
    local_bundle = None
    if args.reader not in API_READERS:
        local_bundle = load_local_reader(args.reader, device=args.device)

    if args.g1_check:
        n_q = args.n or G1_NUM_QUESTIONS
        full_by_lang, none_by_lang = {}, {}
        for lang in langs:
            full_by_lang[lang] = run_condition(
                args.reader, "full", lang, None, n_q, args.compress_dir, args.out_dir,
                local_bundle=local_bundle, suffix=f"__g1_n{n_q}",
            )
            none_by_lang[lang] = run_condition(
                args.reader, "none", lang, None, n_q, args.compress_dir, args.out_dir,
                local_bundle=local_bundle, suffix=f"__g1_n{n_q}",
            )
        rep = evaluate_g1_eligibility(full_by_lang, none_by_lang)
        rep["reader"] = args.reader
        rep_path = Path(args.out_dir) / f"g1_report__{reader_slug(args.reader)}.json"
        rep_path.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(rep, ensure_ascii=False, indent=2))
        return

    rhos = [None] if args.method in RATE_FREE_METHODS else [float(x) for x in args.rhos.split(",") if x.strip()]
    for lang in langs:
        for rho in rhos:
            recs = run_condition(
                args.reader, args.method, lang, rho, args.n, args.compress_dir, args.out_dir,
                local_bundle=local_bundle,
            )
            acc = sum(1 for r in recs if r["correct"]) / len(recs) if recs else 0.0
            print(f"[summary] {args.reader} | {args.method} | {lang} | rho={rho}: "
                  f"n={len(recs)} acc={acc:.4f}")


if __name__ == "__main__":
    main()

