"""Frozen evaluation harness (Gate G6) and Milestone reporting (DECISIONS.md B11; plan.md G6).

Computes:
  1. Raw accuracy and Normalized Context Utilization U = (acc_rho - acc_none) / (acc_full - acc_none),
     with a near-zero denominator guard that warns and returns NaN.
  2. Passage-clustered paired bootstrap (10,000 resamples, fixed seed=713) for 95% confidence intervals:
     - Per-condition accuracy and U
     - H1: acc_zh(M1) - acc_zh(M0) (and M4 vs M0)
     - H2: [acc_zh(M1) - acc_zh(M0)] - [acc_zh(M4) - acc_zh(M0)]
     - Cross-lingual gap: [acc_en(Mk) - acc_zh(Mk)] and its change relative to M0, alongside
       English degradation check [acc_en(Mk) - acc_en(M0)].
  3. Retention summaries (word_retention, cl100k_retention, reader_token_retention).
  4. Error analysis of deleted words by category (punctuation, number_date, qualifier_negation,
     function_stopword, other_content).

Usage:
    python evaluate.py --reader Qwen/Qwen2.5-7B-Instruct --reader-dir results/reader --compress-dir results/compress
"""
import argparse
import json
import math
import random
import unicodedata
import warnings
from collections import Counter, defaultdict
from pathlib import Path

from align import content_words
from compress import DEFAULT_LANGS, DEFAULT_RHOS, DEFAULT_SEED, percentile, read_results

HARNESS_VERSION = "g6-frozen-v1.0"
PRIMARY_RHO = 0.33
BOOTSTRAP_SAMPLES = 10_000
MIN_U_DENOM = 0.05  # Warn and return NaN if |acc_full - acc_none| < 0.05 or acc_full <= acc_none

EN_QUALIFIERS_NEGATIONS = frozenset({
    "not", "no", "never", "neither", "nor", "none", "without", "n't", "cannot",
    "may", "might", "must", "could", "should", "would", "can",
    "almost", "nearly", "about", "approximately", "around", "over", "under",
    "more", "less", "most", "least", "only", "just", "merely", "barely",
    "usually", "often", "always", "sometimes", "rarely", "seldom", "generally",
    "probably", "possibly", "likely", "unlikely", "mainly", "mostly", "largely",
})
ZH_QUALIFIERS_NEGATIONS = frozenset({
    "不", "没", "没有", "未", "非", "莫", "勿", "无", "并非", "未曾", "毫无", "不用", "不会", "不能", "不可",
    "可能", "也许", "大概", "或许", "恐怕", "必定", "一定", "必须",
    "约", "大约", "超过", "不足", "至少", "最多", "至多", "近", "将近", "多达", "高达", "逾",
    "仅", "只", "只有", "只是", "光", "单",
    "通常", "往往", "经常", "常常", "总是", "一向", "偶尔", "有时", "大都", "大多", "主要", "基本",
    "曾", "曾经", "已", "已经", "刚", "刚刚", "将", "即将", "爆发", "初", "末",
})
EN_FUNCTION_WORDS = frozenset({
    "the", "a", "an", "of", "in", "to", "for", "with", "on", "at", "from", "by", "as", "into",
    "is", "are", "was", "were", "be", "been", "being", "am", "have", "has", "had", "do", "does", "did",
    "that", "which", "who", "whom", "whose", "this", "these", "those",
    "and", "or", "but", "if", "because", "while", "during", "through", "between", "among",
    "it", "its", "he", "his", "him", "she", "her", "hers", "they", "their", "them", "we", "our", "us",
})
ZH_FUNCTION_WORDS = frozenset({
    "的", "地", "得", "了", "着", "过", "之",
    "在", "于", "与", "和", "及", "以及", "或", "或者", "而", "并", "并且", "但", "但是", "却", "则", "即",
    "是", "为", "乃", "被", "把", "将", "由", "向", "对", "对于", "关于", "从", "自", "到", "至", "以", "因", "因为",
    "由于", "所以", "因此", "如果", "若", "虽", "虽然", "等", "等等", "中", "内", "上", "下", "里",
    "他", "她", "它", "他们", "她们", "它们", "这", "那", "该", "其", "此", "某",
})
CJK_DIGITS = frozenset("零〇一二三四五六七八九十百千万亿两")
WORD_CATEGORIES = ("punctuation", "number_date", "qualifier_negation", "function_stopword", "other_content")


def compute_u(acc_rho, acc_full, acc_none, min_denom=MIN_U_DENOM):
    """Normalized Context Utilization U = (acc_rho - acc_none) / (acc_full - acc_none).

    Emits a UserWarning and returns float('nan') when the denominator is <= 0 or < min_denom (plan.md G6).
    """
    denom = acc_full - acc_none
    if denom <= 0.0 or abs(denom) < min_denom:
        warnings.warn(
            f"Normalized utility U denominator (acc_full - acc_none = {denom:.4f}) is below threshold "
            f"{min_denom}; returning NaN to avoid unstable ratio.",
            UserWarning,
            stacklevel=2,
        )
        return float("nan")
    return (acc_rho - acc_none) / denom


def clean_json_float(x, ndigits=4):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return None
    return round(float(x), ndigits)


def aggregate_by_passage(question_records, passage_order=None):
    """Group question records by passage_id into ordered lists of (correct_count, total_count).

    Validates that question sets and passage assignments are consistent.
    """
    by_p = defaultdict(lambda: [0, 0])
    seen_order = []
    for r in question_records:
        pid = r["passage_id"]
        if pid not in by_p:
            seen_order.append(pid)
        by_p[pid][0] += int(bool(r["correct"]))
        by_p[pid][1] += 1
    order = passage_order if passage_order is not None else seen_order
    if set(order) != set(by_p.keys()):
        raise ValueError("Passage ID mismatch across conditions in paired evaluation")
    return order, [(by_p[pid][0], by_p[pid][1]) for pid in order]


def generate_bootstrap_weights(n_clusters, n_boot=BOOTSTRAP_SAMPLES, seed=DEFAULT_SEED):
    """Generate B deterministic multinomial cluster-weight vectors of length n_clusters.

    Each replicate draws n_clusters passages with replacement using a fixed seed.
    All models share these exact weight vectors (paired bootstrap).
    """
    rng = random.Random(seed)
    weights = []
    for _ in range(n_boot):
        w = [0] * n_clusters
        for _ in range(n_clusters):
            w[rng.randrange(n_clusters)] += 1
        weights.append(w)
    return weights


def bootstrap_accuracies(cluster_counts, boot_weights):
    """Compute replicate accuracies for one condition given cluster_counts [(s_c, n_c)] and boot_weights."""
    reps = []
    for w in boot_weights:
        num = 0
        den = 0
        for (s_c, n_c), mult in zip(cluster_counts, w):
            if mult:
                num += s_c * mult
                den += n_c * mult
        reps.append(num / den if den else 0.0)
    return reps


def ci_from_replicates(point_est, replicates, alpha=0.05):
    """Return {estimate, ci_low, ci_high} ignoring NaNs if any."""
    valid = sorted(x for x in replicates if x is not None and not math.isnan(x))
    if not valid:
        return {"estimate": clean_json_float(point_est), "ci_low": None, "ci_high": None}
    lo = percentile(valid, alpha / 2.0)
    hi = percentile(valid, 1.0 - alpha / 2.0)
    return {
        "estimate": clean_json_float(point_est),
        "ci_low": clean_json_float(lo),
        "ci_high": clean_json_float(hi),
    }


def diff_ci(point_a, point_b, reps_a, reps_b, alpha=0.05):
    """Paired difference (A - B) point estimate, 95% CI, and whether 95% CI strictly excludes 0."""
    diff_est = point_a - point_b
    diff_reps = [a - b for a, b in zip(reps_a, reps_b)]
    res = ci_from_replicates(diff_est, diff_reps, alpha=alpha)
    res["ci_above_zero"] = (res["ci_low"] is not None and res["ci_low"] > 0.0)
    res["ci_below_zero"] = (res["ci_high"] is not None and res["ci_high"] < 0.0)
    return res


def double_diff_ci(point_m1, point_m0, point_m4, reps_m1, reps_m0, reps_m4, alpha=0.05):
    """Paired double difference [acc(M1) - acc(M0)] - [acc(M4) - acc(M0)] for H2."""
    est = (point_m1 - point_m0) - (point_m4 - point_m0)
    reps = [(m1 - m0) - (m4 - m0) for m1, m0, m4 in zip(reps_m1, reps_m0, reps_m4)]
    res = ci_from_replicates(est, reps, alpha=alpha)
    res["ci_above_zero"] = (res["ci_low"] is not None and res["ci_low"] > 0.0)
    res["ci_below_zero"] = (res["ci_high"] is not None and res["ci_high"] < 0.0)
    return res


def classify_word(word_str, lang):
    """Classify a single segmented content word into one of 5 error-analysis categories."""
    s = word_str.strip()
    if not s:
        return "punctuation"
    if all(unicodedata.category(c)[0] in ("P", "S") for c in s):
        return "punctuation"

    low = unicodedata.normalize("NFKC", s).casefold()
    if lang == "zh":
        if low in ZH_QUALIFIERS_NEGATIONS:
            return "qualifier_negation"
        if low in ZH_FUNCTION_WORDS:
            return "function_stopword"
    else:
        if low in EN_QUALIFIERS_NEGATIONS:
            return "qualifier_negation"
        if low in EN_FUNCTION_WORDS:
            return "function_stopword"

    if any(c.isdigit() for c in low) or (lang == "zh" and any(c in CJK_DIGITS for c in low) and len(low) >= 2):
        return "number_date"

    return "other_content"


def analyze_deleted_words(passages_by_sha, compress_records, spans_by_sha, lang):
    """Error analysis: break down kept vs deleted words across categories for a (method, lang, rho) run.

    Accepts either embedded orig_text/word_spans on each compress record or external lookup dicts.
    """
    total_by_cat = Counter()
    dropped_by_cat = Counter()
    kept_by_cat = Counter()

    for rec in compress_records:
        h = rec.get("passage_sha256")
        text = rec.get("orig_text") if "orig_text" in rec else (passages_by_sha or {}).get(h)
        spans = rec.get("word_spans") if "word_spans" in rec else (spans_by_sha or {}).get(h)
        labels = rec.get("labels")
        if text is None or spans is None or labels is None or len(spans) != len(labels):
            continue
        for i in content_words(spans):
            s, e = spans[i]
            w_str = text[s:e]
            cat = classify_word(w_str, lang)
            total_by_cat[cat] += 1
            if labels[i] == 0:
                dropped_by_cat[cat] += 1
            else:
                kept_by_cat[cat] += 1

    total_content = sum(total_by_cat.values())
    total_dropped = sum(dropped_by_cat.values())
    categories = {}
    for cat in WORD_CATEGORIES:
        tot = total_by_cat[cat]
        drp = dropped_by_cat[cat]
        categories[cat] = {
            "total_words": tot,
            "dropped_words": drp,
            "kept_words": kept_by_cat[cat],
            "drop_rate_within_category": clean_json_float(drp / tot if tot else 0.0),
            "share_of_all_dropped": clean_json_float(drp / total_dropped if total_dropped else 0.0),
        }
    return {
        "total_content_words": total_content,
        "total_dropped_words": total_dropped,
        "overall_drop_rate": clean_json_float(total_dropped / total_content if total_content else 0.0),
        "by_category": categories,
    }


def evaluate_language_suite(cond_records_map, n_boot=BOOTSTRAP_SAMPLES, seed=DEFAULT_SEED, min_u_denom=MIN_U_DENOM):
    """Evaluate all available conditions for a single language using passage-clustered paired bootstrap.

    cond_records_map: dict mapping condition_key (e.g. 'full', 'none', 'm0__rho0.33') -> list of reader records.
    Returns dict with per-condition metrics, replicate cache (for cross-model/cross-lingual tests), and H1/H2 tests.
    """
    if "full" not in cond_records_map or "none" not in cond_records_map:
        raise ValueError("Both 'full' and 'none' reader results are required to compute U and paired CIs")

    passage_order, full_clusters = aggregate_by_passage(cond_records_map["full"])
    _, none_clusters = aggregate_by_passage(cond_records_map["none"], passage_order=passage_order)
    boot_weights = generate_bootstrap_weights(len(passage_order), n_boot=n_boot, seed=seed)

    clusters_by_cond = {"full": full_clusters, "none": none_clusters}
    for cond, recs in cond_records_map.items():
        if cond not in ("full", "none"):
            _, clusters_by_cond[cond] = aggregate_by_passage(recs, passage_order=passage_order)

    point_acc = {}
    reps_acc = {}
    for cond, cl in clusters_by_cond.items():
        num = sum(s for s, _ in cl)
        den = sum(n for _, n in cl)
        point_acc[cond] = num / den if den else 0.0
        reps_acc[cond] = bootstrap_accuracies(cl, boot_weights)

    acc_full = point_acc["full"]
    acc_none = point_acc["none"]
    reps_full = reps_acc["full"]
    reps_none = reps_acc["none"]

    metrics = {}
    for cond, recs in sorted(cond_records_map.items()):
        p_acc = point_acc[cond]
        r_acc = reps_acc[cond]
        acc_ci = ci_from_replicates(p_acc, r_acc)

        if cond == "full":
            p_u = compute_u(acc_full, acc_full, acc_none, min_denom=min_u_denom)
            u_ci = {"estimate": clean_json_float(p_u), "ci_low": clean_json_float(p_u), "ci_high": clean_json_float(p_u)}
        elif cond == "none":
            p_u = compute_u(acc_none, acc_full, acc_none, min_denom=min_u_denom)
            u_ci = {"estimate": clean_json_float(p_u), "ci_low": clean_json_float(p_u), "ci_high": clean_json_float(p_u)}
        else:
            p_u = compute_u(p_acc, acc_full, acc_none, min_denom=min_u_denom)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                r_u = [compute_u(a, f, n, min_denom=min_u_denom) for a, f, n in zip(r_acc, reps_full, reps_none)]
            u_ci = ci_from_replicates(p_u, r_u)

        w_rets = [r["word_retention"] for r in recs if r.get("word_retention") is not None]
        metrics[cond] = {
            "n_questions": len(recs),
            "n_passages": len(passage_order),
            "raw_accuracy": acc_ci,
            "normalized_utility_U": u_ci,
            "mean_word_retention": clean_json_float(sum(w_rets) / len(w_rets)) if w_rets else None,
        }

    return {
        "passage_order": passage_order,
        "point_acc": point_acc,
        "reps_acc": reps_acc,
        "metrics": metrics,
    }


def evaluate_hypotheses(lang_suites, rhos=DEFAULT_RHOS, primary_rho=PRIMARY_RHO):
    """Compute H1, H2, English degradation, and cross-lingual gap tests across compression rates."""
    out = {"primary_rho": primary_rho, "by_rho": {}}
    zh_suite = lang_suites.get("zh")
    en_suite = lang_suites.get("en")

    for rho in rhos:
        r_key = f"rho{rho}"
        entry = {"is_primary_analysis": math.isclose(float(rho), float(primary_rho))}
        c_m0 = f"m0__rho{rho}"
        c_m1 = f"m1__rho{rho}"
        c_m4 = f"m4__rho{rho}"

        # H1 and H2 on target language zh
        if zh_suite and c_m0 in zh_suite["point_acc"]:
            p_zh = zh_suite["point_acc"]
            r_zh = zh_suite["reps_acc"]
            if c_m1 in p_zh:
                entry["H1_zh_M1_minus_M0"] = diff_ci(p_zh[c_m1], p_zh[c_m0], r_zh[c_m1], r_zh[c_m0])
            if c_m4 in p_zh:
                entry["control_zh_M4_minus_M0"] = diff_ci(p_zh[c_m4], p_zh[c_m0], r_zh[c_m4], r_zh[c_m0])
            if c_m1 in p_zh and c_m4 in p_zh:
                entry["H2_zh_double_diff_M1_vs_M4"] = double_diff_ci(
                    p_zh[c_m1], p_zh[c_m0], p_zh[c_m4], r_zh[c_m1], r_zh[c_m0], r_zh[c_m4]
                )

        # English degradation check (G7) and Cross-lingual En-Zh gap analysis (Milestone)
        if en_suite and c_m0 in en_suite["point_acc"]:
            p_en = en_suite["point_acc"]
            r_en = en_suite["reps_acc"]
            for m_tag, c_mk in (("M1", c_m1), ("M4", c_m4)):
                if c_mk in p_en:
                    entry[f"en_change_{m_tag}_minus_M0"] = diff_ci(p_en[c_mk], p_en[c_m0], r_en[c_mk], r_en[c_m0])

        if en_suite and zh_suite:
            p_en, r_en = en_suite["point_acc"], en_suite["reps_acc"]
            p_zh, r_zh = zh_suite["point_acc"], zh_suite["reps_acc"]
            if len(en_suite["passage_order"]) == len(zh_suite["passage_order"]):
                for m_tag, c_mk in (("M0", c_m0), ("M1", c_m1), ("M4", c_m4)):
                    if c_mk in p_en and c_mk in p_zh:
                        entry[f"crosslingual_gap_en_minus_zh_{m_tag}"] = diff_ci(
                            p_en[c_mk], p_zh[c_mk], r_en[c_mk], r_zh[c_mk]
                        )
                if c_m0 in p_en and c_m0 in p_zh:
                    gap_m0_est = p_en[c_m0] - p_zh[c_m0]
                    gap_m0_reps = [e - z for e, z in zip(r_en[c_m0], r_zh[c_m0])]
                    for m_tag, c_mk in (("M1", c_m1), ("M4", c_m4)):
                        if c_mk in p_en and c_mk in p_zh:
                            gap_mk_est = p_en[c_mk] - p_zh[c_mk]
                            gap_mk_reps = [e - z for e, z in zip(r_en[c_mk], r_zh[c_mk])]
                            gap_change = diff_ci(gap_mk_est, gap_m0_est, gap_mk_reps, gap_m0_reps)
                            en_delta = p_en[c_mk] - p_en[c_m0]
                            zh_delta = p_zh[c_mk] - p_zh[c_m0]
                            gap_change["en_delta"] = clean_json_float(en_delta)
                            gap_change["zh_delta"] = clean_json_float(zh_delta)
                            gap_change["driven_by_en_degradation"] = (gap_change["estimate"] < 0 and en_delta < 0 and zh_delta <= 0)
                            entry[f"gap_change_{m_tag}_vs_M0"] = gap_change

        out["by_rho"][r_key] = entry
    return out


def format_markdown_report(eval_report):
    """Format a ready-to-include Markdown summary table for the Milestone report."""
    lines = [
        f"# Belebele Evaluation Results (Harness `{eval_report['harness_version']}`)\n",
        f"- Reader: `{eval_report['reader']}`",
        f"- Bootstrap: `{eval_report['bootstrap_samples']}` passage-clustered paired resamples (`seed={eval_report['seed']}`)",
        f"- Primary pre-specified analysis: **Raw Accuracy at $\\rho = {eval_report['primary_rho']}$**\n",
        "## 1. Main Results Table\n",
        "| Language | Condition | $\\rho$ | Word Ret. | Reader Tok. Ret. | Raw Acc (95% CI) | Norm. Utility $U$ (95% CI) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for lang, ldata in eval_report["languages"].items():
        for cond, m in ldata["conditions"].items():
            parts = cond.split("__rho")
            method = parts[0]
            rho_str = parts[1] if len(parts) > 1 else "—"
            wr = f"{m['mean_word_retention']:.3f}" if m.get("mean_word_retention") is not None else "—"
            rr_tok = f"{m['mean_reader_token_retention']:.3f}" if m.get("mean_reader_token_retention") is not None else "—"

            acc = m["raw_accuracy"]
            acc_str = (
                f"{acc['estimate']:.3f} [{acc['ci_low']:.3f}, {acc['ci_high']:.3f}]"
                if acc["estimate"] is not None and acc["ci_low"] is not None
                else "N/A"
            )
            u = m["normalized_utility_U"]
            u_str = (
                f"{u['estimate']:.3f} [{u['ci_low']:.3f}, {u['ci_high']:.3f}]"
                if u["estimate"] is not None and u["ci_low"] is not None
                else "NaN"
            )
            lines.append(f"| {lang} | `{method}` | {rho_str} | {wr} | {rr_tok} | {acc_str} | {u_str} |")

    lines.append("\n## 2. Hypothesis Tests (Paired Bootstrap 95% CI)\n")
    lines.append("| $\\rho$ | Primary? | Contrast | $\\Delta$ Acc | 95% CI | CI > 0? |")
    lines.append("|---:|:---:|---|---:|---:|:---:|")
    for r_key, hdata in eval_report.get("hypotheses", {}).get("by_rho", {}).items():
        rho_val = r_key.replace("rho", "")
        prim = "**Yes**" if hdata.get("is_primary_analysis") else "No"
        for k, v in hdata.items():
            if not isinstance(v, dict) or "estimate" not in v:
                continue
            est = f"{v['estimate']:+.4f}" if v["estimate"] is not None else "N/A"
            ci = f"[{v['ci_low']:+.4f}, {v['ci_high']:+.4f}]" if v["ci_low"] is not None else "N/A"
            sig = "Yes" if v.get("ci_above_zero") else "No"
            lines.append(f"| {rho_val} | {prim} | `{k}` | {est} | {ci} | {sig} |")

    lines.append("\n## 3. Error Analysis (Deleted Word Shares by Category)\n")
    lines.append("| Language | Condition | Punct. Share | Number/Date Share | Qualifier/Neg. Share | Function Share | Other Content Share |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for lang, ldata in eval_report["languages"].items():
        for cond, m in ldata["conditions"].items():
            ea = m.get("error_analysis")
            if not ea or not ea.get("total_dropped_words"):
                continue
            cats = ea["by_category"]
            p_sh = cats["punctuation"]["share_of_all_dropped"] or 0.0
            n_sh = cats["number_date"]["share_of_all_dropped"] or 0.0
            q_sh = cats["qualifier_negation"]["share_of_all_dropped"] or 0.0
            f_sh = cats["function_stopword"]["share_of_all_dropped"] or 0.0
            o_sh = cats["other_content"]["share_of_all_dropped"] or 0.0
            lines.append(f"| {lang} | `{cond}` | {p_sh:.3f} | {n_sh:.3f} | {q_sh:.3f} | {f_sh:.3f} | {o_sh:.3f} |")

    return "\n".join(lines) + "\n"


def discover_reader_files(reader_dir, reader_slug_str, lang):
    """Find all reader JSONL files for (reader_slug, lang) and map condition key -> records."""
    reader_dir = Path(reader_dir)
    prefix = f"{reader_slug_str}__"
    cond_map = {}
    for p in sorted(reader_dir.glob(f"{prefix}*__{lang}*.jsonl")):
        if "__g1_" in p.name:
            continue
        stem = p.stem[len(prefix):]  # e.g. m0__en__rho0.33 or full__en
        if f"__{lang}__rho" in stem:
            method, rho_part = stem.split(f"__{lang}__rho", 1)
            cond_key = f"{method}__rho{rho_part}"
        elif stem.endswith(f"__{lang}"):
            method = stem[:-len(f"__{lang}")]
            cond_key = method
        else:
            continue
        cond_map[cond_key] = read_results(p)
    return cond_map


def attach_compression_metadata(cond_metrics, compress_dir, lang):
    """Attach token retention stats and deleted-word error analysis from compress JSONLs."""
    compress_dir = Path(compress_dir)
    for cond_key, m in cond_metrics.items():
        if "__rho" in cond_key:
            method, rho_part = cond_key.split("__rho", 1)
            c_path = compress_dir / f"{method}__{lang}__rho{rho_part}.jsonl"
        else:
            c_path = compress_dir / f"{cond_key}__{lang}.jsonl"
        if not c_path.exists():
            m["mean_reader_token_retention"] = None
            m["mean_cl100k_retention"] = None
            m["error_analysis"] = None
            continue
        c_recs = read_results(c_path)
        rd = [r["reader_token_retention"] for r in c_recs if r.get("reader_token_retention") is not None]
        cl = [r["cl100k_retention"] for r in c_recs if r.get("cl100k_retention") is not None]
        m["mean_reader_token_retention"] = clean_json_float(sum(rd) / len(rd)) if rd else None
        m["mean_cl100k_retention"] = clean_json_float(sum(cl) / len(cl)) if cl else None
        if cond_key not in ("full", "none"):
            m["error_analysis"] = analyze_deleted_words(None, c_recs, None, lang)
        else:
            m["error_analysis"] = None


def main(argv=None):
    ap = argparse.ArgumentParser(description="Gate G6 frozen evaluation harness for Belebele prompt compression.")
    ap.add_argument("--reader", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--langs", default=",".join(DEFAULT_LANGS))
    ap.add_argument("--rhos", default=",".join(str(r) for r in DEFAULT_RHOS))
    ap.add_argument("--reader-dir", default="results/reader")
    ap.add_argument("--compress-dir", default="results/compress")
    ap.add_argument("--bootstrap", type=int, default=BOOTSTRAP_SAMPLES)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out-json", default="results/eval_report.json")
    ap.add_argument("--out-md", default="results/eval_report.md")
    args = ap.parse_args(argv)

    slug = args.reader.split("/")[-1]
    langs = [l.strip() for l in args.langs.split(",") if l.strip()]
    rhos = [float(x) for x in args.rhos.split(",") if x.strip()]

    lang_suites = {}
    lang_reports = {}
    for lang in langs:
        cond_map = discover_reader_files(args.reader_dir, slug, lang)
        if not cond_map:
            print(f"[evaluate] No reader files found for reader={slug} lang={lang} in {args.reader_dir}")
            continue
        suite = evaluate_language_suite(cond_map, n_boot=args.bootstrap, seed=args.seed)
        attach_compression_metadata(suite["metrics"], args.compress_dir, lang)
        lang_suites[lang] = suite
        lang_reports[lang] = {"conditions": suite["metrics"]}

    hypotheses = evaluate_hypotheses(lang_suites, rhos=rhos, primary_rho=PRIMARY_RHO) if lang_suites else {}
    report = {
        "harness_version": HARNESS_VERSION,
        "reader": args.reader,
        "primary_rho": PRIMARY_RHO,
        "bootstrap_samples": args.bootstrap,
        "seed": args.seed,
        "languages": lang_reports,
        "hypotheses": hypotheses,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md_text = format_markdown_report(report)
    out_md = Path(args.out_md)
    out_md.write_text(md_text, encoding="utf-8")
    print(md_text)


if __name__ == "__main__":
    main()
