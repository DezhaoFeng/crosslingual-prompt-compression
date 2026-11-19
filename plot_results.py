"""Generate publication-quality visualizations for the cross-lingual prompt compression project.

Reads:
  - results/eval_report.json
  - results/compress/g0_report.json
  - results/compress/*.jsonl (optional, for per-passage G0 violin/box distributions)
  - data/labels/summary.json

Writes 300-DPI PNG and vector PDF figures to results/figures/:
  1. fig1_rate_accuracy_curves.{png,pdf}     - Rate vs. Accuracy & Normalized Utility U (with 95% CIs)
  2. fig2_g0_retention_diagnostics.{png,pdf} - Gate G0 word retention distributions & m0_orig failure modes
  3. fig3_hypothesis_forest_plot.{png,pdf}   - Paired bootstrap 95% CI forest plot (Cross-lingual gap, H1, H2)
  4. fig4_deleted_word_categories.{png,pdf}  - Error analysis: deleted word composition & category drop rates
  5. fig5_distillation_label_bias.{png,pdf}  - Training label distillation yield & punctuation/retention bias

Works both on the frozen G0/G1/G6 baseline report and automatically includes M1/M4/teacher once evaluated.

Usage:
    python plot_results.py
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Publication-grade Okabe-Ito colorblind-safe palette
METHOD_STYLE = {
    "m1": {"label": "M1 (Zh Fine-tuned)", "color": "#D55E00", "marker": "s", "ls": "-", "zorder": 6},
    "m4": {"label": "M4 (En Control)", "color": "#009E73", "marker": "^", "ls": "-", "zorder": 5},
    "m0": {"label": "M0 (B2 Seg. Fixed)", "color": "#0072B2", "marker": "o", "ls": "-", "zorder": 4},
    "m0_orig": {"label": "M0 Orig (Official)", "color": "#CC79A7", "marker": "v", "ls": "--", "zorder": 3},
    "random": {"label": "Random (Budget-matched)", "color": "#7F7F7F", "marker": "d", "ls": ":", "zorder": 2},
    "teacher": {"label": "Teacher (DeepSeek Flash v4)", "color": "#E69F00", "marker": "*", "ls": "-.", "zorder": 7},
}

CATEGORY_STYLE = [
    ("function_stopword", "Function / Stopwords", "#0072B2"),
    ("punctuation", "Punctuation", "#56B4E9"),
    ("qualifier_negation", "Qualifiers / Negations", "#E69F00"),
    ("number_date", "Numbers / Dates", "#CC79A7"),
    ("other_content", "Other Content Words", "#D55E00"),
]

LANG_TITLES = {"en": "English (Belebele eng_Latn)", "zh": "Chinese (Belebele zho_Hans)"}


def configure_matplotlib():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10.5,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.2,
        "figure.titlesize": 13.5,
        "figure.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
    })


def save_fig(fig, out_dir, stem):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def plot_fig1_rate_accuracy(eval_rep, out_dir):
    """Fig 1: 2x2 grid — Top row: Raw Accuracy vs rho (en, zh); Bottom row: Normalized Utility U vs rho (en, zh)."""
    langs = [l for l in ("en", "zh") if l in eval_rep.get("languages", {})]
    if not langs:
        return None
    fig, axes = plt.subplots(2, len(langs), figsize=(6.2 * len(langs), 8.6), sharex=True)
    if len(langs) == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    rhos = [0.2, 0.33, 0.5]
    for col, lang in enumerate(langs):
        conds = eval_rep["languages"][lang]["conditions"]
        ax_acc = axes[0, col]
        ax_u = axes[1, col]

        # Highlight primary rho = 0.33
        for ax in (ax_acc, ax_u):
            ax.axvspan(0.315, 0.345, color="#F0E442", alpha=0.18, zorder=0, label="_nolegend_")
            ax.axvline(0.33, color="#B59B00", ls=":", lw=1.0, alpha=0.7, zorder=1)

        # Full and None reference bounds
        if "full" in conds:
            f_acc = conds["full"]["raw_accuracy"]["estimate"]
            ax_acc.axhline(f_acc, color="#222222", ls="--", lw=1.3, alpha=0.85, label=f"Full Context ({f_acc:.3f})")
        if "none" in conds:
            n_acc = conds["none"]["raw_accuracy"]["estimate"]
            ax_acc.axhline(n_acc, color="#666666", ls="-.", lw=1.2, alpha=0.85, label=f"No Passage ∅ ({n_acc:.3f})")

        ax_u.axhline(1.0, color="#222222", ls="--", lw=1.2, alpha=0.75, label="Full Context (U = 1.0)")
        ax_u.axhline(0.0, color="#666666", ls="-.", lw=1.2, alpha=0.75, label="No Passage ∅ (U = 0.0)")

        for method, st in METHOD_STYLE.items():
            xs, acc_y, acc_lo, acc_hi = [], [], [], []
            u_y, u_lo, u_hi = [], [], []
            for r in rhos:
                ck = f"{method}__rho{r}"
                if ck not in conds:
                    continue
                m = conds[ck]
                a = m["raw_accuracy"]
                u = m["normalized_utility_U"]
                if a["estimate"] is not None:
                    xs.append(r)
                    acc_y.append(a["estimate"])
                    acc_lo.append(a["estimate"] - (a["ci_low"] if a["ci_low"] is not None else a["estimate"]))
                    acc_hi.append((a["ci_high"] if a["ci_high"] is not None else a["estimate"]) - a["estimate"])
                    u_est = u["estimate"] if u["estimate"] is not None else np.nan
                    u_y.append(u_est)
                    u_lo.append(u_est - u["ci_low"] if (u["ci_low"] is not None and not np.isnan(u_est)) else 0.0)
                    u_hi.append(u["ci_high"] - u_est if (u["ci_high"] is not None and not np.isnan(u_est)) else 0.0)

            if not xs:
                continue
            ax_acc.errorbar(
                xs, acc_y, yerr=[acc_lo, acc_hi], label=st["label"], color=st["color"],
                marker=st["marker"], ls=st["ls"], lw=2.0, markersize=6.5, capsize=3.5, zorder=st["zorder"],
            )
            ax_u.errorbar(
                xs, u_y, yerr=[u_lo, u_hi], label=st["label"], color=st["color"],
                marker=st["marker"], ls=st["ls"], lw=2.0, markersize=6.5, capsize=3.5, zorder=st["zorder"],
            )

        ax_acc.set_title(LANG_TITLES.get(lang, lang))
        ax_acc.set_ylabel("Raw Reading Comp. Accuracy" if col == 0 else "")
        ax_acc.set_ylim(0.42, 0.98)
        ax_acc.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.9)

        ax_u.set_xlabel("Target Word Retention Rate ($\\rho$)")
        ax_u.set_ylabel("Normalized Context Utility ($U$)" if col == 0 else "")
        ax_u.set_xticks(rhos)
        ax_u.set_xticklabels(["0.20\n(5x comp.)", "0.33 (Primary)\n(3x comp.)", "0.50\n(2x comp.)"])
        ax_u.set_xlim(0.16, 0.54)
        ax_u.set_ylim(-0.22, 1.08)

    fig.suptitle("Belebele Reading Comprehension vs. Compression Rate (95% Paired Bootstrap CIs)", y=1.01)
    fig.tight_layout()
    return save_fig(fig, out_dir, "fig1_rate_accuracy_curves")


def plot_fig2_g0_diagnostics(g0_rep, out_dir):
    """Fig 2: Gate G0 Diagnostic — Actual Word Retention (p10-p90 & min-max) and All-Dropped Anomaly Counts."""
    if not g0_rep:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.0))
    ax_ret, ax_tok = axes
    rhos = [0.2, 0.33, 0.5]
    x_base = np.arange(len(rhos))

    # Left panel: Word retention target adherence (m0 B2 vs m0_orig on En & Zh)
    series = [
        ("m0", "en", "M0 B2 (En) — Exact Budget", "#0072B2", "o", -0.18),
        ("m0", "zh", "M0 B2 (Zh) — Exact Budget", "#D55E00", "s", -0.06),
        ("m0_orig", "en", "M0 Orig (En) — Official", "#56B4E9", "^", 0.06),
        ("m0_orig", "zh", "M0 Orig (Zh) — Official (No-Space Bug)", "#CC79A7", "v", 0.18),
    ]
    for idx, r in enumerate(rhos):
        ax_ret.hlines(r, idx - 0.28, idx + 0.28, colors="#222222", linestyles="--", lw=1.4,
                      label="Target Budget $\\rho$" if idx == 0 else "_nolegend_")

    for method, lang, label, color, marker, offset in series:
        xs, means, p10s, p90s, mins, maxs, drops = [], [], [], [], [], [], []
        for idx, r in enumerate(rhos):
            key = f"{method}__{lang}__rho{r}"
            if key not in g0_rep:
                continue
            wr = g0_rep[key]["word_retention"]
            xs.append(x_base[idx] + offset)
            means.append(wr["mean"])
            p10s.append(wr["mean"] - wr["p10"])
            p90s.append(wr["p90"] - wr["mean"])
            mins.append(wr["min"])
            maxs.append(wr["max"])
            drops.append(g0_rep[key]["anomalies"]["all_dropped"])
        if not xs:
            continue
        ax_ret.errorbar(
            xs, means, yerr=[p10s, p90s], fmt=marker, color=color, label=label,
            markersize=7, lw=2.2, capsize=4,
        )
        # Light vertical whiskers for min-max range
        for x_pos, mn, mx, dr in zip(xs, mins, maxs, drops):
            ax_ret.vlines(x_pos, mn, mx, color=color, alpha=0.35, lw=1.2, linestyles=":")
            if dr > 0:
                ax_ret.annotate(
                    f"{dr} all-dropped\n(0 words kept)",
                    xy=(x_pos, mn), xytext=(x_pos + 0.02, mn + 0.035),
                    fontsize=8.2, color="#B00020", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#B00020", lw=0.9),
                )

    ax_ret.set_title("A. Gate G0 Word Retention: Official vs. B2 (Mean & p10–p90)")
    ax_ret.set_xticks(x_base)
    ax_ret.set_xticklabels([f"$\\rho = {r}$" for r in rhos])
    ax_ret.set_ylabel("Actual Word Retention Ratio")
    ax_ret.set_ylim(-0.03, 0.60)
    ax_ret.legend(loc="upper left", frameon=True, facecolor="white")

    # Right panel: Word Retention vs. Qwen2.5 Reader-Token Retention under B2 Inference
    width = 0.14
    bars_cfg = [
        ("m0", "en", "word_retention", "En Word Budget (B2)", "#222222", -2.0 * width),
        ("m0", "en", "reader_token_retention", "En Tok. Ret. (M0)", "#56B4E9", -1.0 * width),
        ("m0", "zh", "reader_token_retention", "Zh Tok. Ret. (M0)", "#0072B2", 0.0 * width),
        ("m4", "zh", "reader_token_retention", "Zh Tok. Ret. (M4 En-Ctrl)", "#009E73", 1.0 * width),
        ("m1", "zh", "reader_token_retention", "Zh Tok. Ret. (M1 Zh-Sup)", "#D55E00", 2.0 * width),
    ]
    bars_cfg = [b for b in bars_cfg if f"{b[0]}__{b[1]}__rho0.33" in g0_rep]
    n_b = len(bars_cfg)
    for idx_b, (method, lang, field, label, color, _) in enumerate(bars_cfg):
        shift = (idx_b - (n_b - 1) / 2.0) * width
        vals = []
        for r in rhos:
            key = f"{method}__{lang}__rho{r}"
            d = g0_rep.get(key, {}).get(field)
            vals.append(d["mean"] if d else 0.0)
        bars = ax_tok.bar(x_base + shift, vals, width=width * 0.90, color=color, label=label, alpha=0.92)
        for b, v in zip(bars, vals):
            if v > 0:
                ax_tok.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.2f}",
                            ha="center", va="bottom", fontsize=7.6)

    ax_tok.set_title("B. Word Budget vs. Qwen2.5 Reader-Token Retention")
    ax_tok.set_xticks(x_base)
    ax_tok.set_xticklabels([f"$\\rho = {r}$" for r in rhos])
    ax_tok.set_ylabel("Mean Retention Ratio (488 Passages)")
    ax_tok.set_ylim(0, 0.74)
    ax_tok.legend(loc="upper left", frameon=True, facecolor="white")

    fig.suptitle("Gate G0 Compression Rate & Token Retention Diagnostics on Belebele", y=1.02)
    fig.tight_layout(w_pad=3.0)
    return save_fig(fig, out_dir, "fig2_g0_retention_diagnostics")


def plot_fig3_hypothesis_forest(eval_rep, out_dir):
    """Fig 3: Forest plot of paired bootstrap 95% CIs for cross-lingual gap and H1/H2 contrasts."""
    by_rho = eval_rep.get("hypotheses", {}).get("by_rho", {})
    if not by_rho:
        return None

    contrast_labels = [
        ("H1_zh_M1_minus_M0", "H1: $\\mathrm{Acc}_{\\mathrm{zh}}(M_1) - \\mathrm{Acc}_{\\mathrm{zh}}(M_0)$", "#D55E00"),
        ("control_zh_M4_minus_M0", "Control: $\\mathrm{Acc}_{\\mathrm{zh}}(M_4) - \\mathrm{Acc}_{\\mathrm{zh}}(M_0)$", "#009E73"),
        ("H2_zh_double_diff_M1_vs_M4", "H2: $[\\mathrm{Acc}_{\\mathrm{zh}}(M_1)-M_0] - [\\mathrm{Acc}_{\\mathrm{zh}}(M_4)-M_0]$", "#CC79A7"),
        ("en_change_M1_minus_M0", "G7 En Check: $\\mathrm{Acc}_{\\mathrm{en}}(M_1) - \\mathrm{Acc}_{\\mathrm{en}}(M_0)$", "#E69F00"),
        ("en_change_M4_minus_M0", "G7 En Check: $\\mathrm{Acc}_{\\mathrm{en}}(M_4) - \\mathrm{Acc}_{\\mathrm{en}}(M_0)$", "#56B4E9"),
        ("crosslingual_gap_en_minus_zh_M0", "Cross-Lingual Gap ($M_0$): $\\mathrm{Acc}_{\\mathrm{en}} - \\mathrm{Acc}_{\\mathrm{zh}}$", "#0072B2"),
        ("crosslingual_gap_en_minus_zh_M1", "Cross-Lingual Gap ($M_1$): $\\mathrm{Acc}_{\\mathrm{en}} - \\mathrm{Acc}_{\\mathrm{zh}}$", "#D55E00"),
        ("crosslingual_gap_en_minus_zh_M4", "Cross-Lingual Gap ($M_4$): $\\mathrm{Acc}_{\\mathrm{en}} - \\mathrm{Acc}_{\\mathrm{zh}}$", "#009E73"),
    ]

    rows = []
    for r_key in ("rho0.5", "rho0.33", "rho0.2"):
        if r_key not in by_rho:
            continue
        rho_str = r_key.replace("rho", "")
        is_prim = by_rho[r_key].get("is_primary_analysis", False)
        for c_key, c_title, color in contrast_labels:
            if c_key in by_rho[r_key]:
                d = by_rho[r_key][c_key]
                if d.get("estimate") is not None:
                    tag = f"$\\rho={rho_str}$ ★ " if is_prim else f"$\\rho={rho_str}$   "
                    rows.append((tag + c_title, d["estimate"], d["ci_low"], d["ci_high"], color, is_prim))

    if not rows:
        return None

    rows = list(reversed(rows))
    fig_h = max(3.6, 0.58 * len(rows) + 1.4)
    fig, ax = plt.subplots(figsize=(9.6, fig_h))

    ax.axvline(0.0, color="#222222", ls="--", lw=1.3, zorder=1)
    y_pos = np.arange(len(rows))

    for idx, (label, est, lo, hi, color, is_prim) in enumerate(rows):
        if is_prim:
            ax.axhspan(idx - 0.38, idx + 0.38, color="#F0E442", alpha=0.18, zorder=0)
        err_lo = est - (lo if lo is not None else est)
        err_hi = (hi if hi is not None else est) - est
        ax.errorbar(
            [est], [idx], xerr=[[err_lo], [err_hi]], fmt="o", color=color,
            markersize=7.5, lw=2.3, capsize=4.5, zorder=3,
        )
        ax.text(
            hi + 0.004 if hi is not None else est + 0.004, idx,
            f"{est:+.3f} [{lo:+.3f}, {hi:+.3f}]",
            va="center", ha="left", fontsize=9.2, fontweight="bold" if is_prim else "normal",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("Accuracy Difference $\\Delta$ (10,000 Passage-Clustered Paired Bootstrap Resamples)")
    ax.set_title("Paired Bootstrap 95% Confidence Intervals (★ = Pre-specified Primary Rate $\\rho = 0.33$)")
    all_lo = min((r[2] for r in rows if r[2] is not None), default=-0.05)
    all_hi = max((r[3] for r in rows if r[3] is not None), default=0.15)
    span = max(0.1, all_hi - all_lo)
    ax.set_xlim(min(-0.03, all_lo - 0.12 * span), all_hi + 0.45 * span)
    fig.tight_layout()
    return save_fig(fig, out_dir, "fig3_hypothesis_forest_plot")


def plot_fig4_deleted_word_categories(eval_rep, out_dir, rho=0.33):
    """Fig 4: Error Analysis at primary rho=0.33 — Stacked share of deleted words & category deletion rates."""
    langs = [l for l in ("en", "zh") if l in eval_rep.get("languages", {})]
    if not langs:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14.6, 5.4))
    ax_share, ax_rate = axes

    # Panel A: Stacked bar chart of share_of_all_dropped across available methods at rho=0.33
    bar_entries = []
    for lang in langs:
        conds = eval_rep["languages"][lang]["conditions"]
        for m_key in ("m0_orig", "m0", "m1", "m4", "random"):
            ck = f"{m_key}__rho{rho}"
            if ck in conds and conds[ck].get("error_analysis"):
                bar_entries.append((f"{lang.upper()}\n{METHOD_STYLE[m_key]['label'].split(' (')[0]}",
                                    lang, m_key, conds[ck]["error_analysis"]["by_category"]))

    if not bar_entries:
        plt.close(fig)
        return None

    x = np.arange(len(bar_entries))
    bottoms = np.zeros(len(bar_entries))
    for cat_key, cat_label, color in CATEGORY_STYLE:
        vals = np.array([e[3][cat_key]["share_of_all_dropped"] or 0.0 for e in bar_entries])
        bars = ax_share.bar(x, vals, width=0.68, bottom=bottoms, label=cat_label, color=color, edgecolor="white", lw=0.7)
        for b, v, bot in zip(bars, vals, bottoms):
            if v >= 0.08:
                ax_share.text(b.get_x() + b.get_width() / 2, bot + v / 2, f"{v * 100:.1f}%",
                              ha="center", va="center", fontsize=7.3,
                              color="white" if cat_key != "punctuation" else "#111111",
                              fontweight="bold")
        bottoms += vals

    ax_share.set_title(f"A. Composition of Deleted Words at Primary $\\rho = {rho}$")
    ax_share.set_xticks(x)
    ax_share.set_xticklabels([e[0] for e in bar_entries])
    ax_share.set_ylabel("Share of All Deleted Words")
    ax_share.set_ylim(0, 1.05)
    ax_share.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=True)

    # Panel B: Within-category drop rate comparing M0 (EN), M0 (ZH), M4 (ZH), and M1 (ZH)
    cat_keys = [c[0] for c in CATEGORY_STYLE]
    cat_names = ["Function\nWords", "Punctuation", "Qualifiers /\nNegations", "Numbers /\nDates", "Other\nContent"]
    x_cat = np.arange(len(cat_keys))
    candidate_series = [
        ("en", "m0", "M0 (EN Ref.)", "#56B4E9"),
        ("zh", "m0", "M0 (ZH Base)", "#0072B2"),
        ("zh", "m4", "M4 (ZH w/ En-Ctrl)", "#009E73"),
        ("zh", "m1", "M1 (ZH w/ Zh-Sup)", "#D55E00"),
    ]
    focus_series = []
    for lang, m_key, s_label, color in candidate_series:
        conds = eval_rep.get("languages", {}).get(lang, {}).get("conditions", {})
        ck = f"{m_key}__rho{rho}"
        if ck in conds and conds[ck].get("error_analysis"):
            focus_series.append((s_label, conds[ck]["error_analysis"]["by_category"], color))

    w = 0.78 / max(1, len(focus_series))
    for s_idx, (s_label, by_cat, color) in enumerate(focus_series):
        shift = (s_idx - (len(focus_series) - 1) / 2) * w
        rates = [by_cat[ck]["drop_rate_within_category"] or 0.0 for ck in cat_keys]
        bars = ax_rate.bar(x_cat + shift, rates, width=w * 0.90, label=s_label, color=color, alpha=0.92)
        for b, v in zip(bars, rates):
            ax_rate.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v * 100:.0f}%",
                         ha="center", va="bottom", fontsize=7.6)

    ax_rate.axhline(1.0 - rho, color="#444444", ls="--", lw=1.1, label=f"Target Drop Rate ({1.0 - rho:.2f})")
    ax_rate.set_title(f"B. Within-Category Deletion Probability at $\\rho = {rho}$")
    ax_rate.set_xticks(x_cat)
    ax_rate.set_xticklabels(cat_names)
    ax_rate.set_ylabel("Fraction of Category Deleted")
    ax_rate.set_ylim(0, 1.14)
    ax_rate.legend(loc="upper right", frameon=True, facecolor="white")

    fig.suptitle("Error Analysis: What Types of Words Do the Compressors Delete?", y=1.02)
    fig.tight_layout(w_pad=2.5)
    return save_fig(fig, out_dir, "fig4_deleted_word_categories")


def plot_fig5_distillation_bias(labels_summary, out_dir):
    """Fig 5: Training label distillation yield (58% Zh vs 85% En) and punctuation/retention bias."""
    if not labels_summary:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.8))
    ax_status, ax_bias = axes

    splits = [("en_train", "En Train\n(1,900 gen)"), ("zh_train", "Zh Train\n(2,700 gen)"),
              ("en_dev", "En Dev\n(400 gen)"), ("zh_dev", "Zh Dev\n(600 gen)")]
    status_order = [
        ("ok", "Strict OK (usable)", "#009E73"),
        ("ok_relaxed", "Relaxed OK (usable)", "#56B4E9"),
        ("uncertain", "Uncertain / Boundary (rejected)", "#E69F00"),
        ("added", "Added / Paraphrased (rejected)", "#D55E00"),
        ("reordered", "Reordered / SegErr (rejected)", "#CC79A7"),
    ]

    x = np.arange(len(splits))
    bottoms = np.zeros(len(splits))
    for st_key, st_label, color in status_order:
        shares = []
        for sp_key, _ in splits:
            d = labels_summary[sp_key]
            gen = d["generated"]
            st_map = d["status"]
            if st_key == "reordered":
                cnt = st_map.get("reordered", 0) + st_map.get("segment_error", 0)
            else:
                cnt = st_map.get(st_key, 0)
            shares.append(cnt / gen if gen else 0.0)
        vals = np.array(shares)
        bars = ax_status.bar(x, vals, width=0.55, bottom=bottoms, label=st_label, color=color, edgecolor="white", lw=0.7)
        for b, v, bot in zip(bars, vals, bottoms):
            if v >= 0.08:
                ax_status.text(b.get_x() + b.get_width() / 2, bot + v / 2, f"{v * 100:.1f}%",
                               ha="center", va="center", fontsize=8.5, fontweight="bold", color="#111111")
        bottoms += vals

    ax_status.set_title("A. DeepSeek Flash v4 Distillation Yield & Rejections")
    ax_status.set_xticks(x)
    ax_status.set_xticklabels([s[1] for s in splits])
    ax_status.set_ylabel("Share of Generated Chunks")
    ax_status.set_ylim(0, 1.06)
    ax_status.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=True)

    # Panel B: Exported training labels — Mean Word Keep Rate (with p10-p90) & Deleted Punctuation Share
    x_b = np.arange(len(splits))
    w = 0.34
    keep_means = [labels_summary[s]["word_keep_rate"]["mean"] for s, _ in splits]
    keep_lo = [labels_summary[s]["word_keep_rate"]["mean"] - labels_summary[s]["word_keep_rate"]["p10"] for s, _ in splits]
    keep_hi = [labels_summary[s]["word_keep_rate"]["p90"] - labels_summary[s]["word_keep_rate"]["mean"] for s, _ in splits]
    punct_shares = [labels_summary[s]["dropped_punct_share"] for s, _ in splits]

    b1 = ax_bias.bar(x_b - w / 2, keep_means, width=w, yerr=[keep_lo, keep_hi], capsize=4,
                     color="#0072B2", alpha=0.88, label="Word Keep Rate (mean & p10–p90)")
    b2 = ax_bias.bar(x_b + w / 2, punct_shares, width=w,
                     color="#D55E00", alpha=0.88, label="Punctuation Share Among Deleted Words")

    for b, v, hi in zip(b1, keep_means, keep_hi):
        ax_bias.text(b.get_x() + b.get_width() / 2, v + hi + 0.02, f"{v:.2f}",
                     ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#004D80")
    for b, v in zip(b2, punct_shares):
        ax_bias.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v * 100:.1f}%",
                     ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#B00020")

    ax_bias.set_title("B. Supervision Bias: Keep Rate & Deleted Punctuation Share")
    ax_bias.set_xticks(x_b)
    ax_bias.set_xticklabels(["En Train\n(N=1,567)", "Zh Train\n(N=1,567)", "En Dev\n(N=338)", "Zh Dev\n(N=338)"])
    ax_bias.set_ylabel("Ratio / Share")
    ax_bias.set_ylim(0, 1.15)
    ax_bias.legend(loc="upper right", frameon=True, facecolor="white")

    fig.suptitle("Training Data Distillation Quality & Known Supervision Asymmetries (M1 vs. M4)", y=1.02)
    fig.tight_layout(w_pad=3.0)
    return save_fig(fig, out_dir, "fig5_distillation_label_bias")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate evaluation and diagnostic figures.")
    ap.add_argument("--eval-json", default="results/eval_report.json")
    ap.add_argument("--g0-json", default="results/compress/g0_report.json")
    ap.add_argument("--labels-json", default="data/labels/summary.json")
    ap.add_argument("--out-dir", default="results/figures")
    args = ap.parse_args(argv)

    configure_matplotlib()
    eval_rep = json.loads(Path(args.eval_json).read_text(encoding="utf-8")) if Path(args.eval_json).exists() else {}
    g0_rep = json.loads(Path(args.g0_json).read_text(encoding="utf-8")) if Path(args.g0_json).exists() else {}
    labels_rep = json.loads(Path(args.labels_json).read_text(encoding="utf-8")) if Path(args.labels_json).exists() else {}

    generated = []
    for fn, arg in (
        (plot_fig1_rate_accuracy, eval_rep),
        (plot_fig2_g0_diagnostics, g0_rep),
        (plot_fig3_hypothesis_forest, eval_rep),
        (plot_fig4_deleted_word_categories, eval_rep),
        (plot_fig5_distillation_bias, labels_rep),
    ):
        res = fn(arg, args.out_dir)
        if res:
            generated.append(res[0])
            print(f"[plot] wrote {res[0]} and {res[1]}")
    return generated


if __name__ == "__main__":
    main()
