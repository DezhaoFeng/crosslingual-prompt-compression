# Cross-lingual extractive prompt compression

Tests whether Chinese supervision improves LLMLingua-2 beyond matched English supervision.
Swahili full labelling and M2 are stretch goals; the teacher pilot is reported as a feasibility result.
Full results, hypothesis tests (H1/H2), error analysis and figures: **[RESULTS.md](RESULTS.md) (English)** | **[RESULTS_zh.md](RESULTS_zh.md) (中文版)**.
Decisions and pilot results: [DECISIONS.md](DECISIONS.md). Execution plan and gates: [plan.md](plan.md).

| File | What it does |
|---|---|
| `sample_wiki.py` | Wikipedia chunks (en/zh/sw) with a document-level train/dev split; zh conversion markup stripped |
| `segment.py` | One word segmentation for labels, training and inference (XLM-R + jieba for zh) |
| `align.py` | Word keep/drop labels from a deletion-only teacher output, with legality checks |
| `constrained.py` | Constrained decoding: the teacher may only emit whole original words in order |
| `contamination.py` | Removes chunks that contain Belebele passage text (gate G3) |
| `scan_contamination.py` | Filter hit rate on a seeded sample of whole Wikipedia documents, with evidence for manual review |
| `run_teacher.py` | Runs a teacher (local or DeepSeek API) on chunks and labels the outputs (resumable) |
| `report_teacher.py` | Side-by-side comparison of teacher result files |
| `make_review.py` | Audit sheet: original text, then the label-reconstructed text |
| `distill_full.py` | Full distillation in fixed batches until the usable-label target is reached |
| `export_labels.py` | Training labels for M1/M4 with matched sample counts, plus `summary.json` |
| `train_student.py` | Resumable token-classifier fine-tuning for M1 (zh) and M4 (en) from M0 (`ignore_index=-100` for special/pad tokens) |
| `compress.py` | Belebele passage compression (`full`, `none`, `random`, `m0`, `m1`, `m4`, `m0_orig`, `teacher`) and Gate G0/G7 diagnostics |
| `run_reader.py` | Belebele MCQ reader (`Qwen/Qwen2.5-7B-Instruct` option logits or `deepseek-flash` API) and Gate G1 eligibility check |
| `evaluate.py` | Frozen Gate G6 evaluation harness: accuracy, normalized utility $U$, passage-clustered paired bootstrap 95% CIs (H1/H2/gap), and deleted-word error analysis |
| `plot_results.py` | Publication-grade figures (`results/figures/fig{1..5}_*.{png,pdf}`) for rate-utility curves, G0 diagnostics, H1/H2 forest plot, error analysis, and distillation bias |

Setup: `pip install -r requirements.lock` (use a CUDA build of torch on a GPU machine). Tests: `pytest`.

Data: `data/pilot/` contains text from Wikipedia (`wikimedia/wikipedia`, revision recorded in
`data/pilot/manifest.json`), licensed CC BY-SA 4.0; `results/teacher/` contains the same text plus
model outputs. `data/full/` is the full en/zh pool, `data/labels/` the training labels (handoff notes in
[data/labels/README.md](data/labels/README.md)). Status: **Core experiments complete** — labels distilled
(1,567 train + 338 dev per language, DeepSeek Flash v4), M1 (zh) and M4 (en control) trained, and full
Belebele en/zh evaluation completed under frozen harness `g6-frozen-v1.0` ([results/eval_report.md](results/eval_report.md),
figures in `results/figures/`). Both **H1** ($\Delta = +0.0556\ [+0.0241, +0.0867]$ at primary $\rho=0.33$) and
**H2** ($\Delta = +0.0422\ [+0.0089, +0.0751]$ at $\rho=0.33$) are statistically supported.

Contamination scan evidence contains Belebele passages and Wikipedia excerpts; see
[sources, attribution and licensing](results/contamination/README.md).
