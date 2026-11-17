# Cross-lingual extractive prompt compression

Tests whether Chinese supervision improves LLMLingua-2 beyond matched English supervision.
Swahili full labelling and M2 are stretch goals; the teacher pilot is reported as a feasibility result.
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

Setup: `pip install -r requirements.lock` (use a CUDA build of torch on a GPU machine). Tests: `pytest`.

Data: `data/pilot/` contains text from Wikipedia (`wikimedia/wikipedia`, revision recorded in
`data/pilot/manifest.json`), licensed CC BY-SA 4.0; `results/teacher/` contains the same text plus
model outputs. `data/full/` is the full en/zh pool, `data/labels/` the training labels (handoff notes in
[data/labels/README.md](data/labels/README.md)). Status: labels done (1,567 train + 338 dev per language,
DeepSeek Flash teacher); training M1/M4 and Belebele evaluation not started.

Contamination scan evidence contains Belebele passages and Wikipedia excerpts; see
[sources, attribution and licensing](results/contamination/README.md).
