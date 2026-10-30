# Cross-lingual extractive prompt compression

Tests whether target-language supervision (zh, sw) narrows the cross-lingual gap of LLMLingua-2.
Decisions, protocol and all pilot results so far: [DECISIONS.md](DECISIONS.md).

| File | What it does |
|---|---|
| `sample_wiki.py` | Pilot sample of Wikipedia chunks (en/zh/sw) with a document-level train/dev split |
| `segment.py` | One word segmentation for labels, training and inference (XLM-R + jieba for zh) |
| `align.py` | Word keep/drop labels from a deletion-only teacher output, with legality checks |
| `constrained.py` | Constrained decoding: the teacher may only emit whole original words in order |
| `run_teacher.py` | Runs a local teacher on pilot chunks and labels the outputs (resumable) |
| `report_teacher.py` | Side-by-side comparison of teacher result files |

Setup: `pip install -r requirements.lock` (use a CUDA build of torch on a GPU machine). Tests: `pytest`.

Data: `data/pilot/` contains text from Wikipedia (`wikimedia/wikipedia`, revision recorded in
`data/pilot/manifest.json`), licensed CC BY-SA 4.0; `results/teacher/` contains the same text plus
model outputs. Status: teacher selection in progress, no full labelling or training yet.
