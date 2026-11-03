# Contamination scan evidence

`wiki_scan.json` records a seeded scan of whole Wikipedia documents. It contains
selected source excerpts for manual review, not training labels.

The included source text and this evidence compilation are distributed under
[Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/).
This notice applies to the evidence here; it does not assign a license to the project's code.

- Belebele: Meta / the Belebele authors, Lucas Bandarkar et al.,
  *The Belebele Benchmark: a Parallel Reading Comprehension Dataset in 122 Language Variants* (2024).
  Source: [facebook/belebele dataset card at the recorded revision](https://huggingface.co/datasets/facebook/belebele/blob/7899cdfa4e1e0d733fd77c848e2c273cb1d32be2/README.md).
  The `passage` values are selected, unmodified `flores_passage` values from its test split.
  The [official license notice](https://github.com/facebookresearch/belebele#license)
  identifies CC BY-SA 4.0 for Belebele; the separately assembled training set is not used here.
- Wikipedia: the contributors to
  [英国大学列表](https://zh.wikipedia.org/wiki/英国大学列表) and
  [Gertrude Mongella](https://sw.wikipedia.org/wiki/Gertrude_Mongella).
  Source dataset: [wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia),
  revision `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`, configurations `20231101.zh` and `20231101.sw`.
  Article URLs, titles and IDs are retained in the report.
  `doc_context` values are excerpts of the extracted article text; Chinese text is
  converted with OpenCC `t2s` before selection. `first_ngram` values are normalized
  as documented in DECISIONS B10. These are changes made by this project.

The report adds match counts and links the selected source texts for comparison.
No endorsement by the original authors is implied. Source licenses and attribution
must accompany redistribution of this evidence.
