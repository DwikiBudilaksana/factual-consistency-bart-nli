# Factual Consistency Enhancement in Indonesian Abstractive Summarization via NLI-Guided Candidate Reranking with BART

Code and experiment notebooks accompanying the paper *"Factual Consistency Enhancement in Indonesian Abstractive Summarization via NLI-Guided Candidate Reranking with BART"* (submitted to CompGineer 2026).

## Overview

This repository implements a pipeline that improves factual consistency in Indonesian abstractive summarization by:

1. Fine-tuning **BART** (`facebook/bart-base`) on the **Liputan6** dataset for Indonesian abstractive summarization.
2. Generating multiple candidate summaries per document using beam search.
3. Scoring each candidate with a **Natural Language Inference (NLI)** model (`MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`) to estimate how well it is logically entailed by the source document.
4. Re-ranking candidates using a combined score `α · entailment + (1 − α) · g(generation_score)`, selecting the most factually consistent summary.

## Repository Structure

```
.
├── notebooks/
│   ├── 01_training.ipynb              # Fine-tune BART, generate baseline & NLI-reranked summaries, compute ROUGE/entailment
│   ├── 02_nli_validation.ipynb        # Validate the NLI model on IndoNLI (test_lay / test_expert), per-class precision/recall/F1
│   ├── 03_ablation_study_alpha.ipynb  # Selects α (0.0, 0.3, 0.5, 0.7, 1.0) on the VALIDATION split (bug fix, see notebook's markdown note); also usable as a test-set sensitivity check
│   ├── 04_comparison_methods.ipynb    # Compares Baseline vs. NLI Reranking vs. Semantic Similarity Reranking
│   └── exploratory/
│       └── 00_initial_pipeline_draft.ipynb   # Early exploratory draft (different hyperparameters; kept for transparency, NOT the final configuration)
├── results/
│   ├── main_results.json              # Main baseline vs. NLI performance numbers + IndoNLI validation results (re-run after a max_source_length bug fix, confirmed 2026-08-25)
│   ├── ablation_alpha_results.json    # α sweep on the TEST split — sensitivity check only, not the selection procedure (see P0 #2)
│   ├── alpha_selection_validation_results.json  # α sweep on the VALIDATION split — the actual selection procedure (produced once 03_ablation_study_alpha.ipynb is re-run)
│   └── comparison_methods_results.json# Method comparison results
├── data/            # (not included — see "Data" section below)
├── checkpoints/     # (not included — fine-tuned model checkpoint goes here)
└── requirements.txt
```

> **Note on `notebooks/exploratory/00_initial_pipeline_draft.ipynb`**: this was an early exploratory notebook used while iterating on hyperparameters (e.g., a different max input length and batch size). It is **not** the configuration used to produce the results reported in the paper — refer to `01_training.ipynb` for the final, reported configuration.

## Setup

```bash
pip install -r requirements.txt
```

**Environment:** the notebooks were originally developed and run on **Kaggle** (Python 3.12, NVIDIA Tesla T4 GPU, 15.6 GB VRAM). `requirements.txt` pins lower bounds (`>=`) rather than exact versions; the exact resolved versions used for the reported results were the latest available on Kaggle's Python 3.12 GPU image at the time each notebook was run (see per-notebook `pip install` output cells for the resolved package list at run time). File paths in the notebooks have been converted to relative paths (`./data/...`, `./checkpoints/...`); adjust them if your directory layout differs.

## Data

This project uses the publicly available **Liputan6** dataset:

- Koto, F., Lau, J. H., & Baldwin, T. (2020). *Liputan6: A Large-scale Indonesian Dataset for Text Summarization*. Proc. AACL-IJCNLP.
- Dataset: https://github.com/fajri91/sum_liputan6

For NLI model validation, this project uses **IndoNLI**:

- Mahendra, R., Putra, R. A., & Adriani, M. (2021). *IndoNLI: A Natural Language Inference Dataset for Indonesian*. Proc. EMNLP.

Place the downloaded/preprocessed files under `data/` following the paths referenced at the top of each notebook (`TEXT_COLUMN`, `SUMMARY_COLUMN`, `TEST_FILE`, `VALID_FILE`, etc.), or update those constants to match your own layout. `VALID_FILE` (`./data/valid.jsonl`) is required for `03_ablation_study_alpha.ipynb` since its P0 #2 fix — export it from `01_training.ipynb`'s processed validation split the same way you already export `test.jsonl`.

## Pipeline / How to Reproduce

1. **Select α on the validation split first** — run `notebooks/03_ablation_study_alpha.ipynb` to reproduce the sweep over α ∈ {0.0, 0.3, 0.5, 0.7, 1.0} on the **validation** split and confirm the best-performing α. Candidates are generated once and reused across all α values (reranking only, no regeneration). Requires a fine-tuned BART checkpoint from step 2 below to already exist, so in practice run step 2 first, then come back to this step.
2. **Train BART and generate predictions** — run `notebooks/01_training.ipynb`. This fine-tunes BART on Liputan6 (3 epochs, batch size 12, max source length 256, max target length 128, learning rate 2e-5), then generates both baseline (top-1 beam) and NLI-reranked (α = 0.7 by default — update the `ALPHA` config constant first if step 1 selected a different value, 4 candidates) summaries on the test split, producing the main performance numbers (ROUGE-1/2/L, average entailment/contradiction/neutral). Also includes the P0 #1 fix (`max_source_length=256` for generation, previously silently defaulted to 768).
3. **Validate the NLI model** — run `notebooks/02_nli_validation.ipynb` to reproduce the per-class precision/recall/F1 on IndoNLI's `test_lay` and `test_expert` splits.
4. **Compare against an alternative re-ranking strategy** — run `notebooks/04_comparison_methods.ipynb` to reproduce the comparison between Baseline, NLI Reranking, and Semantic Similarity Reranking (cosine similarity via `paraphrase-multilingual-MiniLM-L12-v2`).

> **Note:** steps 1 and 2 have a circular dependency (α selection needs a trained checkpoint; the final α used in step 2 should come from step 1). In practice: run step 2 once with the default α = 0.7 to get a checkpoint, run step 1 to confirm/select α, then re-run step 2's generation cells if the selected α changed.

### Which notebook reproduces which table in the paper

> **Note (2026-08-25):** the manuscript was condensed from an earlier 12-page/11-table draft to 5 pages / 8 tables to meet the CompGineer 2026 IEEE Conference 6-page limit. Two tables from the earlier draft (Training Convergence; Computational Cost) were folded into one-sentence inline mentions, and the test-split α sweep (previously its own table, kept only as a sensitivity check) is now a single sentence rather than a table — the validation-split α selection table (the actual, reviewer-requested selection basis) is retained in full. The qualitative-examples table was reduced from 3 to 2 examples. All notebooks and result files below are unchanged and still reproduce the full, uncondensed set of numbers described in this README, even where the paper itself now only summarizes them in text.

| Paper table | Produced by | Output file |
|---|---|---|
| Table I (dataset statistics) | `01_training.ipynb` (dataset conversion/preprocessing cells) | printed in-notebook |
| Table II (IndoNLI per-class precision/recall/F1) | `02_nli_validation.ipynb` | printed in-notebook |
| Table III (reproducibility summary) | consolidated from Sections III/IV of the paper and this README | n/a (summary table, not a raw output file) |
| Table IV (baseline vs. NLI performance) | `01_training.ipynb` | `results/main_results.json` — re-run with the P0 #1 fix, confirmed 2026-08-25; numbers align with the validation/comparison α=0.7 rows (within ~1-sample noise) |
| Table V (paired t-test, Cohen's d, bootstrap CI) | `05_significance_test.ipynb` | `results/significance_test_results.json` — run 2026-08-25 on GPU; all four metrics significant at p<0.05, entailment gain has medium effect size (d=-0.549), ROUGE decreases have negligible effect size (\|d\|=0.024-0.050) |
| Table VI (α selection, validation split — the selection basis, P0 #2 fix) | `03_ablation_study_alpha.ipynb` (validation-split sweep) | `results/alpha_selection_validation_results.json` — confirms α = 0.7 sits in a saturated entailment/contradiction plateau for α ≥ 0.3, so α = 0.7 is retained |
| Table VII (qualitative examples, 2 of the original 3) | `01_training.ipynb` (prediction inspection cell) | printed in-notebook — re-extracted from the corrected `predictions_baseline.jsonl` / `predictions_nli.jsonl` for IDs 18333, 24661 |
| Table VIII (method comparison) | `04_comparison_methods.ipynb` | `results/comparison_methods_results.json` |
| *(not a numbered table in the condensed paper, but still reproducible)* α ablation, test split (sensitivity check only) | `03_ablation_study_alpha.ipynb` (test-set sweep) | `results/ablation_alpha_results.json` |
| *(not a numbered table in the condensed paper)* Training convergence | `01_training.ipynb` (training cell logs) | printed in-notebook |
| *(not a numbered table in the condensed paper)* Computational cost | timed manually from Kaggle session logs across notebooks | reported in paper text only |
| *(cut from the condensed paper's 3-example set)* Qualitative Example — electricity tariff protest (ID 14557) | `01_training.ipynb` (prediction inspection cell) | printed in-notebook |

## Key Configuration

| Parameter | Value |
|---|---|
| Summarization model | `facebook/bart-base` |
| NLI model | `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` |
| Max source length | 256 tokens |
| Max target length | 128 tokens |
| Train batch size | 12 |
| Learning rate | 2e-5 |
| Gradient accumulation | 2 steps |
| Epochs | 3 |
| Num candidates (beam search) | 4 |
| Reranking weight α | 0.7 (confirmed via validation-split selection, `03_ablation_study_alpha.ipynb` — see `results/alpha_selection_validation_results.json`, run 2026-08-25) |
| Generation score normalization | `g(s) = tanh(generation_score / 10)` |

## Results Summary

> **Updated 2026-08-25** after fixing a `max_source_length` bug (should be 256 during generation, was previously silently defaulting to 768). The numbers below are confirmed from a re-run of `01_training.ipynb` and now align closely with the independently-computed α=0.7 rows in Tables VI/VIII (which always used 256 tokens) — differences are within the ~1-sample gap between the deduplicated test set (10,971) and the raw test set (10,972). **α = 0.7 is now confirmed** via the validation-split selection procedure (`03_ablation_study_alpha.ipynb`) — see `results/alpha_selection_validation_results.json`.

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | Avg. Entailment | Avg. Contradiction |
|---|---|---|---|---|---|
| BART (baseline) | 0.3856 | 0.2122 | 0.3174 | 0.3100 | 0.0138 |
| BART + NLI Reranking | 0.3834 | 0.2085 | 0.3156 | **0.4664** | **0.0112** |

See `results/` for the full set of numbers, including the ablation study and method comparison.

## License

The code, notebooks, and configuration files in this repository are released under the [MIT License](LICENSE). This does **not** cover the Liputan6 or IndoNLI datasets or the pretrained model weights (BART, mDeBERTa-v3-base-mnli-xnli, paraphrase-multilingual-MiniLM-L12-v2), which remain under their own respective licenses — see the "Data" section above and each model's Hugging Face model card for their terms of use.

## Citation

If you use this code, please cite the paper (details to be updated upon publication) and the datasets/models above.

## Acknowledgment

Parts of this repository's documentation (this README) were reformatted with the assistance of an AI tool (Claude, Anthropic). All experiments, code logic, and results are the authors' own work.
