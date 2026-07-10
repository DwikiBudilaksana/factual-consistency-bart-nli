# Factual Consistency Enhancement in Indonesian Abstractive Summarization Using BART with Natural Language Inference

Code and experiment notebooks accompanying the paper *"Factual Consistency Enhancement in Indonesian Abstractive Summarization Using BART with Natural Language Inference"* (submitted to CompGineer 2026).

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
│   ├── 03_ablation_study_alpha.ipynb  # Ablation over the re-ranking weight α (0.0, 0.3, 0.5, 0.7, 1.0)
│   ├── 04_comparison_methods.ipynb    # Compares Baseline vs. NLI Reranking vs. Semantic Similarity Reranking
│   └── exploratory/
│       └── 00_initial_pipeline_draft.ipynb   # Early exploratory draft (different hyperparameters; kept for transparency, NOT the final configuration)
├── results/
│   ├── main_results.json              # Main baseline vs. NLI performance numbers + IndoNLI validation results
│   ├── ablation_alpha_results.json    # Ablation study results
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

The notebooks were originally developed and run on **Kaggle** (NVIDIA Tesla T4 GPU). File paths in the notebooks have been converted to relative paths (`./data/...`, `./checkpoints/...`); adjust them if your directory layout differs.

## Data

This project uses the publicly available **Liputan6** dataset:

- Koto, F., Lau, J. H., & Baldwin, T. (2020). *Liputan6: A Large-scale Indonesian Dataset for Text Summarization*. Proc. AACL-IJCNLP.
- Dataset: https://github.com/fajri91/sum_liputan6

For NLI model validation, this project uses **IndoNLI**:

- Mahendra, R., Putra, R. A., & Adriani, M. (2021). *IndoNLI: A Natural Language Inference Dataset for Indonesian*. Proc. EMNLP.

Place the downloaded/preprocessed files under `data/` following the paths referenced at the top of each notebook (`TEXT_COLUMN`, `SUMMARY_COLUMN`, `TEST_FILE`, etc.), or update those constants to match your own layout.

## Pipeline / How to Reproduce

1. **Train BART and generate predictions** — run `notebooks/01_training.ipynb`. This fine-tunes BART on Liputan6 (3 epochs, batch size 12, max source length 256, max target length 128, learning rate 2e-5), then generates both baseline (top-1 beam) and NLI-reranked (α = 0.7, 4 candidates) summaries on the test split, producing the main performance numbers (ROUGE-1/2/L, average entailment/contradiction/neutral).
2. **Validate the NLI model** — run `notebooks/02_nli_validation.ipynb` to reproduce the per-class precision/recall/F1 on IndoNLI's `test_lay` and `test_expert` splits.
3. **Run the α ablation study** — run `notebooks/03_ablation_study_alpha.ipynb` to reproduce the sweep over α ∈ {0.0, 0.3, 0.5, 0.7, 1.0}. Candidates are generated once and reused across all α values (reranking only, no regeneration).
4. **Compare against an alternative re-ranking strategy** — run `notebooks/04_comparison_methods.ipynb` to reproduce the comparison between Baseline, NLI Reranking, and Semantic Similarity Reranking (cosine similarity via `paraphrase-multilingual-MiniLM-L12-v2`).

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
| Re-ranking weight α | 0.7 |
| Generation score normalization | `g(s) = tanh(generation_score / 10)` |

## Results Summary

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | Avg. Entailment | Avg. Contradiction |
|---|---|---|---|---|---|
| BART (baseline) | 0.3819 | 0.2090 | 0.3145 | 0.3229 | 0.0133 |
| BART + NLI Reranking | 0.3789 | 0.2045 | 0.3113 | **0.4940** | **0.0108** |

See `results/` for the full set of numbers, including the ablation study and method comparison.

## Citation

If you use this code, please cite the paper (details to be updated upon publication) and the datasets/models above.

## Acknowledgment

Parts of this repository's documentation (this README) were reformatted with the assistance of an AI tool (Claude, Anthropic). All experiments, code logic, and results are the authors' own work.
