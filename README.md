# On the Predictive Power of Representation Dispersion in Language Models

<!-- 
[**Paper PDF**](./nips2025_Representation_Dispersion_in_LMs.pdf) | [Project Website (if available)]() | [ArXiv (if available)]() 
-->

---

## Overview

**Representation dispersion**—the average pairwise distance among hidden vectors in a language model—turns out to be a simple yet powerful indicator of model quality. In our paper, we empirically demonstrate that:

* **Higher dispersion** in the embedding space of a language model is **strongly correlated with lower perplexity** and improved predictive accuracy, across diverse model families (LLaMA, Qwen, etc.) and domains (Wikipedia, news, scientific abstracts).
* Dispersion can be leveraged for **label-free model diagnostics, model selection, layer selection for retrieval augmentation**, and even as an **auxiliary training objective** to improve perplexity and generalization.

<!-- ![Main Figure: Illustration of the relationship between embedding dispersion and perplexity, and how this metric can be used for model selection, evaluation, and training.](link_to_fig_if_possible) -->

---

## Table of Contents
<!-- * [Citing](#citing) -->
* [Installation](#installation)
* [Usage: Main Analysis Pipeline](#usage-main-analysis-pipeline)

  * [Step 1: Calculate Perplexity (PPL)](#step-1-calculate-perplexity-ppl)
  * [Step 2: Group and Uniformly Sample Chunks](#step-2-group-and-uniformly-sample-chunks)
  * [Step 3: Compute Dispersion and Plot](#step-3-compute-dispersion-and-plot)
* [Applications](#applications)

---

## Installation

```bash
git clone https://github.com/yanhong-lbh/rep_dispersion.git
cd rep_dispersion
pip install -r requirements.txt
```

* Requires **Python 3.8+**
* Most experiments use [PyTorch](https://pytorch.org/), [Transformers](https://github.com/huggingface/transformers), and [datasets](https://github.com/huggingface/datasets).

---

## Usage: Main Analysis Pipeline

Here’s how to reproduce the key experiments and plots from the paper.

### Step 1: Calculate Perplexity (PPL)

Calculate token-level perplexity (PPL) for all data chunks for your chosen model(s) and dataset(s).
This is the slowest step and is parallelizable.

```bash
# Example: Calculate PPL for 100,000 chunks of Wikitext using LLaMA-3.2-1b
bash analysis/step1_calculate_ppl.sh
```

This script ultimately runs:

```bash
python analysis/preprocess.py \
    --dataset_name wikitext \
    --model_name llama-3.2-1b \
    --chunk_len 512 \
    --n_chunks 100000 \
    --random_sample \
    --random_seed 42 \
    --ppl_only
```

**Outputs:**
A directory like `data/wikitext_llama-3.2-1b_512_100000_random_42/` containing thousands of `ppl_*.json` files.

---

### Step 2: Group and Uniformly Sample Chunks

Group all chunks by PPL and perform uniform sampling in PPL-space to get a representative set for downstream analysis.

```bash
python analysis/step2_group_and_sample.py
```

**Outputs:**

* `all_models_grouped_ppl_512_wikitext.json` — all chunks grouped by PPL
* `all_models_grouped_ppl_uniform_512_wikitext.json` — uniform samples for each model

---

### Step 3: Compute Dispersion and Plot

Compute the representation dispersion for all sampled chunks and plot dispersion vs. perplexity.

```bash
python analysis/step3_compute_dispersion_and_plot.py
```

* This will:

  1. Extract hidden states for the sampled indices
  2. Compute pairwise dispersion (cosine distance) within each group
  3. Plot dispersion vs. mean PPL and save figures to `plots_postprocessed/`

---

## Applications

COMING SOON!! (should be done by mid-July)

* **Label-Free Diagnostics:** Assess LLM quality on new data without requiring ground-truth labels.
* **Model Selection:** Use dispersion to pick better checkpoints or models for deployment.
* **Layer Selection for Retrieval:** Select the most informative layers for retrieval-augmented generation.
* **Auxiliary Objective:** Incorporate dispersion as an explicit training regularizer to improve generalization and reduce overfitting.
