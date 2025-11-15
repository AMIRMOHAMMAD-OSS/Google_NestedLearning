# NL-HOPE (PyTorch)

A PyTorch implementation of **Nested Learning (NL)** and the **HOPE** sequence model.

> **Ali Behrouz, Meisam Razaviyayn, Peiling Zhong, Vahab Mirrokni**  
> *Nested Learning: The Illusion of Deep Learning Architectures*, NeurIPS 2025.

---

## 🔗 Resources

- 🧠 **Paper:** *Nested Learning: The Illusion of Deep Learning Architectures* (NeurIPS 2025)  
- 💻 **This repo (full PyTorch implementation):**  
  https://github.com/AMIRMOHAMMAD-OSS/Google_NestedLearning  
- 📓 **Hands-on Colab tutorial (PyTorch, train on any HF dataset):**  

  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMIRMOHAMMAD-OSS/Google_NestedLearning/blob/main/Google_Nested_learning.ipynb)

The Colab notebook gives a **minimal, didactic example** of a HOPE-style model and training loop entirely in one file, while this repository contains a more **modular and extensible** implementation.

---

## 1. Conceptual Overview

### 1.1 What is Nested Learning?

Standard deep learning treats a model as:

> *one parameter vector + one global optimizer + one gradient flow.*

**Nested Learning (NL)** reframes this as:

> *a system of nested optimization problems*, each with:
> - its own *context flow* (what it sees over time),
> - its own *update frequency*,
> - its own *learning rule*,
> - and its own *memory*.

Under NL, familiar components become **associative memories**:

- A linear layer trained by SGD is a memory mapping inputs → “surprise signals” (gradients).
- Momentum buffers are memories of past gradients.
- Linear attention memories (like KV states) are online compressions of the sequence.

This lens helps explain:

- In-context learning as **on-the-fly memory compression** of context,
- Optimizers as **learned memory systems**,
- The limits of “just stack more layers” in terms of effective computation depth.

---

### 1.2 What is HOPE?

**HOPE** is a sequence model that operationalizes NL with two main components:

1. **Fast associative memory (working memory)**  
   - A linear-attention–style state update (`FastKV`):  
     \[
     M_{t+1} = M_t + \phi(k_t)^\top v_t
     \]  
   - Acts as an **online memory** over the current context.
   - Updates are local and do *not* require backpropagation through the update rule.

2. **Continuum Memory System (CMS)**  
   - A chain of MLP blocks, each living at a different **time scale** (update frequency).
   - For example:
     - Level 0: updates every token,
     - Level 1: updates every few tokens,
     - Level 2: updates every many tokens.
   - This recovers a **continuum of memories** from short-term to long-term.

Together, HOPE becomes a **self-modifying sequence model**:
- outer parameters trained in the usual way,
- inner components that *adapt during inference* via fast, local update rules.

---

### 1.3 How is this different from a standard Transformer?

Key differences:

1. **Multiple time scales, not a single static backbone**
   - Transformers: layers are fixed after pretraining; all updates happen off-line.
   - HOPE: separates **fast** (per-step) and **slow** (infrequent) components via CMS.

2. **Optimizers as memories**
   - NL views optimizers as **gradient memories** (e.g., momentum ≈ associative memory).
   - This repo includes variants where inner updates use **L2-regression–style rules** rather than pure dot-product Hebbian updates.

3. **Explicit in-context adaptation**
   - In a Transformer, in-context learning is emergent.
   - In HOPE, context is explicitly compressed into a fast memory + CMS levels, and parts of the model are **designed** to update on-the-fly.

4. **Better alignment with continual learning**
   - Fast levels adapt quickly but can forget,
   - Slow levels change rarely and store more stable structure,
   - This avoids relying on a single “short-term memory” bottleneck.

5. **Self-modification without meta-gradients**
   - Inner learning rules are **local** and do not require second-order gradients or unrolled meta-learning.
   - This keeps the architecture more scalable and easier to implement.

---

## 2. What’s in this Repository?

### 2.1 Core components

- **Fast associative memory (FastKV)**
  - Linear-attention–style fast weight update.
  - Implements key–value accumulation and retrieval with a feature map φ(·).
  - File: `nl/modules/fast_kv.py`

- **Continuum Memory System (CMS)**
  - Chain of MLP “memory levels” with configurable update frequencies.
  - Each level has its own optimizer / update schedule.
  - File: `nl/memory/cms.py`

- **HOPE language model**
  - Combines token embeddings, FastKV, CMS, and an inner learning rule.
  - Includes the L2-regression–style gradient step derived in the paper.
  - File: `nl/models/hope.py`

### 2.2 Training / evaluation scripts

- **Training**  
  `scripts/train.py`  
  - Loads config files (`configs/*.yaml`)  
  - Prepares data via Hugging Face Datasets  
  - Trains HOPE on next-token prediction

- **Perplexity evaluation**  
  `scripts/eval_ppl.py`  
  - Loads a trained checkpoint  
  - Computes validation/test perplexity

- **Sampling**  
  `scripts/sample.py`  
  - Loads a trained checkpoint  
  - Generates text with temperature / top-k / top-p

### 2.3 Configuration

- Example model config (HOPE, Colab-friendly size):  
  `configs/hope_colab.yaml`

- Example data config (WikiText-2):  
  `configs/data_wikitext2.yaml`

You can add your own `.yaml` files to configure:
- model size,
- CMS depth / update frequencies,
- dataset paths or HF dataset names,
- optimization hyperparameters.

---

## 3. Google Colab Tutorial Notebook

If you prefer a **single-file, PyTorch-only walkthrough**, use the Colab notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMIRMOHAMMAD-OSS/Google_NestedLearning/blob/main/Google_Nested_learning.ipynb)

The notebook:

- Re-implements a HOPE-style model directly in the notebook (no imports from this repo).
- Shows how to:
  - load a Hugging Face text dataset,
  - build a simple LM dataset (blocks of tokens),
  - define a FastKV + CMS model,
  - train with cross-entropy LM loss,
  - generate text samples.
- Is designed as a **readable reference** for people who want to understand the approach before diving into the full codebase.

If you just want to **try the idea on your own text or HF dataset**, start from this Colab.

---

## 4. Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/AMIRMOHAMMAD-OSS/Google_NestedLearning
cd Google_NestedLearning

pip install -r requirements.txt
```

Make sure you have a working **Python 3.10+** and **PyTorch with CUDA** if you want GPU training.

---

## 5. Quick Start (WikiText-2)

### 5.1 Train on WikiText-2

```bash
python scripts/train.py   --config configs/hope_colab.yaml   --data configs/data_wikitext2.yaml
```

This will:

- Download WikiText-2 via Hugging Face,
- Tokenize and pack it into fixed-length sequences,
- Train a HOPE model with a moderate hidden size on a single GPU,
- Save checkpoints under `outputs/<exp_name>/`.

### 5.2 Evaluate perplexity

```bash
python scripts/eval_ppl.py   --checkpoint outputs/<exp_name>/last.ckpt   --config_model configs/hope_colab.yaml   --data configs/data_wikitext2.yaml
```

This reports token-level negative log-likelihood and perplexity on the validation or test split.

### 5.3 Sample from the model

```bash
python scripts/sample.py   --checkpoint outputs/<exp_name>/last.ckpt   --prompt "Nested learning suggests"   --max_new_tokens 150   --temperature 0.8   --top_k 50   --top_p 0.95
```

Adjust `prompt`, `temperature`, `top_k`, and `top_p` to explore different generation behaviours.

---

## 6. Training on Your Own Data

You have two main options:

### 6.1 Using a Hugging Face dataset (recommended)

1. Pick a dataset from https://huggingface.co/datasets  
   Examples: `imdb`, `ag_news`, `bookcorpusopen`, etc.

2. Create a data config (e.g. `configs/data_mytext.yaml`) specifying:
   - HF dataset name,
   - optional configuration,
   - which field contains the raw text.

3. Point `scripts/train.py` to your custom data config:

   ```bash
   python scripts/train.py      --config configs/hope_colab.yaml      --data configs/data_mytext.yaml
   ```

You can mirror what is done in `configs/data_wikitext2.yaml` and adapt it for your dataset.

### 6.2 Using raw/local text

If you have plain `.txt` files:

- Either:
  - Wrap them as a Hugging Face `text` dataset in your data config, or
  - Use the Colab notebook and replace the dataset-loading cell with:
    ```python
    from datasets import load_dataset
    ds = load_dataset("text", data_files={"train": "your_file.txt"})
    ```
- Once your text is exposed as a stream of strings, the rest of the pipeline (tokenization, block packing, training, sampling) stays unchanged.

---

## 7. Status & Caveats

- This repository is a **research-oriented** implementation, not a production library.
- The NeurIPS camera-ready paper is a **compressed** version; some details and ablations live in the (forthcoming) arXiv version.
- Where the paper is ambiguous, the implementation follows:
  - standard fast-weight / linear-attention practice for the memory update,
  - a clean separation between inner updates and outer gradients (no backprop through inner loops).

If you notice a discrepancy between the paper and the code, feel free to open an issue or PR.

---

## 8. Citing

### 8.1 Cite the Nested Learning / HOPE paper

```bibtex
@inproceedings{behrouz2025nestedlearning,
  author    = {Ali Behrouz and Meisam Razaviyayn and Peiling Zhong and Vahab Mirrokni},
  title     = {Nested Learning: The Illusion of Deep Learning Architectures},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2025}
}
```

### 8.2 Cite this implementation / repository

If you use this codebase or the Colab notebook in your work, please cite:

```bibtex
@software{amirmohammad_nl_hope_2025,
  author  = {AmirMohammad},
  title   = {{NL-HOPE}: Nested Learning and {HOPE} Sequence Model in PyTorch},
  year    = {2025},
  url     = {https://github.com/AMIRMOHAMMAD-OSS/Google_NestedLearning},
  note    = {GitHub repository},
  version = {latest}
}
```
