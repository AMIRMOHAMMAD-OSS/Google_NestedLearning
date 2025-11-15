# NL-HOPE (PyTorch)

This repository provides a reference PyTorch implementation of **Nested Learning (NL)** and the **HOPE** sequence model introduced in:

**Behrouz, Razaviyayn, Zhong, Mirrokni**  
*Nested Learning: The Illusion of Deep Learning Architectures*, NeurIPS 2025.

Nested Learning reinterprets deep learning systems as *stacks of optimization processes*, each operating at its own update frequency rather than a single monolithic backpropagation flow.  
The HOPE architecture leverages this perspective to build a **self-modifying**, sequence-aware model equipped with a **continuum of memory timescales**.

---

## Overview

### What is Nested Learning?

Traditional deep learning treats the model as one set of parameters updated by one global optimizer.  
Nested Learning (NL) reframes the model as **multiple nested optimization loops**, each with:

- its own “context flow”,
- its own update frequency,
- its own internal learning rule,
- its own memory structure.

Backpropagation becomes just one of several learning processes rather than the only one.

This perspective helps explain phenomena such as:

- in-context learning,
- associative memory formation,
- optimizer behavior (e.g., momentum as a memory system),
- emergent long-range reasoning.

---

### What is HOPE?

HOPE is a sequence model that combines two main ideas:

1. **Fast associative memory**  
   A linear-attention–style module that accumulates key/value statistics using a local learning rule (no backprop through the update rule), acting like a fast memory over the current context.

2. **Continuum Memory System (CMS)**  
   A stack of MLP blocks, each updated at its own frequency (e.g., every 1, 4, 16, 64 tokens).  
   This yields a hierarchy of memories ranging from “fast and reactive” to “slow and stable”.

Together, these form a **self-modifying model**:  
the model updates parts of itself *during inference*, not only during pretraining.

---

## What Is Different About NL and HOPE?

### 1. Multiple timescales instead of a single static backbone

Transformers and many SSMs treat all layers as if they live at the same time scale. Once pretraining is finished, parameters are effectively static.

HOPE explicitly introduces:

- fast components that change every step or batch,
- slower components that only update occasionally,
- and a clear separation of update frequencies via CMS.

### 2. Optimizers as memory modules

In NL, optimizers such as momentum or Adam are interpreted as **associative memories of gradients**.  
They compress the history of gradients into a smaller state and update it with local rules.

HOPE makes this perspective concrete by:

- using an inner learning rule (L2-regression style) to update projections,
- keeping outer parameters trained with standard optimizers,
- avoiding expensive backprop through inner update rules.

### 3. Explicit in-context learning mechanisms

Rather than relying purely on scaling to induce emergent in-context behavior, HOPE:

- uses fast associative memory to track current context,
- uses local objectives for inner updates,
- allows parts of the model to adapt on-the-fly.

### 4. Better alignment with continual learning

Because different CMS levels update at different frequencies, HOPE:

- can adapt quickly at fast levels,
- keeps stable structure at slower levels,
- avoids relying on a single short-term memory bottleneck.

### 5. Self-modification without meta-gradient tricks

Fast updates in HOPE:

- do not require second-order gradients,
- do not require unrolled meta-learning,
- are implemented as straightforward local update rules.

This makes the architecture easier to scale and reason about.

---

## What’s Implemented Here

This repository includes:

- **Fast associative memory**
  - Linear-attention–style matrix memory with Hebbian / regression-style updates.  
  - File: `nl/modules/fast_kv.py`

- **Continuum Memory System (CMS)**
  - Multi-level MLP chain with configurable update frequencies.  
  - File: `nl/memory/cms.py`

- **HOPE language model**
  - Combines fast memory + CMS + an inner learning rule (L2-regression step) for Q/K/V and state updates.  
  - File: `nl/models/hope.py`

- **End-to-end scripts**
  - Training: `scripts/train.py`
  - Perplexity evaluation: `scripts/eval_ppl.py`
  - Sampling: `scripts/sample.py`

- **Config files**
  - Example HOPE config suitable for Colab or small-scale experiments: `configs/hope_colab.yaml`
  - WikiText-2 data config: `configs/data_wikitext2.yaml`

---

## Installation

```bash
git clone https://github.com/AMIRMOHAMMAD-OSS/Google_NestedLearning
cd Google_NestedLearning
pip install -r requirements.txt


#Quick Start
#Train on WikiText-2

python scripts/train.py \
  --config configs/hope_colab.yaml \
  --data configs/data_wikitext2.yaml


#Evaluate Perplexity

python scripts/eval_ppl.py \
  --checkpoint outputs/<exp_name>/last.ckpt \
  --config_model configs/hope_colab.yaml \
  --data configs/data_wikitext2.yaml


#Sample Text

python scripts/sample.py \
  --checkpoint outputs/<exp_name>/last.ckpt \
  --prompt "Nested learning suggests" \
  --max_new_tokens 150 \
  --temperature 0.8 \
  --top_k 50 \
  --top_p 0.95
```


If you use this implementation / codebase, please also cite the repository:

@software{amirmohammad_nl_hope,
  author       = {AmirMohammad},
  title        = {NL-HOPE: Nested Learning and HOPE Sequence Model in PyTorch},
  year         = {2025},
  url          = {https://github.com/AMIRMOHAMMAD-OSS/Google_NestedLearning},
  note         = {GitHub repository},
  version      = {latest}
}


