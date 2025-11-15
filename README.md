# NL-HOPE (PyTorch)

This repository implements **Nested Learning (NL)** and the **HOPE** module from:

> Behrouz, Razaviyayn, Zhong, Mirrokni. *Nested Learning: The Illusion of Deep Learning Architectures*, NeurIPS 2025 (camera-ready summary). :contentReference[oaicite:8]{index=8}

**Highlights**
- **Continuum Memory System (CMS)**: a chain of MLP blocks with different update frequencies (Eq. 30–31). :contentReference[oaicite:9]{index=9}
- **Self-modifying sequence model (HOPE)**: fast associative memory + inner optimizer implementing the L2-regression step variant of GD (Eq. 27–29). :contentReference[oaicite:10]{index=10}
- **Optimizers as memory**: momentum/Adam interpretations; we use an inner "deep" memory update for projections. :contentReference[oaicite:11]{index=11}

## Install


## Train / Eval / Sample
See `scripts/` for end-to-end commands on WikiText-2.
To use your own text, provide `--data.path` to a plain text file or Hugging Face dataset name.

## What’s implemented
- Linear-attention fast weights (`nl/modules/fast_kv.py`).
- CMS (multi-level MLP chain with update frequencies) (`nl/memory/cms.py`).
- HOPE LM (`nl/models/hope.py`) that:
  - dynamically updates Q/K/V projections with an inner optimizer step per batch (no backprop through inner updates),
  - routes token features through CMS with independent update schedules.
- Sampling with temperature/top-k/top-p (`scripts/sample.py`).

## Caveats
The NeurIPS camera-ready paper references additional methods/appendices and an arXiv version. In ambiguous spots, we follow standard fast-weight/linear-attention practice and keep inner updates isolated from outer gradients per NL’s multi-flow viewpoint. :contentReference[oaicite:12]{index=12}

## Citation
If you use this repo, please cite the paper:
