# NL-HOPE (PyTorch)

This repository implements **Nested Learning (NL)** and the **HOPE** module from:

> Behrouz, Razaviyayn, Zhong, Mirrokni. *Nested Learning: The Illusion of Deep Learning Architectures*, NeurIPS 2025 (camera-ready summary). :contentReference[oaicite:8]{index=8}

**Highlights**
- **Continuum Memory System (CMS)**: a chain of MLP blocks with different update frequencies (Eq. 30–31). :contentReference[oaicite:9]{index=9}
- **Self-modifying sequence model (HOPE)**: fast associative memory + inner optimizer implementing the L2-regression step variant of GD (Eq. 27–29). :contentReference[oaicite:10]{index=10}
- **Optimizers as memory**: momentum/Adam interpretations; we use an inner "deep" memory update for projections. :contentReference[oaicite:11]{index=11}

## Install
