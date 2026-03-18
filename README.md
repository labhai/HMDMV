# HMDMV
Official PyTorch implementation of **HMDMV: Hierarchical Mutual Distillation for Multi-View Fusion**.

**Paper:** *Hierarchical mutual distillation for multi-view fusion: Learning from all possible view combinations*  
**Journal:** Pattern Recognition 178 (2026) 113432

## Overview
HMDMV is a multi-view fusion framework designed for both **structured** and **unstructured** multi-view image classification.  
Unlike conventional methods that rely only on single-view or full multi-view fusion, HMDMV learns from **all possible view combinations**:  
- **Single-view**
- **Partial multi-view**
- **Full multi-view**

The framework combines:
- **all-possible view-combination fusion**
- **uncertainty-aware weighting**
- **hierarchical mutual distillation**
- **flexible inference under varying numbers of views**
- **scalable subset-sampling strategy for training efficiency**

## Key Features
- Learns from **all possible subsets of views** instead of only single-view or full-view predictions
- Uses **uncertainty-aware weighting** to reduce the influence of unreliable view combinations during training
- Applies **hierarchical mutual distillation** between single/partial multi-view predictions and the full multi-view prediction
- Supports **flexible inference** when the number of available views differs from training
- Includes a **Hotels-8k reference benchmark release** in this repository

## Release Scope
This public release currently provides:
- the core implementation of **HMDMV**
- the training / validation / test pipeline
- the **Hotels-8k** split files used in our reference benchmark
- a runnable training script for the Hotels-8k setting

Please note that the paper reports experiments on multiple datasets, including **Hotels-8k, GLDv2, Carvana, and VinDr-Mammo**, but dataset-specific preprocessing pipelines for the other benchmarks are **not included** in this public release.

## Repository Structure
```text
.
├── dataset/
│   └── hotels8k.py
├── hotels8k/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── loss/
│   └── hmd_loss.py
├── networks/
│   └── hmdmv.py
├── process/
│   └── train.py
├── scripts/
│   └── run_hotels8k.sh
├── main.py
├── utils.py
├── README.md
└── LICENSE
