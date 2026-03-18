# HMDMV
Official PyTorch implementation of **HMDMV: Hierarchical Mutual Distillation for Multi-View Fusion**.

**Paper:** [Hierarchical mutual distillation for multi-view fusion: Learning from all possible view combinations](https://www.sciencedirect.com/science/article/pii/S0031320326003973)  
**Journal:** Pattern Recognition

## Overview
HMDMV is a multi-view fusion framework designed for both **structured** and **unstructured** multi-view image classification.  
Unlike conventional methods that rely only on single-view or full multi-view fusion, HMDMV learns from **all possible view combinations**:  
- **Single-view**
- **Partial multi-view**
- **Full multi-view**
It also combines **uncertainty-aware weighting** and **hierarchical mutual distillation** to improve robustness under incomplete multi-view settings.

## Release Scope
This public release currently provides:
- the core implementation of **HMDMV**
- the training / validation / test pipeline
- the **Hotels-8k** split files used in our reference benchmark
- a runnable training script for the Hotels-8k setting

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

## Installation
```text
pip install -r requirements.txt

## Dataset Preparation
This repository includes the Hotels-8k split CSV files only.
Please prepare the original Hotels-8k images separately and make sure the paths in the CSV files match your local environment.

## Training
Run the Hotels-8k reference experiment with:
```text
bash scripts/run_hotels8k.sh

Or run directly:
```text
python main.py \
    --dataset hotels8k \
    --num_view 3 \
    --num_classes 7774 \
    --method HMDMV \
    --model_name vit_small_r26_s32_224 \
    --hmd_loss True

## Citation
If you find this repository useful, please cite:
```text
@article{yang2026hmdmv,
  title={Hierarchical mutual distillation for multi-view fusion: Learning from all possible view combinations},
  author={Yang, Jiwoong and Chung, Haejun and Jang, Ikbeom},
  journal={Pattern Recognition},
  volume={178},
  pages={113432},
  year={2026}
}
