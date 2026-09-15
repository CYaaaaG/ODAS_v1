# ODAS Reproduction Project

This repository stores the code, configuration, metadata, and experiment records for reproducing:

> Y. Li et al., “An Open-World Semi-Supervised Recognition Method (ODAS) for Φ-OTDR Disturbance Signals,” IEEE Internet of Things Journal, 2025.

DOI: https://doi.org/10.1109/JIOT.2025.3585466

## Data

The Cao laboratory Φ-OTDR dataset is used first. The raw `.mat` files and large zip archives are intentionally excluded from Git because they are too large for a normal GitHub repository.

Dataset and baseline code:

- GitHub: https://github.com/BJTUSensor/Phi-OTDR_dataset_and_codes
- Google Drive: https://drive.google.com/drive/folders/1-4jGDVrGP-KZ-EvLlURN8Y2fJ_IDux-e?usp=sharing
- Baidu Netdisk: https://pan.baidu.com/s/1i07ssN34U31xzW7YLj5f_A
- Baidu extraction code: `morv`

After downloading and extracting the data, place it under:

```text
Data/
├── train/
│   ├── 01_background/
│   ├── 02_dig/
│   ├── 03_knock/
│   ├── 04_water/
│   ├── 05_shake/
│   ├── 06_walk/
│   └── label.txt
└── test/
    ├── 01_background/
    ├── 02_dig/
    ├── 03_knock/
    ├── 04_water/
    ├── 05_shake/
    ├── 06_walk/
    └── label.txt
```

The local dataset currently contains 15,419 samples. Each sample has shape `10000 x 12` and is stored as a MATLAB `.mat` file.

## Reproduction protocol

The initial open-world setting follows the paper:

- Seen classes: background, digging, knocking
- Unseen classes: watering, shaking, walking
- 50% of seen-class samples are labeled
- The remaining seen samples and all unseen samples are unlabeled

Training must not use the true labels of unlabeled samples. Those labels are reserved for evaluation.

## Planned implementation order

1. Data inspection and preprocessing
2. 32 x 32 input construction
3. Supervised CNN/ResNet-18 baseline
4. Open-world split and Hungarian evaluation
5. ODAS minimum viable reproduction
6. UCM, SSM, AST, UGTS, and CACM modules
7. Ablation, comparison, and sensitivity experiments

## Repository policy

Raw data, zip archives, checkpoints, and experiment caches are excluded through `.gitignore`. They must be downloaded or generated separately on each server.
