
# Test-time Visual Shift (TVS)

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Pytorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)

> This is the official repository for the paper **Test-time Visual Shift (TVS)**. This project aims to improve the adaptation efficiency and robustness of Vision-Language Models when facing out-of-distribution visual shifts during the Test-Time Adaptation (TTA) phase.

## 📖 Introduction

In real-world applications of Vision-Language Models, there is often a distribution discrepancy between the training data and the test data in the deployment environment, leading to severe **Visual Shift** challenges. This project proposes an efficient TVS adaptation framework.
Through our method, the model can not only maintain or improve prediction accuracy when facing dynamic visual shifts but also achieve significant breakthroughs in **Efficiency Gains**, substantially reducing the computational overhead of traditional TTA methods.

## 🛠️ Requirements

We recommend using [Conda](https://docs.conda.io/) or Docker to manage your virtual environment.

```bash
# 1. Clone this repository
git clone [https://github.com/YTyangting/Test-time-Visual-Shift.git](https://github.com/YTyangting/Test-time-Visual-Shift.git)
cd Test-time-Visual-Shift

# 2. Create and activate the Conda environment
conda create -n tvs python=3.9
conda activate tvs

# 3. Install dependencies
pip install -r requirements.txt

```

## 📂 Data Preparation

Please organize your test datasets (e.g., ImageNet-C, commonly used for evaluating visual shifts) under the `data/` directory following this structure:

```text
Test-time-Visual-Shift/
└── data/
    └── target_dataset/
        ├── class_1/
        ├── class_2/
        └── ...

```

## 🚀 Quick Start

### 1. Run the Test-Time Adaptation Framework

You can easily run the TVS method using our provided bash scripts. For example, to launch the adaptation process with a ViT-16 backbone, make sure you are in the project root directory and execute:

```bash
bash scripts/tvs/tvs_vit16.sh imagenet

```

*(Note: You can directly edit the `tvs_vit16.sh` script to modify hyperparameters such as the target dataset, batch size, or adaptation learning rate.)*

### 2. Efficiency Analysis

To reproduce the experimental analysis regarding **Efficiency gains** reported in the paper, you can run the following script to obtain a comprehensive evaluation of GPU memory usage, inference latency, and accuracy:

```bash
python evaluate_efficiency.py --config configs/tvs_config.yaml

```

## 📑 Citation

If you use the code from this repository in your academic research or find this work inspiring, please consider citing our paper:

```bibtex
@inproceedings{ye2026tvs,
  title={Test-time Visual Shift: Efficient Adaptation for Vision-Language Models},
  author={Yang, Ting and others},
  booktitle={Proceedings of the 34th ACM International Conference on Multimedia},
  year={2026}
}

```

## ✉️ Contact

If you have any questions regarding code execution or academic discussions, feel free to open an Issue in the repository or contact us directly via email: [Your Email Address].

```

```
