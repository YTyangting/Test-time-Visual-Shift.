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
