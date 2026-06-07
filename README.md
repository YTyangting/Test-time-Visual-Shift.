# Test-time Visual Shift (TVS)

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Pytorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)

> 这是关于 **Test-time Visual Shift (TVS)** 论文的官方实现代码库。本项目致力于在测试时自适应（Test-Time Adaptation, TTA）阶段，提升视觉-语言模型（Vision-Language Models）应对分布外视觉偏移（Visual Shift）的适应效率和鲁棒性。

## 📖 简介 (Introduction)

在实际的视觉-语言模型应用中，训练数据与部署环境的测试数据往往存在分布不一致，导致严重的**视觉偏移（Visual Shift）**挑战。本项目提出了一种高效的 TVS 自适应框架。
通过我们的方法，模型在面临动态视觉偏移时不仅能够维持或提升预测准确率，而且在**效率增益（Efficiency Gains）**上取得了显著突破，大幅降低了传统 TTA 方法的计算开销。

## 🛠️ 环境依赖 (Requirements)

我们推荐使用 [Conda](https://docs.conda.io/) 或 Docker 来管理您的虚拟环境。

```bash
# 1. 克隆本仓库
git clone [https://github.com/YTyangting/Test-time-Visual-Shift.git](https://github.com/YTyangting/Test-time-Visual-Shift.git)
cd Test-time-Visual-Shift

# 2. 创建并激活 Conda 环境
conda create -n tvs python=3.9
conda activate tvs

# 3. 安装依赖库
pip install -r requirements.txt
