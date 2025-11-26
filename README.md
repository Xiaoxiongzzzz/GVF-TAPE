# Generative Visual Foresight Meets Task-Agnostic Pose Estimation in Robotic Table-Top Manipulation

[![arXiv](https://img.shields.io/badge/arXiv-2509.00361-b31b1b.svg)](https://arxiv.org/abs/2509.00361)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue)](https://huggingface.co/Ackermannnnnn/CoRL_2025_GVF-TAPE/tree/main)
[![Project Page](https://img.shields.io/badge/Project-Website-green)](https://clearlab-sustech.github.io/gvf-tape/)

**[Generative Visual Foresight Meets Task-Agnostic Pose Estimation in Robotic Table-Top Manipulation]**   
 *Chuye Zhang<sup> * </sup>*, *Xiaoxiong Zhang <sup> * </sup>*, *Wei Pan*, *Linfang Zheng<sup> † </sup>*, *Wei Zhang <sup>†</sup>*   
**[9th Conference on Robot Learning (CoRL 2025)]**

---

## 🎥 Highlights

![Teaser](./assets/teaser.webp)

**Abstract:** *We introduce **GVF-TAPE**, a framework that enables robots to learn complex manipulation skills **without any action labels**. The system decouples visual planning from execution: it generates a high-fidelity **RGB-D video plan** (Visual Foresight) and executes it using a pose estimator trained on **random exploration** data. This design allows for scalable, closed-loop manipulation in unstructured real-world environments, generalizing to novel objects and tasks.*

## 🛠️ Installation

We recommend using **Conda** to set up the environment.

```bash
# 1. Clone the repository
git clone git@github.com:Xiaoxiongzzzz/GVF-TAPE.git
cd GVF-TAPE

# 2. Create environment
conda env create -f environment.yml
conda activate my_robot_env

# 3. Install additional dependencies (if needed)
pip install -e .