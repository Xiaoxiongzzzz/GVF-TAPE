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
git clone --recursive git@github.com:Xiaoxiongzzzz/GVF-TAPE.git
cd GVF-TAPE

# 2. Create environment
conda env create -f environment.yml
conda activate gvftape
```

## 👉 Download Weights
We provide model weights for libero-spatial, libero-object and libero-goal on [hugging face](https://huggingface.co/Ackermannnnnn/CoRL_2025_GVF-TAPE/tree/main), you can download the weights by runing the following code.
```python
python ./scripts/download_weights.py
```


## 🦾 Evaluation on LIBERO
Choose suite (spatial/object/goal) in scripts/eval_ik_proprio.py and run 
```
python -m scripts.eval_ik_proprio
```
the results will be saved in outputs/.

## 🏷️ License
This repository is released under the MIT license. 

## 🙏 Acknowledgement

## 📝 Citation

If you find our work useful in your research, please consider citing:

```bibtex
@inproceedings{zhang2025generative,
  title={Generative Visual Foresight Meets Task-Agnostic Pose Estimation in Robotic Table-top Manipulation},
  author={Zhang, Chuye and Zhang, Xiaoxiong and Zheng, Linfang and Pan, Wei and Zhang, Wei},
  booktitle={Conference on Robot Learning},
  pages={2823--2846},
  year={2025},
  organization={PMLR}
}
```

## 📬 Contact & Support
I am always excited to connect with researchers and developers in the community! 🌟

Whether you have **questions about the paper**, **ideas for future work**, or just want to **discuss the latest trends in Robot Learning**, please **do not hesitate** to reach out. I'd love to hear from you!

👉 **[Drop me an email](mailto:12110807@mail.sustech.edu.cn?subject=Hi%20Xiaoxiong%20-%20Regarding%20GVF-TAPE)** 👈

* **Xiaoxiong Zhang**: `12110807 [at] mail.sustech.edu.cn`