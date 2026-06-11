# Scaling up HART: A High-Efficiency Cross-Domain Optimizer for Next-Generation Deep Learning

> **Work in Progress** — Codebase active. Paper coming soon.

## Project Description & Research Goals
This project aims to scale up and comprehensively benchmark HART, a novel cross-domain optimizer designed to improve convergence efficiency, stability, and generalization across diverse deep learning workloads.

While AdamW remains the dominant optimization baseline in modern deep learning, its convergence behavior can become increasingly sensitive under aggressive regularization and large-scale optimization settings. HART was developed to address these limitations through a more structured optimization framework with explicit and fine-grained control mechanisms.

Although HART introduces additional hyperparameters compared to AdamW, preliminary experiments indicate that the optimizer maintains strong stability while enabling more robust optimization trajectories and improved regularization behavior across both Computer Vision (CV) and Natural Language Processing (NLP) domains.

The primary objective of this project is to validate whether these improvements persist under larger-scale training regimes and foundation-model workloads.

---

## Preliminary Results & Empirical Evidence

### Computer Vision Benchmarks (CIFAR-100)
HART has been evaluated on CIFAR-100 across multiple architectures, including:
* Standard/raw CNNs
* ResNet-18
* Vision Transformers (ViTs)

In early HART-v1 experiments, the optimizer consistently demonstrated strong generalization capability, achieving improvements of more than **+3% Test Accuracy** over AdamW on several architectures while also maintaining stable gains on ResNet-18. 
The current optimizer under active development is HART-v2, which incorporates additional refinements beyond the original v1 implementation. Based on ongoing experiments, we expect further performance improvements over the already strong v1 baselines.

### Language Modeling Benchmarks (GPT-2 Small)
HART has also been evaluated on autoregressive language modeling tasks using GPT-2 Small trained entirely from scratch on **WikiText-103** (Sequence Length: 256, Batch Size: 32).

* **AdamW Baseline Configuration:** Learning Rate: 2.5e-4, Warm-up: 0.5 epoch, Cosine Scheduling: 8.5 epochs
* **HART Configuration:** Learning Rate: 2.5e-4, **Warm-up: None**, Cosine Scheduling: 9 epochs

Under this standardized in-domain training setup, HART demonstrated significantly faster early-stage convergence and consistently improved perplexity (PPL) behavior compared to AdamW.

Most notably, during high-intensity regularization experiments targeting the “loss of plasticity” problem, HART achieved a **peak validation perplexity of 18.94** under an extreme weight decay configuration (WD=0.30). By comparison, the strongest AdamW baseline achieved a best PPL of 20.19 at WD=0.02 before suffering severe degradation and representation collapse at higher weight decay settings.

These results indicate an **absolute improvement of 1.25 PPL** over the strongest tuned AdamW configuration while additionally operating in a completely **warm-up-free regime**.

---

## Open Science & Reproducibility Plan
To support transparent evaluation and reproducibility:
* Training code will be publicly released.
* Optimizer implementations will be open-sourced.
* Training checkpoints will be uploaded to GitHub.
* Experimental configurations and hyperparameters will be documented in detail.

---

## Why TPU Resources Are Required
Thus far, all development and experimentation have been conducted using a single consumer-grade NVIDIA RTX 4060 GPU. While this environment has been sufficient for early-stage validation, it severely limits the ability to perform large-scale empirical studies required for modern optimizer research.

Access to Google TRC TPU infrastructure would enable us to:
* Scale experiments to larger foundation models such as GPT-2 Medium and ViT-Large.
* Perform large-scale hyperparameter ablation studies and investigate scaling laws.
* Evaluate robustness under longer-context language modeling settings.
* Validate cross-domain generalization on substantially larger datasets such as ImageNet-1k.
* Complete the comprehensive empirical evaluation necessary for submission to top-tier machine learning conferences (e.g., ICLR).

The proposed research is specifically compute-bound rather than idea-bound, and TPU access would directly accelerate validation of a promising optimizer framework that has already demonstrated encouraging results across multiple domains.

## Preliminary Results & Empirical Evidence

### 1. Language Modeling Benchmark (Test PPL ↓)
Evaluated on autoregressive language modeling tasks trained entirely from scratch.
* **Dataset:** WikiText-103
* **Configuration:** Sequence Length=256, Batch Size=32

| Model | Optimizer | Test PPL | Rank |
| :--- | :--- | :---: | :---: |
| **GPT-2 Small** | **HART** | **18.94** | 🥇 **1st** |
| GPT-2 Small | Lion | 19.15 | 2nd |
| GPT-2 Small | AdamW | 19.58 | 3rd |
| | | | |
| **GPT-2 Medium** | **HART** | **17.45** | 🥇 **1st** |
| GPT-2 Medium | Lion | 17.52 | 2nd |
| GPT-2 Medium | AdamW | 18.55 | 3rd |

---

### 2. General Language Understanding Evaluation (GLUE)
Performance averaged over 3 random seeds (42, 100, 2026). Displaying Mean ± Std (sample std, n-1).

| Task / Metric | AdamW | Lion | HART |
| :--- | :---: | :---: | :---: |
| SST-2 | 87.96 ± 0.64 | 87.54 ± 0.47 | **88.19 ± 0.35** |
| MRPC | 80.42 ± 0.85 | **81.93 ± 0.77** | 81.01 ± 1.01 |
| RTE | **53.91 ± 3.56** | 54.63 ± 1.37 | 53.43 ± 3.12 |
| CoLA | 17.31 ± 3.46 | 18.41 ± 5.26 | **20.02 ± 4.28** |
| QNLI | 82.71 ± 1.01 | 82.93 ± 1.60 | **83.97 ± 0.77** |
| **Overall Mean** | 64.46 ± 0.93 | 65.09 ± 0.79 | **65.32 ± 0.48** |

*(Note: HART demonstrates not only the highest overall mean but also significantly lower variance across seeds, indicating robust convergence stability.)*

---

### 3. Vision Benchmark (Test Accuracy % ↑)
Evaluated across various scale datasets and modern architectures.

**CIFAR-100**
| Architecture | HART | AdamW | Lion | Best Performer |
| :--- | :---: | :---: | :---: | :---: |
| ResNet-18 | 75.00 | **75.27** | 74.09 | AdamW |
| ConvNeXT | **70.53** | 69.49 | 69.77 | **HART** |
| ViT-Tiny | **59.00** | 57.90 | 57.65 | **HART** |

**Tiny-ImageNet**
| Architecture | HART | AdamW | Lion | Best Performer |
| :--- | :---: | :---: | :---: | :---: |
| ResNet-50 | **64.86** | 63.76 | 63.94 | **HART** |
| ConvNeXT | **43.95** | 43.25 | 43.23 | **HART** |
| ViT-Tiny | 44.05 | 43.77 | **46.17** | Lion |

**ImageNet (Subset)**
| Architecture | HART | AdamW | Lion | Best Performer |
| :--- | :---: | :---: | :---: | :---: |
| ResNet-50 | 74.60 | **75.08** | 73.68 | AdamW |
| ConvNeXT | 75.11 | **75.43** | 74.70 | AdamW |
| ViT-Tiny | **68.10** | 67.88 | 68.02 | **HART** |

---

### 4. Transfer Learning: Mini-VTAB Benchmark (Test Accuracy % ↑)
| Task | AdamW | Lion | HART |
| :--- | :---: | :---: | :---: |
| CIFAR-100 | 82.01 ± 0.28 | 80.77 ± 0.20 | **83.76 ± 0.34** 🥇 |
| Flowers102 | **84.73 ± 0.90** 🥇 | 83.11 ± 1.39 | 81.37 ± 0.86 |
| DTD | 64.77 ± 0.98 | 63.17 ± 0.20 | **66.21 ± 0.22** 🥇 |
| EuroSAT | 98.75 ± 0.14 | 98.59 ± 0.24 | **99.00 ± 0.21** 🥇 |
| SVHN | 96.06 ± 0.05 | 90.18 ± 3.19 | **96.14 ± 0.17** 🥇 |
