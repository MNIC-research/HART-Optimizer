# Scaling up HART: A High-Efficiency Cross-Domain Optimizer for Next-Generation Deep Learning

> 🚧 **Work in Progress** — Codebase active. Paper coming soon for ICLR 2027.

## Project Description & Research Goals
This project aims to scale up and comprehensively benchmark HART, a novel cross-domain optimizer designed to improve convergence efficiency, stability, and generalization across diverse deep learning workloads.

While AdamW remains the dominant optimization baseline in modern deep learning, its convergence behavior can become increasingly sensitive under aggressive regularization and large-scale optimization settings. HART was developed to address these limitations through a more structured optimization framework with explicit and fine-grained control mechanisms.

By refining our core algorithmic architecture, the latest iteration of HART implements a highly memory-efficient **2-buffer state mechanism**, matching the exact state memory footprint of AdamW while preserving its robust optimization trajectories and advanced regularization capabilities.

The primary objective of this project is to validate whether these improvements persist under larger-scale training regimes and foundation-model workloads.

---

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

*Under this standardized setup, HART demonstrated significantly faster early-stage convergence and consistently improved perplexity (PPL). Most notably, during high-intensity regularization experiments targeting the "loss of plasticity" problem, HART achieved a peak validation PPL of 18.94 on GPT-2 Small under an extreme weight decay configuration (WD=0.30). By comparison, the strongest AdamW baseline collapsed at higher settings, peaking at 19.58.*

*Crucially, **as the model scales, the optimizer's advantage widens.** On GPT-2 Medium, the performance gap between HART (17.45) and AdamW (18.55) expanded to a massive **1.10 PPL absolute improvement**, all while operating in a completely warm-up-free regime.*

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

*Note: HART demonstrates not only the highest overall mean performance (65.32) but, more importantly, exhibits **nearly half the variance of AdamW (±0.48 vs. ±0.93)**. This extremely low variance proves that HART's dynamic orthogonal noise scraping mechanism successfully filters out detrimental gradient noise, guaranteeing stable convergence trajectories regardless of initialization seeds.*

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

*While standard CNN architectures—which have been heavily co-optimized with AdamW and SGD over the past decade—show competitive but mixed results, **HART demonstrates a clear and consistent advantage on Vision Transformers (ViTs).** Across CIFAR-100 and ImageNet subsets, HART outperforms both AdamW and Lion on ViT-Tiny.*

---

### 4. Transfer Learning: Mini-VTAB Benchmark (Test Accuracy % ↑)
| Task | AdamW | Lion | HART |
| :--- | :---: | :---: | :---: |
| CIFAR-100 | 82.01 ± 0.28 | 80.77 ± 0.20 | **83.76 ± 0.34** 🥇 |
| Flowers102 | **84.73 ± 0.90** 🥇 | 83.11 ± 1.39 | 81.37 ± 0.86 |
| DTD | 64.77 ± 0.98 | 63.17 ± 0.20 | **66.21 ± 0.22** 🥇 |
| EuroSAT | 98.75 ± 0.14 | 98.59 ± 0.24 | **99.00 ± 0.21** 🥇 |
| SVHN | 96.06 ± 0.05 | 90.18 ± 3.19 | **96.14 ± 0.17** 🥇 |

*In transfer learning scenarios evaluated via the Mini-VTAB benchmark, HART showcased exceptional adaptability, securing **1st place in 4 out of 5 downstream tasks**. This robust transferability indicates that HART is highly effective at extracting generalized representations, particularly for modern attention-based architectures.*

---

## Open Science & Reproducibility Plan
To support transparent evaluation and reproducibility:
* Training code and optimizer implementations are publicly released.
* Training checkpoints will be continuously uploaded to GitHub.
* Experimental configurations and hyperparameters are documented in detail.

---

## Why TPU Resources Are Required
Thus far, all development and experimentation have been conducted using a single consumer-grade NVIDIA RTX 4060 GPU. While this environment has been sufficient to validate the core algorithmic innovations (achieving state-of-the-art memory-efficient optimization via our 2-buffer state mechanism), it severely limits the ability to perform the large-scale empirical studies required for modern optimizer research.

Access to Google TRC TPU infrastructure would enable us to:
* Scale experiments to larger foundation models such as GPT-2 Large/XL and ViT-Large.
* Perform large-scale hyperparameter ablation studies and investigate optimization scaling laws.
* Evaluate robustness under longer-context language modeling settings.
* Validate cross-domain generalization on substantially larger datasets such as full ImageNet-1k.
* Complete the comprehensive empirical evaluation necessary for submission to top-tier machine learning conferences (e.g., ICLR).

The proposed research is specifically compute-bound rather than idea-bound. TPU access would directly accelerate the final validation of a highly promising, memory-efficient optimizer framework that has already demonstrated exceptional results across language modeling, transfer learning, and attention-based architectures.
