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
