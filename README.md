# Domain-Adapted Multimodal Pipeline for Liver Histopathology

Code accompanying the manuscript submitted to *Journal of Clinical and Translational Hepatology* (JCTH):

> Small-Sample Domain-Adapted Multimodal Pipeline for Liver Histopathology Image Grading, Report Retrieval, and Structured Report Generation

University-Town Hospital of Chongqing Medical University

## Overview

Three tasks on a de-identified single-center liver pathology image–report dataset
(979 images, 217 reports, 109 patients, strict patient-level split):

1. **Grading** — end-to-end multi-task fine-tuning of PLiP (ViT-B/16) for
   five-class inflammatory grade (G0–G4), binary significant inflammation (G≥2)
   and binary significant fibrosis (S≥3), under five-fold patient-level
   cross-validation.
2. **Cross-modal retrieval** — domain-adaptive contrastive fine-tuning of
   Chinese-CLIP to retrieve the diagnostic report for a histology image.
3. **Conditional report generation** — a ClipCap-style model conditioned on the
   predicted binary grading labels (a real cascade pipeline: image → grading →
   report).

A commercial flagship VLM (Qwen3-VL-Plus) is benchmarked under identical
zero-shot / five-shot protocols.

## Environment

Reported results were produced with:

- Python 3.12.3
- PyTorch 2.12.1+cu130, torchvision 0.27.1+cu130 (CUDA 13.0)
- scikit-learn 1.9.1, Hugging Face Transformers 5.19.0
- Single NVIDIA RTX 4090 GPU (AutoDL cloud instance)

Dependencies: see `requirements.txt` (minimum versions); the exact pinned
environment is in `requirements-lock.txt` (exported with `pip freeze` from the
training instance).

Install:

```bash
pip install -r requirements.txt
```

Exact package versions used for the reported results are pinned in
`requirements-lock.txt` (exported with `pip freeze` from the training instance).

## Script map

| File | Role |
|---|---|
| `train_finetune.py` / `分级V2_微调PLiP.py` | Task 1: PLiP multi-task grading fine-tuning (five-fold patient-level CV) |
| `分级V2_微调PLiP_保存预测.py` | Save per-fold predictions / thresholds |
| `分级V3_微调PLiP.py` | Grading variant used for the external-validation pilot |
| `基线_提取特征.py`, `基线_训练评估.py`, `基线V2_训练评估.py` | Frozen-feature baselines |
| `B1_图文检索.py` | Task 2: zero-shot retrieval baseline |
| `B2_对比微调.py` | Task 2: contrastive fine-tuning of Chinese-CLIP |
| `B2_报告级分析.py` | Report-level aggregation and metrics |
| `A4_难负样本微调.py` | Task 2 ablation: hard-negative second stage |
| `B3_报告生成.py` | Task 3: ClipCap-style report generation (unconditional) |
| `A5_条件生成.py` | Task 3: conditional generation on predicted / ground-truth labels |
| `A6_自监督预训练.py` | Exploratory self-supervised pretraining |
| `B0_大模型分级与生成.py` | Commercial VLM (Qwen3-VL-Plus) zero-/five-shot evaluation |

## Data availability

The de-identified image–report data are available from the corresponding author
upon reasonable request and subject to institutional approval (see manuscript
Data Sharing Statement).

## License

MIT License — see `LICENSE`.
