# Multi-Granular Agreement Fusion for Multimodal Misinformation Detection

This repository accompanies the research manuscript **“Multi-Granular Agreement Fusion for Multimodal Misinformation Detection”** and is intended to host its implementation and reproducibility materials.

## Overview

Fine-grained image–text manipulations can preserve overall semantic consistency while introducing subtle local inconsistencies. Moreover, predictions from different auxiliary heads may disagree, making their reliability important for the final authenticity decision.

We propose **Multi-Granular Agreement Fusion (MAF)**, a framework for binary multimodal misinformation detection that combines three components:

- **Multi-Granular Representation (MGR):** Integrates global image–text semantics, local visual and textual features, cross-modal residuals, and manipulation-state information to capture complementary manipulation cues.
- **Head-Agreement Fusion (HAF):** Uses agreement between modality-level manipulation predictions and state-router predictions, together with state-prediction confidence, to guide adaptive decision fusion while retaining a fixed global-branch weight.
- **Dual-Tail Annotation-Aware Focal Optimization (DAFO):** Emphasizes difficult real and text-only samples through subtle-text reweighting, hard-example mining, state-specific margins, and ranking objectives during training.

## Dataset and Evaluation

Experiments use the **DGM4** benchmark for multimodal media manipulation detection and grounding.

- **Official dataset repository:** [MultiModal-DeepFake](https://github.com/rshaojimmy/MultiModal-DeepFake)
- **News sources:** BBC, The Guardian, USA TODAY, and The Washington Post
- **Primary task:** Binary authenticity detection (authentic vs. manipulated image–text pairs)
- **Evaluation metrics:** Accuracy (ACC), area under the ROC curve (AUC), and equal error rate (EER)

Experiments are conducted separately on the four source-specific subsets. Dataset files and news images are **not redistributed** here; please consult the original dataset repository for access and usage conditions.

## Code and Reproducibility

**Status: Code release in preparation.**

This repository currently provides project information only. The implementation, environment requirements, training and evaluation scripts, configurations, and reproducibility instructions will be added after verification and repository cleanup.

Until those files are available, this repository should **not** be considered a runnable reproduction package. Installation commands and result-reproduction instructions will be documented here once validated.

## Authors

- Jiajie Lin
- Yi Yu
- Zhenguo Yang
- Jianming Wu

## Citation

Citation details and a BibTeX entry will be added when the manuscript has a stable publication or preprint record.
