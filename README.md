# Multi-Granular Agreement Fusion for Multimodal Misinformation Detection

MAF combines multi-granular representation (MGR), head-agreement fusion (HAF), and dual-tail annotation-aware focal optimization (DAFO). This repository provides the source code, configurations, data preparation, trained models, and recorded predictions supporting the PAA manuscript.

**[Dataset access](docs/DATA.md) · [Trained models](checkpoints/README.md) · [Experimental protocol and limitations](docs/REPRODUCIBILITY.md)**

## 1. Install

```bash
git clone https://github.com/decode12/MAF-PAA.git
cd MAF-PAA
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
```

Original training used one NVIDIA A40 (48 GB) and CUDA BF16; the commands above install PyTorch for CUDA 11.8. Dependency installation and CPU inference were tested with Python 3.11.14 and PyTorch 2.5.1; exact dependencies are pinned in [requirements.txt](requirements.txt). For CPU use, replace `cu118` with `cpu` in the installation command. Allow space for 10.7 GB of dataset archives plus extracted images, pretrained CLIP, and model files. Run all commands below from the repository root.

## 2. Obtain data and pretrained CLIP

Data are downloaded from the [original DGM4 provider](https://huggingface.co/datasets/rshaojimmy/DGM4), subject to its terms. These commands verify checksums and reconstruct the original source-specific partitions without a random resplit:

```bash
python datasets/download_assets.py --asset dgm4 --output data/raw/DGM4 --extract
python datasets/prepare_dgm4.py --data-root data/raw/DGM4 \
  --derive-source-metadata --output data/prepared --check-images
python datasets/download_assets.py --asset clip --output pretrained/clip-vit-base-patch16
```

[DATA.md](docs/DATA.md) records versions, split counts, labels, preprocessing, and provider links. Raw news images and captions are obtained from the provider, not redistributed here. CLIP ViT-B/16 is fine-tuned; no feature cache or external evidence is required.

## 3. Evaluate released models

```bash
python scripts/download_checkpoints.py --name main_bbc
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py --config configs/main.json \
  --checkpoint checkpoints/main_bbc.pt --split test \
  --output-dir outputs/bbc_test --device cuda
```

For all four news sources and the three BBC module ablations:

```bash
python scripts/download_checkpoints.py --name all
CUDA_VISIBLE_DEVICES=0 bash scripts/reproduce_main.sh evaluate cuda
CUDA_VISIBLE_DEVICES=0 bash scripts/reproduce_ablation.sh evaluate cuda
```

The eight checkpoints are [public release assets](https://github.com/decode12/MAF-PAA/releases/tag/reproducibility-v1), approximately 4.7 GiB in total; the downloader verifies every SHA-256. The GPU wrappers run experiments sequentially on the selected device. For a small CPU check, use the single-model command with `--device cpu --limit 64` and a new output directory; this is not a full paper evaluation.

## 4. Train and reproduce ablations

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/reproduce_main.sh train cuda
CUDA_VISIBLE_DEVICES=0 bash scripts/reproduce_ablation.sh train cuda
```

Each run starts from pretrained CLIP and new MAF heads, trains for **20 epochs with seed 42**, and tests the **first maximum-validation-ACC** checkpoint at threshold **0.5**. Released MAF checkpoints are not required for training. The ablations completely remove MGR, HAF, or DAFO, respectively. For one source only:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/main.json \
  --source bbc --output-dir runs/main_bbc --device cuda
```

Use fresh output directories for repeated experiments; completed outputs are protected against overwriting. Keep physical batch size 8 and gradient accumulation 4: DAFO operates within each physical batch. Details and ablation definitions are in the [protocol](docs/REPRODUCIBILITY.md).

## 5. Check reported results

These commands recompute tables and analysis from the released predictions, without downloading data or models:

```bash
python scripts/summarize_results.py --output outputs/recomputed_results.csv
python scripts/reproduce_analysis.py --output-dir outputs/analysis --verify-paper --plot
```

| Test source | ACC (%) | AUC (%) | EER (%) |
| --- | ---: | ---: | ---: |
| BBC | 86.00 | 91.16 | 14.54 |
| The Guardian | 89.64 | 94.49 | 10.66 |
| USA TODAY | 84.28 | 89.78 | 16.60 |
| The Washington Post | 82.21 | 87.70 | 18.58 |

Saved-prediction recomputation matches the reported rounding. **Fresh inference is a separate check:** CPU FP32 tests differed from archived CUDA BF16 predictions on 3 of 352 records, with a maximum probability difference of 0.886. Full CUDA reevaluation and complete retraining have not been repeated for this release; no numerical tolerance is claimed. Figures 3/4 use an earlier BBC checkpoint, and third-party baseline implementations are not bundled. See the [protocol and test evidence](docs/REPRODUCIBILITY.md) for these limitations and result provenance.

## Repository layout

| Directory | Contents |
| --- | --- |
| `maf/` | Final model, losses, data loader, training and evaluation |
| `configs/` | Main protocol and three module ablations |
| `datasets/` | Asset downloads, checksums and split reconstruction |
| `scripts/` | Executable training, evaluation and result commands |
| `checkpoints/` | Model download manifest and instructions |
| `results/` | Metrics, provenance, selection history and sanitized predictions |
| `docs/` | Data instructions, reproduction protocol and third-party notices |

## Citation and license

```bibtex
@misc{lin_maf,
  title = {Multi-Granular Agreement Fusion for Multimodal Misinformation Detection},
  author = {Lin, Jiajie and Yu, Yi and Yang, Zhenguo and Wu, Jianming},
  note = {Manuscript submitted to Pattern Analysis and Applications},
  howpublished = {\url{https://github.com/decode12/MAF-PAA}}
}
```

Original MAF code is released under [MIT](LICENSE). Dataset content, pretrained models, and third-party software retain their [original terms](docs/THIRD_PARTY.md).
