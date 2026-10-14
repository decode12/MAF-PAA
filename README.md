# Multi-Granular Agreement Fusion for Multimodal Misinformation Detection

MAF combines multi-granular representation (MGR), head-agreement fusion (HAF), and dual-tail annotation-aware focal optimization (DAFO) for binary manipulation detection on DGM4. This repository contains the implementation and configurations corresponding to the current manuscript submitted to **Pattern Analysis and Applications**.

**Release status:** the final-method source, exact split reconstruction, audited results, frozen prediction records and all eight retained MAF checkpoints are publicly available. Checkpoints are distributed through [GitHub Releases](https://github.com/decode12/MAF-PAA/releases/tag/reproducibility-v1); their anonymous downloads have been verified against the SHA-256 values in `checkpoints/manifest.json`. Fresh training does not require these trained checkpoints. Public artifact availability does not resolve the numerical discrepancy between new CPU inference and archived CUDA BF16 predictions, documented below.

The main results use **seed 42, fresh 20-epoch training, and separate training for each of four news sources**. Historical five-epoch seeds 43/44 are not the current main results. Figures 3 and 4 use a separate earlier BBC checkpoint, documented explicitly below. See [the audit](REPRODUCIBILITY_AUDIT.md), [the protocol](docs/REPRODUCIBILITY.md), and [release validation](PAA_RELEASE_REPORT.md).

## Repository structure

```text
maf/                       Final model, objectives, dataset loader, train/evaluation engine
configs/main.json          Shared four-source protocol; source selected by CLI
configs/ablation/           Three complete module-removal configurations
datasets/                  Official asset downloader and exact split reconstruction
scripts/                   Training, evaluation, table and analysis reproduction
checkpoints/               Verified model manifest and download-status information
results/                   Reported metrics, validation selection and sanitized predictions
docs/                      Data access, protocol and third-party notices
```

No raw news images, captions, feature caches, obsolete models, training daemons, optimizer snapshots, or unrelated projects are distributed. Compressed files under `results/predictions/` are derived prediction data; source code is provided as ordinary readable files.

## Environment and hardware

The release was tested in an isolated **Python 3.11.14** environment with **PyTorch 2.5.1 CPU**, torchvision 0.20.1, Transformers 5.11.0, NumPy 1.26.4 and the pinned versions in `requirements.txt`. Dependencies were installed afresh, not inherited from the original training environment. The original runtime currently contains PyTorch 2.5.1+cu118; an immutable training-time environment lock was not retained. The release pins tested versions rather than claiming an unavailable historical lock.

The original experiments used NVIDIA A40 GPUs (48 GB), CUDA BF16 and disabled cuDNN. Each source is trained on one GPU; two GPUs are not required. CPU inference is supported and considerably slower. Use an otherwise idle GPU for full training. Do not change batch size to match another GPU without disclosing the change: DAFO mining and pairwise ranking operate within each physical batch of 8, not the accumulated batch of 32. Allow disk space for about 10.7 GB of downloaded DGM4 archives, extracted images, 0.6 GB of CLIP assets, about 5 GB for all final models, and several GB per training run.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# CPU evaluation / metric reproduction:
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
# For training on CUDA 11.8, use this command instead of the CPU command:
# python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
python -m pip check
```

Run the following commands from the repository root. They use only public project-relative paths. CLI `--data-root`, `--source-metadata`, and `--clip-model` overrides support data stored elsewhere.

## Data download and preparation

[DGM4's official repository](https://github.com/rshaojimmy/MultiModal-DeepFake) provides the [official dataset download](https://huggingface.co/datasets/rshaojimmy/DGM4). The downloader pins revisions and checks file sizes and SHA-256. Read [DATA.md](docs/DATA.md) for original-provider terms, storage and exact fingerprints.

```bash
python datasets/download_assets.py --asset dgm4 --output data/raw/DGM4 --extract
python datasets/prepare_dgm4.py --data-root data/raw/DGM4 \
  --derive-source-metadata --output data/prepared --check-images
```

The source-map reconstruction is explicitly checked against the original VisualNews mapping on all 83,229 used news IDs. It is not a random split or a new assignment. All original record ordering and repeated manipulation variants are retained. Prepared data remain local and ignored by Git.

| Source | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| BBC | 40,811 | 4,424 | 9,370 |
| The Guardian | 109,703 | 11,685 | 27,028 |
| USA TODAY | 31,811 | 2,917 | 7,926 |
| The Washington Post | 25,859 | 3,100 | 6,381 |

Binary targets are Real=0 and Manipulated=1. State order is Real, Text-only, Image-only, Joint. Local annotations supervise training; inference uses the paired image, manipulated caption and tokenizer-derived position masks. Original captions, external articles and retrieved evidence are not inference inputs.

## Pretrained and final weights

Download the exact pretrained CLIP ViT-B/16 assets for both initialization and tokenizer/configuration:

```bash
python datasets/download_assets.py --asset clip --output pretrained/clip-vit-base-patch16
```

The model safetensors revision and all hashes are in `datasets/pretrained_assets.json`; no moving model alias is used by the default configuration. CLIP is jointly fine-tuned, with no frozen-feature cache.

All eight retained MAF checkpoints are available in [the model release](https://github.com/decode12/MAF-PAA/releases/tag/reproducibility-v1). Their purposes, sizes, and download links are in [checkpoints/README.md](checkpoints/README.md); exact SHA-256 values are in `checkpoints/manifest.json`. Download and verify a selected model, or all eight:

```bash
python scripts/download_checkpoints.py --name main_bbc
# Or retrieve all eight retained models:
python scripts/download_checkpoints.py --name all
```

The downloader checks each file against the manifest and also verifies any existing local file before reusing it. Checkpoint exports contain identical original model tensors and portable metadata, without optimizer state or private paths. Their file hashes differ from the original training snapshots; both hashes are recorded. A successful download check establishes artifact integrity, not agreement of a new inference run with the historical predictions.

## Evaluate a fixed model

CPU smoke evaluation on the first 64 official BBC test records:

```bash
python scripts/evaluate.py --config configs/main.json \
  --checkpoint checkpoints/main_bbc.pt --split test --limit 64 \
  --output-dir outputs/bbc_smoke --device cpu
```

The output is explicitly marked `partial` and is not a paper result. For complete canonical evaluation, omit `--limit`, use CUDA BF16 and the default evaluation batch size 16:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py --config configs/main.json \
  --checkpoint checkpoints/main_bbc.pt --split test \
  --output-dir outputs/bbc_test --device cuda
# All four separate source models, sequentially on the selected device:
CUDA_VISIBLE_DEVICES=0 bash scripts/reproduce_main.sh evaluate cuda
```

The engine records checkpoint SHA-256, split, precision, sample count and fixed threshold 0.5 before inference. It refuses to overwrite completed output. Test outcomes never select checkpoints or adjust thresholds.

## Train and validate the main experiments

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/main.json \
  --source bbc --output-dir runs/main_bbc --device cuda
# Or train and then evaluate each of the four sources sequentially:
CUDA_VISIBLE_DEVICES=0 bash scripts/reproduce_main.sh train cuda
```

Training reads only train and validation records, starts with pretrained CLIP and new AdamW state, and retains the **first maximum validation ACC** checkpoint. No continuation from earlier experiments is used. The fixed protocol is 20 epochs, seed 42, batch 8, accumulation 4, head LR 1e-4, CLIP LR 5e-7, weight decay 1e-4, clipping 1.0 and BF16. There is no LR scheduler. `best.pt`, `last.pt`, configuration, validation history, sampler audit and completion record are written under the chosen run directory. Fresh training does not require published final MAF weights.

## Complete module ablations

```bash
# Train the three BBC variants and evaluate their validation-selected checkpoints:
CUDA_VISIBLE_DEVICES=0 bash scripts/reproduce_ablation.sh train cuda
# Or evaluate the three released variant weights:
CUDA_VISIBLE_DEVICES=0 bash scripts/reproduce_ablation.sh evaluate cuda
```

The configurations remove MGR, HAF or all DAFO objectives. They are complete module removals, not the older internal-component ablations. Table 3 uses BBC **test** records. Its full-model row is the same BBC model as Table 2. Selected epochs are 19, 15 and 10 for w/o MGR, w/o HAF and w/o DAFO, respectively. See the protocol for the precise replacement branches and retained objectives.

## Cross-domain evaluation (extension only)

The current paper reports source-specific experiments, not a cross-domain table. The following evaluates a fixed BBC-trained model on each source without retraining, threshold tuning or target-test selection:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/reproduce_cross_domain.sh bbc cuda
```

This extension has no paper reference values and was **NOT TESTED** as a full experiment during release preparation.

## Reproduce tables and analysis from frozen predictions

These commands need no dataset download, GPU or MAF checkpoint:

```bash
python scripts/summarize_results.py --output outputs/recomputed_results.csv
python scripts/reproduce_analysis.py --output-dir outputs/analysis --verify-paper --plot
```

The first recomputes every MAF row of Tables 2 and 3 from the saved full-split scores and checks the published rounding. It does not claim to rerun the model. Baseline rows are retained with their provenance; locally recorded HAMMER/HAMMER++/ALBEF and published FKA-Owl values are explicitly distinguished. Third-party baseline implementations/weights are not included or relicensed here, so complete baseline reexecution is not provided by this MAF-only release.

The analysis command reconstructs Figure 3 quartiles and the two Figure 4 prediction records. Their model is `analysis_bbc`, selected at epoch 4 of an earlier five-epoch BBC run. It is architecturally the same full MAF but is **not** the current Table 2 checkpoint. News photos and caption content must be obtained from the dataset; this repository does not redistribute them. Plot fonts may depend on installed fonts and do not change computed values.

For freshly computed predictions:

```bash
python scripts/summarize_results.py --predictions outputs/bbc_test/test_predictions.jsonl
```

## Expected results and numerical variation

| Source | ACC (%) | AUC (%) | EER (%) | Selected epoch |
| --- | ---: | ---: | ---: | ---: |
| BBC | 86.00 | 91.16 | 14.54 | 17 |
| The Guardian | 89.64 | 94.49 | 10.66 | 19 |
| USA TODAY | 84.28 | 89.78 | 16.60 | 19 |
| The Washington Post | 82.21 | 87.70 | 18.58 | 19 |

ACC uses probability >=0.5; AUC uses continuous manipulation probabilities; EER uses the interpolated ROC crossing of false-positive and false-negative rates. Saved-score recomputation must match these two-decimal results. CPU FP32 and CUDA BF16 are not interchangeable. The release smoke evaluations observed 3 differing classifications across 352 ordered test records and a maximum absolute probability difference of 0.886. This is not merely a claim of harmless threshold rounding; canonical full-split CUDA BF16 inference remains unverified by the release tests. See `results/validation/cpu_inference.json`. We do not invent a universal tolerance or a multi-seed standard deviation: current final results are single-seed, and full fresh retraining was not repeated during release preparation. Report observed differences with precision and sample counts; do not optimize test thresholds to remove them.

## Reproducibility limitations

- All retained MAF checkpoints are publicly downloadable and checksum-verified. New full-split CUDA BF16 agreement with the archived predictions remains unresolved; public availability does not establish numerical equivalence.
- Current results are seed 42. Existing historical seeds 43/44 used a different five-epoch protocol and are excluded.
- Figures 3/4 use the separately identified earlier model, not the current main model.
- Earlier development accessed the designated test split; complete project-wide test independence cannot be established retrospectively.
- Original final-source content hashes are preserved, but that experiment directory had no verified final Git commit. This repository's commit identifies the portable release.
- Full GPU retraining, new full-split GPU inference, full cross-domain execution and baseline reexecution were not run during release preparation. See the release report for the CPU tests actually performed.

## Citation

The paper is submitted; no accepted venue, DOI or publication year is invented.

```bibtex
@misc{lin_maf,
  title = {Multi-Granular Agreement Fusion for Multimodal Misinformation Detection},
  author = {Lin, Jiajie and Yu, Yi and Yang, Zhenguo and Wu, Jianming},
  note = {Manuscript submitted to Pattern Analysis and Applications},
  howpublished = {\url{https://github.com/decode12/MAF-PAA}}
}
```

## License

Original MAF code is provided under the [MIT License](LICENSE), as authorized by the authors. Dataset content, pretrained models and third-party implementations retain their respective terms; see [third-party notices](docs/THIRD_PARTY.md). MIT does not grant new rights over the news images or captions.
