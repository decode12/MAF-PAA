# Data access and split reconstruction

The experiments use the original DGM4 train/validation/test metadata, filtered by news source without shuffling, resplitting, deduplication, caption correction, or image replacement. No dataset images, captions, articles, or per-record ID manifests are distributed in this repository.

## Official sources and terms

- [DGM4 official project and preparation instructions](https://github.com/rshaojimmy/MultiModal-DeepFake).
- [DGM4 download repository](https://huggingface.co/datasets/rshaojimmy/DGM4), pinned to revision `9efd3f1ca14cb1dac1692cdce0beea6844b968a6`. The dataset card declares Apache-2.0. The associated implementation has a separate [S-Lab non-commercial code license](https://github.com/rshaojimmy/MultiModal-DeepFake/blob/main/LICENSE); these are distinct notices.
- [VisualNews original repository](https://github.com/FuxiaoLiu/VisualNews-Repository) and its [official download directory](https://www.cs.rice.edu/~vo9/visualnews/). VisualNews provides the news ID/source association. No explicit redistribution license for the news content was found in the inspected repository. Dataset access does not transfer the underlying publishers' rights. Obtain assets from their original providers and observe their terms.

These sources were checked on 2026-10-14. Data access and licensing remain controlled by the original providers. The release does not relicense third-party content.

## Download and extract DGM4

Run from the repository root. The downloader uses Python's standard library, pins each URL, and checks every file's SHA-256 and byte count. Full images and metadata require approximately 10.7 GB of downloads; allow additional space for extraction.

```bash
python datasets/download_assets.py --asset dgm4 --output data/raw/DGM4 --extract
```

Downloads come directly from the official Hugging Face repository. The expected archive and metadata checksums are in `datasets/dgm4_assets.json`. Existing files with matching checksums are reused; a conflicting file stops the download. A failed download may leave a `.partial` file, which is overwritten on the next attempt. Metadata alone can be downloaded for split auditing with `--metadata-only`; it is insufficient for inference or training.

The resulting structure is:

```text
data/raw/DGM4/
  metadata/train.json
  metadata/val.json
  metadata/test.json
  origin/bbc/...
  origin/guardian/...
  origin/usa_today/...
  origin/washington_post/...
  manipulation/HFGI/...
  manipulation/StyleCLIP/...
  manipulation/infoswap/...
  manipulation/simswap/...
```

Preserve capitalization and the official image paths. Dataset records contain paths beginning with `DGM4/`; the loader accepts `--data-root data/raw/DGM4` and resolves that prefix without editing the records.

## Reconstruct the exact source mapping

Training used VisualNews `data.json` to map each original news ID to its source. An equivalent reconstruction is possible from the original-image paths already present in the official DGM4 metadata: every one of the 83,229 news IDs used across the three splits has at least one `DGM4/origin/<source>/...` record. This was checked against the actual VisualNews map for every used ID; the mappings are identical. No manipulation labels or predictions are used to choose a source.

The explicit reconstruction option avoids downloading the approximately 91 GB VisualNews `origin.tar` just to obtain its metadata:

```bash
python datasets/prepare_dgm4.py \
  --data-root data/raw/DGM4 \
  --derive-source-metadata \
  --output data/prepared \
  --check-images
```

The command produces `data/prepared/source_by_id.jsonl` containing only local `id` and `source` records, plus `data/prepared/split_audit.json`. Pass the generated `source_by_id.jsonl` as the training/evaluation `--source-metadata` argument. The mapping fingerprint must match `e6f00c11e09e2d18e0d5873a779ebb704fb687704a6f8fb0252622f8b59aff9b`; otherwise preparation fails. This is a verified reconstruction of the original assignment, not a new split.

Alternatively, if the original VisualNews metadata is already available, use:

```bash
python datasets/prepare_dgm4.py \
  --data-root data/raw/DGM4 \
  --source-metadata data/raw/VisualNews/data.json \
  --output data/prepared \
  --check-images
```

The VisualNews [official notebook](https://github.com/FuxiaoLiu/VisualNews-Repository/blob/master/VisualNews-Dataset.ipynb) reads `origin/data.json` from the original release. The original file audited for this paper has SHA-256 `642a28a0065e3a14d6ba74c23ec2b516710086a4df8e6b204d44095ff94f5caf`. A reduced or differently formatted source-map file is accepted only when the mapping of all used IDs and the resulting ordered records exactly match the recorded fingerprints. Full VisualNews articles are not model inputs and are not required.

## Fixed partitions and fingerprints

| Source | Train | Validation | Test |
|---|---:|---:|---:|
| BBC | 40,811 | 4,424 | 9,370 |
| The Guardian | 109,703 | 11,685 | 27,028 |
| USA TODAY | 31,811 | 2,917 | 7,926 |
| The Washington Post | 25,859 | 3,100 | 6,381 |
| Total | 208,184 | 22,126 | 50,705 |

The metadata byte fingerprints also match the official pinned Hugging Face LFS objects:

| File | SHA-256 |
|---|---|
| `metadata/train.json` | `dafd4fe101d3de63b433759fae79b14b465294f7be73f26c9f3bb13b8de87704` |
| `metadata/val.json` | `d4377ecc19cb6528e36c77401482cbfffaa57977faee2c6124e3b1bdda045520` |
| `metadata/test.json` | `b58b29c4cf0f40cdf791189181ad57821f9359e358e93683cc702db3a41c0a18` |

`datasets/expected_splits.json` records counts, four-state counts, binary counts, ordered ID hashes, and ordered canonical-record hashes for all 12 source/split combinations. Original news IDs do not overlap across train, validation, and test. Multiple manipulated variants with the same ID within a split are retained.

The preparation script does not write source-filtered caption files by default: the loader filters the original JSON records directly. Optional `--write-subsets` and `--write-id-manifests` flags generate local JSONL records and ordered ID lists. These outputs must remain under ignored local data directories and are not part of the public release. A mismatching dataset stops verification; there is no automatic fallback to another partition.

## Labels and preprocessing

The state order is `0=Real`, `1=Text-only`, `2=Image-only`, `3=Joint`; the binary target is `0=Real`, `1=Manipulated`. `fake_cls`, `fake_text_pos`, and a valid `fake_image_box` jointly determine these labels, following the submitted loader's fallbacks. Text modification indices refer to zero-based whitespace words. They are mapped to CLIP subwords through character-offset overlap, with padding and special tokens excluded from local supervision.

Images are converted to RGB, resized directly to 224 by 224 pixels with bicubic interpolation, and normalized using CLIP mean `(0.48145466, 0.4578275, 0.40821073)` and standard deviation `(0.26862954, 0.26130258, 0.27577711)`. There is no center crop or random augmentation in this pipeline. CLIP ViT-B/16 yields a 14 by 14 patch grid: a patch is positive when its center is inside the resized manipulation bounding box. Tokenization uses maximum length 77, truncation, and fixed padding. No external-evidence retrieval, article loading, feature cache, or frozen-feature extraction step is required by the final MAF pipeline.

## Pretrained CLIP

```bash
python datasets/download_assets.py --asset clip --output pretrained/clip-vit-base-patch16
```

The exact model is [OpenAI CLIP ViT-B/16](https://huggingface.co/openai/clip-vit-base-patch16). Tokenizer/configuration files are pinned to `57c216476eefef5ab752ec549e440a49ae4ae5f3`. The audited `model.safetensors` comes from the public conversion at revision `fa726d97eb8b97497b912f1b8035b95fb86eb292` (`refs/pr/7`), with SHA-256 `846bf12e84b91c2a65b44833a915241443b39356f1000ac1345c61d7bf8a209e`. These exact files, sizes, pinned URLs, and checksums are in `datasets/pretrained_assets.json`. No model alias silently selects a newer checkpoint.

The official main revision also provides `pytorch_model.bin` with SHA-256 `ec89c7b09c749a60aae3c9cd910516f24b58214a7df060b48962d14c469cfbf0`, but the release downloader deliberately uses the audited safetensors artifact. See [third-party notices](THIRD_PARTY.md).
