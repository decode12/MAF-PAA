# Data access and preprocessing

## Official providers and terms

- **DGM4:** [original project](https://github.com/rshaojimmy/MultiModal-DeepFake) and [dataset download](https://huggingface.co/datasets/rshaojimmy/DGM4), pinned to revision `9efd3f1ca14cb1dac1692cdce0beea6844b968a6`. The dataset card declares Apache-2.0; the associated implementation has a distinct [S-Lab non-commercial license](https://github.com/rshaojimmy/MultiModal-DeepFake/blob/main/LICENSE).
- **VisualNews:** [original repository](https://github.com/FuxiaoLiu/VisualNews-Repository). Its news ID/source mapping was used by the experiments. The equivalent reconstruction below avoids downloading the full VisualNews archive. No explicit news-content redistribution license was found in the inspected repository.
- **CLIP ViT-B/16:** [OpenAI model](https://huggingface.co/openai/clip-vit-base-patch16), with exact files and revisions in [pretrained_assets.json](../datasets/pretrained_assets.json).

Provider links were checked on 2026-10-14. Access and use remain subject to provider terms and underlying publishers' rights. Raw images, captions and articles are not redistributed or relicensed by this repository. See [third-party notices](THIRD_PARTY.md).

## Download and prepare

Run from the repository root:

```bash
python datasets/download_assets.py --asset dgm4 --output data/raw/DGM4 --extract
python datasets/prepare_dgm4.py --data-root data/raw/DGM4 \
  --derive-source-metadata --output data/prepared --check-images
python datasets/download_assets.py --asset clip --output pretrained/clip-vit-base-patch16
```

The downloader checks pinned SHA-256 values and file sizes, reuses verified files and rejects conflicts. DGM4 archives require about 10.7 GB before extraction. `--metadata-only` supports partition auditing but is insufficient for inference or training.

```text
data/raw/DGM4/
  metadata/{train,val,test}.json
  origin/{bbc,guardian,usa_today,washington_post}/...
  manipulation/{HFGI,StyleCLIP,infoswap,simswap}/...
data/prepared/
  source_by_id.jsonl
  split_audit.json
pretrained/clip-vit-base-patch16/
  model.safetensors
  ... tokenizer and configuration files
```

Preserve official image paths and capitalization. The loader resolves the `DGM4/` prefix relative to the configured data root. CLI options `--data-root`, `--source-metadata` and `--clip-model` support alternative local locations.

## Exact partitions

Source reconstruction uses original-image paths `DGM4/origin/<source>/...`, aggregated by news ID. All 83,229 used unique IDs are covered, and every assignment was verified against the original VisualNews mapping. Conflicts, missing assignments or mismatching fingerprints stop preparation. No random split, deduplication, caption correction, image replacement or record reordering is performed.

| Source | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| BBC | 40,811 | 4,424 | 9,370 |
| The Guardian | 109,703 | 11,685 | 27,028 |
| USA TODAY | 31,811 | 2,917 | 7,926 |
| The Washington Post | 25,859 | 3,100 | 6,381 |
| Total | 208,184 | 22,126 | 50,705 |

[dgm4_assets.json](../datasets/dgm4_assets.json) pins official downloads and metadata hashes. [expected_splits.json](../datasets/expected_splits.json) records source-map and ordered-record fingerprints, split counts, and binary/four-state counts. News IDs are disjoint across splits; repeated variants within a split are retained. The loader filters original metadata directly. Optional `--write-subsets` and `--write-id-manifests` generate local files under the ignored data directory.

If official VisualNews metadata is already available, replace `--derive-source-metadata` with `--source-metadata data/raw/VisualNews/data.json`; the same mapping fingerprints must pass. Full articles are not required or used as model inputs.

## Labels and inputs

- Binary labels: **0=Real, 1=Manipulated**. State labels: **0=Real, 1=Text-only, 2=Image-only, 3=Joint**. The submitted loader uses `fake_cls`, `fake_text_pos` and valid `fake_image_box` annotations, including its recorded fallbacks.
- Images: RGB, direct bicubic resize to **224 × 224**, no center crop or random augmentation. CLIP normalization uses mean `(0.48145466, 0.4578275, 0.40821073)` and standard deviation `(0.26862954, 0.26130258, 0.27577711)`.
- Text: matching fast CLIP tokenizer, truncation and fixed padding to **77 tokens**. Zero-based whitespace-word annotations map to subwords by character-offset overlap; padding and special tokens are excluded from local supervision.
- Visual targets: a patch in the **14 × 14** grid is positive when its center lies within the resized manipulation bounding box.

Annotations provide training supervision and evaluation labels, not inference features. The inference token-validity mask comes from tokenizer positions, not manipulation annotations. Input captions are the provided, potentially manipulated captions; original captions and retrieved evidence are not used for inference.

## Pretrained initialization

CLIP tokenizer/configuration files are pinned to `57c216476eefef5ab752ec549e440a49ae4ae5f3`; the audited `model.safetensors` uses public conversion revision `fa726d97eb8b97497b912f1b8035b95fb86eb292` (`refs/pr/7`). All URLs, sizes and checksums are in [pretrained_assets.json](../datasets/pretrained_assets.json). The downloader does not resolve a moving model alias. These initialization/tokenizer assets are needed separately from the [trained MAF models](../checkpoints/README.md).
