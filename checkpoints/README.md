# Final checkpoint availability

All eight model-only checkpoints are publicly available in [GitHub Releases](https://github.com/decode12/MAF-PAA/releases/tag/reproducibility-v1). Each export was checked against the original model tensors, and each complete anonymous download was verified against the SHA-256 recorded in `manifest.json`. Download verification is documented in [`public_checkpoint_downloads.json`](../results/validation/public_checkpoint_downloads.json). Fresh training and saved-prediction metric reproduction do not require the trained checkpoints.

Each artifact contains `model`, selected epoch, original-checkpoint hash, and a small portable configuration. No optimizer state, training path, caption, image or token is included. File hashes change because serialization metadata are removed; model values remain identical. Original snapshot hashes and public-export hashes are distinct fields in the manifest.

| Artifact | Purpose | Selected epoch | Size (MiB) | Public download |
| --- | --- | ---: | ---: | --- |
| `main_bbc.pt` | Tables 2/3 and Figures 3/4 | 17 | 593.8 | [Download](https://github.com/decode12/MAF-PAA/releases/download/reproducibility-v1/main_bbc.pt) |
| `main_guardian.pt` | Main table | 19 | 593.8 | [Download](https://github.com/decode12/MAF-PAA/releases/download/reproducibility-v1/main_guardian.pt) |
| `main_usa_today.pt` | Main table | 19 | 593.8 | [Download](https://github.com/decode12/MAF-PAA/releases/download/reproducibility-v1/main_usa_today.pt) |
| `main_washington_post.pt` | Main table | 19 | 593.8 | [Download](https://github.com/decode12/MAF-PAA/releases/download/reproducibility-v1/main_washington_post.pt) |
| `bbc_without_mgr_complete.pt` | BBC complete-module ablation | 19 | 577.0 | [Download](https://github.com/decode12/MAF-PAA/releases/download/reproducibility-v1/bbc_without_mgr_complete.pt) |
| `bbc_without_haf_complete.pt` | BBC complete-module ablation | 15 | 593.3 | [Download](https://github.com/decode12/MAF-PAA/releases/download/reproducibility-v1/bbc_without_haf_complete.pt) |
| `bbc_without_dafo_complete.pt` | BBC complete-module ablation | 10 | 593.8 | [Download](https://github.com/decode12/MAF-PAA/releases/download/reproducibility-v1/bbc_without_dafo_complete.pt) |
| `analysis_bbc.pt` | Historical development only (five epochs) | 4 | 593.8 | [Download](https://github.com/decode12/MAF-PAA/releases/download/reproducibility-v1/analysis_bbc.pt) |

The underlying CLIP ViT-B/16 assets are publicly downloadable using `datasets/download_assets.py --asset clip`; their versions, licensing notes and SHA-256 values are documented separately. These initialization/tokenizer assets and the trained MAF checkpoints are separate dependencies.

Download from the repository root:

```bash
python scripts/download_checkpoints.py --name main_bbc
# Download the seven current-paper checkpoints:
python scripts/download_checkpoints.py --name paper
# The legacy model remains available explicitly as --name analysis_bbc.
```

Model binaries are release assets and remain ignored by Git; they are not packed into the source history. No Git LFS setup is needed. The downloader verifies the manifest SHA-256 before accepting a downloaded or existing file.

Full CUDA BF16 reevaluation reproduced all four main-test score files exactly. CPU FP32 may differ. For executed tests and precision limitations, see the [reproducibility protocol](../docs/REPRODUCIBILITY.md).
