# Final checkpoint availability

Eight model-only checkpoints have been exported locally and verified against the original tensors. **They have not yet been uploaded to a public model host.** A null `url` in `manifest.json` means the download is unavailable, not an undocumented private dependency. Public inference reproduction remains blocked until uploads are completed. Fresh training and saved-prediction metric reproduction are available now.

Each artifact contains `model`, selected epoch, original-checkpoint hash, and a small portable configuration. No optimizer state, training path, caption, image or token is included. File hashes change because serialization metadata are removed; model values remain identical. Original snapshot hashes and public-export hashes are distinct fields in the manifest.

| Artifact | Purpose | Selected epoch | Size (MiB) | Public download |
| --- | --- | ---: | ---: | --- |
| `main_bbc.pt` | Main table | 17 | 593.8 | Not uploaded |
| `main_guardian.pt` | Main table | 19 | 593.8 | Not uploaded |
| `main_usa_today.pt` | Main table | 19 | 593.8 | Not uploaded |
| `main_washington_post.pt` | Main table | 19 | 593.8 | Not uploaded |
| `bbc_without_mgr_complete.pt` | BBC complete-module ablation | 19 | 577.0 | Not uploaded |
| `bbc_without_haf_complete.pt` | BBC complete-module ablation | 15 | 593.3 | Not uploaded |
| `bbc_without_dafo_complete.pt` | BBC complete-module ablation | 10 | 593.8 | Not uploaded |
| `analysis_bbc.pt` | Figures 3/4 (earlier five-epoch model) | 4 | 593.8 | Not uploaded |

The underlying CLIP ViT-B/16 assets are publicly downloadable using `datasets/download_assets.py --asset clip`; their versions, licensing notes and SHA-256 values are documented separately. The absence of final MAF downloads must not be confused with pretrained CLIP availability.

The intended host is GitHub Releases, whose per-file capacity is sufficient for these approximately 0.6 GB files. Model files are ignored by Git and will not be packed into the source history. A maintainer must upload the verified assets, verify their anonymous download URLs and checksums, and only then populate manifest URLs. Git LFS is not silently enabled, and no paid storage allocation is assumed.
