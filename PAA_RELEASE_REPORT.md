# PAA reproducibility release report

Prepared 2026-10-14 for *Multi-Granular Agreement Fusion for Multimodal Misinformation Detection*.

## Publication and completion status

Target: https://github.com/decode12/MAF-PAA

Base repository commit: `dd0bf746b027daf170c752f8688aee9b755f9d95`.
Release branch: `release/paa-reproducibility`.
Verified source-release commit: `f2cde8ffcc5ecc2478b39b097dc23f143cacf571`. Both `main` and `release/paa-reproducibility` were advanced normally from the base commit. No pre-existing history was removed or force-pushed. Later report-only commits document these completed checks.

Anonymous HTTPS access to README, model source and a prediction archive succeeded with byte-identical SHA-256 hashes. An unauthenticated fresh clone of `main` retrieved the complete release. Its Git tree equals the audited local tree `6d32ea61c7959628a3a486b8cc96b5a49ded79f4`. The table and analysis reproduction commands both passed when executed from that fresh clone in the isolated environment. The initial source release contained 57 tracked files (about 3.4 MB), including the eight authorized compressed prediction files. The current file inventory is maintained in `docs/RELEASE_FILES.txt`.

Automatic publication review initially blocked eight per-record prediction archives. The author subsequently explicitly authorized publishing these eight sanitized archives (about 3.1 MB, containing record IDs, labels and derived model scores; no image/text content or private paths). This blocker is resolved; no rejected upload was bypassed.

**Status: source and all eight retained MAF checkpoints are publicly available and anonymously verified; complete numerical reproduction remains unresolved.** On 2026-10-14, the model-only exports were published as assets of [release `reproducibility-v1`](https://github.com/decode12/MAF-PAA/releases/tag/reproducibility-v1). Every complete anonymous asset download matched its exported SHA-256. The checkpoint manifest records public URLs, byte sizes, exported hashes and original snapshot hashes. The completed download checks are recorded in `results/validation/public_checkpoint_downloads.json`. The earlier public-download blocker has been resolved; model binaries remain outside Git history.

The remaining unresolved item is numerical agreement between fresh CPU inference and the archived CUDA BF16 predictions. Original and cleaned implementations agree exactly on the checked CPU inputs, but the CPU outputs do not always agree with the archived GPU outputs. No GPU was used during release validation because both devices were occupied. Publishing and downloading the weights does not resolve this discrepancy.

## Files and exclusions

Published source consists of `maf/`, `configs/`, `datasets/`, `scripts/`, pinned requirements, MIT license, documentation, checkpoint checksums/status, and compact audited experimental outputs under `results/`. `docs/RELEASE_FILES.txt` lists the actual source file set. Source is directly browsable; the compressed JSONL files contain derived predictions, not an archive-only code release.

Excluded: historical architectures and internal search variants not in the final table, diffusion/evidence attempts, feature pilots, autoresearch controllers, notification/tmux/mail code, unrelated IJCAI/TMM projects, drafts, raw datasets, large logs, intermediate checkpoints and optimizer snapshots. The earlier BBC analysis model is retained as an explicit dependency of Figures 3 and 4. Original experiments and manuscript scientific results were not overwritten.

## Provenance conclusions

| Result | Model asset | Selected epoch | ACC / AUC / EER (%) |
| --- | --- | ---: | --- |
| Table 2 BBC | `main_bbc` | 17 / 20 | 86.00 / 91.16 / 14.54 |
| Table 2 Guardian | `main_guardian` | 19 / 20 | 89.64 / 94.49 / 10.66 |
| Table 2 USA TODAY | `main_usa_today` | 19 / 20 | 84.28 / 89.78 / 16.60 |
| Table 2 Washington Post | `main_washington_post` | 19 / 20 | 82.21 / 87.70 / 18.58 |
| Table 3 Full | `main_bbc` | 17 / 20 | 86.00 / 91.16 / 14.54 |
| Table 3 w/o MGR | `bbc_without_mgr_complete` | 19 / 20 | 85.67 / 91.84 / 14.73 |
| Table 3 w/o HAF | `bbc_without_haf_complete` | 15 / 20 | 85.61 / 86.90 / 16.98 |
| Table 3 w/o DAFO | `bbc_without_dafo_complete` | 10 / 20 | 85.47 / 91.89 / 15.40 |

All these rows use seed 42, first maximum validation ACC and test threshold 0.5. The four principal runs are fresh 20-epoch runs with new AdamW state. Original checkpoint hashes, complete epoch selections and raw metric precision are in `results/provenance.json` and `results/selection_history.csv`.

Current seeds 43/44 cannot be supplied: only historical five-epoch versions exist. Current source code is identified by its frozen SHA-256; an original final-source Git commit was not found and is not invented. The current manuscript has no cross-domain table. The provided cross-domain command is explicitly an extension.

Figures 3/4 depend on the separate `analysis_bbc` checkpoint selected at epoch 4 of five epochs. All Figure 3 group counts and numeric values and both Figure 4 prediction records were independently reconstructed. These are not attributed to the 20-epoch BBC model.

## Data and pretrained access

Official DGM4 access is documented in `docs/DATA.md`; the downloader pins the official repository revision and verifies hashes. All 281,015 original metadata records were checked, including existing image paths. All 83,229 used news IDs can be assigned using official original-image paths, and the derived mapping exactly matches the original VisualNews association. Original split ID intersections are empty; repeated variants within a split remain intact. No raw image, caption or article is redistributed.

The exact CLIP ViT-B/16 safetensors revision, tokenizer/config revisions, licenses and hashes are in `datasets/pretrained_assets.json` and `docs/THIRD_PARTY.md`. Independently downloaded official tokenizer/configuration files exactly matched the existing preprocessing on 32 real captions. All official image-archive directory layouts were checked with small HTTP range reads. Full multi-GB redownload was **NOT TESTED** because existing original assets were already available and checksum-audited.

Eight retained model-only assets, about 5 GB in total, are publicly available through [GitHub Releases](https://github.com/decode12/MAF-PAA/releases/tag/reproducibility-v1). Every model tensor was checked against the original checkpoint. Private arguments and optimizer state were removed. `checkpoints/manifest.json` distinguishes original snapshot SHA-256 from export SHA-256 and provides the verified anonymous download URLs. Full-file anonymous download and SHA-256 verification passed for all eight exports on 2026-10-14. No paid LFS allocation or inaccessible private URL is substituted.

## Executed validation

| Check | Result | Scope |
| --- | --- | --- |
| Isolated environment installation/imports | PASS | Python 3.11.14, CPU PyTorch 2.5.1; `pip check` clean |
| Actual data preparation and label equivalence | PASS | All 281,015 records and image-path checks |
| Official tokenizer compatibility | PASS | 32 real captions; IDs, masks, offsets and config equal |
| Cleaned/original architecture parity | PASS | Four variants; initialization, RNG, outputs, loss and synthetic-test gradients |
| Real-data parity against original implementation | PASS | Four real BBC records per variant; strict checkpoint load, train/eval outputs and loss equal |
| Annotation-independent predictions | PASS | Ground-truth labels/targets changed without changing predictions; tokenizer position masks preserved |
| New real main-checkpoint inference | PASS, partial only | First 64 test records for each of four sources, CPU FP32 |
| New real ablation-checkpoint inference | PASS, partial only | First 32 BBC test records for each of three removals, CPU FP32 |
| Minimal training pipeline | PASS | One real batch of 8 train records, 16 validation records; optimizer step, selected checkpoint save/reload |
| Saved-score table reproduction | PASS | All seven distinct full test prediction files, eight MAF table rows |
| Saved-score Figures 3/4 reproduction | PASS | 9,370 BBC predictions, all quartiles, both case records |
| CLI/config/shell consistency | PASS | Help/import/config loading and shell syntax verified |
| Secret/private-path/English-comment scan | PASS | Source and decompressed published prediction records |
| Anonymous public clone and file verification | PASS | Source release commit above; model source, README and prediction archive SHA-256 checks |
| Reproduction from the anonymous clone | PASS | Eight MAF table rows and Figures 3/4 saved-score analysis |
| New full-split CUDA BF16 inference | NOT TESTED | Both GPUs occupied; not claimed equivalent to CPU |
| Full 20-epoch retraining | NOT TESTED | No long training launched |
| Cross-domain extension | NOT TESTED | No corresponding result table in current paper |
| Baseline reexecution | NOT PROVIDED / NOT TESTED | Original methods retain their own licenses |
| Anonymous final-model download | PASS | All eight complete public release assets; anonymous access, byte size and exported SHA-256 verified |

Machine-readable evidence is in `results/validation/`. Local smoke-test paths were supplied as CLI overrides; the commands use the same public entry points documented in README.

Commands actually exercised include:

```bash
python -m pip install -r requirements.txt
python -m pip check
python datasets/prepare_dgm4.py --data-root data/raw/DGM4 --derive-source-metadata --output data/prepared --check-images
python scripts/train.py --help
python scripts/evaluate.py --help
python scripts/evaluate.py --config configs/main.json --source bbc --checkpoint checkpoints/main_bbc.pt --split test --limit 64 --batch-size 8 --device cpu --output-dir outputs/validation/main_bbc
python scripts/summarize_results.py --output outputs/recomputed_results.csv
python scripts/reproduce_analysis.py --output-dir outputs/analysis --verify-paper --plot
python scripts/download_checkpoints.py --name main_bbc
```

The last command in the original source-release validation verified an already present local artifact. The subsequent public-release checks on 2026-10-14 independently downloaded all eight complete assets without authentication and verified their SHA-256 values; see `results/validation/public_checkpoint_downloads.json`. This distinction preserves the original test history. Other sources and all three ablation configuration/checkpoint pairs were similarly evaluated on CPU. The one-batch train test used a separate configuration with `epochs=1`, `limit_train=8` and `limit_eval=16`, and was never added to the paper's result set.

## Numerical comparison and unresolved discrepancy

Saved full-test probabilities reproduce the paper's ACC/AUC/EER exactly to two decimals for every MAF row. That is a successful artifact/metric audit, not new full inference.

Across the 352 new CPU FP32 evaluations there were three classification differences against the corresponding archived CUDA BF16 records. Maximum absolute probability differences reached 0.886 on Guardian, 0.412 on USA TODAY and 0.313 for w/o DAFO. These differences are **not** dismissed as harmless rounding around 0.5. The same input tensors, original checkpoint tensors and original versus public source were independently checked on eight Guardian examples: original and cleaned CPU FP32 outputs were identical, excluding the cleanup and input ordering as the explanation for those checked samples. Explicit CPU BF16 also failed to reproduce the archived CUDA results and can overflow the state-logit computation; it is not enabled by the release CPU engine. A canonical CUDA BF16 rerun is still required to resolve the remaining execution-mode discrepancy. No metric tolerance or result replacement was invented.

The exact sample-limited metrics are in `results/validation/cpu_inference.json`. They are not substituted for full-test paper numbers.

## PAA requirements assessment

- Public readable source, configurations, official data-access and preprocessing instructions: **PASS**, published and verified through an anonymous fresh clone.
- English source comments and README: PASS.
- Minimal final-method code with necessary checkpoint compatibility: PASS.
- Public final-checkpoint access and integrity: **PASS**, all eight complete anonymous downloads match the manifest.
- Complete numerical reproduction from new inference: **NOT YET ESTABLISHED**; canonical full-split CUDA BF16 validation remains required.
- Universal reproduction of every compared baseline: **NOT CLAIMED**; published FKA-Owl and locally adapted baselines are distinguished in provenance.
- Independent researchers can download the retained model weights and run the documented evaluation commands. A claim that new inference has reproduced all reported numbers is **NOT SUPPORTED** until canonical numeric validation is complete.

## Manuscript declarations

The current manuscript identifies the public source repository and documents access to data, preprocessing, pretrained initialization, retraining, evaluation and recorded predictions. Its wording does not promise code only upon publication or claim that all model outputs have been independently reproduced. The statements below reflect that current wording. Publishing the weights does not require changing the manuscript or its prepared submission archive.

The earlier source-release declaration edit passed two LaTeX compilation passes: 23 pages rendered, no errors and no overfull boxes. This records that earlier verification; no new manuscript edit or compilation is attributed to the checkpoint upload. Scientific results were not edited.

**Code availability**

The MAF source code, experimental configurations, and scripts for data preparation, training, and evaluation are publicly available at https://github.com/decode12/MAF-PAA. The repository README provides environment setup instructions, links to the dataset and the pretrained initialization model, and commands for retraining and evaluating the source-specific models and module ablations. Recorded prediction outputs supporting the reported MAF results are also provided in this repository.

**Data availability**

The experiments in this study used the publicly available DGM4 dataset, accessible through the original authors' repository at https://github.com/rshaojimmy/MultiModal-DeepFake. Dataset download instructions, preprocessing scripts, and procedures for reconstructing the source-specific training, validation, and test partitions are provided in the MAF repository at https://github.com/decode12/MAF-PAA. Dataset access and use are subject to the original providers' terms. No new dataset was collected or created in this study.

Retain the manuscript's formal DGM4 citation when inserting this text. Public artifact availability and new-inference numerical agreement are separate checks; do not claim complete numerical reproduction while the recorded discrepancy remains unresolved.
