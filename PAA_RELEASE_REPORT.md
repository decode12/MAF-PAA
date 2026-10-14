# PAA reproducibility release report

Prepared 2026-10-14 for *Multi-Granular Agreement Fusion for Multimodal Misinformation Detection*.

## Publication and completion status

Target: https://github.com/decode12/MAF-PAA

Base repository commit: `dd0bf746b027daf170c752f8688aee9b755f9d95`.
Release branch: `release/paa-reproducibility`.
The final publication commit is recorded after anonymous remote verification. No pre-existing history is removed or force-pushed.

Automatic publication review initially blocked eight per-record prediction archives. The author subsequently explicitly authorized publishing these eight sanitized archives (about 3.1 MB, containing record IDs, labels and derived model scores; no image/text content or private paths). This blocker is resolved; no rejected upload was bypassed.

**Status: source release prepared; complete checkpoint-based independent reproduction remains incomplete.** The repository is publicly accessible and the connected account has push permission. Final MAF weights have been exported locally, but there is no configured local GitHub Releases or model-host upload authentication. They are not represented as already public. The connected source-control tools can publish source files but do not expose a large-release-asset upload operation.

A second unresolved item is numerical agreement between fresh CPU inference and the archived CUDA BF16 predictions. Original and cleaned implementations agree exactly on the checked CPU inputs, but the CPU outputs do not always agree with the archived GPU outputs. No GPU was used because both devices were occupied.

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

Eight final model-only assets, about 5 GB in total, are available in the maintainer's local release directory. Every model tensor was checked against the original checkpoint. Private arguments and optimizer state were removed. `checkpoints/manifest.json` distinguishes original snapshot SHA-256 from export SHA-256. Public asset URLs remain null until upload and anonymous verification succeed. No paid LFS allocation or inaccessible private URL is substituted.

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
| New full-split CUDA BF16 inference | NOT TESTED | Both GPUs occupied; not claimed equivalent to CPU |
| Full 20-epoch retraining | NOT TESTED | No long training launched |
| Cross-domain extension | NOT TESTED | No corresponding result table in current paper |
| Baseline reexecution | NOT PROVIDED / NOT TESTED | Original methods retain their own licenses |
| Anonymous final-model download | BLOCKED | Weights not uploaded |

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

The last command verified the already present local artifact; it did **not** demonstrate a public final-weight download. Other sources and all three ablation configuration/checkpoint pairs were similarly evaluated. The one-batch train test used a separate configuration with `epochs=1`, `limit_train=8` and `limit_eval=16`, and was never added to the paper's result set.

## Numerical comparison and unresolved discrepancy

Saved full-test probabilities reproduce the paper's ACC/AUC/EER exactly to two decimals for every MAF row. That is a successful artifact/metric audit, not new full inference.

Across the 352 new CPU FP32 evaluations there were three classification differences against the corresponding archived CUDA BF16 records. Maximum absolute probability differences reached 0.886 on Guardian, 0.412 on USA TODAY and 0.313 for w/o DAFO. These differences are **not** dismissed as harmless rounding around 0.5. The same input tensors, original checkpoint tensors and original versus public source were independently checked on eight Guardian examples: original and cleaned CPU FP32 outputs were identical, excluding the cleanup and input ordering as the explanation for those checked samples. Explicit CPU BF16 also failed to reproduce the archived CUDA results and can overflow the state-logit computation; it is not enabled by the release CPU engine. A canonical CUDA BF16 rerun is still required to resolve the remaining execution-mode discrepancy. No metric tolerance or result replacement was invented.

The exact sample-limited metrics are in `results/validation/cpu_inference.json`. They are not substituted for full-test paper numbers.

## PAA requirements assessment

- Public readable source, configurations, official data-access and preprocessing instructions: prepared and validated; remote publication is recorded below after verification.
- English source comments and README: PASS.
- Minimal final-method code with necessary checkpoint compatibility: PASS.
- Complete public checkpoint-based reproduction: **NOT YET SATISFIED**.
- Universal reproduction of every compared baseline: **NOT CLAIMED**; published FKA-Owl and locally adapted baselines are distinguished in provenance.
- Paper-ready claim that independent researchers can already download final weights and reproduce all numbers: **NOT SUPPORTED** until upload and canonical numeric validation are complete.

## Suggested manuscript declarations

Use the following only after the source release is publicly visible. The weight limitation is deliberately explicit.

**Code availability**

The source code, experimental configurations, data-preparation scripts, and recorded prediction outputs are publicly available at https://github.com/decode12/MAF-PAA. The repository documents the reproduction protocol and its current limitations. Public download links for the trained model checkpoints are not yet available.

**Data availability**

The experiments in this study used the publicly available DGM4 dataset, accessible through the original authors' repository at https://github.com/rshaojimmy/MultiModal-DeepFake. The code repository provides instructions and scripts for reconstructing the original source-specific partitions. Dataset access and use are subject to the original providers' terms. No new dataset was collected or created in this study.

Retain the manuscript's formal DGM4 citation when inserting this text. Do not state that complete checkpoint-based reproducibility has been achieved while the listed blockers remain.
