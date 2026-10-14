# Reproducibility protocol

This document describes the experiments that generated the current MAF manuscript. `results/provenance.json` maps each retained result to the original configuration, checkpoint SHA-256, training history, source snapshot and evaluation output. Paths beginning with `results/`, `paper/` or `autoresearch-runs/` inside the `*_original` provenance fields are evidence locations relative to the original project, not runtime dependencies. The release does not depend on that project or those directories.

## Which experiment is being reproduced?

| Purpose | Run IDs | Epoch budget | Seed | Selection |
| --- | --- | ---: | ---: | --- |
| Table 2, MAF | `main_bbc`, `main_guardian`, `main_usa_today`, `main_washington_post` | 20 | 42 | First maximum validation ACC |
| Table 3, complete model | `main_bbc` | 20 | 42 | First maximum validation ACC |
| Table 3, complete module removals | `ablation_without_mgr_complete`, `ablation_without_haf_complete`, `ablation_without_dafo_complete` | 20 | 42 | First maximum validation ACC |
| Figures 3 and 4 | `analysis_bbc_legacy` | 5 | 42 | First maximum validation ACC |

The three ablation checkpoint names use the prefix `bbc_` in `checkpoints/manifest.json`; the analysis checkpoint is named `analysis_bbc`. These names are explicitly mapped in `results/provenance.json`.

The current main results are single-seed experiments, not means over seeds 42/43/44. Earlier five-epoch experiments with seeds 43 and 44 exist locally but do not support the current tables. They are deliberately excluded from this minimal release. The current paper has no cross-domain result table. Cross-source evaluation supported by the release is an additional experiment, not a reproduction of an unreported table.

## Data and preprocessing

Obtain DGM4 from its original provider as described in `DATA.md`. Raw news images, captions and annotations are not redistributed here. Preserve the original DGM4 `train.json`, `val.json` and `test.json` files, including duplicate records and their order. The experiment originally used the Visual News ID-to-source mapping. The release also supports the explicit `--derive-source-metadata` option: it derives source assignments only from original-image records with the path prefix `DGM4/origin/<source>/`, aggregates them by record ID, and rejects conflicts or missing assignments. Across all three official metadata files, these original records cover all 83,229 used unique IDs; the resulting mapping was checked ID by ID and is identical to the original Visual News mapping. This avoids requiring the full Visual News download. It does not guess sources from manipulated-image paths. The preparation script verifies the original metadata hashes, ordered-record hashes, mapping hash and split counts. Alternatively, use `--source-metadata` with official Visual News metadata. The two options are mutually exclusive; no random resplit is introduced.

| Source | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| BBC | 40,811 | 4,424 | 9,370 |
| The Guardian | 109,703 | 11,685 | 27,028 |
| USA TODAY | 31,811 | 2,917 | 7,926 |
| The Washington Post | 25,859 | 3,100 | 6,381 |

The exact source metadata and split SHA-256 values are provided in `results/provenance.json`. Every saved main-test prediction was checked against its original source-filtered test record: ordered ID, image path, binary label and manipulation state all matched.

Images are converted to RGB, directly resized to 224 by 224 pixels with bicubic interpolation, converted to tensors and normalized with CLIP mean `(0.48145466, 0.4578275, 0.40821073)` and standard deviation `(0.26862954, 0.26130258, 0.27577711)`. This is a direct square resize, not the default center-crop CLIP evaluation transform. Text uses the matching fast CLIP tokenizer, fixed-length padding and truncation to 77 tokens. Word-level manipulation annotations are zero-based whitespace-word indices and are mapped through character offsets to subword tokens. Padding and special tokens are excluded from token supervision. Visual patch targets are positive when a patch center lies inside the scaled manipulation bounding box.

Binary labels are Real=0 and Manipulated=1. State labels are Real=0, Text-only=1, Image-only=2 and Joint=3. Subtype ACC is binary authenticity accuracy conditioned on the ground-truth state; it is not four-class router accuracy. Ground-truth manipulation annotations supply training losses and evaluation labels, not model inference features. The token-validity mask used in inference is computed from tokenizer offsets and padding/special-token positions; it is not a ground-truth manipulation mask. Original unmanipulated captions are not inference inputs.

## Main training protocol

All four current main runs start independently from pretrained `openai/clip-vit-base-patch16`, newly initialized MAF heads and a new AdamW optimizer. They do not continue an earlier five-epoch experiment. Both CLIP encoders are trainable. The hidden feature width is 256 and dropout is 0.1. Training uses physical batch size 8, gradient accumulation over four batches, head/module learning rate 1e-4, CLIP learning rate 5e-7, weight decay 1e-4, gradient clipping at 1.0, CUDA bfloat16, gradient checkpointing, disabled cuDNN and zero data-loader workers. There is no learning-rate scheduler in these MAF runs. Validation and final evaluation use batch size 16. The original hardware is NVIDIA A40.

The final protocol isolates data-loader random generators by split. The training sampler receives seed 42; validation and test generator offsets are 1,000,000 and 2,000,000. Selection compares validation ACC with a strict `>` update, retaining the first maximum. Selected epochs are BBC=17 and Guardian=USA TODAY=Washington Post=19. `results/selection_history.csv` retains every epoch's validation ACC/AUC and the selected flag.

Some archived configuration fields were remnants of earlier methods and were inactive in the final code path. In particular, archived `lambda_base=0.25` and `lambda_ground=0.25` do not describe MAF's actual final objective. The implemented active loss coefficients are final BCE 0.62, global BCE 0.12, balanced focal 0.30, dual-tail hard mining 0.14, state-specific margin 0.16, state CE 0.16, state-derived binary BCE 0.07, summed modality-presence BCE 0.10, summed grounding BCE 0.06, global ranking 0.08 and Text–Real ranking 0.10. Positive grounding weights are 5 for text and 8 for image patches. The public implementation retains the actual objective rather than activating historical configuration values.

The original serialized state includes a small number of inactive compatibility parameters. Removing their initialization naively changes subsequent random initialization, even though those parameters do not affect inference. The release implementation and checkpoint conversion must preserve or explicitly account for this fact. Architecture parity and new-forward test results are reported separately from the saved-artifact audit.

## Complete module ablations

Each ablation is trained independently for 20 epochs on BBC with seed 42 and the same split-specific sampler protocol. The selected epochs are 19 without MGR, 15 without HAF and 10 without DAFO.

- Without MGR removes local projections, grounding, cross-modal residuals and multi-granular state/context heads. Global-only replacement heads provide the inputs required by the remaining HAF machinery; this is not a local-feature zeroing experiment.
- Without HAF removes adaptive agreement/confidence fusion, modality-presence heads and the corresponding state-binary/presence auxiliary objectives. It directly uses the multi-granular classification logit.
- Without DAFO removes its entire focal, annotation weighting, state weighting/balancing, hard-mining, margin and ranking objective. The unchanged classification and remaining auxiliary objectives are retained.

The complete-model ablation originally used a separately serialized BBC checkpoint. Its entire model state was compared tensor by tensor with `main_bbc`: all tensors are identical. Only runtime metadata and serialized file hashes differ. One BBC model asset therefore reproduces both rows. All Table 3 results use the BBC **test** subset; the epoch-selection statistics remain validation results.

## Metrics and numerical behavior

Final binary decisions use probability >=0.5, with Fake positive. ACC uses those decisions, AUC uses continuous final manipulation probabilities, and EER is computed at the linearly interpolated crossing of FPR and FNR on the ROC curve, grouping tied scores and keeping all ROC points. EER is not accuracy at an optimized decision threshold. Percentages in the paper are rounded to two decimal places.

| Source | ACC (%) | AUC (%) | EER (%) |
| --- | ---: | ---: | ---: |
| BBC | 86.00 | 91.16 | 14.54 |
| The Guardian | 89.64 | 94.49 | 10.66 |
| USA TODAY | 84.28 | 89.78 | 16.60 |
| The Washington Post | 82.21 | 87.70 | 18.58 |

CPU FP32 and CUDA BF16 need not produce bitwise identical logits or classifications near 0.5. No empirically validated universal tolerance is claimed. Recomputing metrics from the supplied frozen predictions should reproduce the reported rounded values exactly; a new inference run must report its precision, full-split sample count and observed differences. Do not silently relax tolerances, tune a test threshold, select checkpoints on test ACC or substitute a higher-scoring historical run. Complete fresh-training numerical reproducibility has not been rerun as part of preparing this release.

## Analysis figures use a different frozen checkpoint

Figures 3 and 4 use the historical BBC seed-42 model selected at epoch 4 of a five-epoch run. It has the same full MAF architecture/objective but a different training budget and historical default sampler behavior. It is retained solely because those figures in the current manuscript depend on it. Its source is traceable to Git commit `f077c8569702d120c575677c87abdc1fd0a1f81c`; the final 20-epoch source snapshot instead has SHA-256 `cd58e939e6a8022cb5f3a238300693e5a302d79a106568b93f20b84730fbb7e2` and no verified original Git commit.

For Figure 3, sort BBC test samples by increasing reliability `g` with `sample_id` as the tie-breaker, then apply `numpy.array_split` into four groups. Counts are 2343, 2343, 2342 and 2342. State-derived binary accuracy uses `1-q_Real >=0.5`. Correction and harm rates divide by every sample in the quartile, not only the globally wrong or correct samples. Net ACC change is corrected minus harmed, in percentage points. `results/haf_quartiles.csv` stores the plotted statistics.

For Figure 4, the successful and failed examples are news IDs 1462685 and 1550713. `results/case_study_records.json` contains identifiers and derived model outputs only. Obtain their actual image/text records through the original dataset. The original captions shown for illustration were not model inputs. No claim is made that these historical example probabilities are outputs of the current 20-epoch BBC model.

## Baselines and scope limitations

Table 2 contains locally obtained HAMMER, HAMMER++ and ALBEF binary-adaptation records and **published** FKA-Owl intra-domain values. FKA-Owl was not reproduced locally. Baseline encoders, preprocessing, budgets and validation-selection rules differ from MAF. Local baseline selection used validation AUC rather than MAF's validation ACC. These are reference comparisons, not architecture- or budget-matched ablations.

This minimal MAF release does not bundle third-party baseline implementations, their model weights or their data. Baseline result records and source provenance are provided, but independent end-to-end reproduction of all baseline rows using only this repository is **NOT TESTED / NOT PROVIDED**. Obtain the original implementations under their own licenses; the MAF MIT license does not relicense them.

The current manuscript's HAMMER++ USA TODAY row is 76.87/82.22/25.89. A later continuation evaluated another validation-selected checkpoint at 76.52/82.77/25.23. This release preserves the submitted table and discloses that difference; it does not silently choose between the two using test performance.

Earlier development had accessed the same designated test records. The project cannot retrospectively claim a completely untouched test set throughout development. The audited final checkpoint selection itself uses the documented validation criterion.

## Artifact verification versus executed reproduction

The preparation audit ran CPU-only checks of checkpoint SHA-256, configuration, epoch-selection histories, exact original test-record alignment, and independent metric recomputation from saved predictions. These checks passed for all four MAF main runs and three ablations. They are **not** newly executed model inference. A separate release validation report records actual dependency installation, model-forward and checkpoint-evaluation commands. Any validation not explicitly executed there remains **NOT TESTED**.

Pretrained CLIP access, final checkpoint downloads and dataset access are separate dependencies. See `checkpoints/README.md` and `DATA.md` for their current access status. Availability of code or compressed prediction files alone is not sufficient to claim checkpoint-based independent reproduction.
