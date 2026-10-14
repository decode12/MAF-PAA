# Experimental protocol and reproducibility

## Result-to-model mapping

All listed results use the official **test** split. Checkpoints are selected using validation ACC, with a strict improvement retaining the first maximum. Test predictions use a fixed probability threshold of 0.5.

| Paper result | Checkpoint name | Epoch budget | Selected epoch | Seed |
| --- | --- | ---: | ---: | ---: |
| Table 2: BBC; Table 3: full model; Figures 3/4 | `main_bbc` | 20 | 17 | 42 |
| Table 2: Guardian | `main_guardian` | 20 | 19 | 42 |
| Table 2: USA TODAY | `main_usa_today` | 20 | 19 | 42 |
| Table 2: Washington Post | `main_washington_post` | 20 | 19 | 42 |
| Table 3: without MGR | `bbc_without_mgr_complete` | 20 | 19 | 42 |
| Table 3: without HAF | `bbc_without_haf_complete` | 20 | 15 | 42 |
| Table 3: without DAFO | `bbc_without_dafo_complete` | 20 | 10 | 42 |

[provenance.json](../results/provenance.json) maps results to original configurations, evaluation records, checkpoint hashes and source snapshots. Its `*_original` paths are historical evidence locations, not runtime dependencies. [selection_history.csv](../results/selection_history.csv) records per-epoch validation metrics; [reported_results.csv](../results/reported_results.csv) records paper metrics; the [checkpoint manifest](../checkpoints/manifest.json) supplies public URLs and checksums. `checkpoint_sha256` in the result CSV refers to the original training file; `sha256` in the download manifest refers to the sanitized public export. Their model tensors are identical. The separately serialized full-model ablation has identical model tensors to `main_bbc`, so no duplicate asset is needed.

Current tables are single-seed results, not averages over seeds 42/43/44. Historical seeds 43/44 used a different five-epoch protocol. The paper reports separate source-specific training, not a cross-domain experiment.

## Training and evaluation

Data versions, source reconstruction, label mapping and input preprocessing are specified in [DATA.md](DATA.md). Both CLIP encoders are trainable. Each run starts independently from pretrained CLIP, new MAF heads and new AdamW state; it does not resume an earlier experiment. Released checkpoints are model-only exports, not optimizer-resume snapshots.

| Setting | Value |
| --- | --- |
| Hidden width / dropout | 256 / 0.1 |
| Training batch / gradient accumulation | 8 / 4 |
| Validation and test batch | 16 |
| Head LR / CLIP LR | 1e-4 / 5e-7 |
| Weight decay / gradient clipping | 1e-4 / 1.0 |
| Scheduler | None |
| Precision / hardware | CUDA BF16 / NVIDIA A40 |
| cuDNN / gradient checkpointing / loader workers | Disabled / enabled / 0 |

The sampler uses independent generators: seed 42 for training, with offsets 1,000,000 for validation and 2,000,000 for test. Mining and ranking use the physical batch of 8, not the accumulated batch of 32.

Active loss coefficients are: final BCE 0.62, global BCE 0.12, focal 0.30, hard mining 0.14, margin 0.16, state CE 0.16, state-binary BCE 0.07, summed presence BCE 0.10, summed grounding BCE 0.06, global ranking 0.08, and Text–Real ranking 0.10. Positive grounding weights are 5 for text and 8 for image. Archived `lambda_base=0.25` and `lambda_ground=0.25` fields were inactive and are not the implemented objective. Inactive compatibility parameters are retained because their initialization affects subsequent random initialization.

ACC uses Fake probability >=0.5; AUC uses continuous probabilities. EER uses the linearly interpolated FPR/FNR crossing, retaining all ROC points and grouping tied scores. Subtype ACC is binary accuracy conditioned on Real/Text-only/Image-only/Joint, not router accuracy. Saved-score metrics must match the paper after rounding to two decimal places. Test scores must not select checkpoints or tune thresholds.

## Complete module ablations

Each variant is independently trained with the main BBC protocol. Table 3 uses BBC **test** records; validation statistics are used only for checkpoint selection.

- **Without MGR:** removes local projections, grounding, cross-modal residuals and multi-granular state/context heads. Global-only replacement heads supply the inputs needed by HAF; features are not merely zeroed.
- **Without HAF:** removes agreement/confidence fusion, modality-presence heads and state-binary/presence auxiliary losses. The multi-granular logit directly supplies the prediction.
- **Without DAFO:** removes all DAFO focal, annotation/state weighting and balancing, hard-mining, margin and ranking objectives. Classification and remaining auxiliary objectives are retained.

See [ablation configurations](../configs/ablation/) and the corresponding rows of [reported_results.csv](../results/reported_results.csv).

## Figures 3 and 4

Both figures use **main_bbc, selected epoch 17 from the 20-epoch seed-42 run**, with all 9,370 BBC test predictions from canonical CUDA BF16 inference. Their numbers match the latest submitted manuscript. The four main prediction bundles retain their original global/final scores and now also include the verified forward-pass diagnostics required for analysis. No raw captions, images or private paths are included.

Figure 3 sorts records by the **stored** `fusion_gate`, breaking ties by the original-metadata `sample_id`, then uses `numpy.array_split` for 2343/2343/2342/2342 records. State-derived accuracy uses `1-q_Real >=0.5`. Correction/harm rates divide by all records in each quartile; net ACC change is their difference in percentage points. Near-saturated gates can tie across quartile boundaries: these are rank groups, not equal-width intervals. The analysis describes within-model predictions; the independently trained HAF ablation is Table 3.

| Group | n | State accuracy (%) | Corrected | Harmed | Net change (pp) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q1 | 2343 | 79.39 | 41 | 31 | 0.43 |
| Q2 | 2343 | 88.60 | 0 | 0 | 0.00 |
| Q3 | 2342 | 85.40 | 0 | 0 | 0.00 |
| Q4 | 2342 | 90.86 | 0 | 0 | 0.00 |

Full-subset ACC changes from 85.89% (global branch) to 86.00% (MAF). The trend in state accuracy is not strictly monotonic. Unrounded values are in [haf_quartiles.csv](../results/haf_quartiles.csv).

| Figure 4 case | News ID | Manipulated / original test row | Global p(Fake) | MAF p(Fake) | q_Text | g |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Success | 1445998 | 28625 / 23178 | 0.436 | 0.985 | 0.931 | 0.910 |
| Failure | 1545723 | 10597 / 7890 | 0.019 | 0.026 | 0.135 | 0.555 |

Rows are **zero-based indices in the full original test metadata**, not BBC-filtered positions. Both cases are Text-only and use native 464×261 images. [Case records](../results/case_study_records.json) supply full-precision predictions, original-pair identifiers and image hashes. With `--data-root`, the analysis script checks the metadata version, same news ID/image, original label and image hash before extracting the captions and rendering Figure 4. Original captions are explanatory references, never inference inputs. The repository renders verbatim metadata (including `Europe s`); the manuscript artwork normalizes this punctuation to `Europe’s`. The model input is unchanged. Fonts/layout can differ from the manually arranged paper artwork; all data and cases are the same. `--font` accepts an installed font; the default is the freely available DejaVu Serif.

The old five-epoch analysis, old case IDs 1462685/1550713 and old quartiles are retained in [results/history](../results/history/), excluded from the default commands. The historical model remains downloadable explicitly and is not used for current results.

## Verified behavior and limitations

| Check | Outcome |
| --- | --- |
| Isolated dependency installation, imports and real-sample forward passes | PASS; pinned Python 3.11 / PyTorch 2.5.1 environment |
| All eight complete anonymous checkpoint downloads and SHA-256 checks | [PASS](../results/validation/public_checkpoint_downloads.json); historical release verification |
| Full four-source CUDA BF16 main-checkpoint reevaluation | [PASS](../results/validation/cuda_inference.json): 50,705 records, exact archived probabilities |
| Saved-score recomputation of all eight MAF rows in Tables 2/3 | PASS |
| Current Figure 3 quartiles and Figure 4 predictions / original pairs | PASS; executable `reproduce_analysis.py --verify-paper --plot --data-root ...` |
| Current cleanup: anonymous BBC download, 16-record CPU test, three ablation forwards/losses, one full-model optimizer step on 8 training records | [PASS](../results/validation/repository_check.json); no formal checkpoint changed |
| Full 20-epoch retraining and new full-split CUDA ablation reevaluation | NOT TESTED for this release |
| Complete third-party baseline reexecution | NOT PROVIDED / NOT TESTED |

The full CUDA BF16 reevaluation used batch 16, cuDNN disabled and released model tensors; it reproduced every main-test global/final probability and logit exactly. This supersedes the earlier unresolved canonical-GPU check. The verification evidence contains original/public prediction hashes and checkpoint hashes; it is an existing completed reevaluation, not a new training run.

CPU FP32 is supported for functional checks but is **not numerically interchangeable** with canonical CUDA BF16. Earlier CPU tests disagreed on 3 of 352 labels across four main models and three ablations, with maximum absolute probability difference 0.886 ([CPU evidence](../results/validation/cpu_inference.json)). These are not merely rounding errors; no universal tolerance is claimed. An [eight-record comparison](../results/validation/numerical_variation_review.json) verified identical inputs, tensors and CPU predictions between original/released code. CPU BF16 produced nonfinite outputs and is not enabled. Use CUDA BF16/batch 16 to compare with reported values, and preserve all differences rather than tuning against test results.

Dependencies are pinned to a tested environment ([CPU package snapshot](environment-cpu-tested.txt)); an immutable training-time environment lock was not retained. The final training source has SHA-256 `cd58e939e6a8022cb5f3a238300693e5a302d79a106568b93f20b84730fbb7e2`, with no verified original final Git commit. Earlier development accessed the designated test split; project-wide test independence cannot be retrospectively claimed. Four sources are independent in-domain experiments; no current cross-domain result is reported. Historical seeds 43/44 and transfer experiments are not replicas of this protocol.

## Baseline provenance

Table 2 combines local HAMMER, HAMMER++ and ALBEF binary-adaptation results with **published** [FKA-Owl Table 2 intra-domain results](https://arxiv.org/html/2403.01988v2#S4.T2). FKA-Owl was not reproduced locally. Encoders, preprocessing and budgets differ; local baselines used validation AUC selection instead of MAF's validation ACC. These are reference comparisons, not capacity- or budget-matched ablations. Baseline evidence is recorded in [provenance.json](../results/provenance.json); third-party implementations and weights are not bundled here.

The submitted HAMMER++ USA TODAY row is 76.87/82.22/25.89 (ACC/AUC/EER). A later continuation yielded 76.52/82.77/25.23 from another validation-selected checkpoint. Both records are disclosed in [provenance.json](../results/provenance.json); the submitted row is preserved, rather than choosing between them using test performance.
