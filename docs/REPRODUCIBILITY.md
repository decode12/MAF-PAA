# Experimental protocol and reproducibility

## Result-to-model mapping

All listed results use the official **test** split. Checkpoints are selected using validation ACC, with a strict improvement retaining the first maximum. Test predictions use a fixed probability threshold of 0.5.

| Paper result | Checkpoint name | Epoch budget | Selected epoch | Seed |
| --- | --- | ---: | ---: | ---: |
| Table 2: BBC; Table 3: full model | `main_bbc` | 20 | 17 | 42 |
| Table 2: Guardian | `main_guardian` | 20 | 19 | 42 |
| Table 2: USA TODAY | `main_usa_today` | 20 | 19 | 42 |
| Table 2: Washington Post | `main_washington_post` | 20 | 19 | 42 |
| Table 3: without MGR | `bbc_without_mgr_complete` | 20 | 19 | 42 |
| Table 3: without HAF | `bbc_without_haf_complete` | 20 | 15 | 42 |
| Table 3: without DAFO | `bbc_without_dafo_complete` | 20 | 10 | 42 |
| Figures 3/4: earlier BBC analysis | `analysis_bbc` | 5 | 4 | 42 |

[provenance.json](../results/provenance.json) maps results to original configurations, evaluation records, checkpoint hashes and source snapshots. Its `*_original` paths are historical evidence locations, not runtime dependencies. [selection_history.csv](../results/selection_history.csv) records per-epoch validation metrics; [reported_results.csv](../results/reported_results.csv) records paper metrics; the [checkpoint manifest](../checkpoints/manifest.json) supplies public URLs and checksums. The separately serialized full-model ablation has identical model tensors to `main_bbc`, so no duplicate asset is needed.

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

The sampler uses independent generators: seed 42 for training, with offsets 1,000,000 for validation and 2,000,000 for test. Mining and ranking use the physical batch of 8, not the accumulated batch of 32. The earlier analysis run used the historical default sampler behavior.

Active loss coefficients are: final BCE 0.62, global BCE 0.12, focal 0.30, hard mining 0.14, margin 0.16, state CE 0.16, state-binary BCE 0.07, summed presence BCE 0.10, summed grounding BCE 0.06, global ranking 0.08, and Text–Real ranking 0.10. Positive grounding weights are 5 for text and 8 for image. Archived `lambda_base=0.25` and `lambda_ground=0.25` fields were inactive and are not the implemented objective. Inactive compatibility parameters are retained because their initialization affects subsequent random initialization.

ACC uses Fake probability >=0.5; AUC uses continuous probabilities. EER uses the linearly interpolated FPR/FNR crossing, retaining all ROC points and grouping tied scores. Subtype ACC is binary accuracy conditioned on Real/Text-only/Image-only/Joint, not router accuracy. Saved-score metrics must match the paper after rounding to two decimal places. Test scores must not select checkpoints or tune thresholds.

## Complete module ablations

Each variant is independently trained with the main BBC protocol. Table 3 uses BBC **test** records; validation statistics are used only for checkpoint selection.

- **Without MGR:** removes local projections, grounding, cross-modal residuals and multi-granular state/context heads. Global-only replacement heads supply the inputs needed by HAF; features are not merely zeroed.
- **Without HAF:** removes agreement/confidence fusion, modality-presence heads and state-binary/presence auxiliary losses. The multi-granular logit directly supplies the prediction.
- **Without DAFO:** removes all focal, annotation/state weighting and balancing, hard-mining, margin and ranking objectives. Classification and remaining auxiliary objectives are retained.

See [ablation configurations](../configs/ablation/) and the corresponding rows of [reported_results.csv](../results/reported_results.csv).

## Figures 3 and 4

These figures use the same full architecture but the **earlier five-epoch BBC model**, not the current Table 2 model. Its predictions are in `results/predictions/analysis_bbc_legacy.jsonl.gz`.

Figure 3 sorts BBC test records by reliability `g`, breaking ties by `sample_id`, then uses `numpy.array_split` for quartiles of 2343/2343/2342/2342 records. State-derived accuracy uses `1-q_Real >=0.5`. Correction and harm rates divide by all records in each quartile; net ACC change is corrected minus harmed, in percentage points. The script reconstructs `g` from saved float32 state/presence probabilities (maximum difference from stored `g`: 5.96e-8) and verifies [the plotted statistics](../results/haf_quartiles.csv).

Figure 4 uses news IDs 1462685 and 1550713. [Case records](../results/case_study_records.json) provide IDs and model outputs; images and captions come from DGM4. Original captions shown for illustration were not model inputs. `scripts/reproduce_analysis.py --verify-paper --plot` reconstructs the statistics and prediction records, not the original news-photo composite.

## Verified behavior and unresolved differences

| Check | Outcome |
| --- | --- |
| Isolated dependency installation, imports and real-sample forward passes | PASS |
| All eight complete anonymous checkpoint downloads and SHA-256 checks | [PASS](../results/validation/public_checkpoint_downloads.json) |
| Frozen predictions: MAF rows of Tables 2/3 and Figures 3/4 statistics | PASS |
| Loaded-checkpoint CPU inference: 352 test records across four main models and three ablations | Executed; [differences recorded](../results/validation/cpu_inference.json) |
| New full-split CUDA BF16 inference and complete 20-epoch retraining | NOT TESTED during release preparation |
| Complete third-party baseline reexecution | NOT PROVIDED / NOT TESTED |

CPU FP32 inference disagreed with archived CUDA BF16 classifications on 3 of 352 records. Maximum absolute probability differences reached 0.886 for Guardian, 0.412 for USA TODAY and 0.313 for the DAFO ablation. These differences cannot be described as harmless threshold rounding. No universal tolerance is supported, and matching saved predictions does not establish matching new inference.

An [eight-record Guardian comparison](../results/validation/numerical_variation_review.json) found identical input tensors, checkpoint tensors and CPU outputs between the original and released implementations, while both differed from archived GPU predictions. This excludes the code cleanup as the cause for those checked records, but does not establish the cause of the historical discrepancy. Diagnostic CPU BF16 produced nonfinite values in 4/8 records and is not enabled in the public CPU engine. Canonical CUDA agreement remains **UNRESOLVED**.

Dependencies are pinned to a tested environment ([CPU package snapshot](environment-cpu-tested.txt)); an immutable training-time environment lock was not retained. The final training source has SHA-256 `cd58e939e6a8022cb5f3a238300693e5a302d79a106568b93f20b84730fbb7e2`, with no verified original final Git commit. The earlier analysis source traces to commit `f077c8569702d120c575677c87abdc1fd0a1f81c`. Earlier development accessed the designated test split; project-wide test independence cannot be retrospectively claimed.

## Baseline provenance

Table 2 combines local HAMMER, HAMMER++ and ALBEF binary-adaptation results with **published** FKA-Owl intra-domain results. FKA-Owl was not reproduced locally. Encoders, preprocessing and budgets differ; local baselines used validation AUC selection instead of MAF's validation ACC. These are reference comparisons, not capacity- or budget-matched ablations. Baseline evidence is recorded in [provenance.json](../results/provenance.json); third-party implementations and weights are not bundled here.

The submitted HAMMER++ USA TODAY row is 76.87/82.22/25.89 (ACC/AUC/EER). A later continuation yielded 76.52/82.77/25.23 from another validation-selected checkpoint. Both records are disclosed in [provenance.json](../results/provenance.json); the submitted row is preserved, rather than choosing between them using test performance.
