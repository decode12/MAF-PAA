# Reproducibility audit of the MAF submission

Audited against the current manuscript source and PDF on 2026-10-14. This is an evidence audit, not a claim that all training has been executed again. No original training result was overwritten, no unrelated project was modified and no GPU training was started for this audit.

## Result-to-artifact mapping

The machine-readable source of truth is [`results/provenance.json`](results/provenance.json). Original evidence paths are relative to the PAA project and have SHA-256 values; they are not required runtime paths. Every table entry is retained in [`results/reported_results.csv`](results/reported_results.csv), and all epoch-selection observations in [`results/selection_history.csv`](results/selection_history.csv).

| Manuscript result | Configuration / checkpoint name | Seed | Selected epoch / budget | Split | ACC / AUC / EER (%) | Audit status |
| --- | --- | ---: | --- | --- | --- | --- |
| Table 2, BBC MAF | `main_bbc` | 42 | 17 / 20 | Test | 86.00 / 91.16 / 14.54 | PASS: saved artifacts |
| Table 2, Guardian MAF | `main_guardian` | 42 | 19 / 20 | Test | 89.64 / 94.49 / 10.66 | PASS: saved artifacts |
| Table 2, USA TODAY MAF | `main_usa_today` | 42 | 19 / 20 | Test | 84.28 / 89.78 / 16.60 | PASS: saved artifacts |
| Table 2, Washington Post MAF | `main_washington_post` | 42 | 19 / 20 | Test | 82.21 / 87.70 / 18.58 | PASS: saved artifacts |
| Table 3, MAF | `main_bbc` | 42 | 17 / 20 | BBC test | 86.00 / 91.16 / 14.54 | PASS: identical full-model tensors |
| Table 3, w/o MGR | `bbc_without_mgr_complete` | 42 | 19 / 20 | BBC test | 85.67 / 91.84 / 14.73 | PASS: saved artifacts |
| Table 3, w/o HAF | `bbc_without_haf_complete` | 42 | 15 / 20 | BBC test | 85.61 / 86.90 / 16.98 | PASS: saved artifacts |
| Table 3, w/o DAFO | `bbc_without_dafo_complete` | 42 | 10 / 20 | BBC test | 85.47 / 91.89 / 15.40 | PASS: saved artifacts |
| Figures 3 and 4 | `analysis_bbc` | 42 | 4 / 5 | BBC test | See derived analysis records | Historical full MAF required by the current figures |

“PASS: saved artifacts” means actual checkpoint hashes, stored configurations, complete validation histories and frozen prediction metrics were checked independently. It does not mean a new inference or retraining run was performed by this audit. The exact raw metrics and hashes are in provenance. All four principal test files were checked record by record against the original DGM4 test metadata and the Visual News source mapping. Validation selection is first maximum ACC and final threshold is 0.5.

All current main MAF models were trained fresh for 20 epochs, with no optimizer continuation. Prior results combining five-epoch checkpoints and continuation are excluded. The archived full-ablation BBC checkpoint and the fresh-20 BBC checkpoint have different file hashes but **identical model tensors**. One released model can reproduce both.

## Four manipulation states

Real, Text-only, Image-only and Joint test accuracies and sample counts are provided in [`results/subtype_results.csv`](results/subtype_results.csv). These are binary predictions conditioned on manipulation state, not four-way classification accuracy. They were recomputed from saved predictions, with no new threshold fitting. Original image/text data are not included.

## Seeds 43 and 44 and cross-domain results

Verified historical configurations for all four sources use five epochs and seeds 42, 43 or 44. BBC checkpoints were historically identified as EXP029/EXP032/EXP033; the other sources have corresponding source-and-seed experiments. They do **not** generate the current 20-epoch main table and must not be presented as current three-seed replicas. Except for the BBC seed-42 analysis model still required by Figures 3 and 4, these historical models and code are excluded.

Older BBC-to-other-source transfer experiments exist, but the current manuscript contains no cross-domain result table. A generic cross-source evaluation capability can be provided as an extension; its outputs are not paper results. No unavailable fresh-20 seed-43/44 or cross-domain results are fabricated.

## Actual implementation and source provenance

The executed final snapshot is `code/experiments/ablation_protocol_v2/train.py`, SHA-256 `cd58e939e6a8022cb5f3a238300693e5a302d79a106568b93f20b84730fbb7e2`. All four main run snapshots share this hash. The original folder is not a Git repository, so an original final-code commit cannot be truthfully supplied. The new release commit identifies the portable public implementation; the source hash and parity checks connect it to the experiment snapshot.

The historical analysis-model source is independently traceable to commit `f077c8569702d120c575677c87abdc1fd0a1f81c`, with source SHA-256 `f1260c0f7581ff297d85ed1799985a65e0ce24c9f875993d3e3208b899d5555d`. Its model is necessary for existing figures; historical naming does not make it dispensable.

The minimal scientific dependencies are the MAF model and objective, DGM4 image/text loader and annotation mapping, source-filtering metadata, pretrained CLIP ViT-B/16, training and fixed-checkpoint evaluation, metric aggregation, three complete-module ablations and the retained analysis-model outputs. There is no external evidence retrieval or precomputed feature-cache dependency. Archived inactive correction heads require care when converting checkpoints or reproducing seeded initialization. Historical configuration lambda fields are not substituted for the hard-coded final MAF objective.

## Other compared methods

HAMMER, HAMMER++ and ALBEF values match local result records, while FKA-Owl values are cited from the original publication's intra-domain table. See `results/provenance.json` for record hashes and source details. FKA-Owl is not labeled as a local reproduction. Baseline selection used validation AUC and method-specific training budgets/preprocessing, unlike MAF's validation-ACC selection.

The repository does not bundle baseline implementations and weights. Their independent execution from this MAF-only release is **NOT PROVIDED / NOT TESTED**, and their original licenses remain applicable. The baseline rows support faithful table reconstruction, not a claim of universal end-to-end baseline reproducibility. In particular, the current HAMMER++ USA TODAY row differs from a later 20-epoch continuation result; the release preserves the actual manuscript rather than silently replacing it.

## Data, checkpoint access and disclosure

Raw DGM4 images/captions and Visual News data are not republished. Official access directions, licensing notes and local preparation are described in [`docs/DATA.md`](docs/DATA.md). Exact metadata hashes and counts are recorded in provenance. The verified `--derive-source-metadata` preparation option reconstructs every used source assignment from `DGM4/origin/<source>/` records in the three official metadata splits, without downloading Visual News. All 83,229 used unique IDs are covered and match the original Visual News mapping exactly; manipulated-image paths are never used to guess a source. Mapping SHA-256 is `e6f00c11e09e2d18e0d5873a779ebb704fb687704a6f8fb0252622f8b59aff9b`. No random resplitting, deduplication or test-driven model selection is introduced.

The prediction bundles contain only identifiers, labels and derived model scores. They contain no image files, captions, image paths, raw annotation records, private server paths or credentials. New public model assets should contain sanitized inference weights rather than original optimizer snapshots and private runtime arguments. Check [`checkpoints/README.md`](checkpoints/README.md) and the checkpoint manifest for actual download availability. If the verified weights are not accessible through a public URL, independent evaluation remains blocked even when code and saved predictions are available; this must remain disclosed in the release report.

## Executed audit checks

- PASS: actual hashes of all four fresh-20 main checkpoints and all complete-module ablation checkpoints match frozen evaluation records.
- PASS: every current main run and ablation has 20 complete validation epochs, with the recorded checkpoint selected by the first maximum validation ACC.
- PASS: analysis-model SHA-256 and first-maximum-validation-ACC selection at epoch 4 of five epochs.
- PASS: independent ACC/AUC/EER recomputation from four full test prediction files and three ablation test files.
- PASS: all 24 Table 2 / Table 3 rows match the manuscript's numeric entries to two decimal places.
- PASS: current source-filtered test records, IDs, paths, labels and states match original metadata without changing order or split.
- PASS: full-ablation and fresh-20 BBC model states are tensor-identical.
- PASS: all Figure 3 quartile statistics and both Figure 4 case probabilities/agreement/reliability values match independent recomputation or lookup from the sanitized analysis prediction bundle.
- PASS: all compressed prediction records satisfy the restricted identifier/label/derived-score schema with no original news text, images or private paths.
- NOT TESTED by this artifact audit: clean-environment installation, new forward passes, new full-checkpoint inference, retraining and baseline reexecution. These are separate release-validation tasks; see the final release report for commands actually executed.

## Remaining limitations

1. Figures 3 and 4 use a historical five-epoch model, not the current 20-epoch model. Its provenance is disclosed and its outputs are kept separate.
2. No final-protocol seed-43/44 experiments or current cross-domain result table are available.
3. Earlier development accessed the designated test split. Complete project-wide test-set independence cannot be reconstructed retrospectively.
4. CUDA BF16 and CPU FP32 may differ near the threshold. No unmeasured numerical tolerance or training variance is asserted.
5. Public data-provider access and public final-weight access must work for an external researcher; local artifact availability alone does not satisfy that condition.
6. Baseline implementation/weight reproduction is outside this minimal MAF release and is not falsely reported as tested.
7. The final historical source snapshot has a verified content hash but no verified original Git commit.

## Excluded material

Earlier architectures, component-search trials not reported in the final paper, diffusion/evidence experiments, frozen-feature pilots, autoresearch controllers and notification code, old three-seed/domain-transfer artifacts, continuation experiments superseded by fresh training, large training logs, optimizer snapshots, temporary files, manuscript drafts and unrelated IJCAI/TMM projects are not public-source dependencies and are excluded. The single older analysis checkpoint is the explicit exception because it still generates current figures.
