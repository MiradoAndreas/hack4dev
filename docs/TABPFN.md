# Local TabPFN-v2 experiment

**Built with PriorLabs-TabPFN.** **The fixed RF + TabPFN-v2 blend improves development macro F1 to 0.5842**, from **0.5679** for the previous RF–fine-tuned-MLP hybrid (**+0.0163**, or 1.63 percentage points). It also lowers severe errors from **9.53% to 8.30%**. Use it as the leading development candidate for the macro-F1 objective, pending independent geographic/competition evaluation. TabPFN-v2 alone scores **0.5709**. These findings do not establish the performance of TabPFN-3.5 or statistical significance.

## Skill installation before testing

The `tabpfn` skill was installed from [magnus919/agent-skills](https://github.com/magnus919/agent-skills/tree/4f03205616b01360374f458f471791188f448591/tabpfn), pinned to commit `4f03205616b01360374f458f471791188f448591`, using the skill-installer helper. The archive download failed; the helper's Git fallback succeeded. Its model, operations and evaluation instructions were read before the model pilot. The existing local scikit-learn evaluation skill was also used. The installed skill will be automatically discoverable on the next conversation turn; it was read explicitly for this run.

The skill's doctor recorded package versions. Its synthetic three-way forward-time manifest audit was not treated as validation of this task: this experiment uses the already established spatial cross-validation protocol, with no calibration split or chronological forecasting claim. Actual model loading, prediction, group/context checks, full score recomputation and fresh-process reconstruction were performed separately.

## Exact model and access boundary

This test uses the **original public TabPFN-v2 classifier**, not TabPFN-3.5, 3.5-Fast, the newer default real-data-fine-tuned v2 alias, or a hosted service. The existing machine has no CUDA GPU. Newer gated checkpoints require account/license-access steps that were not performed on the user's behalf. This v2 test does not determine the performance of those later models.

- Package: **TabPFN 9.2.0**, with the existing CPU PyTorch environment.
- Weight repository: `Prior-Labs/TabPFN-v2-clf`.
- Immutable revision: `275f19d2add9c2c0de38f9ebaded3856599fa2e7`.
- Filename: `tabpfn-v2-classifier.ckpt`.
- SHA-256: `f65a35685aeef42e31b796d9bfa34e68d6fc780bc98e7bff7763802964cf435f`.
- Checkpoint size: 29,016,968 bytes; its digest was checked against the upstream LFS record.
- Weight license: **Prior Labs License 1.1 (May 2025)**, retained locally with the model card. The package code license and weight license are separate. This is local research evaluation, not a competition-rule determination or production deployment.

The package's current v2 alias points to a different checkpoint, so the filename is specified explicitly. An optional `tabpfn` dependency was added to `pyproject.toml` and `uv.lock` together. No Torch upgrade or GPU installation was performed. All installed packages passed `uv pip check`.

## Fixed inference and evaluation procedure

Each run uses exactly **25 Amsterdam labels/class (100 true labeled context rows)**, matching the saved baseline samples. No Madrid labels enter TabPFN's context. TabPFN's external synthetic pretraining differs from the RF's source-city supervision; these are complete-procedure comparisons with matched target-label budgets, not identical total prior information.

The 54-feature allowlist and column order are unchanged. Coordinates, IDs, construction years, age classes, folds and eligibility flags are excluded from predictors. No external scaling or imputation is applied: the package's fitted preprocessing is constructed from the permitted training context. Missing feature values are retained; infinite values are rejected. The validation rows and spatial groups are excluded from the labeled context.

Settings were fixed before the new evaluation: four inference estimators, CPU with four Torch threads, `fit_with_cache`, one preprocessing worker, no probability balancing, fixed seeds `10042 + 100*fold + repetition`, and query batches of 256. There is no dataset-specific gradient fine-tuning, calibration, threshold optimization, checkpoint selection by Amsterdam scores or outer-fold mixture search. Runtime pilots selected the practical CPU/batching setup without using validation scores.

The classifier's four output columns are aligned to age classes 1–4. The new hybrid is fixed as:

```text
P(TabPFN–RF hybrid) = 0.25 × P(Madrid RF)
                   + 0.25 × P(Amsterdam RF)
                   + 0.50 × P(TabPFN-v2)
```

Five spatial folds and three matched sampling repetitions are evaluated in full. Repetitions average within folds before averaging the five fold means. Baseline RF and RF–fine-tuned-MLP predictions are reused from the verified transfer experiment on the exact same rows. All 15 TabPFN contexts/prediction runs are newly computed. There are no dropped failing runs.

## Results

| Model | Macro F1 | Fold SD | Weighted F1 | Severe errors |
|---|---:|---:|---:|---:|
| Fixed RF + TabPFN-v2 blend | 0.5842 | 0.0174 | 0.5931 | 0.0830 |
| TabPFN-v2 alone | 0.5709 | 0.0171 | 0.5766 | 0.0925 |
| Previous RF + fine-tuned MLP | 0.5679 | 0.0158 | 0.5769 | 0.0953 |
| Original RF blend | 0.5494 | 0.0258 | 0.5636 | 0.0863 |

Severe errors span two or more age classes. Fold SD is descriptive, not a confidence interval. Native probabilities are not assumed calibrated; log loss and multiclass Brier are also exported. A probability-metric improvement does not replace macro-F1 comparison.

### Per-class F1

| Model | Class 1 | Class 2 | Class 3 | Class 4 |
|---|---:|---:|---:|---:|
| Fixed RF + TabPFN-v2 blend | 0.6601 | 0.5769 | 0.5633 | 0.5365 |
| Previous RF + fine-tuned MLP | 0.6514 | 0.5571 | 0.5437 | 0.5192 |
| TabPFN-v2 alone | 0.6535 | 0.5340 | 0.5654 | 0.5307 |
| Original RF blend | 0.6337 | 0.5872 | 0.4847 | 0.4919 |

TabPFN–RF hybrid minus the current RF–neural hybrid, by fold: **+0.0095, +0.0247, +0.0293, +0.0116, +0.0066**. The new blend leads on all five fold means and improves all four class-level mean F1 scores relative to the previous neural hybrid. Class 2 remains below the original RF-only blend, so it does not dominate every baseline in every class.

These are previously used spatial development folds, not an untouched geographic or competition test. No statistical-significance claim is made. Source/target age-boundary differences remain a limitation of the RF source task; target-only TabPFN context does not use those source labels.

## Runtime, privacy and reproducibility

The measured total context-fit plus held-out prediction time is **1678 seconds** (28.0 minutes), with peak process RSS of approximately **1137 MiB**. Peak RSS is the process high-water mark, not a measurement of model-only memory. Four inference estimators and all 77,976 held-out context/location predictions were retained. Reported probability-row counts include repeated comparison methods.

Inference ran locally using an explicit, hash-verified checkpoint, offline Hugging Face mode, and a guard that rejects attempted socket connections during model construction, context fitting and prediction. No hosted TabPFN client was used and no project data were uploaded for prediction. Weight/skill/package acquisition required network access but sent no project tables.

A setup reproducibility check found a maximum violating absolute difference of about 1.2e-6 when comparing the same rows in 32-row versus 256-row query batches. No scores or model settings were used to address this. The export/reload probe was changed to the same 256-row batch shape as evaluation; the initial diagnostic artifacts were preserved, and the first full run was recomputed with that check. Fresh-process reconstruction then reloads the exact checkpoint, seed, 100-label context and schema and compares held-out probabilities at matching batch shape.

The log-loss helper emitted probability-sum warnings when float32 TabPFN outputs were blended with float64 RF outputs. All arrays passed finite, nonnegative and row-sum checks within 1e-6. After prediction completed, log loss and Brier were recomputed using float64 per-row normalization, preserving every original probability and checking that all classification metrics were unchanged. The original prediction code and pre-correction metrics are archived. Fresh runs use the same float64 convention directly.

The labeled context is part of the inference artifact. Example contexts, model weights and validation artifacts are local generated data, excluded from version control. The exported example excludes Amsterdam fold 0 and is not a separately trained final competition model.

## Run and inspect

Open [tabpfn_experiment.ipynb](../notebooks/experiments/tabpfn_experiment.ipynb) for the exact checkpoint, score plots, per-class results, paired folds, timings and reloaded context inference. By default it verifies saved results; set `RETRAIN = True` to resume/reconstruct the full experiment. [Experiment implementation](../scripts/tabpfn_experiment.py), [public checkpoint downloader](../scripts/download_tabpfn_checkpoint.py), [score CSV](results/tabpfn_scores.csv).

```bash
uv sync --locked --extra tabpfn --extra deep-learning
uv run --extra tabpfn --extra deep-learning python -m scripts.download_tabpfn_checkpoint
uv run --extra tabpfn --extra deep-learning python -m scripts.tabpfn_experiment
# Resume validated completed run files after an interruption:
uv run --extra tabpfn --extra deep-learning python -m scripts.tabpfn_experiment --resume
uv run --extra tabpfn --extra deep-learning python -m scripts.tabpfn_experiment --verify
# Reconstruct and predict in a fresh process:
uv run --extra tabpfn --extra deep-learning python -m scripts.tabpfn_experiment --reload
```

Prerequisites are the preprocessed Amsterdam feature table, baseline configuration/memberships, and saved `data/torch_transfer/` predictions and manifest. Acquisition does not automatically accept a gated checkpoint agreement. Do not substitute a different checkpoint without creating a separate experiment directory and recorded comparison.

Artifacts under `data/tabpfn_experiment/` include immutable-checkpoint metadata and license, all per-run probabilities and timings, class/run/fold scores, exact adaptation memberships, split audits, example context/probe, model/skill records, and hashes. Completed run files are individually hashed for resumability.

Verification checks **60 run scores from 311,904 probability rows**, exact budgets and held-out spatial exclusions, class alignment, fixed mixture arithmetic, fold aggregation/SD, all recorded input/output hashes, and fresh-process context reconstruction against 256 held-out probability vectors.

## Run worksheet

The installed skill's worksheet is summarized here: target `age_class`, four-class classification, local retrospective spectral inputs, unique locations grouped into spatial blocks, no forward-time forecasting claim, no extra labels, no calibration/threshold selection, and unchanged incumbent artifacts for rollback. The checkpoint/license/context/version records above define the exact task scope.

| Stage | Status | Evidence |
|---|---|---|
| Inventory and schema/group checks | Pass | `skill_doctor.json`, input hashes, memberships, split audit |
| Local import/device/network probe | Pass | CPU pilot and blocked socket connections during actual inference |
| All 15 load/context/predict runs | Pass | Complete per-run Parquet files and timing records |
| Optional gradient fine-tuning | Not requested | No weight updates |
| Fresh-process reconstruction | Pass | Same checkpoint, seed and context reproduce 256 held-out probability vectors |
| Matched development comparison | Pass | 60 score recomputations, paired folds and class metrics |
| Independent competition test | Not available | No independent-test performance claim |
| Serving/final production adaptation | Not requested | Fold-excluded example only; prior models preserved |

## Attribution and sources

Built with PriorLabs-TabPFN. Model attribution: Hollmann et al., *Accurate predictions on small data with a tabular foundation model*, Nature 637, 319–326 (2025), [DOI](https://doi.org/10.1038/s41586-024-08328-6). Exact [weight repository](https://huggingface.co/Prior-Labs/TabPFN-v2-clf/tree/275f19d2add9c2c0de38f9ebaded3856599fa2e7) and [license](https://huggingface.co/Prior-Labs/TabPFN-v2-clf/blob/275f19d2add9c2c0de38f9ebaded3856599fa2e7/LICENSE.txt). API behavior was checked against the installed 9.2.0 implementation and [official source](https://github.com/PriorLabs/TabPFN). Those sources motivate the method; the table above is computed from this project's saved predictions.

## Conclusion

**The fixed RF + TabPFN-v2 blend improves development macro F1 to 0.5842**, from **0.5679** for the previous RF–fine-tuned-MLP hybrid (**+0.0163**, or 1.63 percentage points). It also lowers severe errors from **9.53% to 8.30%**. Use it as the leading development candidate for the macro-F1 objective, pending independent geographic/competition evaluation. TabPFN-v2 alone scores **0.5709**. These findings do not establish the performance of TabPFN-3.5 or statistical significance.
