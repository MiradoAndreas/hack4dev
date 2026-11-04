# Experiment 3: regularized CatBoost

**CatBoost is implemented and evaluated, but does not improve the current model.** Its best new variant, the fixed CatBoost–neural hybrid, scores **0.5618 macro F1**, versus **0.5679** for the current RF–neural hybrid (**−0.0060**). Retain the RF–fine-tuned-MLP hybrid as the current macro-F1 candidate. This comparison does not rule out other CatBoost settings or prove a statistically reliable ranking.

## What was implemented

The experiment uses the 54-feature allowlist and the exact original **five spatial outer folds × three sampling repetitions**, with **25 Amsterdam labels/class (100 total)** in each adaptation run. It evaluates Madrid-only CatBoost, target-only CatBoost, their fixed 50/50 probability blend, and a CatBoost–neural hybrid, alongside the RF baseline and current RF–neural hybrid.

```text
P(CatBoost–neural hybrid) = 0.25 × P(Madrid CatBoost)
                         + 0.25 × P(Amsterdam CatBoost)
                         + 0.50 × P(existing fine-tuned MLP)
```

All blending weights were fixed before outer evaluation. The fine-tuned MLP is unchanged; verified saved neural and comparison probabilities are reused. New CatBoost models are trained here. This is not neural fine-tuning of a boosted model.

## Selection and leakage controls

**Source model:** depth 4, learning rate 0.03, L2 leaf regularization 10, balanced class weights. A model trained on Madrid folds 1–4 evaluates 150, 300 and 600 trees on spatially disjoint Madrid fold 0. It selects **600 trees**, then refits on all **76,263 Madrid locations** using the selected setting. Madrid checkpoint macro F1 is **0.5443**; this is a selected source-development diagnostic, not an untouched test score. No Amsterdam data select the source setting.

CatBoost retains the **original Madrid labels** with the 1960 first-class boundary, matching the RF source procedure. The existing neural model was pretrained using harmonized 1945-boundary labels, as documented in the [transfer report](TORCH_TRANSFER.md). These source-supervision differences are explicit; organizer confirmation of class definitions remains outstanding.

**Target models:** each allocated 100-label sample is split into three inner spatial folds; a two-fold fallback is allowed if necessary to preserve all training classes. The candidates are depths **3, 4, 5** and tree counts **150, 300**, with learning rate 0.03 and L2 leaf regularization 10. The six settings are ranked by mean inner macro F1; ties prefer smaller depth then fewer trees. The chosen model is refitted on all 100 allocated labels. Tree prefixes share one 300-tree fit per inner fold/depth; no outer labels select tree counts.

Median imputation is fitted separately inside every inner, source and final target fit. No scaling is used. Coordinates and IDs are used only for matching and spatial exclusions. Construction years, age classes, fold IDs and eligibility indicators are excluded from predictors. Each target fit and selection excludes outer validation locations and groups. Prediction class columns are explicitly aligned before averaging.

## Results

| Model | Macro F1 | Fold SD | Weighted F1 | Severe errors |
|---|---:|---:|---:|---:|
| Current RF + fine-tuned MLP | 0.5679 | 0.0158 | 0.5769 | 0.0953 |
| CatBoost + fine-tuned MLP | 0.5618 | 0.0180 | 0.5673 | 0.0978 |
| Original RF blend | 0.5494 | 0.0258 | 0.5636 | 0.0863 |
| Madrid/Amsterdam CatBoost blend | 0.5412 | 0.0351 | 0.5506 | 0.0927 |
| Target-only CatBoost (inner-selected) | 0.5252 | 0.0277 | 0.5401 | 0.1286 |
| Madrid CatBoost (0 target labels) | 0.4562 | 0.0281 | 0.4640 | 0.1011 |

All table scores average repetitions within folds first, then average the five fold means. Fold SD is descriptive, not a confidence interval. Severe errors span two or more age classes. The zero-target-label source diagnostic repeats the same predictions across repetitions for alignment; these are not independent retraining runs.

The CatBoost–neural hybrid's paired fold differences against the current RF–neural hybrid are **-0.0201, +0.0043, -0.0059, -0.0068, -0.0019**. It leads on 1 of five fold means and trails on 4. Its severe-error rate is **9.78%**, compared with **9.53%** for the current hybrid.

### Per-class F1

| Model | Class 1 | Class 2 | Class 3 | Class 4 |
|---|---:|---:|---:|---:|
| Current RF + fine-tuned MLP | 0.6514 | 0.5571 | 0.5437 | 0.5192 |
| CatBoost + fine-tuned MLP | 0.6471 | 0.5184 | 0.5615 | 0.5204 |
| Madrid/Amsterdam CatBoost blend | 0.6432 | 0.5058 | 0.5351 | 0.4808 |
| Target-only CatBoost (inner-selected) | 0.6419 | 0.5016 | 0.5202 | 0.4370 |
| Original RF blend | 0.6337 | 0.5872 | 0.4847 | 0.4919 |

Per-class precision, recall and supports are also exported. All local folds have informed prior development, and spatial blocks can touch. The results do not establish independent competition-test performance, statistical significance, or a guaranteed gain after repeated model comparisons.

## Inspect, reproduce and predict

Open [catboost_experiment.ipynb](../notebooks/experiments/catboost_experiment.ipynb). Default cells verify and display saved results; set `RETRAIN = True` to retrain. The notebook includes the candidate-selection code, all selected settings, score and severe-error plots, class results, paired differences and reloaded inference. The [implementation](../scripts/catboost_experiment.py) provides training, native CatBoost model export, prediction and verification.

```bash
uv sync --locked --extra boosting --extra deep-learning
uv run --extra boosting --extra deep-learning python -m scripts.catboost_experiment
uv run --extra boosting --extra deep-learning python -m scripts.catboost_experiment --verify
```

The new optional `boosting` extra pins **CatBoost 1.2.10**; `pyproject.toml` and `uv.lock` were updated together. Prerequisites are the preprocessed city tables, baseline memberships, and saved `data/torch_transfer/` experiment, including the fine-tuned example weights and neural scaler. This run used CPU and two CatBoost threads and took approximately **153 seconds**.

Outputs under `data/catboost_experiment/` include full outer and inner probabilities, per-class scores, exact label memberships, source checkpoint results, all 15 selections, native `.cbm` files, source/target imputers, copied neural weights/scaler, split audits and a hash manifest. The reloaded bundle is a **fold-0/repetition-0 example**, not a final competition adaptation. [Machine-readable score table](results/catboost_scores.csv).

Verification recomputes **90 outer run scores from 467,856 probability rows**, plus **270 inner-fold scores**; checks exact label budgets and group exclusions; reproduces all inner selections; and checks output hashes and actual exported-model predictions. The notebook independently checks the neural hybrid formula on all outer predictions and fold-level aggregation/SD.

## Skills and primary sources

The installed **scikit-learn skill** and its model-evaluation reference informed the training-only preprocessing, spatial inner/outer separation and matched comparisons. The **Firecrawl developer-index skill** guided API checks. Firecrawl's API failed DNS resolution, so official documentation was retrieved using the web fallback. The installed `CatBoostClassifier.predict_proba` signature was also checked for tree-prefix prediction support.

Official sources: [CatBoost parameter tuning](https://catboost.ai/docs/en/concepts/parameter-tuning), [tree-prefix probability prediction](https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier_staged_predict_proba), and the [CatBoost 1.2.10 package record](https://pypi.org/project/catboost/1.2.10/). These sources support the APIs and methods; they do not predict this dataset's score.

## Conclusion

**CatBoost is implemented and evaluated, but does not improve the current model.** Its best new variant, the fixed CatBoost–neural hybrid, scores **0.5618 macro F1**, versus **0.5679** for the current RF–neural hybrid (**−0.0060**). Retain the RF–fine-tuned-MLP hybrid as the current macro-F1 candidate. This comparison does not rule out other CatBoost settings or prove a statistically reliable ranking.
