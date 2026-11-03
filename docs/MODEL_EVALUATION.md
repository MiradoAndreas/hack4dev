# Model evaluation report

Madrid random forest: **macro F1 0.5755 ± 0.0153**, accuracy 0.5942, balanced accuracy 0.5646, MCC 0.4392. Most-frequent dummy macro F1: 0.1313 ± 0.0000.
Amsterdam source-only (zero local labels): **macro F1 0.3462 ± 0.0152**.

These are spatial development cross-validation results, independently recomputed from saved predictions. No untouched test set is available locally. Final Madrid training-set predictions are not used as evaluation.

| Labels/class | Original blend macro F1 | Shrunk alignment macro F1 | Paired Δ F1 | Improved folds |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 0.4737 ± 0.0279 | 0.4824 ± 0.0212 | +0.0087 | 5/5 |
| 25 | 0.5494 ± 0.0258 | 0.5519 ± 0.0242 | +0.0025 | 4/5 |
| 50 | 0.5887 ± 0.0254 | 0.5906 ± 0.0232 | +0.0019 | 3/5 |
| 100 | 0.6168 ± 0.0232 | 0.6132 ± 0.0231 | -0.0035 | 1/5 |
| 200 | 0.6321 ± 0.0233 | 0.6295 ± 0.0199 | -0.0026 | 2/5 |

Means first average three sampling repetitions within each fold; SD is across five fold means. It is not a confidence interval. Five budgets are all reported; no retrospective hybrid model is scored.

## Class-level results at 25 labels/class

| Class | Baseline precision | Baseline recall | Baseline F1 | Alignment precision | Alignment recall | Alignment F1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.7228 | 0.5687 | 0.6337 | 0.7206 | 0.5714 | 0.6353 |
| 2 | 0.5169 | 0.6836 | 0.5872 | 0.5404 | 0.6151 | 0.5737 |
| 3 | 0.6134 | 0.4075 | 0.4847 | 0.5602 | 0.4541 | 0.4977 |
| 4 | 0.4487 | 0.5639 | 0.4919 | 0.4263 | 0.6224 | 0.5008 |

Lowest mean class F1 for alignment at 25/class: class 3 (0.4977); inspect its precision/recall tradeoff and confusion rows before proposing changes.

## Score trade-offs at 25 labels/class

Compared with the original blend, alignment changes accuracy from 56.52% to 56.14%, balanced accuracy from 55.59% to 56.57%, and MCC from 0.4038 to 0.4015. Errors spanning at least two classes rise from 8.63% to 9.85%. The small macro-F1 gain therefore does not imply better performance on every criterion.

## Stability and probability quality

- At 25/class, the conditional block-resampling interval for paired F1 change is [-0.0050, +0.0090], including zero. This is a limited fixed-model uncertainty diagnostic, not evidence of a confirmed generalization gain.
- On fold 0 / repetition 0 at 25/class, baseline versus alignment log loss is 1.0546 versus 1.0629, Brier is 0.5826 versus 0.5843, and ten-bin ECE is 0.0891 versus 0.0980. Probability quality does not improve on these criteria in this saved example fold.
- Paired spatial-block bootstrap intervals and draws are exported. They are conditional on the saved fitted models and fixed adaptation samples. They do not include full retraining uncertainty, cross-validation dependence or correction for trying several methods, and are not independent-test confidence intervals.
- Probabilities were freshly generated from the saved source model and ten saved baseline/alignment models. All predictions matched recorded results. Log loss, multiclass Brier, macro OVR AUC, macro average precision and ten-bin ECE are exported for Amsterdam fold 0 / repetition 0 only; they are not five-fold averages.
- Reliability diagrams show probability behavior without fitting calibration on evaluation data. Probability metrics from one fold cannot establish calibration throughout the city.

## Evaluation limits and practical conclusion

- The shrunk alignment shows modest development gains at 5, 25 and 50 labels/class, and regressions at 100 and 200. The original baseline remains the reference; there is no demonstrated universal transfer improvement.
- The development outer-fold results have been viewed repeatedly and informed experiment design. Neither the apparent best method nor the best budget has an unbiased final-test score here.
- Spatial blocks stay separate, but touching blocks may remain correlated. Current class 1/2 historical boundaries and endpoint conventions require organizer confirmation, and area-weighted construction-year labels can be ambiguous in mixed-age pixels.
- Error slices by building coverage, observation history and proximity to class boundaries are exploratory and retain their counts/class coverage. Class-distance error is ordinal, not construction-year error.
- Reserve organizer test data for a frozen final procedure. More elaborate calibration, feature selection or blend-weight tuning must use training/inner folds, not these evaluation labels.

## Reproduction and artifacts

Run `evaluation.ipynb` after preprocessing, baseline and transfer experiments. It fits no models and does not alter prior models or predictions. Detailed tables, figures, probability predictions, input hashes and the evaluation manifest are in `data/evaluation/`.

Verification: 465 evaluation-run scores recomputed from 2,517,798 saved prediction rows; full city/location coverage, label agreement, folds and all 75 adaptation budgets checked. All probability predictions from saved models reproduce recorded labels.

## Skills and software

Applied the downloaded scikit-learn model-evaluation and verification-before-completion skills. They are directly readable instruction copies in `docs/upstream-skills/`, not registered `.agents` installations. No runtime dependencies changed.
Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. https://doi.org/10.48550/arXiv.2609.00065.
