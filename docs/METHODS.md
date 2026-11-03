# Methods

## Task and evaluation unit

Predict one of four building-age classes for each spectral observation location. Madrid is the source city; Amsterdam is the target city. A location observed over multiple years is **one labeled example**, not one example per year.

The primary comparison uses 25 Amsterdam labels per class. Earlier experiments also cover 5, 50, 100 and 200 labels per class; those results are supplementary and must retain their budget labels.

## Data and labels

| Property | Madrid | Amsterdam |
|---|---:|---:|
| Supervised locations | 76,263 | 25,992 |
| Class 1 locations | 13,794 | 7,568 |
| Class 2 locations | 27,152 | 8,802 |
| Class 3 locations | 18,672 | 6,745 |
| Class 4 locations | 16,645 | 2,877 |
| Current class-1 upper boundary | 1960 | 1945 |

Targets are derived from the supplied area-weighted mean construction year. Only usable, finite targets within 1850–2024 are retained; inconsistent location labels are excluded. Current bins are right-closed:

1. Up to the city-specific historical boundary.
2. After that boundary through 1984.
3. After 1984 through 2004.
4. After 2004 through 2024.

These definitions reproduce the current workflow. Organizer confirmation is still required: source and target class-1/2 intervals differ, and the mean year of a mixed-age pixel is not necessarily its dominant building-age class.

## Spectral preparation

Use observations from 1984–2024 with coverage of at least 0.15. Select the first supplied QA-valid scene slot with complete finite values for all six bands: Blue, Green, Red, NIR, SWIR1 and SWIR2. All bands for a year come from the same scene slot.

The baseline uses **54 unscaled spectral/temporal features**: nine summaries per band—mean, standard deviation, median, minimum, maximum, early mean, late mean, late-minus-early difference and annual slope. The early/late feature split is 2003; this differs from the 2004 target-class boundary. IDs, coordinates, construction years and availability indicators are excluded from predictors.

Median imputation is fitted inside each training pipeline. Spectral summaries are calculated per location without reading its target. Future predictions must use the same observation-history window and feature order.

A follow-up experiment tested 84 robust and seasonal summaries, including quartiles, IQRs, period medians, change summaries and fixed day-of-year quarters. Empty periods remain missing until training-only imputation. Physical indices remain disabled because reflectance scale and offset have not been confirmed; `scripts/audit_acquisitions.py` contains a metadata-gated NDVI utility.

## Recommended model

The Madrid source pipeline contains a random forest with 150 trees, minimum leaf size 2, square-root feature sampling and balanced bootstrap class weighting. The Amsterdam target forest uses the same fixed settings and exactly the permitted adaptation sample.

For each location:

```text
transfer probability = 0.5 × source probability + 0.5 × target probability
predicted class      = class with the largest transfer probability
```

No outer evaluation labels choose the blend weight. Source-only and target-only models are retained as controls. Saved Amsterdam bundles are fold-excluded demonstrations, not a claim that a final production adaptation has been selected.

## Validation and metrics

Use five spatial outer folds and three seeded target-sampling repetitions. Group locations into coordinate-key blocks; exclude the evaluated locations and blocks from adaptation. Select any tuned model using spatial inner folds restricted to the allocated target labels. Average repetitions within each fold, then average the five fold means.

Primary metric: **macro F1**, which weights each age class equally. Also report weighted F1, per-class precision/recall/F1, accuracy and errors spanning at least two age classes. Ordinal class distance is not construction-year error.

Follow-up validation tested 2- and 5-coordinate-key-unit exclusion buffers and 40-key-unit blocks. Physical coordinate units and building footprints are not confirmed, so these are location/block checks rather than verified building-level independence. Buffering changes the eligible adaptation sample; larger blocks also change outer-fold membership.

Fold SD is descriptive, not a confidence interval. Paired 500-draw spatial-block bootstrap intervals are conditional diagnostics for fixed fitted models. They omit retraining uncertainty, residual spatial dependence, cross-fold dependence and correction for repeated experimentation.

## Additional experiments

- Full and shrunk moment alignment; shrinkage prior count fixed at 40.
- Source-tree leaf representations with nested adapter selection.
- A 32-configuration target-forest/blend grid with spatial inner selection.
- At 25/class: robust features, regularized forests, cumulative ordinal forests, sigmoid calibration and inner-selected abstention.
- Calibration confined to training partitions; complete held-out probability exports for the 25/class follow-up.
- Six-band grouped SHAP explanations using training backgrounds, multiple background sizes/seeds and probability-reconstruction checks.

Candidate performance is reported in [Results](RESULTS.md). None of these experiments justifies a universal improvement claim.

## Interpretation boundaries

All local data have informed development. Nested validation within an experiment does not erase reuse of outer results across experiments. No untouched organizer test was available. Acquisition quality and spectral distributions differ between cities; the raw-data audit flags these differences without automatically changing physical units, QA rules or labels.

Official method references: [random forests](https://scikit-learn.org/stable/modules/ensemble.html#forest), [pipelines](https://scikit-learn.org/stable/modules/compose.html), [cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html), [probability calibration](https://scikit-learn.org/stable/modules/calibration.html), [SHAP](https://shap.readthedocs.io/) and [USGS NDVI](https://www.usgs.gov/landsat-missions/landsat-normalized-difference-vegetation-index).
