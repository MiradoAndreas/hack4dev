# First random-forest baseline

Madrid spatial CV macro F1: **0.5755 ± 0.0153**. Most-frequent dummy: **0.1313 ± 0.0000**.
Amsterdam without adaptation: **0.3462 ± 0.0152** (zero local labels).

Scores are means ± sample SD across five spatial folds. Amsterdam target/transfer scores average three sampling repetitions within each fold first. Fold SD is not a confidence interval.

| Amsterdam labels per class | Target-only RF | 50/50 source + target |
| ---: | ---: | ---: |
| 5 | 0.4322 ± 0.0288 | 0.4737 ± 0.0279 |
| 25 | 0.5393 ± 0.0232 | 0.5494 ± 0.0258 |
| 50 | 0.5822 ± 0.0240 | 0.5887 ± 0.0254 |
| 100 | 0.6127 ± 0.0224 | 0.6168 ± 0.0232 |
| 200 | 0.6302 ± 0.0209 | 0.6321 ± 0.0233 |

## Method and validation

- 54 unscaled spectral/temporal predictors, 76,263 Madrid and 25,992 Amsterdam locations; observations restricted to 1984–2024.
- Median imputation is fitted inside each training pipeline. IDs, coordinates, construction years and availability indicators are excluded from predictors.
- Random forest: 150 trees, minimum leaf size 2, square-root feature sampling, bootstrap class weighting. No hyperparameter search.
- Five fixed spatial folds group 20 coordinate-key blocks. Each Amsterdam adaptation sample excludes the evaluated fold and its blocks.
- Target-only and transfer use identical nested label samples. Transfer averages the fixed Madrid model and target model probabilities with predefined equal weights.
- All scores are development cross-validation. Organizer test sets were not accessed. Spatial block boundaries can touch; no buffer was applied.
- Class boundaries reproduce existing right-closed bins (Madrid event 1960; Amsterdam event 1945). Organizer confirmation remains needed.

## Artifacts and reproduction

Run `preprocessing_age_labels.ipynb`, then `baseline.ipynb`. Trained pipelines, model bundles, fold/class metrics, predictions, memberships, plots and exact configuration are in `data/baseline/` (generated, gitignored). The Stage 1 pipeline is `madrid_source_model.joblib`; five saved Amsterdam bundles use fold 0 / repetition 0 and reference that source artifact. They are demonstration adaptations, not models trained using the evaluation fold.

The notebook verifies every F1 score against predictions, all label budgets and spatial exclusions, model reload predictions, and exported files. No scores are inferred from training accuracy.

## Skills

Applied the downloaded K-Dense-AI scikit-learn skill and obra verification-before-completion skill. Provenance is in `docs/upstream-skills/`. These directly read copies are not registered `.agents` installations.

Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. https://doi.org/10.48550/arXiv.2609.00065.
