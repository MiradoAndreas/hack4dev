# Spatial nested grid search for transfer learning

Frozen Madrid source forest plus an Amsterdam target forest. Exhaustive search evaluates 32 configurations: max_depth ∈ {None, 12}, max_features ∈ {sqrt, 0.5}, min_samples_leaf ∈ {2, 1}, Madrid source weight ∈ {0.5, 0.25, 0.75, 0}. 150 trees and balanced_subsample class weighting remain fixed.

Each configuration is selected by mean macro F1 over up to three spatial inner folds (two only if needed to retain all training classes). Imputation fits inside each inner training pipeline. The selected forest is refitted on the same limited adaptation sample. A zero source weight permits rejecting harmful transfer.

| Amsterdam labels/class | Original blend | Grid-selected | Paired Δ F1 | Improved folds |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 0.4737 ± 0.0279 | 0.4543 ± 0.0434 | -0.0194 | 0/5 |
| 25 | 0.5494 ± 0.0258 | 0.5424 ± 0.0235 | -0.0070 | 0/5 |
| 50 | 0.5887 ± 0.0254 | 0.5858 ± 0.0284 | -0.0029 | 3/5 |
| 100 | 0.6168 ± 0.0232 | 0.6174 ± 0.0247 | +0.0007 | 2/5 |
| 200 | 0.6321 ± 0.0233 | 0.6355 ± 0.0236 | +0.0034 | 3/5 |

Means average three matched sampling repetitions within each of five outer spatial folds first. ± is sample SD across fold means, not a confidence interval. Outer locations and spatial blocks are excluded from adaptation, preprocessing fits and selection. No additional Amsterdam labels are used. Baseline predictions are reconstructed and matched exactly for all 75 runs.

## Selected configurations

The optimum is conditional on the grid and each permitted training sample. There is no single universal optimum selected from outer scores. Most frequently selected inner-CV configuration per budget (a descriptive count, not a separately evaluated fixed model):

| Labels/class | max_depth | max_features | min_samples_leaf | Madrid weight | Selected runs |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | None | sqrt | 2 | 0.5 | 4/15 |
| 25 | None | 0.5 | 1 | 0.25 | 4/15 |
| 50 | None | sqrt | 2 | 0.25 | 3/15 |
| 100 | None | sqrt | 2 | 0.25 | 3/15 |
| 200 | 12 | sqrt | 2 | 0.25 | 4/15 |

## Limits and reproduction

These are iterative development results: these outer folds were inspected in previous experiments. No untouched organizer test data is available locally or used here. Spatial blocks can touch, and city-specific class boundaries still require confirmation. A higher development score does not establish a statistically confirmed gain or final-test performance. Do not retune the grid using these outer scores.

Run preprocessing_age_labels.ipynb, baseline.ipynb, then transfer_grid_search.ipynb. The notebook calls scripts/grid_transfer.py. Artifacts are in data/transfer_grid_search/: full candidate/fold scores, selected parameters for every run, inner memberships, held-out predictions, per-class precision/recall/F1, paired fold differences, hashes, versions and five demonstration models excluding outer fold 0. Dependencies are unchanged.

## Skills

Web discovery confirmed the [K-Dense scikit-learn skill](https://github.com/K-Dense-AI/scientific-agent-skills/blob/main/skills/scikit-learn/SKILL.md). Applied its directly readable copy and model-evaluation/pipeline references in docs/upstream-skills/, plus the verification-before-completion instructions. These are not registered .agents installations.

Kassis et al. (2026), Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. https://doi.org/10.48550/arXiv.2609.00065

Method reference: [scikit-learn nested cross-validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).
