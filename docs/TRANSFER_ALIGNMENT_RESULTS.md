# Inductive feature-alignment transfer results

Two fixed procedures align target feature means and scales toward a class-balanced Madrid reference, using only the budget-limited adaptation sample. Both combine aligned-source and target-forest probabilities with fixed equal weights.

| Labels/class | Original blend | Full alignment | Δ F1 | Shrunk alignment | Δ F1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 0.4737 ± 0.0279 | 0.4580 ± 0.0217 | -0.0158 | 0.4824 ± 0.0212 | +0.0087 |
| 25 | 0.5494 ± 0.0258 | 0.5437 ± 0.0242 | -0.0057 | 0.5519 ± 0.0242 | +0.0025 |
| 50 | 0.5887 ± 0.0254 | 0.5837 ± 0.0213 | -0.0050 | 0.5906 ± 0.0232 | +0.0019 |
| 100 | 0.6168 ± 0.0232 | 0.6107 ± 0.0241 | -0.0060 | 0.6132 ± 0.0231 | -0.0035 |
| 200 | 0.6321 ± 0.0233 | 0.6283 ± 0.0185 | -0.0038 | 0.6295 ± 0.0199 | -0.0026 |

Five outer spatial folds and three matched label-sampling repetitions; repetitions average within a fold before mean/SD. SD is descriptive, not a confidence interval. All samples and target-forest seeds match baseline, verified by reconstructing every old prediction.

All source reference statistics use Madrid training data. Target statistics, imputation and target trees fit only the allocated Amsterdam labels; evaluation locations/blocks are excluded. No added local labels or runtime dependencies. Prior count 40 and variance floor 0.01 are fixed design choices, not selected from outer scores.

This follow-up was motivated by baseline distribution shift and early source-leaf results. These are iterative development results; no variant is declared an unbiased winner based on this comparison. Confirm gains on untouched organizer data. Label-boundary and touching-block limitations remain.

Run `transfer_alignment.ipynb` after preprocessing and baseline. Helpers are in `scripts/moment_transfer.py`; artifacts and exact settings are in `data/transfer_alignment/`. Ten saved demonstration adapters exclude Amsterdam fold 0. The separate leaf-selection experiment is retained in `transfer_improved.ipynb` and `docs/TRANSFER_RESULTS.md`.

Skills: downloaded scikit-learn and verification-before-completion instructions in `docs/upstream-skills/`. Citation: Kassis et al. (2026), *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*, https://doi.org/10.48550/arXiv.2609.00065.
