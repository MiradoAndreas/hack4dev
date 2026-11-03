# Source-representation transfer experiment

Frozen Madrid forest leaves are one-hot encoded and used by a regularized Amsterdam classifier. Eleven recipes are selected using spatial inner validation restricted to each permitted label budget; the selected components are then refitted on those same labels. No additional Amsterdam labels or external datasets are used.

| Labels/class | Original blend | Nested selected transfer | Paired Δ macro F1 | Leaf-only C=1 ablation |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 0.4737 ± 0.0279 | 0.4652 ± 0.0391 | -0.0086 | 0.4235 ± 0.0317 |
| 25 | 0.5494 ± 0.0258 | 0.5435 ± 0.0252 | -0.0059 | 0.5074 ± 0.0183 |
| 50 | 0.5887 ± 0.0254 | 0.5889 ± 0.0242 | +0.0002 | 0.5443 ± 0.0195 |
| 100 | 0.6168 ± 0.0232 | 0.6136 ± 0.0269 | -0.0031 | 0.5897 ± 0.0165 |
| 200 | 0.6321 ± 0.0233 | 0.6289 ± 0.0186 | -0.0032 | 0.6115 ± 0.0188 |

Average F1 across the five budgets: 0.5680; average paired change: -0.0041.
Budgets with positive mean paired change: 1 of 5.

## Validation and limitations

- Five outer spatial folds, three identical seeded label-sampling repetitions per fold. Means average repetitions within a fold first; SD is across five fold means, not a confidence interval.
- Inner grouped validation uses only the 5/25/50/100/200 labels per class. It refits imputation, leaf vocabularies and heads inside each inner training split. No evaluation labels or evaluation-feature statistics are fitted.
- Candidate choices are based on mean inner macro F1, with a predefined 0.005 tie tolerance favoring earlier candidates. Diagnostics are saved, including unstable low-data selections and fallbacks.
- Reconstructed original-blend predictions match every original evaluation run, verifying that comparisons reuse exactly the same samples and random states.
- The C=1 leaf-only result is a prespecified ablation. It is not used retrospectively to choose a winner or change the nested selector.
- The baseline scores had already been viewed and informed this follow-up design. These are iterative development results, not a new independent test; confirm any gains on organizer-held-out data.
- Class boundaries remain city-specific; block size is in coordinate-key units and neighboring blocks can touch. Existing label and spatial-independence limitations remain.

## Reproduction and artifacts

Run `preprocessing_age_labels.ipynb`, `baseline.ipynb`, then `transfer_improved.ipynb`. The callable training/inference code is `scripts/transfer_adaptation.py`. Generated artifacts are in `data/transfer_improved/`, including five example fold-0 adapters, inner scores, selected recipes, class metrics, predictions, charts, input hashes and exact settings. The source model is reused from `data/baseline/madrid_source_model.joblib`. No runtime dependencies changed.

## Skills

Applied the downloaded scikit-learn and verification-before-completion instructions in `docs/upstream-skills/` (directly readable copies, not registered installations).
Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. https://doi.org/10.48550/arXiv.2609.00065.
