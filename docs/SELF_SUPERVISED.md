# Self-supervised learning + MLP experiment

The fixed denoising-autoencoder + MLP configuration achieved **0.5304 macro F1 ± 0.0254 spatial-fold SD**, with **0.5473 weighted F1** and **0.5462 accuracy** at 25 Amsterdam labels per class (100 labels per run).

| Method | Macro F1 | Fold SD | Weighted F1 | Accuracy |
|---|---:|---:|---:|---:|
| Denoising SSL + MLP | 0.5304 | 0.0254 | 0.5473 | 0.5462 |
| Raw-feature MLP | 0.5339 | 0.0253 | 0.5485 | 0.5464 |
| Original RF blend | 0.5494 | 0.0258 | 0.5636 | 0.5652 |

All scores use the original five spatial development folds and three matched adaptation-label samples per fold. Repetitions are averaged within folds first. Fold SD is descriptive, not a confidence interval. These previously used development folds are not an independent competition test.

## Method and controls

The self-supervised task reconstructs clean spectral summaries after randomly masking 20% of standardized inputs. An MLP regressor implements a 54 → 64 → 16 → 64 → 54 autoencoder. Its first two hidden layers form a frozen encoder. A 16 → 32 → 4 MLP predicts age class from the embedding using the saved 25/class Amsterdam samples. The classifier uses L-BFGS and fixed L2 regularization; outer labels never select architecture or training duration.

Each outer fold pretrains on 20,000 sampled Madrid locations and 20,000 sampled Amsterdam locations outside the validation blocks. Madrid age labels are not read. Within the pretraining pool, a group-disjoint 10% checkpoint split selects the reconstruction epoch, up to 60 epochs with patience eight. Median imputation and standardization fit only on the pretraining fit partition. Fresh masks are used for training each epoch; checkpoint masks are fixed. Neither features nor labels from held-out Amsterdam blocks enter any fitting stage.

The raw-feature MLP control has the same classifier width, regularization, solver and target labels, but its imputation/scaling fit only on the labeled sample. Thus this is a comparison of the complete SSL procedure, including access to unlabeled data, rather than an isolated effect of the denoising loss. The RF blend also uses supervised Madrid labels; its source supervision budget differs from SSL.

Only the baseline's explicit 54-feature allowlist enters the models. Coordinates define spatial exclusions but are not predictors. Construction year, age class and label-derived eligibility fields are excluded from self-supervision.

## Conclusion

After increasing the classifier iteration budget from 1,500 to 5,000 in response to convergence warnings, all raw MLP fits and 14 of 15 SSL MLP fits completed without a convergence warning. One SSL fit still reported a warning at the fixed budget; its predictions remain included, and optimization may be incomplete. No architecture or regularization was changed in response to validation scores.

This configuration scores **0.0035 lower macro F1 than the raw MLP** and **0.0190 lower than the RF blend**. It provides no mean-score improvement and does not support replacing the recommended RF blend. The small SSL-versus-MLP difference does not establish a reliable ranking beyond these runs.

The autoencoder learns to reconstruct its input, but reconstruction error does not measure age discrimination. A frozen compressed representation may discard useful class information; this experiment does not determine the cause of the observed difference or rule out other self-supervised objectives or fine-tuning. Touching spatial blocks, prior development on these folds, and the absence of an untouched geographic test remain limitations.

## Reproduce and inspect

Open [self_supervised.ipynb](../notebooks/experiments/self_supervised.ipynb) and run all cells, or train with:

```bash
uv run python -m scripts.self_supervised
```

The [training helper](../scripts/self_supervised.py) uses existing locked dependencies. Generated artifacts are under `data/self_supervised/` and excluded from version control, consistent with the other experiments. They include scores, per-class metrics, all MLP validation probabilities, pretraining memberships, reconstruction histories, split checks, example model artifacts, and a SHA-256 experiment manifest. Notebook cells verify saved prediction scores and display loss curves, fold scores and confusion matrices.

Implementation guidance: the installed scikit-learn skill and its model-evaluation/supervised-learning references. Public API contracts were checked against the official [MLPRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html) and [MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html) documentation.
