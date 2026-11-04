# RF + scratch-trained MLP ensemble

This ensemble was **not previously implemented**: the earlier transfer experiment trained the scratch network but blended RF only with the fine-tuned network. We have now retrained the scratch networks and tested the fixed blend.

The new blend reaches **0.5664 macro F1**, improving on the RF baseline (**0.5494**) by **+0.0170**, but trailing the existing fine-tuned hybrid (**0.5679**) by **−0.0015**. Its severe-error rate is also higher than the fine-tuned hybrid's. Retain the existing fine-tuned hybrid as the current macro-F1 candidate; this experiment provides no mean-score improvement over it. The small F1 difference does not establish a statistically reliable ranking.

## Model and protocol

```text
P(new ensemble) = 0.25 × P(Madrid RF)
                + 0.25 × P(Amsterdam RF)
                + 0.50 × P(scratch MLP)
```

The scratch MLP is **54 → 64 → 32 → 4**, with ReLU and dropout 0.2. All neural weights start randomly. AdamW trains all layers at learning rate **0.001**, weight decay **0.01**, for **300 fixed full-batch epochs**. There is one network per fold/repetition, not a five-seed neural ensemble. Class IDs 1–4 map to 0–3 for cross entropy; probabilities align to classes 1–4 before averaging.

To isolate the new blend from the earlier scratch control, median imputation and standardization fit only on Madrid folds 1–4. Thus the neural model uses source features for preprocessing, but **no Madrid labels or pretrained weights**. The Madrid RF component still uses Madrid labels. This distinction matters: the whole ensemble is not source-label-free.

Each of the five original spatial folds uses the same three baseline adaptation samples, **25 Amsterdam locations/class (100 total)**. All 15 scratch models are freshly trained. Their full validation probabilities reproduce the saved scratch-control probabilities within 1e-6. RF and fine-tuned-hybrid probabilities are reused from the verified transfer experiment. The exported fold-0 scratch bundle also uses the actual saved RF pipelines; its reloaded inference matches the evaluation probabilities within 1e-6.

The blend weight, architecture, epoch count and seeds are fixed before the new evaluation. No outer labels select them. No extra Amsterdam labels are used, and held-out locations/groups never enter target training. Repetitions are averaged within folds before averaging fold means. The same Madrid preprocessing is shared across runs. Source RF training and source-scaler variability are not measured here.

## Results

| Model | Macro F1 | Fold SD | Weighted F1 | Severe errors |
|---|---:|---:|---:|---:|
| Existing RF + fine-tuned MLP | **0.5679** | 0.0158 | 0.5769 | 0.0953 |
| **New RF + scratch MLP** | **0.5664** | 0.0204 | **0.5774** | 0.1013 |
| Scratch MLP alone | 0.5534 | 0.0199 | 0.5634 | 0.1107 |
| Original RF blend | 0.5494 | 0.0258 | 0.5636 | **0.0863** |

Severe errors span two or more age classes. The new ensemble improves on RF on all five fold means. Compared with the fine-tuned hybrid, it leads on three folds and trails on two, with its largest loss on fold 4. Its mean accuracy is 0.5772 versus 0.5767 for the fine-tuned hybrid; the primary metric remains macro F1. Fold SD is descriptive, not a confidence interval. No significance test is claimed.

### Per-class F1

| Model | Class 1 | Class 2 | Class 3 | Class 4 |
|---|---:|---:|---:|---:|
| Existing RF + fine-tuned MLP | 0.6514 | 0.5571 | 0.5437 | 0.5192 |
| New RF + scratch MLP | 0.6467 | 0.5557 | 0.5596 | 0.5036 |
| Original RF blend | 0.6337 | 0.5872 | 0.4847 | 0.4919 |

The scratch hybrid improves class 3 relative to the fine-tuned hybrid, but loses ground in the other three classes. It does not dominate the current model.

## Run and inspect

Open [rf_scratch_blend.ipynb](../notebooks/experiments/rf_scratch_blend.ipynb). It verifies and displays the experiment by default; set `RETRAIN = True` to retrain all scratch networks. It includes score plots, training curves, paired fold differences and a reloaded inference example. The [implementation](../scripts/rf_scratch_blend.py) contains training, fixed blending, artifact export, prediction and verification. No dependency changes are required.

```bash
uv run --extra deep-learning python -m scripts.rf_scratch_blend
uv run --extra deep-learning python -m scripts.rf_scratch_blend --verify
```

Prerequisites are the preprocessed city tables, baseline models/memberships and saved `data/torch_transfer/` experiment. Run the existing baseline and transfer workflow first if absent. The earlier [complete hybrid notebook](../notebooks/workflow/hybrid_rf_pytorch.ipynb) remains the standalone training notebook for the fine-tuned hybrid.

Artifacts under `data/rf_scratch_blend/` include all held-out probabilities, run/fold/class metrics, training curves, label memberships, paired differences, a preprocessing pipeline, model weights and hashes. [Score CSV](results/rf_scratch_blend_scores.csv). The model bundle is a **fold-0/repetition-0 demonstration**, trained without that fold, not a new final competition model.

Verification checks **60 run scores from 311,904 probability rows**, the fixed mixture formula, all 15 budgets and spatial exclusions, recorded hashes, and actual reloaded model inference. The notebook also recomputes fold aggregation and SD from the run table.

Guidance: installed scikit-learn skill and its model-evaluation reference, plus official [probability-voting documentation](https://scikit-learn.org/stable/modules/ensemble.html#voting-classifier). All reported scores are development results on previously used folds, not independent-test performance.

## Conclusion

**Implemented and evaluated: 0.5664 macro F1. Keep the existing fine-tuned hybrid at 0.5679 for now.** The scratch ensemble is competitive, but does not improve the primary mean score and has more severe errors. A multi-seed scratch ensemble remains untested; do not tune the blend weight on these outer-fold results.
