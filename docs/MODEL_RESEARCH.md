# Research shortlist for the Amsterdam building-age task

Research checked on **4 November 2026**. This is a recommendation report, not a new training experiment. **None of the five proposed procedures has a measured score on this project.** Rankings below are judgments about task fit and practical experiment value, not a prediction of leaderboard order.

The recommended shortlist is **TabPFN, TabICLv2, ExtraTrees, semi-supervised TabDPT, and temporal change-point features combined with the existing classifier**. The first two address the 100-label setting directly; ExtraTrees is a low-cost alternative to the successful RF; TabDPT offers a way to use unlabeled training locations; change-point features address information lost by spectral aggregation.

## Project evidence that determines the shortlist

The inputs are 54 numerical spectral summaries, with annual six-band observations available upstream for 1984–2024. The label budget is **25 Amsterdam locations/class: 100 labels per run**. Madrid provides 76,263 source labels; Amsterdam has 25,992 supervised development locations, but unused labels must remain hidden from each training run. Locations, rather than annual observations, are the independent labeled units.

| Already evaluated procedure | Development macro F1 | Severe-error rate |
|---|---:|---:|
| RF + Madrid-pretrained/Amsterdam-fine-tuned MLP | **0.5679** | 0.0953 |
| RF + scratch MLP | 0.5664 | 0.1013 |
| CatBoost + fine-tuned MLP | 0.5618 | 0.0978 |
| New scratch MLP alone | 0.5534 | 0.1107 |
| Original RF source/target blend | 0.5494 | **0.0863** |

These are matched spatial development results, not untouched test scores. See [transfer results](TORCH_TRANSFER.md), [scratch ensemble](RF_SCRATCH_BLEND.md), and [CatBoost](CATBOOST.md). The benchmark to beat is **0.5679 macro F1**, while monitoring the current hybrid's **9.53% severe-error rate**.

Madrid and Amsterdam also use different historical age boundaries in the original source labels. Target-only foundation-model context avoids that mismatch initially. Arbitrarily pooling all source and target labels would change the problem and could overwhelm the 100 Amsterdam examples.

## 1. TabPFN — first model family to test

**Evidence.** The peer-reviewed TabPFN study establishes a foundation-model approach for small tabular prediction. The newer September 2026 TabPFN-3.5 technical report includes grouped and temporal benchmark settings; importantly, it does not establish a universal advantage on every non-IID subset. [Nature paper](https://doi.org/10.1038/s41586-024-08328-6), [3.5 report, Section 2.2](https://arxiv.org/html/2609.17895).

**Proposed experiment.** Use the exact 100 Amsterdam labels as the model's labeled context, on the current 54 features. Begin with a pinned local checkpoint, without Madrid context or dataset-specific weight updates. Evaluate standalone probabilities and a predeclared 50/50 blend with the original RF blend. This tests a learned prior without assuming that source-city supervision is beneficial.

**Practical choice.** TabPFN-3.5-Fast is a current speed-oriented option; full 3.5 is another option if compute permits. On the existing CPU-only machine, an older small v2 checkpoint is also a reasonable pilot, but its results must not be reported as 3.5 results. Official guidance recommends GPU inference, supports batched queries, and documents checkpoint-specific CPU limits. Model access and weight licensing are separate from the package's code license. [Official implementation](https://github.com/PriorLabs/TabPFN).

**Uncertainty.** Published small-data results and the current vendor report are not evidence for a particular Amsterdam macro F1. Competition permission for external pretrained weights is not established by the repository documents.

## 2. TabICLv2 — strongest open alternative for the same experiment

**Evidence.** The ICML 2026 paper's Appendix L.3 tests 16 training/validation examples. TabICLv2 is competitive there despite a pretraining minimum of 300 rows. The few-shot classification comparison is binary and measured by AUC; it does not directly establish four-class spatial macro-F1 performance. [Paper, Appendix L.3](https://arxiv.org/html/2602.11139).

**Proposed experiment.** Use the same 100-label Amsterdam context, unchanged features, no fine-tuning initially, and compare standalone predictions and a fixed RF blend. Keep this separate from TabPFN so results identify which pretrained prior helps. Do not append thousands of Madrid rows in the first comparison.

**Practical choice.** Pin the v2 classification checkpoint rather than using an unspecified package default. The official project publishes inference and pretraining code, weights and a scikit-learn interface. Check batching and compute on a pilot fold before running the full 15 comparisons. [Official implementation](https://github.com/soda-inria/tabicl).

**Uncertainty.** Context inference is attractive for small samples, but the synthetic pretraining distribution may not match spectral correlations or cross-city shifts. No published result found reproduces this Amsterdam/Madrid protocol.

## 3. ExtraTrees — cheap, evidence-backed alternative to the RF component

**Evidence.** The same tiny-data benchmark ranks ExtraTrees ahead of RF, CatBoost and TabM for its binary classification tasks. This is limited cross-task evidence, not a guaranteed ranking here. ExtraTrees uses randomized split thresholds, providing a distinct tree ensemble rather than another RF seed. [Benchmark table L.3](https://arxiv.org/html/2602.11139), [official estimator documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html).

**Proposed experiment.** Train separate Madrid and Amsterdam ExtraTrees classifiers and compare target-only, fixed source/target averaging, and a fixed blend with the existing fine-tuned MLP. Start with a small declared regularization grid, such as minimum leaf sizes 1, 2 and 4, selected only inside the allocated Amsterdam labels. Keep all class and feature orders aligned.

**Practical choice.** It is already available through the installed scikit-learn dependency, needs no GPU or external checkpoint, and can use the existing imputation pipeline and sample memberships. A repository search found no earlier ExtraTrees experiment in the executable helpers or public experiment reports.

**Uncertainty.** Random thresholds can trade lower variance for additional bias. ExtraTrees may share many of RF's errors; improved standalone performance would not automatically imply a better neural blend.

## 4. TabDPT with controlled pseudo-label expansion

**Evidence.** The NeurIPS 2025 paper explicitly evaluates a semi-supervised procedure with **10 true labels per class** on seven datasets. It expands context using high-confidence predictions on unlabeled training rows. The evidence concerns accuracy, not this project's macro F1, and the reported mean advantage is not large enough to justify a guaranteed-gain claim. [Paper, Section 4.6](https://arxiv.org/html/2410.18164).

**Proposed experiment.** Start from the 100 Amsterdam labels. First evaluate ordinary TabDPT; then add a bounded set of confident pseudo-labeled Amsterdam locations from training groups only. Maintain class-wise caps as a declared adaptation to avoid letting a majority predicted class dominate. Keep a no-pseudo-label control. Select any pseudo-label count/threshold only using inner spatial validation, rebuilding the pseudo-label pool inside each inner split.

**Practical choice.** The inference repository currently exposes the v1.3 family. That checkpoint is newer than the original paper's few-shot experiment; distinguish a reproduction from a new adaptation. Its dependencies include FAISS, so test Python 3.14 wheel compatibility in an isolated environment before changing the locked project setup. [Official inference repository](https://github.com/layer6ai-labs/TabDPT-inference), [dependency declaration](https://github.com/layer6ai-labs/TabDPT-inference/blob/main/pyproject.toml).

**Uncertainty.** Confident errors can reinforce themselves under geographic shift. No outer-fold features, labels or neighbors may enter the unlabeled training pool under the project's existing inductive protocol. Whether competition rules allow use of additional unlabeled target data must be checked.

## 5. Temporal segmentation / change-point features + RF–MLP hybrid

**Evidence.** Building-construction research uses Landsat temporal segmentation to identify construction-related changes; this is closer to the target than generic tabular benchmarks. LandTrendr produces segmented annual trajectories and breakpoint information. Related building studies use auxiliary information that this project does not currently have. [Construction-time study](https://doi.org/10.1109/JSTARS.2024.3409157), [LandTrendr output specification](https://emapr.github.io/LT-GEE/lt-gee-outputs.html), [related building study and its data requirements](https://isprs-archives.copernicus.org/articles/XLIX-B3-2026/589/2026/).

**Proposed experiment.** Use the already supplied, QA-filtered six-band annual series to derive breakpoint year, pre/post-change magnitude, slope, persistence, fit residual and no-detectable-change indicators. Append these to the existing summary features and compare the same classifier with and without them. A local segmented-regression implementation inspired by this literature is a new method; it must not be labeled an exact reproduction of LandTrendr or BATSCCD unless it actually is one.

**Practical choice.** This needs a new feature pipeline, not more labels or necessarily new imagery. It targets information discarded by the current means, extrema and single global slopes. Do not invent building footprints or calculate physical spectral indices before confirming scale/offset metadata. Include observation-gap and acquisition-change diagnostics so sensor artifacts are not mistaken for construction.

**Important limit.** The history begins in 1984, so it cannot directly observe the construction of the pre-1945 or 1946–1984 classes. The target is an area-weighted mean year, so a detected change is not automatically its true construction date. These are supplemental predictors, especially for the two later classes, not deterministic age labels.

## Why the shortlist differs from the earlier suggestions

- **TabM is still credible but no longer a top-five priority for this budget.** Its efficient ensembling and numerical embeddings have broad tabular evidence, but broad-dataset performance is not a 100-label guarantee. The small-sample comparison above motivates trying ExtraTrees and foundation models first. [TabM paper](https://arxiv.org/html/2410.24210).
- **SCARF is an alternative if foundation weights cannot be used.** It has evidence for contrastive pretraining with limited labels, unlike merely reconstructing inputs, but does not provide a demonstrated Amsterdam/Madrid solution. [Authors' publication record](https://research.google/pubs/scarf-self-supervised-contrastive-learning-using-random-feature-corruption/).
- **A temporal CNN/L-TAE requires a sequence pipeline and additional architecture decisions.** The change-point feature experiment provides a more interpretable first test of whether temporal information helps this target. [Satellite temporal CNN study](https://arxiv.org/abs/1811.10166).
- **A large image segmentation/ViT building-age model is not directly usable on the 54 summaries.** A recent building-age preprint requires multi-source rasters and building footprint information. Do not transfer its performance claims to these pixel-summary inputs. [October 2026 building-age preprint](https://arxiv.org/abs/2610.16320).
- **A new boosting grid or another plain MLP is lower priority.** Existing local comparisons already cover those families; different settings remain possible, but a broader grid is not the strongest next hypothesis.

## Evaluation plan and decision rule

Preserve the exact five spatial folds and three 25/class adaptation samples. Report macro F1, weighted F1, per-class precision/recall/F1, paired fold differences and severe errors. A 100-label target context does not authorize reading all other target labels. Features and labels from outer evaluation groups must stay out of preprocessing fits, pseudo-label pools, training and checkpoint selection.

For source-supervised procedures, state the source labels and their boundaries explicitly. For externally pretrained models, record package versions, checkpoint names/hashes, feature preprocessing, permitted training context and inference batching. Source-label budgets and external pretraining differ from the existing neural models; compare complete procedures honestly.

Begin with standalone candidates and fixed blends. Optimize weights only with inner predictions; never search the stored outer-fold probabilities for the most favorable mixture. These folds have been reused across many experiments, so a higher score is a candidate-discovery result. Confirm the leading procedure on a reserved geographic test or the organizer's untouched evaluation.

Benchmark evidence also argues against automatic foundation-model optimism: BeyondArena finds that grouped and temporal problems can favor conventional models. The later TabPFN-3.5 report improves those results but uses differing benchmark details, including exposing group identifiers in some evaluations. Neither is an exact match to our coordinate-excluded spectral protocol. [BeyondArena study](https://arxiv.org/abs/2606.30410), [3.5 grouped-protocol discussion](https://arxiv.org/html/2609.17895).

## Research provenance and access limits

The installed **Firecrawl research-paper/index skills** guided literature discovery and method verification; the **scikit-learn skill** guided task-fit and evaluation checks. Firecrawl's paper API failed DNS resolution. This report therefore used ordinary web search, official repositories, primary papers, publisher records and accessible full-text passages; it does **not** claim a successful Firecrawl corpus or citation-graph search.

The Nature page had an access redirect; its publisher search record was available. The BATSCCD full-text page returned 403, so only publisher-visible passages were used; it is contextual evidence rather than an implementation specification. The LandTrendr construction-time paper's DOI page required JavaScript; indexed publisher methodology passages were available. TabPFN-3.5, TabICLv2, TabDPT and TabM full-text HTML, official code repositories, and the related ISPRS building-study PDF were accessible. Vendor technical reports are distinguished from peer-reviewed papers above.

Rerun topic: few-shot four-class spectral building-age classification with 100 balanced Amsterdam labels, Madrid transfer and spatial group evaluation. Search families: tabular foundation models and small-sample benchmarks; unlabeled-data context expansion; spectral construction-time detection; benchmark behavior under grouped distribution shift. No new models were installed or trained for this research task.

## Conclusion

**Test TabPFN first, then TabICLv2. Run ExtraTrees as the cheapest fully local comparison.** Follow with controlled semi-supervised TabDPT and temporal segmentation features if the first comparisons do not improve the hybrid. Keep **0.5679** as the current measured development benchmark. There is no evidence-based numerical forecast for any of these five on this dataset, and no claim of an independent-test gain.
