# Competition submission

## Recommended result to present

Present the **fixed 50/50 Madrid-source + Amsterdam-target random-forest blend** at **25 labeled Amsterdam locations per class**.

> With 100 target-city adaptation labels per run, the transfer baseline achieved macro F1 **0.5494 ± 0.0258 fold SD** and weighted F1 **0.5636** on five spatial development folds. Shrunk alignment reached **0.5519 macro F1**, but its small gain remained uncertain. More complex model-selection experiments did not consistently improve performance.

Show shrunk alignment as an experimental comparison, not a confirmed replacement. Do not report the 200/class result of 0.6355 as a 25/class result. Label every score as development cross-validation; no untouched competition-test result is available.

## Handoff checklist

| Deliverable | Status |
|---|---|
| Problem, labels and method description | [Methods](METHODS.md) |
| Primary and supplementary comparisons | [Results](RESULTS.md) and [complete CSV](results/all_model_scores.csv) |
| Reproducible environment | `pyproject.toml`, `uv.lock`, `.python-version` |
| Data preparation and training | `notebooks/workflow/`, supported by `scripts/` |
| Madrid source model | Local `data/baseline/madrid_source_model.joblib` |
| 25/class adaptation demonstration | Local `data/baseline/amsterdam_fold0_n25_rep0.joblib`; excludes fold 0 |
| Artifact verification | Input/model hashes, saved memberships and verification scripts |
| Presentation figure | [Results figure](assets/results.png) |
| Slide deck and team details | Prepare and review before submission |
| Organizer label/physical-metadata confirmation | Pending |
| Untouched final-test evaluation | Pending: dataset not supplied |

The fold-excluded Amsterdam demonstration is not a final deployment model fitted to a separately selected adaptation set. Preserve that distinction when packaging or describing it.

## Suggested presentation

1. **Challenge:** building-age classification across cities with scarce local labels.
2. **Data:** Madrid source, Amsterdam target, four classes and 100 adaptation labels.
3. **Method:** coherent spectral observations, temporal features, random forests and fixed probability blending.
4. **Results:** target-only 0.5393, original blend 0.5494, shrunk alignment 0.5519 macro F1 on matched original folds.
5. **Robustness:** spatial-buffer checks and unsuccessful complexity experiments; include class-level weaknesses.
6. **Conclusion:** retain the simple baseline; prioritize label quality, spectral consistency and independent testing.

## Build a clean archive

```bash
uv run python scripts/package_submission.py --include-models
```

The archive is written to `dist/competition_submission.zip`. It includes curated documentation, notebooks, executable helpers and environment files. With `--include-models`, it also includes the available baseline source and 25/class demonstration artifacts, configuration and a dedicated model hash manifest. Raw datasets, detailed local experiment outputs and private review lookups are excluded.

Before uploading, open the archive, add the final team/presentation material, confirm competition packaging rules and verify all claims against [Results](RESULTS.md). Building an archive does not submit it to the competition.
