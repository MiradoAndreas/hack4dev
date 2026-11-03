# Reproduction guide

## Environment and inputs

The project requires Python ≥3.12; `.python-version` selects Python 3.14. Use the locked environment:

```bash
uv sync --locked
uv run --with jupyter jupyter lab
```

Required local datasets:

```text
data/madrid_train.parquet
data/amsterdam_data.parquet
```

Raw data and fitted artifacts remain local. Package versions, ordered feature schemas, seeds, memberships and hashes are saved beside each experiment. Load only trusted model artifacts in a compatible environment.

## Notebook execution order

Run notebooks top to bottom in fresh kernels. Their first cell finds the repository root and sets it as the working directory, so paths behave consistently when launched from any notebook folder.

| Order | Notebook | Purpose |
|---:|---|---|
| 1 | [Data audit](../notebooks/workflow/eda.ipynb) | Dataset integrity, quality and feasibility |
| 2 | [Preprocessing](../notebooks/workflow/preprocessing_age_labels.ipynb) | Labels, features, spatial folds and sample memberships |
| 3 | [Baseline](../notebooks/workflow/baseline.ipynb) | Madrid model and Amsterdam controls/blend |
| 4 | [Leaf adapters](../notebooks/experiments/transfer_improved.ipynb) | Supplementary nested adapter experiment |
| 5 | [Alignment](../notebooks/experiments/transfer_alignment.ipynb) | Full and shrunk moment alignment |
| 6 | [Grid search](../notebooks/experiments/transfer_grid_search.ipynb) | Nested forest/blend search |
| 7 | [25/class experiments](../notebooks/experiments/improvement_25.ipynb) | Robust features, calibration and spatial sensitivity |
| 8 | [Evaluation](../notebooks/workflow/evaluation.ipynb) | Baseline/adapter/alignment diagnostics and model checks |

Steps 1–3 reproduce the recommended baseline. The remaining experiments provide supporting comparisons. The evaluation notebook expects the baseline, leaf-adapter and alignment artifacts. The 25/class and grid notebooks expose explicit training switches; saved-output verification is their default route where configured.

The four original numbered notebooks are retained under [notebooks/reference](../notebooks/reference) for the challenge specification and starter workflow. Their filenames are preserved. Their results are not substituted for the audited competition pipeline above.

## Command-line experiments and verification

For a live demonstration with the existing saved baseline artifacts:

```bash
uv run python -m scripts.demo_model
```

This loads the saved 25/class adaptation, verifies hashes and spatial separation,
and predicts 5,197 locations in its excluded fold. It prints source-only,
target-only and blended scores, a recomputed five-fold comparison, class metrics,
a confusion matrix and eight fixed random examples. The blend does not beat
target-only on every fold; present both the single-fold outcome and full
development comparison. This command does not retrain or establish final-test
performance.

After preprocessing and baseline artifacts exist:

```bash
uv run python -m scripts.grid_transfer
uv run python -m scripts.verify_grid_transfer

uv run python -m scripts.audit_acquisitions
uv run python -m scripts.check_improvement_invariants
uv run python -m scripts.improve_25
uv run python -m scripts.verify_improvement_25
```

The grid search and full spatial-sensitivity run train many forests. Do not rerun them merely to read the published result tables.

Optional explanation checks use a temporary dependency overlay:

```bash
uv run python -m scripts.independent_evaluation --freeze
uv run --with 'shap==0.52.0' python -m scripts.explain_improvement_25
uv run python -m scripts.check_independent_evaluation
```

The freeze command creates a fixed demonstration candidate for future evaluation; it does not promote the unsuccessful selector to production. Explanations do not fit predictive models.

## Artifact map

| Folder | Contents |
|---|---|
| `data/eda/` | Data audit tables |
| `data/preprocessed_age_labels/` | Deterministic features, labels, folds and configuration |
| `data/baseline/` | Recommended source/target pipelines, predictions and memberships |
| `data/transfer_improved/` | Leaf-adapter experiment |
| `data/transfer_alignment/` | Alignment models and results |
| `data/transfer_grid_search/` | Inner scores, selected parameters and held-out results |
| `data/evaluation/` | Recomputed metrics, error slices and probability diagnostics |
| `data/improvement_25/` | Complete 25/class probability exports, checks and review queue |
| `reports/` | Detailed per-run reports; separate from curated documentation |

The complete published score table is [all_model_scores.csv](results/all_model_scores.csv). Refresh it from saved class metrics with:

```bash
uv run python -m scripts.export_scores
```

## Future independent evaluation

No independent dataset was supplied. Before scoring one, complete the data contract using organizer/data-owner evidence: age definitions, reflectance encoding, prediction date, feature construction and independence from development.

```bash
uv run python -m scripts.independent_evaluation \
  --input /path/features.parquet \
  --contract /path/confirmed-contract.json \
  --output /path/final-report.json
```

The evaluator checks the feature schema and frozen model hashes, rejects known development IDs and overlapping coordinates, and prevents silent report replacement or rescoring of the same model/dataset pair. Add `--monitor` for unlabeled distribution monitoring; it produces no predictive-performance score.

## Documentation and artifact integrity

Notebook outputs are cleared in the repository to keep review and diffs compact. Curated tables and the results figure remain available without executing training.

During the documentation reorganization, only report wording/locations and a module docstring changed in the grid and fixed-budget training helpers. Their computational syntax trees were verified unchanged. Original training-source hashes and copies are retained locally under `data/documentation_migration/`; current verification hashes identify the reorganized implementation. Model files and saved prediction scores were preserved.
