# Repository Guidelines

## Local working instructions

Read the relevant installed instructions under `.agents/skills/<name>/SKILL.md`
when applying data analysis, model evaluation, statistics, or explanation tools.
Keep these local development resources separate from competition deliverables.
Public documentation is maintained in `docs/`; see `README.md` for its index.
Preserve evidence-backed evaluation claims and distinguish development results
from independent-test performance.

## Project Structure & Module Organization

This project explores building-age classification using spectral data from Amsterdam and Madrid. The workflow lives in four reference notebooks under `notebooks/reference/`: `1-Introduction.ipynb`, `2-Reading_Data.ipynb`, `3-Preprocessing.ipynb`, and `4-Modelling.ipynb`. Follow their numbered order. The active competition workflow is under `notebooks/workflow/`, with supplementary comparisons under `notebooks/experiments/`. The recommended model is a fixed source/target random-forest probability blend.

Raw Parquet datasets live in `data/`; preprocessing writes `data/preprocessed/preprocessed_data.pkl`. Executable training and verification helpers are under `scripts/`. Dependencies are declared in `pyproject.toml` and locked in `uv.lock`. There is no separate source package or test directory.

## Build, Test, and Development Commands

- `uv sync --locked`: install the locked project dependencies.
- `uv run python -m scripts.verify_improvement_25`: verify saved 25/class experiment artifacts; this does not retrain models.
- `uv run --with jupyter jupyter lab`: launch Jupyter locally without adding it to project dependencies.

The project requires Python 3.12 or newer; `.python-version` selects Python 3.14. Run notebooks sequentially and check dataset paths before execution. Notebook bootstrap cells normalize the working directory to the repository root. The original reference workflow uses `data/preprocessed/preprocessed_data.pkl`; the active competition workflow uses `data/preprocessed_age_labels/`.

## Coding Style & Naming Conventions

Use four-space indentation, `snake_case` for functions and variables, and uppercase names for configuration constants. Group imports near the start of notebooks. Preserve numbered notebook filenames and use Markdown cells to explain feature engineering and experiments. Declare dependency changes in `pyproject.toml` and update `uv.lock` together. No formatter or linter is configured.

## Testing Guidelines

No automated test framework or coverage threshold is configured. Validate changes by restarting the kernel and running affected notebooks from top to bottom, including upstream preprocessing when needed. Preserve fixed random seeds for reproducibility. Report macro F1, class-level results, and validation splits; distinguish in-sample scores from held-out and cross-city evaluation. Keep evaluation data out of preprocessing fits and training.

## Commit & Pull Request Guidelines

There is no commit history establishing conventions. Use concise, imperative subjects, such as `Fix preprocessing output path`. PRs should describe the change, affected notebooks, execution checks, and relevant metric comparisons. Link related issues when available. Avoid unrelated notebook-output churn, and never commit credentials or local environment files.
