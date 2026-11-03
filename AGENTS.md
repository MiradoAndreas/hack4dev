# Repository Guidelines

## Agent Skills

See `docs/AGENT_SKILLS.md` for the requested upstream skills, their project
roles, and installation status. Install at project scope with
`bash scripts/setup-agent-skills.sh`; check file presence with
`bash scripts/setup-agent-skills.sh --check`. Once installed, read the relevant
`.agents/skills/<name>/SKILL.md` before applying a skill. Use exploratory data
analysis for data audits, scikit-learn for baselines and evaluation, transformers
for compatible pretrained-model experiments, systematic debugging for failures,
and verification before completion for evidence-backed completion claims.
Use pytorch-lightning for optional neural training, shap for model explanations,
scientific-slides for presentations, huggingface-gradio for optional demos, and
writing-plans for implementation plans. The user selected huggingface-llm-trainer
for the requested huggingface-trainer name; use it for explicitly authorized
TRL/Unsloth jobs, not as the default spectral-data trainer.
Skill installation alone must not add model runtime dependencies or change the
notebook workflow. Do not claim a skill is available until discovery is confirmed.

## Project Structure & Module Organization

This project explores building-age classification using spectral data from Amsterdam and Madrid. The workflow lives in four root-level notebooks: `1-Introduction.ipynb`, `2-Reading_Data.ipynb`, `3-Preprocessing.ipynb`, and `4-Modelling.ipynb`. Follow their numbered order. `METRIC_LEARNING_APPROACH.md` describes a proposed metric-learning approach; the modelling notebook currently uses random forests.

Raw Parquet datasets live in `data/`; preprocessing writes `data/preprocessed/preprocessed_data.pkl`. `main.py` is a placeholder entry point. Dependencies are declared in `pyproject.toml` and locked in `uv.lock`. There is no separate source package or test directory.

## Build, Test, and Development Commands

- `uv sync --locked`: install the locked project dependencies.
- `uv run python main.py`: check that the environment can run the placeholder script; this does not train a model.
- `uv run --with jupyter jupyter lab`: launch Jupyter locally without adding it to project dependencies.

The project requires Python 3.12 or newer; `.python-version` selects Python 3.14. Run notebooks sequentially and check dataset paths before execution. In particular, modelling currently reads `../data/preprocessed/preprocessed_data.pkl`, while preprocessing writes under `./data/preprocessed`; reconcile paths with the notebook working directory.

## Coding Style & Naming Conventions

Use four-space indentation, `snake_case` for functions and variables, and uppercase names for configuration constants. Group imports near the start of notebooks. Preserve numbered notebook filenames and use Markdown cells to explain feature engineering and experiments. Declare dependency changes in `pyproject.toml` and update `uv.lock` together. No formatter or linter is configured.

## Testing Guidelines

No automated test framework or coverage threshold is configured. Validate changes by restarting the kernel and running affected notebooks from top to bottom, including upstream preprocessing when needed. Preserve fixed random seeds for reproducibility. Report macro F1, class-level results, and validation splits; distinguish in-sample scores from held-out and cross-city evaluation. Keep evaluation data out of preprocessing fits and training.

## Commit & Pull Request Guidelines

There is no commit history establishing conventions. Use concise, imperative subjects, such as `Fix preprocessing output path`. PRs should describe the change, affected notebooks, execution checks, and relevant metric comparisons. Link related issues when available. Avoid unrelated notebook-output churn, and never commit credentials or local environment files.
