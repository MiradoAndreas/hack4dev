# Project agent skills

Installation is pending: the managed shell could not resolve GitHub, and the
project `.agents` and `.codex` directories are read-only in that environment.
No upstream skill files have been installed or verified yet.

## Install and verify

From a network-enabled terminal with write access to this repository:

```bash
bash scripts/setup-agent-skills.sh
bash scripts/setup-agent-skills.sh --check
```

The installer targets Codex at project scope, without a global installation.
Canonical skill files should be available under `.agents/skills/<name>/SKILL.md`.
It installs the eleven selected skills below, including the original five, with
their bundled resources. Existing installations are validated and skipped rather
than overwritten. The check verifies readable instructions and required metadata;
successful agent discovery must also be confirmed
in a new agent turn/session by checking its available-skills list.

The user explicitly selected `huggingface-llm-trainer` for the requested
`huggingface-trainer` name, which is not in the current upstream catalog.
The LLM trainer uses TRL or Unsloth with Hugging Face Jobs. It should not be
assumed suitable for spectral time series.
Source: https://github.com/huggingface/skills#available-skills

## When to use each skill

| Skill | Upstream | Project use |
| --- | --- | --- |
| exploratory-data-analysis | K-Dense-AI/scientific-agent-skills | Audit Parquet schemas, missingness, class balance, city distributions, and repeated building/year observations before modeling. |
| scikit-learn | K-Dense-AI/scientific-agent-skills | Maintain random-forest baselines and preprocessing pipelines; evaluate macro F1 and class-level results with building-aware and cross-city splits. |
| transformers | K-Dense-AI/scientific-agent-skills | Investigate compatible pretrained models only when a transfer-learning experiment calls for them. Check spectral/time-series input compatibility first. |
| systematic-debugging | obra/superpowers | Diagnose notebook path errors, data-shape problems, leakage, and training failures from evidence before fixing them. |
| verification-before-completion | obra/superpowers | Run relevant checks and inspect their actual outputs before claiming notebooks, metrics, or submissions are ready. |
| pytorch-lightning | K-Dense-AI/scientific-agent-skills | Structure optional embedding-network training, checkpoints, and logging. Use one Lightning import namespace consistently. |
| shap | K-Dense-AI/scientific-agent-skills | Explain a fixed evaluated model using a training/reference background population; specify the explained output and verify additivity. Explanations do not replace predictive evaluation. |
| scientific-slides | K-Dense-AI/scientific-agent-skills | Present held-out and cross-city metrics, methods, limitations, and cited figures in the final hackathon presentation. |
| huggingface-gradio | huggingface/skills | Build an optional prediction demo around saved artifacts; keep training separate from inference. |
| huggingface-llm-trainer | huggingface/skills | Plan and run explicitly authorized TRL/Unsloth fine-tuning on Hugging Face Jobs; estimate hardware and budget first. Not the default trainer for the spectral-data baseline. |
| writing-plans | obra/superpowers | Plan multi-step implementation work with explicit files, commands, expected results, and the upstream review workflow. |

## Dependency and evaluation boundaries

The repository uses numbered notebooks, Python >=3.12, and Python 3.14 in
`.python-version`. Its existing dependencies already include scikit-learn,
NumPy, pandas, Matplotlib, PyArrow, and fastparquet. Installing instructions for
the agent does not require adding Python runtime packages. No dependency or lock
file changes are needed for this setup.

Transformers, PyTorch, Lightning, SHAP, and Gradio are not currently declared.
Before implementing functionality that needs them, verify Python/platform support,
including the selected Python 3.14 interpreter, then update `pyproject.toml` and
`uv.lock` together. Do not replace the random-forest baseline merely because a
transformers skill is available.

Installing these skills does not authorize launching cloud jobs, publishing
models or demos, or uploading local datasets. Such actions require explicit task
authorization. Read each installed upstream `SKILL.md` and its relevant bundled
references before use; external service credentials are only configured when a
task actually needs them. Follow the existing project instructions and user
scope when an upstream skill references additional skills or tools that are not
installed. Report missing requirements rather than inventing replacements.

Follow notebooks 1–4. Resolve the preprocessing output versus modeling input
path before execution. Keep all preprocessing fits and metric-learning triplets
restricted to training buildings. Keep Madrid adaptation labels separate from
Madrid evaluation labels. Report held-out and cross-city results separately;
an embedding visualization alone does not demonstrate transfer performance.

No notebooks or training runs are required to validate installation of agent
instruction files. Model changes still require the notebook execution checks in
`AGENTS.md`.
