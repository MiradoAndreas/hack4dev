# hack4dev

Transfer-learning grid search: run `preprocessing_age_labels.ipynb` and
`baseline.ipynb`, then `.venv/bin/python -m scripts.grid_transfer`.
Open [transfer_grid_search.ipynb](transfer_grid_search.ipynb) to verify and review
the results, or set `RUN_GRID_SEARCH = True` there to repeat training.
The experiment searches 32 configurations using spatial inner validation and
evaluates them on separate outer folds. Generated models and predictions stay
under `data/transfer_grid_search/`.
