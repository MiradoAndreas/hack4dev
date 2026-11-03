"""Exhaustive, spatially nested grid search for the frozen-source RF blend.

Run from transfer_grid_search.ipynb or `python -m scripts.grid_transfer`.
Selection sees only the supplied adaptation sample, never the outer fold.
"""
from pathlib import Path
import hashlib
import json
import platform
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import ParameterGrid

from scripts.transfer_adaptation import CLASSES, inner_splits, probabilities, target_forest

ROOT = Path(__file__).resolve().parents[1]
# Include the unchanged baseline. Ordering resolves exact ties reproducibly.
FOREST_GRID = {
    'max_depth': [None, 12],
    'max_features': ['sqrt', 0.5],
    'min_samples_leaf': [2, 1],
}
SOURCE_WEIGHTS = [0.5, 0.25, 0.75, 0.0]


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def macro_f1(labels, prediction):
    return f1_score(labels, prediction, labels=CLASSES, average='macro', zero_division=0)


def candidates():
    return [{**forest, 'source_weight': weight}
            for forest in ParameterGrid(FOREST_GRID) for weight in SOURCE_WEIGHTS]


def fit_grid(sample, features, source_prob, rf_parameters, seed):
    """Search every combination on inner spatial folds, then refit the winner.

    Reuse each forest's probabilities for all four weights. This is exhaustive
    ParameterGrid search, with identical fits/scores to fitting each weight
    separately; it avoids four redundant forest fits. Imputation is refit in
    every training pipeline. Highest mean inner macro F1 wins (exact ties use
    grid order). There is no outer-data argument or outer-score fallback.
    """
    if not sample.location_id.is_unique or len(sample) != len(source_prob):
        raise ValueError('Nonunique or misaligned adaptation data')
    labels = sample.age_class.to_numpy(dtype=int)
    splits = inner_splits(labels, sample.spatial_group.to_numpy(), seed)
    if not splits:
        raise ValueError('Cannot form class-complete spatial inner training folds')
    grid = candidates()
    records, memberships = [], []
    for split, (train, valid) in enumerate(splits):
        assert set(sample.iloc[train].spatial_group).isdisjoint(sample.iloc[valid].spatial_group)
        for role, positions in [('train', train), ('validation', valid)]:
            membership = sample.iloc[positions][['location_id', 'spatial_group']].copy()
            membership['inner_fold'], membership['role'] = split, role
            memberships.append(membership)
        for forest_id, forest in enumerate(ParameterGrid(FOREST_GRID)):
            model = target_forest(seed, {**rf_parameters, **forest})
            model.fit(sample.iloc[train][features], labels[train])
            target_prob = probabilities(model, sample.iloc[valid][features])
            for offset, weight in enumerate(SOURCE_WEIGHTS):
                prediction = CLASSES[(weight * source_prob[valid] +
                                      (1 - weight) * target_prob).argmax(axis=1)]
                records.append({'candidate': forest_id * len(SOURCE_WEIGHTS) + offset,
                                'inner_fold': split,
                                'macro_f1': macro_f1(labels[valid], prediction)})
    scores = pd.DataFrame(records)
    means = scores.groupby('candidate').macro_f1.mean().reindex(range(len(grid)))
    best = int(means.idxmax())
    chosen = grid[best]
    model = target_forest(seed, {**rf_parameters,
                                **{k: v for k, v in chosen.items() if k != 'source_weight'}})
    model.fit(sample[features], labels)
    artifact = {'target_pipeline': model, 'source_weight': chosen['source_weight'],
                'feature_names': list(features), 'class_order': CLASSES.copy(),
                'best_candidate': best, 'best_params': chosen,
                'inner_macro_f1': float(means[best]), 'seed': seed,
                'adaptation_location_ids': sample.location_id.tolist(),
                'selection_used_outer_labels': False}
    return artifact, scores, pd.concat(memberships, ignore_index=True)


def predict_grid(artifact, source, table):
    x = table[artifact['feature_names']]
    weight = artifact['source_weight']
    proba = weight * probabilities(source, x) + (1 - weight) * probabilities(artifact['target_pipeline'], x)
    return CLASSES[proba.argmax(axis=1)], proba


def run_experiment():
    baseline = ROOT / 'data/baseline'
    inputs = ROOT / 'data/preprocessed_age_labels'
    output = ROOT / 'data/transfer_grid_search'
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((baseline / 'experiment.json').read_text())
    features = config['feature_names']
    budgets = config['preprocessing_configuration']['budgets']
    manifest = json.loads((baseline / 'manifest.json').read_text())
    for name in ['madrid_source_model.joblib', 'adaptation_membership.csv',
                 'validation_predictions.parquet', 'summary.csv']:
        assert sha256(baseline / name) == manifest[name]['sha256']
    assert sha256(inputs / 'amsterdam_supervised_features.parquet') == config['input_hashes']['amsterdam_supervised_features.parquet']
    table = pd.read_parquet(inputs / 'amsterdam_supervised_features.parquet')
    assert table.location_id.is_unique
    assert table.groupby('spatial_group').fold.nunique().eq(1).all()
    source = joblib.load(baseline / 'madrid_source_model.joblib')
    source_hash = sha256(baseline / 'madrid_source_model.joblib')
    source_prob = probabilities(source, table[features])
    membership = pd.read_csv(baseline / 'adaptation_membership.csv')
    positions = pd.Series(np.arange(len(table)), index=table.location_id)
    old_predictions = pd.read_parquet(baseline / 'validation_predictions.parquet')
    old_predictions = old_predictions.loc[old_predictions.strategy.eq('source_target_blend')]
    grid = candidates()
    settings = {'forest_grid': FOREST_GRID, 'source_weights': SOURCE_WEIGHTS,
                'fixed_rf_parameters': config['rf_parameters'], 'n_candidates': len(grid),
                'budgets': budgets, 'repetitions': config['repetitions'],
                'selection': 'maximum mean inner spatial macro F1; grid order breaks exact ties',
                'outer_results_previously_viewed': True, 'organizer_test_sets_used': False,
                'source_sha256': source_hash, 'helper_sha256': sha256(Path(__file__)),
                'input_hashes': {str(p.relative_to(ROOT)): sha256(p) for p in [
                    inputs / 'amsterdam_supervised_features.parquet',
                    baseline / 'adaptation_membership.csv', baseline / 'experiment.json']},
                'versions': {'python': platform.python_version(), 'sklearn': sklearn.__version__,
                             'numpy': np.__version__, 'pandas': pd.__version__}}
    # Persist the full search space before evaluating any outer fold.
    (output / 'experiment.json').write_text(json.dumps(settings, indent=2) + '\n')
    pd.DataFrame([{'candidate': i, **p} for i, p in enumerate(grid)]).to_csv(output / 'grid.csv', index=False)
    metrics, classes, predictions, selections, searches, inner_memberships = [], [], [], [], [], []
    start = time.perf_counter()
    for fold in range(config['preprocessing_configuration']['n_folds']):
        valid = table.loc[table.fold.eq(fold)]
        valid_pos = positions.loc[valid.location_id].to_numpy()
        for repetition in range(config['repetitions']):
            seed = config['seed'] + 10_000 + 100 * fold + repetition
            for n in budgets:
                run = {'fold': fold, 'repetition': repetition, 'samples_per_class': n}
                members = membership.loc[membership.fold.eq(fold) & membership.repetition.eq(repetition)
                                         & membership.samples_per_class.eq(n)]
                sample_pos = positions.loc[members.location_id].to_numpy()
                sample = table.iloc[sample_pos].reset_index(drop=True)
                assert len(sample) == 4 * n
                assert sample.age_class.value_counts().reindex(CLASSES).eq(n).all()
                assert set(sample.location_id).isdisjoint(valid.location_id)
                assert set(sample.spatial_group).isdisjoint(valid.spatial_group)
                artifact, scores, inner_members = fit_grid(
                    sample, features, source_prob[sample_pos], config['rf_parameters'], seed)
                for key, value in run.items():
                    scores[key], inner_members[key] = value, value
                searches.append(scores)
                inner_memberships.append(inner_members)
                selections.append({**run, 'candidate': artifact['best_candidate'],
                                   'inner_macro_f1': artifact['inner_macro_f1'], **artifact['best_params']})
                tuned = predict_grid(artifact, source, valid)[0]
                # Reconstruct the old model to verify matched samples and seeds.
                reference = target_forest(seed, config['rf_parameters'])
                reference.fit(sample[features], sample.age_class)
                baseline_pred = CLASSES[(.5 * source_prob[valid_pos] +
                                        .5 * probabilities(reference, valid[features])).argmax(axis=1)]
                recorded = old_predictions.loc[old_predictions.fold.eq(fold)
                           & old_predictions.repetition.eq(repetition)
                           & old_predictions.samples_per_class.eq(n)].set_index('location_id')
                np.testing.assert_array_equal(baseline_pred, recorded.loc[valid.location_id, 'y_pred'])
                for strategy, prediction in [('original_blend', baseline_pred), ('grid_selected', tuned)]:
                    y = valid.age_class.to_numpy(dtype=int)
                    metrics.append({**run, 'strategy': strategy, 'macro_f1': macro_f1(y, prediction),
                                    'accuracy': accuracy_score(y, prediction),
                                    'balanced_accuracy': balanced_accuracy_score(y, prediction)})
                    p, r, f, support = precision_recall_fscore_support(y, prediction, labels=CLASSES, zero_division=0)
                    classes.extend([{**run, 'strategy': strategy, 'age_class': int(cls),
                                     'precision': p[i], 'recall': r[i], 'f1': f[i], 'support': int(support[i])}
                                    for i, cls in enumerate(CLASSES)])
                    part = valid[['location_id', 'age_class', 'spatial_group']].rename(columns={'age_class': 'y_true'}).copy()
                    part['y_pred'], part['strategy'] = prediction, strategy
                    for key, value in run.items():
                        part[key] = value
                    predictions.append(part)
                if fold == 0 and repetition == 0:
                    artifact.update({'excluded_Amsterdam_fold': fold, 'samples_per_class': n,
                                     'source_model_file': '../baseline/madrid_source_model.joblib',
                                     'source_sha256': source_hash})
                    path = output / f'grid_fold0_n{n}.joblib'
                    joblib.dump(artifact, path, compress=3)
                    np.testing.assert_array_equal(predict_grid(joblib.load(path), source, valid)[0], tuned)
                print(f'fold={fold} rep={repetition} n={n}: baseline={metrics[-2]["macro_f1"]:.4f} '
                      f'grid={metrics[-1]["macro_f1"]:.4f} best={artifact["best_params"]} '
                      f'elapsed={time.perf_counter()-start:.0f}s', flush=True)
        # Write checkpoints without reusing them for future parameter selection.
        pd.DataFrame(selections).to_csv(output / 'selected_parameters.csv', index=False)
        pd.concat(searches, ignore_index=True).to_csv(output / 'inner_scores.csv', index=False)
    metrics = pd.DataFrame(metrics)
    predictions = pd.concat(predictions, ignore_index=True)
    keys = ['strategy', 'samples_per_class', 'fold', 'repetition']
    assert len(metrics) == 2 * 5 * 3 * len(budgets)
    assert not predictions.duplicated(keys + ['location_id']).any()
    for key, part in predictions.groupby(keys):
        row = metrics.set_index(keys).loc[key]
        assert np.isclose(row.macro_f1, macro_f1(part.y_true, part.y_pred))
    fold_means = metrics.groupby(['strategy', 'samples_per_class', 'fold'], as_index=False)[
        ['macro_f1', 'accuracy', 'balanced_accuracy']].mean()
    summary = fold_means.groupby(['strategy', 'samples_per_class'], as_index=False).agg(
        macro_f1_mean=('macro_f1', 'mean'), macro_f1_fold_sd=('macro_f1', 'std'),
        accuracy_mean=('accuracy', 'mean'), balanced_accuracy_mean=('balanced_accuracy', 'mean'))
    paired = fold_means.pivot(index=['samples_per_class', 'fold'], columns='strategy', values='macro_f1').reset_index()
    paired['delta_macro_f1'] = paired.grid_selected - paired.original_blend
    class_metrics = pd.DataFrame(classes)
    class_summary = class_metrics.groupby(['strategy', 'samples_per_class', 'age_class'], as_index=False)[
        ['precision', 'recall', 'f1']].mean()
    for name, frame in [('fold_metrics', metrics), ('fold_mean_metrics', fold_means),
                        ('summary', summary), ('paired_fold_differences', paired),
                        ('class_metrics', class_metrics), ('class_summary', class_summary)]:
        frame.to_csv(output / f'{name}.csv', index=False)
    pd.concat(inner_memberships, ignore_index=True).to_csv(output / 'inner_membership.csv', index=False)
    predictions.to_parquet(output / 'validation_predictions.parquet', index=False)
    pd.testing.assert_frame_equal(predictions, pd.read_parquet(output / 'validation_predictions.parquet'))
    assert sha256(baseline / 'madrid_source_model.joblib') == source_hash
    write_report(summary, paired, pd.DataFrame(selections))
    (output / 'manifest.json').write_text(json.dumps({p.name: {'sha256': sha256(p), 'bytes': p.stat().st_size}
        for p in sorted(output.iterdir()) if p.is_file() and p.name != 'manifest.json'}, indent=2) + '\n')
    print('Verified all outer scores, baseline reconstructions, five model reloads and exports.', flush=True)
    return summary


def write_report(summary, paired, selections):
    lines = ['# Spatial nested grid search for transfer learning', '',
             'Frozen Madrid source forest plus an Amsterdam target forest. Exhaustive search evaluates '
             '32 configurations: max_depth ∈ {None, 12}, max_features ∈ {sqrt, 0.5}, '
             'min_samples_leaf ∈ {2, 1}, Madrid source weight ∈ {0.5, 0.25, 0.75, 0}. '
             '150 trees and balanced_subsample class weighting remain fixed.', '',
             'Each configuration is selected by mean macro F1 over up to three spatial inner folds '
             '(two only if needed to retain all training classes). Imputation fits inside each inner '
             'training pipeline. The selected forest is refitted on the same limited adaptation sample. '
             'A zero source weight permits rejecting harmful transfer.', '',
             '| Amsterdam labels/class | Original blend | Grid-selected | Paired Δ F1 | Improved folds |',
             '| ---: | ---: | ---: | ---: | ---: |']
    for n in sorted(summary.samples_per_class.unique()):
        rows = summary.loc[summary.samples_per_class.eq(n)].set_index('strategy')
        a, b = rows.loc['original_blend'], rows.loc['grid_selected']
        differences = paired.loc[paired.samples_per_class.eq(n), 'delta_macro_f1']
        lines.append(f'| {n} | {a.macro_f1_mean:.4f} ± {a.macro_f1_fold_sd:.4f} | '
                     f'{b.macro_f1_mean:.4f} ± {b.macro_f1_fold_sd:.4f} | '
                     f'{differences.mean():+.4f} | {int(differences.gt(0).sum())}/5 |')
    lines += ['', 'Means average three matched sampling repetitions within each of five outer spatial folds first. '
              '± is sample SD across fold means, not a confidence interval. Outer locations and spatial '
              'blocks are excluded from adaptation, preprocessing fits and selection. No additional '
              'Amsterdam labels are used. Baseline predictions are reconstructed and matched exactly for all 75 runs.', '',
              '## Selected configurations', '',
              'The optimum is conditional on the grid and each permitted training sample. There is no single '
              'universal optimum selected from outer scores. Most frequently selected inner-CV configuration per budget '
              '(a descriptive count, not a separately evaluated fixed model):', '',
              '| Labels/class | max_depth | max_features | min_samples_leaf | Madrid weight | Selected runs |',
              '| ---: | ---: | ---: | ---: | ---: | ---: |']
    for n, group in selections.groupby('samples_per_class'):
        counts = group.candidate.value_counts()
        cid = int(counts.index[0])
        p = candidates()[cid]
        lines.append(f'| {n} | {p["max_depth"]} | {p["max_features"]} | {p["min_samples_leaf"]} | '
                     f'{p["source_weight"]} | {int(counts.iloc[0])}/15 |')
    lines += ['', '## Limits and reproduction', '',
              'These are iterative development results: these outer folds were inspected in previous experiments. '
              'No untouched organizer test data is available locally or used here. Spatial blocks can touch, '
              'and city-specific class boundaries still require confirmation. A higher development score '
              'does not establish a statistically confirmed gain or final-test performance. Do not retune '
              'the grid using these outer scores.', '',
              'Run preprocessing_age_labels.ipynb, baseline.ipynb, then transfer_grid_search.ipynb. '
              'The notebook calls scripts/grid_transfer.py. Artifacts are in data/transfer_grid_search/: '
              'full candidate/fold scores, selected parameters for every run, inner memberships, '
              'held-out predictions, per-class precision/recall/F1, paired fold differences, hashes, '
              'versions and five demonstration models excluding outer fold 0. Dependencies are unchanged.', '',
              '## Skills', '',
              'Web discovery confirmed the [K-Dense scikit-learn skill]'
              '(https://github.com/K-Dense-AI/scientific-agent-skills/blob/main/skills/scikit-learn/SKILL.md). '
              'Applied its directly readable copy and model-evaluation/pipeline references in docs/upstream-skills/, '
              'plus the verification-before-completion instructions. These are not registered .agents installations.', '',
              'Kassis et al. (2026), Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. '
              'https://doi.org/10.48550/arXiv.2609.00065', '',
              'Method reference: [scikit-learn nested cross-validation]'
              '(https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).']
    (ROOT / 'docs/TRANSFER_GRID_SEARCH_RESULTS.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    print(run_experiment().to_string(index=False))
