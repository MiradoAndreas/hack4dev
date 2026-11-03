"""Independently audit persisted grid-search selections and held-out results."""
import json

import joblib
import numpy as np
import pandas as pd

from scripts.grid_transfer import ROOT, candidates, macro_f1, predict_grid, sha256
from scripts.transfer_adaptation import CLASSES


def verify():
    output = ROOT / 'data/transfer_grid_search'
    settings = json.loads((output / 'experiment.json').read_text())
    manifest = json.loads((output / 'manifest.json').read_text())
    for name, entry in manifest.items():
        assert sha256(output / name) == entry['sha256'], name
    for name, expected in settings['input_hashes'].items():
        assert sha256(ROOT / name) == expected, name
    assert sha256(ROOT / 'scripts/grid_transfer.py') == settings['helper_sha256']
    scores = pd.read_csv(output / 'inner_scores.csv')
    selections = pd.read_csv(output / 'selected_parameters.csv')
    inner = pd.read_csv(output / 'inner_membership.csv')
    predictions = pd.read_parquet(output / 'validation_predictions.parquet')
    metrics = pd.read_csv(output / 'fold_metrics.csv')
    summary = pd.read_csv(output / 'summary.csv')
    table = pd.read_parquet(ROOT / 'data/preprocessed_age_labels/amsterdam_supervised_features.parquet')
    memberships = pd.read_csv(ROOT / 'data/baseline/adaptation_membership.csv')
    run_keys = ['fold', 'repetition', 'samples_per_class']
    all_keys = run_keys + ['strategy']
    assert len(selections) == 75 and len(metrics) == 150
    split_counts = scores.groupby(run_keys).inner_fold.nunique()
    assert len(split_counts) == 75 and split_counts.isin([2, 3]).all()
    assert len(scores) == 32 * int(split_counts.sum())
    assert not selections.duplicated(run_keys).any()
    assert not scores.duplicated(run_keys + ['candidate', 'inner_fold']).any()
    assert not predictions.duplicated(all_keys + ['location_id']).any()
    score_groups = scores.groupby(run_keys)
    inner_groups = inner.groupby(run_keys)
    member_groups = memberships.groupby(run_keys)
    metric_index = metrics.set_index(all_keys)
    for run, selection in selections.set_index(run_keys).iterrows():
        fold, repetition, n = run
        run_scores = score_groups.get_group(run)
        assert set(run_scores.candidate) == set(range(32))
        means = run_scores.groupby('candidate').macro_f1.mean().reindex(range(32))
        best = int(means.idxmax())
        assert int(selection.candidate) == best
        assert np.isclose(selection.inner_macro_f1, means[best])
        parameters = candidates()[best]
        for name, value in parameters.items():
            actual = selection[name]
            if value is None:
                assert pd.isna(actual)
            elif isinstance(value, str):
                assert str(actual) == value
            else:
                assert float(actual) == value
        sample = member_groups.get_group(run)
        assert sample.age_class.value_counts().reindex(CLASSES).eq(n).all()
        outer = table.loc[table.fold.eq(fold)]
        sample_table = table.set_index('location_id').loc[sample.location_id]
        assert set(sample.location_id).isdisjoint(outer.location_id)
        assert set(sample_table.spatial_group).isdisjoint(outer.spatial_group)
        run_inner = inner_groups.get_group(run)
        for _, split in run_inner.groupby('inner_fold'):
            train, valid = split.loc[split.role.eq('train')], split.loc[split.role.eq('validation')]
            assert set(train.location_id).isdisjoint(valid.location_id)
            assert set(train.spatial_group).isdisjoint(valid.spatial_group)
            assert set(split.location_id) == set(sample.location_id)
        inner_valid = run_inner.loc[run_inner.role.eq('validation')]
        assert inner_valid.location_id.is_unique
        assert set(inner_valid.location_id) == set(sample.location_id)
    checked = 0
    for key, part in predictions.groupby(all_keys):
        outer = table.loc[table.fold.eq(key[0])].set_index('location_id')
        assert len(part) == len(outer) and set(part.location_id) == set(outer.index)
        np.testing.assert_array_equal(part.y_true, outer.loc[part.location_id, 'age_class'])
        assert np.isclose(macro_f1(part.y_true, part.y_pred), metric_index.loc[key, 'macro_f1'])
        checked += 1
    fold_means = metrics.groupby(['strategy', 'samples_per_class', 'fold']).macro_f1.mean()
    expected = fold_means.groupby(['strategy', 'samples_per_class']).agg(['mean', 'std'])
    for key, row in summary.set_index(['strategy', 'samples_per_class']).iterrows():
        assert np.isclose(row.macro_f1_mean, expected.loc[key, 'mean'])
        assert np.isclose(row.macro_f1_fold_sd, expected.loc[key, 'std'])
    source_path = ROOT / 'data/baseline/madrid_source_model.joblib'
    assert sha256(source_path) == settings['source_sha256']
    source = joblib.load(source_path)
    outer = table.loc[table.fold.eq(0)]
    for n in settings['budgets']:
        bundle = joblib.load(output / f'grid_fold0_n{n}.joblib')
        assert bundle['selection_used_outer_labels'] is False
        assert set(bundle['adaptation_location_ids']).isdisjoint(outer.location_id)
        actual, proba = predict_grid(bundle, source, outer)
        expected = predictions.loc[predictions.fold.eq(0) & predictions.repetition.eq(0)
                   & predictions.samples_per_class.eq(n) & predictions.strategy.eq('grid_selected')]
        np.testing.assert_array_equal(actual, expected.set_index('location_id').loc[outer.location_id, 'y_pred'])
        assert np.isfinite(proba).all() and np.allclose(proba.sum(axis=1), 1)
    message = (f'PASS: {len(scores)} inner candidate/fold scores; 75 optimal selections; '
               f'all spatial and label-budget exclusions; {checked} held-out scores from '
               f'{len(predictions):,} predictions; summary aggregation; five model reloads; hashes.')
    print(message)
    return message


if __name__ == '__main__':
    verify()
