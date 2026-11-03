"""Independently recompute fixed-budget scores and audit spatial memberships."""
import json

import numpy as np
import pandas as pd

from scripts.improve_25 import ROOT, OUT, METHODS, REGIMES, CLASSES, sha, nearest, metrics


def verify():
    config = json.loads((OUT / 'experiment.json').read_text())
    assert config['budget_per_class'] == 25
    assert sha(ROOT / 'scripts/improve_25.py') == config['code_sha256']
    for path, expected in config['inputs'].items():
        assert sha(ROOT / path) == expected, path
    manifest = json.loads((OUT / 'manifest.json').read_text())
    for name, entry in manifest.items():
        assert sha(OUT / name) == entry['sha256'], name
    table = pd.read_parquet(ROOT / 'data/preprocessed_age_labels/amsterdam_supervised_features.parquet')
    membership = pd.read_parquet(OUT / 'adaptation_membership.parquet')
    inner = pd.read_parquet(OUT / 'inner_membership.parquet')
    probabilities = pd.read_parquet(OUT / 'validation_probabilities.parquet')
    scores = pd.read_csv(OUT / 'metrics.csv').set_index(['regime', 'fold', 'repetition', 'method'])
    settings = pd.read_csv(OUT / 'selections.csv').set_index(['regime', 'fold', 'repetition'])
    search = pd.read_csv(OUT / 'inner_scores.csv')
    assert len(settings) == 60 and len(scores) == 420
    assert not probabilities.duplicated(['regime', 'fold', 'repetition', 'method', 'location_id']).any()
    for regime, _, buffer in REGIMES:
        folds = pd.read_parquet(OUT / f'folds_{regime}.parquet')
        assert folds.location_id.is_unique and set(folds.location_id) == set(table.location_id)
        assert folds.groupby('spatial_group').fold.nunique().eq(1).all()
        meta = folds.merge(table.drop(columns=['fold', 'spatial_group']), on='location_id', validate='one_to_one').set_index('location_id')
        for fold in range(5):
            outer = meta.loc[meta.fold.eq(fold)]
            for repetition in range(3):
                run = membership.loc[membership.regime.eq(regime) & membership.fold.eq(fold) & membership.repetition.eq(repetition)]
                assert len(run) == 100 and run.location_id.is_unique
                assert run.age_class.value_counts().reindex(CLASSES).eq(25).all()
                train = meta.loc[run.location_id]
                assert train.fold.ne(fold).all()
                assert set(train.spatial_group).isdisjoint(outer.spatial_group)
                if buffer:
                    assert nearest(train, outer).min() > buffer
                this_inner = inner.loc[inner.regime.eq(regime) & inner.fold.eq(fold) & inner.repetition.eq(repetition)]
                validation_ids = []
                for _, split in this_inner.groupby('inner_fold'):
                    a = meta.loc[split.loc[split.role.eq('train'), 'location_id']]
                    b = meta.loc[split.loc[split.role.eq('validation'), 'location_id']]
                    assert set(a.index).isdisjoint(b.index)
                    assert set(a.spatial_group).isdisjoint(b.spatial_group)
                    assert set(a.index).union(b.index).issubset(set(train.index))
                    if buffer:
                        assert nearest(a, b).min() > buffer
                    validation_ids.extend(b.index)
                assert len(validation_ids) == 100 and set(validation_ids) == set(train.index)
                run_scores = search.loc[search.regime.eq(regime) & search.fold.eq(fold) & search.repetition.eq(repetition)]
                means = run_scores.groupby('method').macro_f1.mean()
                expected = max(METHODS, key=lambda m: means[m])
                assert settings.loc[(regime, fold, repetition), 'method'] == expected
    checked = 0
    labels = table.set_index('location_id').age_class
    for key, part in probabilities.groupby(['regime', 'fold', 'repetition', 'method']):
        folds = pd.read_parquet(OUT / f'folds_{key[0]}.parquet')
        expected_ids = set(folds.loc[folds.fold.eq(key[1]), 'location_id'])
        assert set(part.location_id) == expected_ids
        np.testing.assert_array_equal(part.y_true, labels.loc[part.location_id])
        p = part[[f'p{c}' for c in CLASSES]].to_numpy()
        assert np.isfinite(p).all() and np.all(p >= 0) and np.allclose(p.sum(axis=1), 1)
        np.testing.assert_array_equal(part.y_pred, CLASSES[p.argmax(axis=1)])
        recalculated = metrics(part.y_true.to_numpy(), p)
        for metric, value in recalculated.items():
            assert np.isclose(value, scores.loc[key, metric]), (key, metric)
        checked += 1
    fold = scores.reset_index().groupby(['regime', 'method', 'fold']).macro_f1.mean()
    expected = fold.groupby(['regime', 'method']).agg(['mean', 'std'])
    summary = pd.read_csv(OUT / 'summary.csv').set_index(['regime', 'method'])
    np.testing.assert_allclose(summary.macro_f1, expected.loc[summary.index, 'mean'])
    np.testing.assert_allclose(summary.fold_sd, expected.loc[summary.index, 'std'])
    message = f'PASS: {checked} probability-based scores, 60 exact 25/class budgets and inner selections, all spatial buffers/groups, input hashes and summaries.'
    print(message)
    (OUT / 'verification.txt').write_text(message + '\n')
    return message


if __name__ == '__main__':
    verify()
