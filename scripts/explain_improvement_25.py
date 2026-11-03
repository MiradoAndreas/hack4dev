"""Exact six-band coalition explanations with background sensitivity checks.

Requires the optional SHAP overlay. Uses training backgrounds and a fixed random
held-out subset; no explanation is used to select features or tune the model.
"""
import json

import joblib
import numpy as np
import pandas as pd
import shap

from scripts.improve_25 import ROOT, OUT, BANDS, predict, probabilities, sha


def explain():
    frozen = json.loads((OUT / 'frozen_candidate.json').read_text())
    assert sha(ROOT / frozen['model']) == frozen['model_sha256']
    bundle = joblib.load(ROOT / frozen['model'])
    table = pd.read_parquet(ROOT / 'data/preprocessed_age_labels/amsterdam_supervised_features.parquet')
    table = table.merge(pd.read_parquet(OUT / 'derived_features.parquet'), left_on='location_id', right_index=True, validate='one_to_one')
    train = table.set_index('location_id').loc[bundle['adaptation_location_ids']]
    valid = table.loc[table.fold.eq(0)].sample(n=8, random_state=42)
    all_features = [c for c in table if any(c.startswith(b + '_') for b in BANDS)]
    x = valid[all_features].to_numpy()
    source = None
    if bundle['source_required']:
        assert sha(ROOT / frozen['source']) == frozen['source_sha256']
        source = joblib.load(ROOT / frozen['source'])
        source.named_steps['classifier'].n_jobs = 1

    def full_prediction(a):
        frame = pd.DataFrame(a, columns=all_features)
        sp = probabilities(source, frame[source.feature_names_in_]) if source is not None else np.full((len(frame), 4), .25)
        return predict(bundle['estimator'], frame, sp)

    records, errors = [], []
    for size in [16, 32]:
        for seed in [42, 43]:
            background = train[all_features].sample(n=size, random_state=seed).to_numpy()
            for row_number, row in enumerate(x):
                def game(coalitions):
                    repeated = np.tile(background, (len(coalitions), 1))
                    masks = np.repeat(coalitions, size, axis=0)
                    for group, band in enumerate(BANDS):
                        columns = [j for j, f in enumerate(all_features) if f.startswith(band + '_')]
                        take = masks[:, group] > .5
                        repeated[np.ix_(take, columns)] = row[columns]
                    return full_prediction(repeated).reshape(len(coalitions), size, 4).mean(axis=1)

                exp = shap.Explainer(game, np.zeros((1, 6)), algorithm='exact', feature_names=BANDS)(np.ones((1, 6)))
                reconstruction = exp.base_values + exp.values.sum(axis=1)
                expected = full_prediction(row[None, :])
                np.testing.assert_allclose(reconstruction, expected, rtol=1e-5, atol=1e-6)
                errors.append(float(np.max(np.abs(reconstruction - expected))))
                for band_number, band in enumerate(BANDS):
                    for k in range(4):
                        records.append({'background_n': size, 'seed': seed, 'row_number': row_number,
                                        'band_group': band, 'age_class': k+1,
                                        'probability_attribution': float(exp.values[0, band_number, k])})
            print(f'Explained 8 held-out rows, background={size}, seed={seed}', flush=True)
    values = pd.DataFrame(records)
    values.to_csv(OUT / 'grouped_shap.csv', index=False)
    importance = values.assign(magnitude=values.probability_attribution.abs()).groupby(['background_n', 'seed', 'band_group', 'age_class'], as_index=False).magnitude.mean()
    importance.to_csv(OUT / 'grouped_shap_importance.csv', index=False)
    stability = []
    pivot = importance.pivot(index=['band_group', 'age_class'], columns=['background_n', 'seed'], values='magnitude')
    for a in pivot:
        for b in pivot:
            if a < b:
                for k in range(1, 5):
                    corr = pivot.xs(k, level='age_class')[[a, b]].corr(method='spearman').iloc[0, 1]
                    stability.append({'first': str(a), 'second': str(b), 'age_class': k, 'spearman': corr})
    pd.DataFrame(stability).to_csv(OUT / 'shap_background_stability.csv', index=False)
    (OUT / 'shap_checks.json').write_text(json.dumps({'shap_version': shap.__version__, 'scope': '8 fixed random fold0 held-out locations',
        'background_sizes': [16, 32], 'seeds': [42, 43], 'groups': BANDS, 'algorithm': 'exact six-band coalition Shapley values',
        'output': 'four class probabilities', 'max_additivity_error': max(errors),
        'minimum_rank_correlation': float(pd.DataFrame(stability).spearman.min()),
        'limitations': 'Small exploratory subset; empirical training backgrounds; inter-band dependence not preserved; no causal claim'}, indent=2) + '\n')


if __name__ == '__main__':
    explain()
