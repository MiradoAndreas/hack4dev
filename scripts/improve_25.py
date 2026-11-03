"""Fixed-budget development experiments for building-age classification.

No new labels, task definitions, or final-test independence are inferred.
Run via improvement_25.ipynb or python -m scripts.improve_25.
"""
from pathlib import Path
import hashlib
import json
import platform
import time

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import sklearn
from scipy.spatial import cKDTree
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import isotonic_regression
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_recall_fscore_support
from sklearn.pipeline import Pipeline

from scripts.transfer_adaptation import CLASSES, inner_splits, probabilities, target_forest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/improvement_25'
BANDS = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2']
METHODS = ['original_blend', 'target_only', 'robust_rf', 'regularized_rf', 'ordinal_rf', 'calibrated_rf']
# Frozen before looking at outer outcomes. Coordinate units have not been confirmed.
REGIMES = [('original', 20, 0), ('buffer2', 20, 2), ('buffer5', 20, 5), ('block40_buffer2', 40, 2)]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def f1(y, p):
    return f1_score(y, p, labels=CLASSES, average='macro', zero_division=0)


def nearest(train, valid):
    return cKDTree(valid[['px_key', 'py_key']]).query(train[['px_key', 'py_key']], k=1)[0]


def split_training(sample, seed, buffer):
    splits = inner_splits(sample.age_class.to_numpy(), sample.spatial_group.to_numpy(), seed)
    if not splits:
        raise ValueError('Insufficient class-complete spatial folds')
    result = []
    for train, valid in splits:
        if buffer:
            train = train[nearest(sample.iloc[train], sample.iloc[valid]) > buffer]
        if set(sample.iloc[train].age_class) != set(CLASSES):
            raise ValueError('Buffer removes a training class; do not silently relax validation')
        assert set(sample.iloc[train].spatial_group).isdisjoint(sample.iloc[valid].spatial_group)
        if buffer:
            assert nearest(sample.iloc[train], sample.iloc[valid]).min() > buffer
        result.append((train, valid))
    assert sorted(np.concatenate([v for _, v in result])) == list(range(len(sample)))
    return result


def derive_features():
    """Deterministic per-location summaries; no learned statistics or target access."""
    path = ROOT / 'data/amsterdam_data.parquet'
    metadata = pq.ParquetFile(path)
    assert metadata.metadata.num_rows < 2_000_000
    cols = ['px_key', 'py_key', 'year', 'coverage']
    cols += [f'{b}_{s}' for s in [1, 2, 3] for b in BANDS]
    cols += [f'{k}_{s}' for s in [1, 2, 3] for k in ['qa_valid', 'doy']]
    raw = pd.read_parquet(path, columns=cols)
    assert not raw.duplicated(['px_key', 'py_key', 'year']).any()
    values = np.full((len(raw), 6), np.nan)
    doy = np.full(len(raw), np.nan)
    chosen = np.zeros(len(raw), dtype=int)
    for slot in [1, 2, 3]:
        x = raw[[f'{b}_{slot}' for b in BANDS]].to_numpy(dtype=float)
        valid = raw[f'qa_valid_{slot}'].fillna(False).to_numpy(dtype=bool) & np.isfinite(x).all(axis=1)
        take = (chosen == 0) & valid
        values[take] = x[take]
        doy[take] = raw.loc[take, f'doy_{slot}']
        chosen[take] = slot
    keep = raw.year.between(1984, 2024).to_numpy() & raw.coverage.between(.15, 1).to_numpy() & (chosen > 0)
    obs = raw.loc[keep, ['px_key', 'py_key', 'year']].copy()
    obs[BANDS] = values[keep]
    obs['doy'] = doy[keep]
    obs['location_id'] = 'Amsterdam:' + obs.px_key.astype(str) + ':' + obs.py_key.astype(str)
    grouped = obs.groupby('location_id', sort=True)[BANDS]
    pieces = []
    for q, suffix in [(.25, 'q25'), (.5, 'robust_median'), (.75, 'q75')]:
        pieces.append(grouped.quantile(q).add_suffix('_' + suffix))
    pieces.append((grouped.quantile(.75) - grouped.quantile(.25)).add_suffix('_iqr'))
    for name, mask in [('early', obs.year.le(2003)), ('late', obs.year.gt(2003)),
                       ('first5', obs.year.le(1988)), ('last5', obs.year.ge(2020))]:
        pieces.append(obs.loc[mask].groupby('location_id')[BANDS].median().add_suffix('_' + name + '_median'))
    # Four fixed day-of-year quarters, with empty quarters preserved as missing.
    for season in range(4):
        mask = obs.doy.between(1, 366) & ((obs.doy - 1) // 92).clip(upper=3).eq(season)
        pieces.append(obs.loc[mask].groupby('location_id')[BANDS].median().add_suffix(f'_season{season}_median'))
    derived = pd.concat(pieces, axis=1)
    for band in BANDS:
        derived[f'{band}_robust_late_minus_early'] = derived[f'{band}_late_median'] - derived[f'{band}_early_median']
        derived[f'{band}_last5_minus_first5'] = derived[f'{band}_last5_median'] - derived[f'{band}_first5_median']
    derived.to_parquet(OUT / 'derived_features.parquet')
    coverage = raw.groupby(['px_key', 'py_key']).coverage.median().reset_index()
    coverage['location_id'] = 'Amsterdam:' + coverage.px_key.astype(str) + ':' + coverage.py_key.astype(str)
    coverage[['location_id', 'coverage']].to_parquet(OUT / 'coverage.parquet', index=False)
    audit = {'raw_rows': len(raw), 'retained_observations': int(keep.sum()),
             'selected_invalid_doy': int((~obs.doy.between(1, 366)).sum()),
             'feature_count': len(derived.columns), 'label_columns_read': False,
             'missing_fraction': float(derived.isna().to_numpy().mean()),
             'indices_enabled': False, 'indices_reason': 'Reflectance units/offset not confirmed',
             'formulas': 'Per-location q25/median/q75/IQR; period and DOY-quarter medians; late-early differences'}
    (OUT / 'feature_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    return derived


def forest(seed, regularized=False):
    return target_forest(seed, {'n_estimators': 150, 'max_features': .5 if regularized else 'sqrt',
                               'min_samples_leaf': 5 if regularized else 2,
                               'max_depth': 12 if regularized else None,
                               'class_weight': 'balanced_subsample', 'n_jobs': 1})


def fit_model(method, sample, basic, robust, source_prob, seed, buffer):
    features = basic if method in ['original_blend', 'target_only'] else robust
    x = sample[features]
    y = sample.age_class.to_numpy(dtype=int)
    if method == 'ordinal_rf':
        models = []
        for boundary in [1, 2, 3]:
            model = forest(seed)
            model.fit(x, (y <= boundary).astype(int))
            models.append(model)
        return {'method': method, 'models': models, 'features': features}
    model = forest(seed, method == 'regularized_rf')
    if method == 'calibrated_rf':
        splits = split_training(sample, seed + 71, buffer)
        model = CalibratedClassifierCV(model, method='sigmoid', cv=splits, ensemble=False, n_jobs=1)
    model.fit(x, y)
    return {'method': method, 'model': model, 'features': features}


def predict(artifact, table, source_prob):
    x = table[artifact['features']]
    if artifact['method'] == 'ordinal_rf':
        cdf = np.column_stack([m.predict_proba(x)[:, list(m.classes_).index(1)] for m in artifact['models']])
        # Project the three cumulative probabilities to a monotone CDF per row.
        cdf = np.vstack([isotonic_regression(row, increasing=True) for row in cdf])
        p = np.diff(np.column_stack([np.zeros(len(x)), cdf, np.ones(len(x))]), axis=1)
    else:
        p = probabilities(artifact['model'], x)
        if artifact['method'] == 'original_blend':
            p = .5 * p + .5 * source_prob
    assert np.isfinite(p).all() and (p >= -1e-12).all() and np.allclose(p.sum(axis=1), 1)
    return p


def metrics(y, p):
    pred = CLASSES[p.argmax(axis=1)]
    onehot = y[:, None] == CLASSES[None, :]
    return {'macro_f1': f1(y, pred), 'accuracy': accuracy_score(y, pred),
            'log_loss': log_loss(y, p, labels=CLASSES),
            'brier': float(np.mean(np.sum((p - onehot)**2, axis=1))),
            'severe_errors': float((np.abs(y - pred) >= 2).mean())}


def run():
    OUT.mkdir(exist_ok=True)
    paths = [ROOT / 'data/amsterdam_data.parquet', ROOT / 'data/preprocessed_age_labels/amsterdam_supervised_features.parquet',
             ROOT / 'data/baseline/adaptation_membership.csv', ROOT / 'data/baseline/madrid_source_model.joblib']
    baseline = json.loads((ROOT / 'data/baseline/experiment.json').read_text())
    source_manifest = json.loads((ROOT / 'data/baseline/manifest.json').read_text())
    assert sha(paths[-1]) == source_manifest[paths[-1].name]['sha256']
    settings = {'budget_per_class': 25, 'repetitions': 3, 'regimes': REGIMES, 'methods': METHODS,
                'selection': 'max mean inner spatial macro F1, method order breaks ties',
                'abstention': 'inner OOF tenth-percentile confidence (nominal 90% training coverage)',
                'independent_test_available': False, 'outer_results_previously_viewed': True,
                'coordinate_units': 'unconfirmed key units; no metre claims',
                'inputs': {str(p.relative_to(ROOT)): sha(p) for p in paths}, 'code_sha256': sha(__file__),
                'versions': {'python': platform.python_version(), 'sklearn': sklearn.__version__, 'pandas': pd.__version__, 'numpy': np.__version__}}
    (OUT / 'experiment.json').write_text(json.dumps(settings, indent=2) + '\n')
    derived = derive_features()
    table = pd.read_parquet(paths[1]).merge(derived, left_on='location_id', right_index=True, validate='one_to_one')
    basic, robust = baseline['feature_names'], list(derived.columns)
    assert not {'age_class', 'weighted_mean_year', 'px_key', 'py_key'}.intersection(robust)
    source = joblib.load(paths[-1])
    source.named_steps['classifier'].n_jobs = 1
    source_p = probabilities(source, table[basic])
    position = pd.Series(np.arange(len(table)), index=table.location_id)
    members = pd.read_csv(paths[2])
    old = pd.read_parquet(ROOT / 'data/baseline/validation_predictions.parquet')
    old = old.loc[old.strategy.eq('source_target_blend') & old.samples_per_class.eq(25)]
    rows, selections, inner_scores, memberships, predictions, class_rows, operations, inner_members = [], [], [], [], [], [], [], []
    started = time.monotonic()
    for regime, block_size, buffer in REGIMES:
        work = table.copy()
        if block_size != 20:
            from sklearn.model_selection import StratifiedGroupKFold
            work['spatial_group'] = (work.px_key // block_size).astype(str) + ':' + (work.py_key // block_size).astype(str)
            work['fold'] = -1
            splitter = StratifiedGroupKFold(5, shuffle=True, random_state=42)
            for fold, (_, valid_index) in enumerate(splitter.split(work, work.age_class, work.spatial_group)):
                work.loc[work.index[valid_index], 'fold'] = fold
        work[['location_id', 'fold', 'spatial_group']].to_parquet(OUT / f'folds_{regime}.parquet', index=False)
        for fold in range(5):
            valid = work.loc[work.fold.eq(fold)]
            eligible = work.loc[work.fold.ne(fold)]
            if buffer:
                eligible = eligible.loc[nearest(eligible, valid) > buffer]
            for repetition in range(3):
                seed = 42 + 10_000 + 100 * fold + repetition
                key = {'regime': regime, 'fold': fold, 'repetition': repetition}
                if regime == 'original':
                    ids = members.loc[members.fold.eq(fold) & members.repetition.eq(repetition) & members.samples_per_class.eq(25), 'location_id']
                    sample = work.set_index('location_id', drop=False).loc[ids].reset_index(drop=True)
                else:
                    sample = pd.concat([eligible.loc[eligible.age_class.eq(c)].sample(n=25, random_state=seed + int(c)) for c in CLASSES]).reset_index(drop=True)
                assert sample.age_class.value_counts().reindex(CLASSES).eq(25).all()
                assert set(sample.location_id).isdisjoint(valid.location_id)
                assert set(sample.spatial_group).isdisjoint(valid.spatial_group)
                if buffer:
                    assert nearest(sample, valid).min() > buffer
                memberships.append(sample[['location_id', 'age_class', 'spatial_group']].assign(**key))
                sp = source_p[position.loc[sample.location_id]]
                vp = source_p[position.loc[valid.location_id]]
                splits = split_training(sample, seed, buffer)
                oof = {method: np.zeros((100, 4)) for method in METHODS}
                for inner_fold, (train, check) in enumerate(splits):
                    for role, ix in [('train', train), ('validation', check)]:
                        inner_members.append(sample.iloc[ix][['location_id', 'spatial_group']].assign(**key, inner_fold=inner_fold, role=role))
                    for method in METHODS:
                        model = fit_model(method, sample.iloc[train].reset_index(drop=True), basic, robust, sp[train], seed, buffer)
                        p = predict(model, sample.iloc[check], sp[check])
                        oof[method][check] = p
                        inner_scores.append({**key, 'inner_fold': inner_fold, 'method': method,
                                             'macro_f1': f1(sample.iloc[check].age_class, CLASSES[p.argmax(axis=1)])})
                means = {m: np.mean([r['macro_f1'] for r in inner_scores if all(r[k] == v for k, v in key.items()) and r['method'] == m]) for m in METHODS}
                chosen = max(METHODS, key=lambda m: means[m])
                threshold = float(np.quantile(oof[chosen].max(axis=1), .1))
                selections.append({**key, 'method': chosen, 'inner_macro_f1': means[chosen], 'confidence_threshold': threshold})
                for method in METHODS:
                    model = fit_model(method, sample, basic, robust, sp, seed, buffer)
                    p = predict(model, valid, vp)
                    if regime == 'original' and method == 'original_blend':
                        recorded = old.loc[old.fold.eq(fold) & old.repetition.eq(repetition)].set_index('location_id')
                        np.testing.assert_array_equal(CLASSES[p.argmax(axis=1)], recorded.loc[valid.location_id, 'y_pred'])
                    names = [method, 'inner_selected'] if method == chosen else [method]
                    for name in names:
                        y = valid.age_class.to_numpy(dtype=int)
                        pred = CLASSES[p.argmax(axis=1)]
                        rows.append({**key, 'method': name, **metrics(y, p)})
                        prec, rec, f, support = precision_recall_fscore_support(y, pred, labels=CLASSES, zero_division=0)
                        class_rows.extend([{**key, 'method': name, 'age_class': int(c), 'precision': prec[j], 'recall': rec[j], 'f1': f[j], 'support': int(support[j])} for j, c in enumerate(CLASSES)])
                        part = valid[['location_id', 'age_class', 'spatial_group']].rename(columns={'age_class': 'y_true'}).assign(**key, method=name, y_pred=pred)
                        for j, c in enumerate(CLASSES):
                            part[f'p{c}'] = p[:, j]
                        predictions.append(part)
                    if method == chosen:
                        accept = p.max(axis=1) >= threshold
                        accepted_metrics = metrics(y[accept], p[accept]) if accept.any() else {k: np.nan for k in metrics(y, p)}
                        operations.append({**key, 'threshold': threshold, 'coverage': accept.mean(), 'n_accepted': int(accept.sum()), **accepted_metrics})
                        if repetition == 0:
                            artifact = {'estimator': model, 'source_required': method == 'original_blend', 'source_sha256': sha(paths[-1]),
                                        'adaptation_location_ids': sample.location_id.tolist(), 'excluded_fold': fold,
                                        'regime': regime, 'threshold': threshold, 'class_order': CLASSES,
                                        'selection_used_outer_labels': False, 'versions': settings['versions']}
                            joblib.dump(artifact, OUT / f'selected_{regime}_fold{fold}.joblib', compress=3)
                            loaded = joblib.load(OUT / f'selected_{regime}_fold{fold}.joblib')
                            np.testing.assert_allclose(predict(loaded['estimator'], valid, vp), p)
                print(f'{regime} fold={fold} rep={repetition} selected={chosen} elapsed={time.monotonic()-started:.0f}s', flush=True)
                pd.DataFrame(rows).to_csv(OUT / 'metrics_checkpoint.csv', index=False)
    for name, data in [('metrics', rows), ('selections', selections), ('inner_scores', inner_scores), ('class_metrics', class_rows), ('operational', operations)]:
        pd.DataFrame(data).to_csv(OUT / f'{name}.csv', index=False)
    pd.concat(memberships).to_parquet(OUT / 'adaptation_membership.parquet', index=False)
    pd.concat(inner_members).to_parquet(OUT / 'inner_membership.parquet', index=False)
    pred = pd.concat(predictions, ignore_index=True)
    pred.to_parquet(OUT / 'validation_probabilities.parquet', index=False)
    scores = pd.DataFrame(rows)
    fold_scores = scores.groupby(['regime', 'method', 'fold'], as_index=False)[['macro_f1', 'accuracy', 'log_loss', 'brier', 'severe_errors']].mean()
    fold_scores.to_csv(OUT / 'fold_metrics.csv', index=False)
    summary = fold_scores.groupby(['regime', 'method']).agg(macro_f1=('macro_f1', 'mean'), fold_sd=('macro_f1', 'std'), accuracy=('accuracy', 'mean'), log_loss=('log_loss', 'mean'), brier=('brier', 'mean'), severe_errors=('severe_errors', 'mean')).reset_index()
    summary.to_csv(OUT / 'summary.csv', index=False)
    paired_uncertainty(pred, summary)
    create_review_queue(table, pred)
    (OUT / 'manifest.json').write_text(json.dumps({p.name: {'sha256': sha(p), 'bytes': p.stat().st_size} for p in sorted(OUT.iterdir()) if p.is_file() and p.name != 'manifest.json'}, indent=2) + '\n')
    print(summary.to_string(index=False), flush=True)
    return summary


def paired_uncertainty(pred, summary):
    def score(x):
        tp = np.diagonal(x, axis1=-2, axis2=-1)
        den = x.sum(axis=-1) + x.sum(axis=-2)
        return np.divide(2 * tp, den, out=np.zeros_like(tp, dtype=float), where=den > 0).mean(axis=-1)
    rows = []
    for regime in summary.regime.unique():
        for method in METHODS[1:] + ['inner_selected']:
            rng = np.random.default_rng(42)
            aggregate = np.zeros((500, 2))
            for _, sub in pred.loc[pred.regime.eq(regime) & pred.method.isin(['original_blend', method])].groupby('fold'):
                blocks, ix = np.unique(sub.spatial_group, return_inverse=True)
                counts = np.zeros((len(blocks), 2, 3, 4, 4), dtype=np.int64)
                np.add.at(counts, (ix, sub.method.map({'original_blend': 0, method: 1}), sub.repetition, sub.y_true-1, sub.y_pred-1), 1)
                w = rng.multinomial(len(blocks), np.full(len(blocks), 1 / len(blocks)), size=500)
                aggregate += score((w @ counts.reshape(len(blocks), -1)).reshape(500, 2, 3, 4, 4)).mean(axis=-1) / 5
            low, high = np.quantile(aggregate[:, 1] - aggregate[:, 0], [.025, .975])
            indexed = summary.loc[summary.regime.eq(regime)].set_index('method')
            rows.append({'regime': regime, 'method': method, 'delta_f1': indexed.loc[method, 'macro_f1'] - indexed.loc['original_blend', 'macro_f1'], 'conditional_low': low, 'conditional_high': high})
    pd.DataFrame(rows).to_csv(OUT / 'paired_uncertainty.csv', index=False)


def create_review_queue(table, pred):
    """Tokenized sample for human review; never relabel using predictions."""
    sub = pred.loc[pred.regime.eq('original') & pred.method.eq('inner_selected') & pred.repetition.eq(0)]
    sub = sub.merge(table[['location_id', 'weighted_mean_year', 'fold']], on=['location_id', 'fold'], validate='one_to_one')
    sub['class4_false_positive'] = sub.y_pred.eq(4) & sub.y_true.ne(4)
    sub['class3_miss'] = sub.y_true.eq(3) & sub.y_pred.ne(3)
    sub['boundary'] = np.min(np.abs(sub.weighted_mean_year.to_numpy()[:, None] - [1945, 1984, 2004]), axis=1) <= 2
    queue = sub.loc[sub.class4_false_positive | sub.class3_miss | sub.boundary]
    queue = queue.groupby(['fold', 'y_true'], group_keys=False).sample(n=5, random_state=42)
    queue['review_token'] = queue.location_id.map(lambda v: hashlib.sha256(v.encode()).hexdigest()[:16])
    queue[['review_token', 'fold', 'y_true', 'y_pred', 'class4_false_positive', 'class3_miss', 'boundary']].to_csv(OUT / 'label_review_queue.csv', index=False)
    # Explicitly separate operational lookup from the redacted EDA report.
    queue[['review_token', 'location_id']].to_parquet(OUT / 'private_review_lookup.parquet', index=False)


if __name__ == '__main__':
    run()
