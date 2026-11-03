"""Score a frozen candidate once, or monitor new feature-table distributions.

Requires an explicitly confirmed data contract. Never fits or updates models.
Only locally generated, hash-verified joblib artifacts are loaded.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scripts.improve_25 import ROOT, OUT, CLASSES, sha, predict, probabilities, metrics


def freeze():
    """Fixed primary regime/fold, chosen before inspecting outer scores."""
    model_path = OUT / 'selected_buffer2_fold0.joblib'
    settings = json.loads((OUT / 'experiment.json').read_text())
    assert sha(model_path) == json.loads((OUT / 'manifest.json').read_text())[model_path.name]['sha256']
    bundle = joblib.load(model_path)
    features = bundle['estimator']['features']
    train = pd.read_parquet(ROOT / 'data/preprocessed_age_labels/amsterdam_supervised_features.parquet')
    derived = pd.read_parquet(OUT / 'derived_features.parquet')
    train = train.merge(derived, left_on='location_id', right_index=True, validate='one_to_one')
    train = train.set_index('location_id').loc[bundle['adaptation_location_ids']]
    reference = json.loads(train[features].quantile([0, .25, .5, .75, 1]).to_json())
    manifest = {'model': str(model_path.relative_to(ROOT)), 'model_sha256': sha(model_path),
                'source': 'data/baseline/madrid_source_model.joblib', 'source_sha256': bundle['source_sha256'],
                'features': features, 'class_order': [1, 2, 3, 4], 'threshold': bundle['threshold'],
                'target_labels': 100, 'selection': 'buffer2 fold0 repetition0 inner-selected candidate; not outer-best',
                'code_sha256': sha(ROOT / 'scripts/improve_25.py'), 'versions': settings['versions'],
                'training_reference_quantiles': reference,
                'minimum_separation_key_units': 2, 'development_experiment_sha256': sha(OUT / 'experiment.json'),
                'independent_test_performed': False}
    (OUT / 'frozen_candidate.json').write_text(json.dumps(manifest, indent=2) + '\n')
    contract = {'confirmed_age_bins': False, 'confirmed_reflectance_units_and_offset': False,
                'confirmed_prediction_date_and_available_history': False,
                'confirmed_same_feature_engineering': False, 'independent_of_all_development': False,
                'building_footprints_available': False,
                'note': 'Complete using organizer/data-owner evidence. Do not mark true from predictions.'}
    contract_path = OUT / 'data_contract_template.json'
    if not contract_path.exists():
        contract_path.write_text(json.dumps(contract, indent=2) + '\n')
    return manifest


def evaluate(path, contract_path, output, monitor=False):
    frozen = json.loads((OUT / 'frozen_candidate.json').read_text())
    contract = json.loads(Path(contract_path).read_text())
    required = ['confirmed_age_bins', 'confirmed_reflectance_units_and_offset',
                'confirmed_prediction_date_and_available_history', 'confirmed_same_feature_engineering']
    if not monitor:
        required.append('independent_of_all_development')
    if any(contract.get(k) is not True for k in required):
        raise ValueError('Unconfirmed data contract: ' + ', '.join(k for k in required if contract.get(k) is not True))
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.suffix != '.parquet':
        raise ValueError('Provide a regular local Parquet feature table')
    path = path.resolve()
    if Path(output).exists():
        raise FileExistsError('Refusing to replace an existing evaluation report')
    table = pd.read_parquet(path)
    needed = frozen['features'] + ['location_id', 'city', 'px_key', 'py_key']
    if not monitor:
        needed += ['age_class']
    if not set(needed).issubset(table):
        raise ValueError('Feature schema incomplete; use the frozen feature engineering')
    if table.empty or table.location_id.duplicated().any() or table[needed].select_dtypes(include='number').isin([np.inf, -np.inf]).any().any():
        raise ValueError('Empty, duplicate or nonfinite table')
    if not monitor:
        from scripts.improve_25 import nearest
        for city in ['Madrid', 'Amsterdam']:
            development = pd.read_parquet(ROOT / f'data/preprocessed_age_labels/{city.lower()}_supervised_features.parquet')
            same_city = table.loc[table.city.eq(city)]
            if set(table.location_id).intersection(development.location_id):
                raise ValueError('Locations were already used in development')
            if len(same_city) and nearest(same_city, development).min() <= frozen['minimum_separation_key_units']:
                raise ValueError('Evaluation coordinates overlap or touch the development exclusion buffer')
        if not table.age_class.isin(CLASSES).all() or set(table.age_class) != set(CLASSES):
            raise ValueError('Evaluation needs valid labels for all four classes')
    model_path, source_path = ROOT / frozen['model'], ROOT / frozen['source']
    assert sha(model_path) == frozen['model_sha256'] and sha(source_path) == frozen['source_sha256']
    bundle = joblib.load(model_path)
    if bundle['source_required']:
        source = joblib.load(source_path)
        source_prob = probabilities(source, table[source.feature_names_in_])
    else:
        source_prob = np.full((len(table), 4), .25)
    p = predict(bundle['estimator'], table, source_prob)
    accepted = p.max(axis=1) >= frozen['threshold']
    result = {'input_sha256': sha(path), 'model_sha256': frozen['model_sha256'],
              'n_locations': len(table), 'coverage': float(accepted.mean()), 'monitor_only': monitor}
    if not monitor:
        receipts = OUT / 'independent_receipts'
        receipts.mkdir(exist_ok=True)
        receipt = receipts / (frozen['model_sha256'] + '_' + sha(path) + '.json')
        with receipt.open('x') as stream:
            json.dump({'status': 'scoring_started', 'output': str(output),
                       'input_sha256': sha(path), 'contract_sha256': sha(contract_path)}, stream, indent=2)
        result.update(metrics(table.age_class.to_numpy(dtype=int), p))
        from sklearn.metrics import classification_report, confusion_matrix
        pred = CLASSES[p.argmax(axis=1)]
        result['classes'] = classification_report(table.age_class, pred, labels=CLASSES, output_dict=True, zero_division=0)
        result['confusion'] = confusion_matrix(table.age_class, pred, labels=CLASSES).tolist()
        result['accepted'] = metrics(table.age_class.to_numpy(dtype=int)[accepted], p[accepted]) if accepted.any() else None
    drift = []
    for feature in frozen['features']:
        q = frozen['training_reference_quantiles'][feature]
        lo, hi = q['0.0'], q['1.0']
        scale = q['0.75'] - q['0.25'] if q['0.75'] is not None and q['0.25'] is not None else None
        x = table[feature]
        observed_median = x.median()
        drift.append({'feature': feature, 'missing_fraction': float(x.isna().mean()),
                      'outside_training_range_fraction': float(((x < lo) | (x > hi)).mean()) if lo is not None and hi is not None else None,
                      'median_shift_over_training_iqr': float((observed_median-q['0.5']) / scale) if scale and np.isfinite(observed_median) else None})
    result['drift'] = drift
    # Exclusive output creation prevents silent replacement of a final-test report.
    with Path(output).open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--input')
    parser.add_argument('--contract')
    parser.add_argument('--output')
    parser.add_argument('--monitor', action='store_true')
    args = parser.parse_args()
    if args.freeze:
        freeze()
    elif all([args.input, args.contract, args.output]):
        evaluate(args.input, args.contract, args.output, args.monitor)
    else:
        parser.error('Use --freeze or provide --input, --contract and --output')
