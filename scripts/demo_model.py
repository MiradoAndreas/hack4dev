"""Demonstrate the saved 25/class baseline on its excluded Amsterdam fold."""
from pathlib import Path
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT / 'data/baseline'
    manifest = json.loads((folder / 'manifest.json').read_text())
    bundle_path = folder / 'amsterdam_fold0_n25_rep0.joblib'
    source_path = folder / 'madrid_source_model.joblib'
    for path in [bundle_path, source_path, folder / 'validation_predictions.parquet']:
        with path.open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == manifest[path.name]['sha256']
    config = json.loads((folder / 'experiment.json').read_text())
    table_path = ROOT / 'data/preprocessed_age_labels/amsterdam_supervised_features.parquet'
    with table_path.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == config['input_hashes'][table_path.name]
    bundle = joblib.load(bundle_path)
    source = joblib.load(source_path)
    source.named_steps['classifier'].n_jobs = 1
    table = pd.read_parquet(table_path)
    valid = table.loc[table.fold.eq(0)].copy()
    train = table.set_index('location_id').loc[bundle['adaptation_location_ids']]
    assert len(train) == 100 and train.age_class.value_counts().reindex([1, 2, 3, 4]).eq(25).all()
    assert set(train.index).isdisjoint(valid.location_id)
    assert set(train.spatial_group).isdisjoint(valid.spatial_group)
    x = valid[bundle['feature_names']]
    classes = np.asarray(bundle['class_order'])

    def probabilities(model):
        return model.predict_proba(x)[:, [list(model.classes_).index(c) for c in classes]]

    source_p = probabilities(source)
    target_p = probabilities(bundle['target_pipeline'])
    blended = bundle['source_weight'] * source_p + (1 - bundle['source_weight']) * target_p
    predictions = classes[blended.argmax(axis=1)]
    all_recorded = pd.read_parquet(folder / 'validation_predictions.parquet')
    recorded = all_recorded.loc[all_recorded.city.eq('Amsterdam') & all_recorded.strategy.eq('source_target_blend') & all_recorded.fold.eq(0) & all_recorded.repetition.eq(0) & all_recorded.samples_per_class.eq(25)].set_index('location_id')
    np.testing.assert_array_equal(predictions, recorded.loc[valid.location_id, 'y_pred'])
    print(f'PASS: model/input hashes; exactly 25 adaptation labels/class; evaluation locations and spatial groups excluded; recorded predictions reproduced.')
    print(f'Live inference: {len(valid):,} locations, Amsterdam fold 0 / repetition 0. Models are loaded, not retrained.\n')
    rows = []
    for name, p in [('Source only: 0 local labels', source_p), ('Target only: 25/class', target_p), ('50/50 blend: 25/class', blended)]:
        pred = classes[p.argmax(axis=1)]
        rows.append({'model': name, 'macro_f1': f1_score(valid.age_class, pred, labels=classes, average='macro'),
                     'weighted_f1': f1_score(valid.age_class, pred, labels=classes, average='weighted')})
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    aggregate = []
    for strategy, budget in [('source_only', 0), ('target_only', 25), ('source_target_blend', 25)]:
        sub = all_recorded.loc[all_recorded.city.eq('Amsterdam') & all_recorded.strategy.eq(strategy) & all_recorded.samples_per_class.eq(budget)]
        run_scores = [{'fold': fold, 'macro_f1': f1_score(g.y_true, g.y_pred, labels=classes, average='macro')}
                      for (fold, _), g in sub.groupby(['fold', 'repetition'])]
        means = pd.DataFrame(run_scores).groupby('fold').macro_f1.mean()
        aggregate.append({'model': strategy, 'labels_per_class': budget, 'macro_f1_mean': means.mean(), 'fold_sd': means.std()})
    print('\nFull five-fold DEVELOPMENT comparison, recomputed from saved predictions:')
    print(pd.DataFrame(aggregate).round(4).to_string(index=False))
    print('\nBlend class metrics:')
    print(classification_report(valid.age_class, predictions, labels=classes, digits=4, zero_division=0))
    print('Confusion matrix: rows=true class, columns=predicted class; class order 1,2,3,4')
    print(confusion_matrix(valid.age_class, predictions, labels=classes))
    rng = np.random.default_rng(42)
    selected = rng.choice(len(valid), size=8, replace=False)
    examples = pd.DataFrame({'example': np.arange(1, 9), 'true_class': valid.age_class.to_numpy()[selected],
                             'predicted_class': predictions[selected], 'model_confidence': blended.max(axis=1)[selected]})
    print('\nEight fixed random examples, including errors:')
    print(examples.round(3).to_string(index=False))
    print('\nThis is one held-out DEVELOPMENT fold, not an untouched final test. The five-fold blend mean is macro F1 0.5494 ± 0.0258 fold SD.')
    print('Probability confidence is not a guarantee of correctness or proven calibration.')


if __name__ == '__main__':
    main()
