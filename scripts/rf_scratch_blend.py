"""Fixed 50/50 original RF blend + scratch MLP on matched spatial folds.

Run: uv run --extra deep-learning python -m scripts.rf_scratch_blend
Verify: uv run --extra deep-learning python -m scripts.rf_scratch_blend --verify
"""
import json
from pathlib import Path
import platform
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn

from scripts.self_supervised_torch import CLASSES, seed_all, sha
from scripts.torch_transfer import TransferMLP, probabilities, metrics
from scripts.compare_torch import checked_probabilities

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/rf_scratch_blend'
CONFIG = dict(seed=42, hidden=[64, 32], dropout=0.2, epochs=300,
              learning_rate=0.001, weight_decay=0.01, rf_weight=0.5,
              folds=5, repetitions=3, labels_per_class=25, source_scaler_checkpoint_fold=0)
METRICS = ['macro_f1', 'weighted_f1', 'accuracy', 'balanced_accuracy', 'severe_error_rate']


def train_scratch(x, y, seed):
    """Random initialization; source weights and source labels are never used."""
    seed_all(seed)
    model = TransferMLP(x.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    history = []
    for epoch in range(CONFIG['epochs']):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(x), y)
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
        history.append(dict(epoch=epoch+1, training_loss=loss.item()))
    model.eval()
    return model, history


def predict_bundle(folder, frame):
    """Load the exported fold-0 example and predict from feature columns only."""
    folder = Path(folder)
    bundle = joblib.load(folder / 'example_sklearn.joblib')
    checkpoint = torch.load(folder / 'example_scratch.pt', weights_only=True, map_location='cpu')
    features = checkpoint['features']
    if not len(frame) or not set(features).issubset(frame.columns):
        raise ValueError('Provide nonempty data containing the 54 expected features.')
    model = TransferMLP(len(features))
    model.load_state_dict(checkpoint['state_dict'])
    x = torch.tensor(bundle['preprocessing'].transform(frame[features]), dtype=torch.float32)
    p_mlp = probabilities(model, x)
    def rf_p(model):
        assert set(model.classes_) == set(CLASSES)
        return model.predict_proba(frame[features])[:, [list(model.classes_).index(c) for c in CLASSES]]
    p = .25 * rf_p(bundle['source_rf']) + .25 * rf_p(bundle['target_rf']) + .5 * p_mlp
    assert np.isfinite(p).all() and (p >= 0).all()
    np.testing.assert_allclose(p.sum(1), 1, atol=1e-6)
    return p


def aligned(frame, method, fold, repetition, valid):
    s = frame.loc[frame.method.eq(method) & frame.fold.eq(fold) & frame.repetition.eq(repetition)].set_index('location_id')
    assert s.index.is_unique and set(s.index) == set(valid.location_id)
    return checked_probabilities(s.loc[valid.location_id], valid.age_class.to_numpy(dtype=int))


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    torch.set_num_threads(2)
    paths = [ROOT / p for p in [
        'data/preprocessed_age_labels/madrid_supervised_features.parquet',
        'data/preprocessed_age_labels/amsterdam_supervised_features.parquet',
        'data/baseline/experiment.json', 'data/baseline/adaptation_membership.csv',
        'data/torch_transfer/predictions.parquet', 'data/torch_transfer/experiment.json',
        'data/baseline/madrid_source_model.joblib', 'data/baseline/amsterdam_fold0_n25_rep0.joblib',
        'scripts/torch_transfer.py', 'scripts/self_supervised_torch.py', 'scripts/compare_torch.py']]
    previous = json.loads(paths[5].read_text())
    assert previous['outputs']['predictions.parquet'] == sha(paths[4])
    assert previous['code_sha256'] == sha(paths[8])
    for name, digest in previous['inputs'].items():
        assert sha(ROOT / name) == digest, f'Changed transfer input: {name}'
    features = json.loads(paths[2].read_text())['feature_names']
    assert len(features) == 54
    source = pd.read_parquet(paths[0])
    target = pd.read_parquet(paths[1])
    assert source.location_id.is_unique and target.location_id.is_unique
    fit = source.fold.ne(CONFIG['source_scaler_checkpoint_fold'])
    # Match the earlier scratch control: source features for preprocessing only.
    preprocessing = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler())
    preprocessing.fit(source.loc[fit, features])
    joblib.dump(preprocessing, OUT / 'preprocessing.joblib')
    members = pd.read_csv(paths[3])
    reference = pd.read_parquet(paths[4])
    manifest = dict(config=CONFIG, features=features, independent_test=False,
        fixed_before_new_evaluation=True, outer_folds_previously_used_for_development=True,
        method='0.25 Madrid RF + 0.25 Amsterdam RF + 0.50 randomly initialized MLP',
        neural_source_labels_used=False, neural_source_weights_used=False,
        preprocessing='Median and standardization fit only on Madrid folds 1-4; matches prior scratch control',
        target_labels_per_run=100, tuning='None; no outer-selected weights, epochs or seeds',
        inputs={str(p.relative_to(ROOT)): sha(p) for p in paths}, code_sha256=sha(__file__),
        versions=dict(torch=torch.__version__, python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__))
    (OUT / 'experiment.json').write_text(json.dumps(manifest, indent=2)+'\n')
    rows, parts, class_rows, histories, audits = [], [], [], [], []
    for fold in range(5):
        valid = target.loc[target.fold.eq(fold)]
        eligible = target.loc[target.fold.ne(fold)]
        assert set(valid.spatial_group).isdisjoint(eligible.spatial_group)
        y = valid.age_class.to_numpy(dtype=int)
        valid_x = torch.tensor(preprocessing.transform(valid[features]), dtype=torch.float32)
        for repetition in range(3):
            ids = members.loc[members.fold.eq(fold) & members.repetition.eq(repetition) & members.samples_per_class.eq(25), 'location_id']
            assert len(ids) == 100 and ids.is_unique
            sample = eligible.set_index('location_id', drop=False).loc[ids]
            assert sample.age_class.value_counts().reindex(CLASSES).eq(25).all()
            assert set(sample.spatial_group).isdisjoint(valid.spatial_group)
            seed = 10042 + 100 * fold + repetition
            x = torch.tensor(preprocessing.transform(sample[features]), dtype=torch.float32)
            labels = torch.tensor(sample.age_class.to_numpy(dtype=np.int64)-1)
            model, history = train_scratch(x, labels, seed)
            p_scratch = probabilities(model, valid_x)
            # Fresh training must reproduce the original scratch implementation.
            np.testing.assert_allclose(p_scratch, aligned(reference, 'scratch', fold, repetition, valid), atol=1e-6)
            p_rf = aligned(reference, 'original_blend', fold, repetition, valid)
            candidates = dict(original_blend=p_rf, scratch=p_scratch,
                rf_fine_tuned_blend=aligned(reference, 'rf_fine_tuned_blend', fold, repetition, valid),
                rf_scratch_blend=.5*p_rf + .5*p_scratch)
            for method, p in candidates.items():
                pred = CLASSES[p.argmax(1)]
                frame = pd.DataFrame(dict(location_id=valid.location_id.to_numpy(), spatial_group=valid.spatial_group.to_numpy(),
                    method=method, fold=fold, repetition=repetition, y_true=y, y_pred=pred,
                    **{f'p{c}': p[:, i] for i, c in enumerate(CLASSES)}))
                checked_probabilities(frame, y)
                parts.append(frame)
                rows.append(dict(method=method, fold=fold, repetition=repetition, **metrics(y, p)))
                pr, re, f1, support = precision_recall_fscore_support(y, pred, labels=CLASSES, zero_division=0)
                class_rows.extend(dict(method=method, fold=fold, repetition=repetition, age_class=int(c),
                    precision=pr[i], recall=re[i], f1=f1[i], support=int(support[i])) for i, c in enumerate(CLASSES))
            histories.extend(dict(fold=fold, repetition=repetition, **h) for h in history)
            audits.append(dict(fold=fold, repetition=repetition, seed=seed, target_labels=100,
                               overlap_locations=0, overlap_spatial_groups=0))
            if fold == 0 and repetition == 0:
                source_rf = joblib.load(paths[6])
                target_bundle = joblib.load(paths[7])
                assert set(target_bundle['adaptation_location_ids']) == set(ids)
                joblib.dump(dict(source_rf=source_rf, target_rf=target_bundle['target_pipeline'], preprocessing=preprocessing,
                    features=features, excluded_fold=0, adaptation_ids=ids.tolist()), OUT / 'example_sklearn.joblib', compress=3)
                torch.save(dict(state_dict=model.state_dict(), features=features, config=CONFIG), OUT / 'example_scratch.pt')
                # Actual inference via exported RF and MLP weights must match saved blending.
                np.testing.assert_allclose(predict_bundle(OUT, valid), candidates['rf_scratch_blend'], atol=1e-6)
        print(f'Fold {fold+1}/5 complete; {time.monotonic()-started:.1f}s elapsed', flush=True)
    runs = pd.DataFrame(rows)
    runs.to_csv(OUT / 'run_metrics.csv', index=False)
    folds = runs.groupby(['method', 'fold'])[METRICS].mean()
    folds.to_csv(OUT / 'fold_metrics.csv')
    summary = folds.groupby('method').mean()
    summary['macro_f1_fold_sd'] = folds.groupby('method').macro_f1.std()
    summary['delta_vs_rf'] = summary.macro_f1 - summary.loc['original_blend', 'macro_f1']
    summary['delta_vs_fine_tuned_hybrid'] = summary.macro_f1 - summary.loc['rf_fine_tuned_blend', 'macro_f1']
    summary.to_csv(OUT / 'summary.csv')
    summary.to_csv(ROOT / 'docs/results/rf_scratch_blend_scores.csv')
    pd.concat(parts, ignore_index=True).to_parquet(OUT / 'predictions.parquet', index=False)
    pd.DataFrame(class_rows).to_csv(OUT / 'class_metrics.csv', index=False)
    pd.DataFrame(histories).to_csv(OUT / 'training_history.csv', index=False)
    pd.DataFrame(audits).to_csv(OUT / 'split_audit.csv', index=False)
    members.loc[members.samples_per_class.eq(25)].to_csv(OUT / 'adaptation_membership.csv', index=False)
    paired = folds.macro_f1.unstack('method')
    paired['scratch_hybrid_minus_rf'] = paired.rf_scratch_blend-paired.original_blend
    paired['scratch_hybrid_minus_fine_tuned_hybrid'] = paired.rf_scratch_blend-paired.rf_fine_tuned_blend
    paired.to_csv(OUT / 'paired_fold_scores.csv')
    manifest['elapsed_seconds'] = time.monotonic()-started
    manifest['outputs'] = {p.name: sha(p) for p in OUT.iterdir() if p.is_file() and p.name != 'experiment.json'}
    (OUT / 'experiment.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(summary.to_string(), flush=True)
    return summary


def verify():
    manifest = json.loads((OUT / 'experiment.json').read_text())
    assert manifest['code_sha256'] == sha(__file__)
    for name, digest in manifest['inputs'].items():
        assert sha(ROOT / name) == digest, f'Changed input: {name}'
    for name, digest in manifest['outputs'].items():
        assert sha(OUT / name) == digest, f'Changed output: {name}'
    predictions = pd.read_parquet(OUT / 'predictions.parquet')
    runs = pd.read_csv(OUT / 'run_metrics.csv').set_index(['method', 'fold', 'repetition'])
    assert len(runs) == 60 and runs.index.is_unique
    for key, frame in predictions.groupby(['method', 'fold', 'repetition']):
        p = checked_probabilities(frame, frame.y_true)
        assert frame.location_id.is_unique
        for name, value in metrics(frame.y_true.to_numpy(), p).items():
            np.testing.assert_allclose(value, runs.loc[key, name], atol=1e-12)
    for (fold, repetition), frame in predictions.groupby(['fold', 'repetition']):
        groups = {m: s.set_index('location_id').sort_index() for m,s in frame.groupby('method')}
        cols = [f'p{c}' for c in CLASSES]
        np.testing.assert_allclose(groups['rf_scratch_blend'][cols],
            .5*groups['original_blend'][cols]+.5*groups['scratch'][cols], atol=1e-12)
    target = pd.read_parquet(ROOT / 'data/preprocessed_age_labels/amsterdam_supervised_features.parquet')
    members = pd.read_csv(OUT / 'adaptation_membership.csv')
    for (fold, repetition), sample in members.groupby(['fold', 'repetition']):
        assert len(sample) == 100 and sample.location_id.is_unique
        train = target.set_index('location_id').loc[sample.location_id]
        valid = target.loc[target.fold.eq(fold)]
        assert train.age_class.value_counts().reindex(CLASSES).eq(25).all()
        assert set(train.spatial_group).isdisjoint(valid.spatial_group)
    valid = target.loc[target.fold.eq(0)]
    saved = predictions.loc[predictions.method.eq('rf_scratch_blend') & predictions.fold.eq(0) & predictions.repetition.eq(0)]
    saved = saved.set_index('location_id').loc[valid.location_id]
    np.testing.assert_allclose(predict_bundle(OUT, valid), saved[[f'p{c}' for c in CLASSES]], atol=1e-6)
    print(f'Verified {len(runs)} run scores, {len(predictions):,} probability rows, fixed weights, 15 label budgets and exported inference.')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    verify() if args.verify else run()
