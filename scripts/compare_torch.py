"""Fixed residual MLP comparison on existing 25/class spatial development folds.

Run with the existing torch environment: python -m scripts.compare_torch
Reuses and verifies existing raw/SSL MLP and RF predictions; never tunes on outer labels.
"""
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support

from scripts.self_supervised_torch import CLASSES, score, seed_all, sha

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/torch_comparison'
CONFIG = dict(width=32, residual_blocks=2, dropout=0.2, epochs=300,
              learning_rate=0.001, weight_decay=0.01, folds=5, repetitions=3,
              labels_per_class=25, rf_blend_weight=0.5, device='cpu')


class ResidualMLP(nn.Module):
    """Small pre-normalized residual MLP, inspired by tabular ResNet.

    LayerNorm replaces batch normalization for the tiny adaptation sample.
    This is not an exact reproduction of the RTDL benchmark architecture.
    """
    def __init__(self, n_features):
        super().__init__()
        width = CONFIG['width']
        self.input = nn.Linear(n_features, width)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.ReLU(),
                          nn.Dropout(CONFIG['dropout']), nn.Linear(width, width),
                          nn.Dropout(CONFIG['dropout']))
            for _ in range(CONFIG['residual_blocks'])])
        self.output = nn.Sequential(nn.LayerNorm(width), nn.ReLU(), nn.Linear(width, 4))

    def forward(self, x):
        x = self.input(x)
        for block in self.blocks:
            x = x + block(x)
        return self.output(x)


def checked_probabilities(frame, y):
    np.testing.assert_array_equal(frame.y_true, y)
    p = frame[[f'p{c}' for c in CLASSES]].to_numpy()
    assert np.isfinite(p).all() and (p >= 0).all()
    np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-6)
    np.testing.assert_array_equal(frame.y_pred, CLASSES[p.argmax(axis=1)])
    return p


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    started = time.monotonic()
    paths = [ROOT / p for p in [
        'data/preprocessed_age_labels/amsterdam_supervised_features.parquet',
        'data/baseline/experiment.json', 'data/baseline/adaptation_membership.csv',
        'data/baseline/validation_predictions.parquet',
        'data/self_supervised_torch/predictions.parquet',
        'data/self_supervised_torch/experiment.json',
        'data/improvement_25/validation_probabilities.parquet']]
    features = json.loads(paths[1].read_text())['feature_names']
    assert len(features) == 54
    # Existing SSL run must have used exactly the same input tables and source code.
    previous = json.loads(paths[5].read_text())
    for name, digest in previous['inputs'].items():
        assert sha(ROOT / name) == digest, f'Stale SSL input: {name}'
    assert previous['code_sha256'] == sha(ROOT / 'scripts/self_supervised_torch.py')
    assert previous['outputs']['predictions.parquet'] == sha(paths[4])
    manifest = dict(config=CONFIG, features=features, independent_test=False,
                    protocol='Original five spatial development folds, three matched repetitions; fixed architecture and epoch budget',
                    outer_labels_used_for_training_or_checkpoint=False,
                    outer_features_used_for_fitting=False,
                    source_supervision='RF uses Madrid labels; neural models do not',
                    inputs={str(p.relative_to(ROOT)): sha(p) for p in paths},
                    code_sha256=sha(__file__), versions=dict(torch=torch.__version__, python=platform.python_version()))
    (OUT / 'experiment.json').write_text(json.dumps(manifest, indent=2) + '\n')
    data = pd.read_parquet(paths[0])
    members = pd.read_csv(paths[2])
    baseline = pd.read_parquet(paths[3])
    baseline = baseline.loc[baseline.strategy.eq('source_target_blend') & baseline.samples_per_class.eq(25)]
    saved = pd.read_parquet(paths[4])
    rf = pd.read_parquet(paths[6], filters=[('regime', '=', 'original'), ('method', '=', 'original_blend')])
    assert data.location_id.is_unique
    rows, class_rows, predictions, histories, audits = [], [], [], [], []
    for fold in range(CONFIG['folds']):
        valid = data.loc[data.fold.eq(fold)]
        eligible = data.loc[data.fold.ne(fold)]
        assert set(valid.spatial_group).isdisjoint(eligible.spatial_group)
        y = valid.age_class.to_numpy(dtype=int)
        for repetition in range(CONFIG['repetitions']):
            seed = 10042 + 100 * fold + repetition
            ids = members.loc[members.fold.eq(fold) & members.repetition.eq(repetition)
                              & members.samples_per_class.eq(25), 'location_id']
            assert ids.is_unique and len(ids) == 100
            sample = eligible.set_index('location_id', drop=False).loc[ids]
            assert sample.age_class.value_counts().reindex(CLASSES).eq(25).all()
            assert set(sample.location_id).isdisjoint(valid.location_id)
            audits.append(dict(fold=fold, repetition=repetition, labels=100,
                               overlap_locations=0, overlap_spatial_groups=0))
            seed_all(seed)
            preprocessor = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler())
            train_x = torch.tensor(preprocessor.fit_transform(sample[features]), dtype=torch.float32)
            valid_x = torch.tensor(preprocessor.transform(valid[features]), dtype=torch.float32)
            labels = torch.tensor(sample.age_class.to_numpy(dtype=np.int64) - 1)
            model = ResidualMLP(len(features))
            optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
            model.train()
            for epoch in range(CONFIG['epochs']):
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(model(train_x), labels)
                assert torch.isfinite(loss)
                loss.backward()
                optimizer.step()
                histories.append(dict(fold=fold, repetition=repetition, epoch=epoch+1, training_loss=loss.item()))
            model.eval()
            with torch.inference_mode():
                p_res = torch.cat([model(batch).softmax(1) for batch in valid_x.split(2048)]).numpy()
            reference = baseline.loc[baseline.fold.eq(fold) & baseline.repetition.eq(repetition)].set_index('location_id').loc[valid.location_id]
            source = rf.loc[rf.fold.eq(fold) & rf.repetition.eq(repetition)].set_index('location_id')
            assert source.index.is_unique and set(source.index) == set(valid.location_id)
            source = source.loc[valid.location_id]
            p_rf = checked_probabilities(source, y)
            np.testing.assert_array_equal(reference.y_true, y)
            np.testing.assert_array_equal(reference.y_pred, source.y_pred)
            candidates = {'original_blend': p_rf, 'residual_mlp': p_res,
                          'rf_residual_blend': (p_rf + p_res) / 2}
            for method in ['raw_mlp', 'ssl_mlp']:
                s = saved.loc[saved.method.eq(method) & saved.fold.eq(fold) & saved.repetition.eq(repetition)].set_index('location_id')
                assert s.index.is_unique and set(s.index) == set(valid.location_id)
                candidates[method] = checked_probabilities(s.loc[valid.location_id], y)
            for method, p in candidates.items():
                pred = CLASSES[p.argmax(axis=1)]
                frame = pd.DataFrame(dict(location_id=valid.location_id.to_numpy(), y_true=y, y_pred=pred,
                                          method=method, fold=fold, repetition=repetition,
                                          **{f'p{c}': p[:, i] for i, c in enumerate(CLASSES)}))
                checked_probabilities(frame, y)
                rows.append(dict(method=method, fold=fold, repetition=repetition, **score(y, p)))
                precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=CLASSES, zero_division=0)
                class_rows.extend(dict(method=method, fold=fold, repetition=repetition, age_class=int(c),
                                       precision=precision[i], recall=recall[i], f1=f1[i], support=int(support[i]))
                                  for i, c in enumerate(CLASSES))
                predictions.append(frame)
        print(f'Fold {fold+1}/5 complete ({time.monotonic()-started:.1f}s)', flush=True)
    runs = pd.DataFrame(rows)
    runs.to_csv(OUT / 'run_metrics.csv', index=False)
    metrics = ['macro_f1', 'weighted_f1', 'accuracy', 'balanced_accuracy']
    folds = runs.groupby(['method', 'fold'])[metrics].mean()
    folds.to_csv(OUT / 'fold_metrics.csv')
    summary = folds.groupby('method').mean()
    summary['macro_f1_fold_sd'] = folds.groupby('method').macro_f1.std()
    summary['delta_vs_rf'] = summary.macro_f1 - summary.loc['original_blend', 'macro_f1']
    summary.to_csv(OUT / 'summary.csv')
    summary.to_csv(ROOT / 'docs/results/torch_comparison_scores.csv')
    paired = folds.macro_f1.unstack('method')
    paired.subtract(paired.original_blend, axis=0).to_csv(OUT / 'paired_fold_deltas.csv')
    pd.DataFrame(class_rows).to_csv(OUT / 'class_metrics.csv', index=False)
    pd.DataFrame(audits).to_csv(OUT / 'split_audit.csv', index=False)
    pd.DataFrame(histories).to_csv(OUT / 'training_history.csv', index=False)
    pd.concat(predictions, ignore_index=True).to_parquet(OUT / 'predictions.parquet', index=False)
    manifest['elapsed_seconds'] = time.monotonic() - started
    manifest['outputs'] = {p.name: sha(p) for p in OUT.iterdir() if p.is_file() and p.name != 'experiment.json'}
    (OUT / 'experiment.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(summary.to_string(), flush=True)


def verify():
    """Recompute saved scores and check artifact hashes without training."""
    manifest = json.loads((OUT / 'experiment.json').read_text())
    for name, digest in manifest['inputs'].items():
        assert sha(ROOT / name) == digest, f'Changed input: {name}'
    assert sha(__file__) == manifest['code_sha256']
    for name, digest in manifest['outputs'].items():
        assert sha(OUT / name) == digest, f'Changed output: {name}'
    predictions = pd.read_parquet(OUT / 'predictions.parquet')
    runs = pd.read_csv(OUT / 'run_metrics.csv').set_index(['method', 'fold', 'repetition'])
    assert len(runs) == 75 and runs.index.is_unique
    for key, frame in predictions.groupby(['method', 'fold', 'repetition']):
        assert frame.location_id.is_unique
        p = checked_probabilities(frame, frame.y_true)
        for metric, value in score(frame.y_true, p).items():
            np.testing.assert_allclose(value, runs.loc[key, metric], atol=1e-12)
    print(f'Verified {len(runs)} run scores, {len(predictions):,} probability rows and artifact hashes.')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true', help='Verify without retraining')
    args = parser.parse_args()
    verify() if args.verify else run()
