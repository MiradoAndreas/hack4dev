"""Supervised Madrid pretraining and Amsterdam adaptation at 25 labels/class.

Run: uv run --extra deep-learning python -m scripts.torch_transfer
Verify: uv run --extra deep-learning python -m scripts.torch_transfer --verify
"""
from copy import deepcopy
import json
from pathlib import Path
import platform
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn

from scripts.self_supervised_torch import CLASSES, seed_all, sha, score
from scripts.compare_torch import checked_probabilities

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/torch_transfer'
CONFIG = dict(seed=42, hidden=[64, 32], dropout=0.2,
              target_boundaries=[1945, 1984, 2004, 2024],
              source_checkpoint_fold=0, source_max_epochs=80, source_patience=12,
              source_batch_size=512, source_learning_rate=0.001,
              target_epochs=300, head_learning_rate=0.001,
              encoder_learning_rate=0.0001, weight_decay=0.01,
              folds=5, repetitions=3, labels_per_class=25,
              rf_blend_weight=0.5, device='cpu')
METRICS = ['macro_f1', 'weighted_f1', 'accuracy', 'balanced_accuracy', 'severe_error_rate']
METHODS = ['source_only', 'frozen_encoder', 'fine_tuned', 'scratch', 'scratch_matched_lr',
           'rf_fine_tuned_blend', 'original_blend', 'raw_mlp']


class TransferMLP(nn.Module):
    def __init__(self, n_features=54):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(n_features, 64), nn.ReLU(), nn.Dropout(CONFIG['dropout']),
                                     nn.Linear(64, 32), nn.ReLU(), nn.Dropout(CONFIG['dropout']))
        self.head = nn.Linear(32, 4)

    def forward(self, x):
        return self.head(self.encoder(x))


def metrics(y, p):
    return dict(**score(y, p), severe_error_rate=float(np.mean(np.abs(y - CLASSES[p.argmax(1)]) >= 2)))


def probabilities(model, x):
    model.eval()
    with torch.inference_mode():
        return torch.cat([model(batch).softmax(1) for batch in x.split(2048)]).numpy()


def source_labels(source):
    # Only the supervised target vector uses construction year; never a predictor.
    years = source.weighted_mean_year.to_numpy()
    assert np.isfinite(years).all() and (years >= 1850).all() and (years <= 2024).all()
    labels = np.searchsorted(CONFIG['target_boundaries'], years, side='left') + 1
    assert set(labels) == set(CLASSES)
    return labels


def pretrain(source, features):
    seed_all(CONFIG['seed'])
    labels = source_labels(source)
    fit = source.fold.ne(CONFIG['source_checkpoint_fold']).to_numpy()
    check = ~fit
    assert set(source.loc[fit, 'spatial_group']).isdisjoint(source.loc[check, 'spatial_group'])
    preprocessing = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler())
    train_x = torch.tensor(preprocessing.fit_transform(source.loc[fit, features]), dtype=torch.float32)
    check_x = torch.tensor(preprocessing.transform(source.loc[check, features]), dtype=torch.float32)
    train_y = torch.tensor(labels[fit] - 1, dtype=torch.long)
    weights = len(train_y) / (4 * torch.bincount(train_y).float())
    model = TransferMLP(len(features))
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['source_learning_rate'], weight_decay=CONFIG['weight_decay'])
    best_f1, best_epoch, stale, history = -np.inf, 0, 0, []
    for epoch in range(CONFIG['source_max_epochs']):
        model.train()
        total = 0
        for idx in torch.randperm(len(train_x)).split(CONFIG['source_batch_size']):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(train_x[idx]), train_y[idx], weight=weights)
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        p = probabilities(model, check_x)
        checkpoint_f1 = f1_score(labels[check], CLASSES[p.argmax(1)], labels=CLASSES, average='macro', zero_division=0)
        history.append(dict(epoch=epoch+1, training_loss=total/len(train_x), checkpoint_macro_f1=checkpoint_f1))
        if checkpoint_f1 > best_f1 + 1e-5:
            best_f1, best_epoch, stale = checkpoint_f1, epoch+1, 0
            best_state = deepcopy(model.state_dict())
        else:
            stale += 1
        if (epoch+1) % 10 == 0:
            print(f'Madrid epoch {epoch+1}: checkpoint F1 {checkpoint_f1:.4f}', flush=True)
        if stale >= CONFIG['source_patience']:
            break
    model.load_state_dict(best_state)
    model.eval()
    pd.DataFrame(history).to_csv(OUT / 'source_history.csv', index=False)
    source[['location_id', 'spatial_group', 'fold']].assign(
        pretraining_partition=np.where(fit, 'fit', 'checkpoint'),
        original_age_class=source.age_class.to_numpy(),
        harmonized_age_class=labels).to_parquet(OUT / 'source_membership.parquet', index=False)
    joblib.dump(preprocessing, OUT / 'source_preprocessor.joblib')
    torch.save(dict(state_dict=model.state_dict(), config=CONFIG, features=features), OUT / 'source_weights.pt')
    details = dict(n_fit=int(fit.sum()), n_checkpoint=int(check.sum()),
                   best_epoch=best_epoch, checkpoint_macro_f1=best_f1,
                   relabeled_locations=int(np.sum(labels != source.age_class.to_numpy())),
                   fit_class_counts=np.bincount(labels[fit], minlength=5)[1:].tolist(),
                   checkpoint_class_counts=np.bincount(labels[check], minlength=5)[1:].tolist())
    print(f'Madrid pretraining complete: {details}', flush=True)
    return model, preprocessing, details


def adapt(source_model, x, y, method, seed):
    # The same seed produces the same fresh head for all four controls.
    seed_all(seed)
    model = TransferMLP(x.shape[1])
    if method in ['fine_tuned', 'frozen_encoder']:
        model.encoder.load_state_dict(source_model.encoder.state_dict())
    if method == 'frozen_encoder':
        model.encoder.requires_grad_(False)
        optimizer = torch.optim.AdamW(model.head.parameters(), lr=CONFIG['head_learning_rate'], weight_decay=CONFIG['weight_decay'])
    else:
        encoder_lr = CONFIG['head_learning_rate'] if method == 'scratch' else CONFIG['encoder_learning_rate']
        optimizer = torch.optim.AdamW([
            {'params': model.encoder.parameters(), 'lr': encoder_lr},
            {'params': model.head.parameters(), 'lr': CONFIG['head_learning_rate']}], weight_decay=CONFIG['weight_decay'])
    before = deepcopy(model.encoder.state_dict())
    history = []
    for epoch in range(CONFIG['target_epochs']):
        model.train()
        if method == 'frozen_encoder':
            # Frozen representations must not change through dropout.
            model.encoder.eval()
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(x), y)
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
        history.append(dict(epoch=epoch+1, training_loss=loss.item()))
    delta = sum(float((model.encoder.state_dict()[k] - v).square().sum()) for k, v in before.items())
    assert delta == 0 if method == 'frozen_encoder' else delta > 0
    model.eval()
    return model, history, delta


def aligned(frame, fold, repetition, valid):
    s = frame.loc[frame.fold.eq(fold) & frame.repetition.eq(repetition)].set_index('location_id')
    assert s.index.is_unique and set(s.index) == set(valid.location_id)
    s = s.loc[valid.location_id]
    return checked_probabilities(s, valid.age_class.to_numpy(dtype=int))


def run():
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    paths = [ROOT / p for p in [
        'data/preprocessed_age_labels/madrid_supervised_features.parquet',
        'data/preprocessed_age_labels/amsterdam_supervised_features.parquet',
        'data/baseline/experiment.json', 'data/baseline/adaptation_membership.csv',
        'data/baseline/validation_predictions.parquet',
        'data/torch_comparison/predictions.parquet', 'data/torch_comparison/experiment.json']]
    comparison = json.loads(paths[-1].read_text())
    assert comparison['outputs']['predictions.parquet'] == sha(paths[-2])
    for name, digest in comparison['inputs'].items():
        assert sha(ROOT / name) == digest, f'Stale comparison input: {name}'
    source = pd.read_parquet(paths[0])
    target = pd.read_parquet(paths[1])
    assert source.location_id.is_unique and target.location_id.is_unique
    assert set(source.location_id).isdisjoint(target.location_id)
    features = json.loads(paths[2].read_text())['feature_names']
    assert len(features) == 54 and not set(features).intersection(['weighted_mean_year', 'age_class', 'fold', 'px_key', 'py_key'])
    members = pd.read_csv(paths[3])
    baseline = pd.read_parquet(paths[4])
    baseline = baseline.loc[baseline.strategy.eq('source_target_blend') & baseline.samples_per_class.eq(25)]
    reference = pd.read_parquet(paths[5])
    reference = reference.loc[reference.method.isin(['original_blend', 'raw_mlp'])]
    manifest = dict(config=CONFIG, features=features,
        protocol='Madrid supervised pretraining; fixed-budget Amsterdam fine-tuning; original 5 spatial folds x 3 matched repetitions',
        independent_test=False, outer_folds_previously_used_for_development=True,
        target_labels_per_run=100, source_label_boundaries='Amsterdam: 1945/1984/2004/2024',
        target_training='Fresh head; fixed 300 epochs; no target checkpoint or outer selection',
        preprocessing='Fit on Madrid fit partition only; shared across all new neural methods',
        inputs={str(p.relative_to(ROOT)): sha(p) for p in paths}, code_sha256=sha(__file__),
        versions=dict(python=platform.python_version(), torch=torch.__version__, sklearn=sklearn.__version__, numpy=np.__version__, pandas=pd.__version__))
    (OUT / 'experiment.json').write_text(json.dumps(manifest, indent=2)+'\n')
    source_model, preprocessing, details = pretrain(source, features)
    manifest['pretraining'] = details
    target_tensor = torch.tensor(preprocessing.transform(target[features]), dtype=torch.float32)
    source_p = probabilities(source_model, target_tensor)
    rows, predictions, classes, histories, audits = [], [], [], [], []
    for fold in range(CONFIG['folds']):
        valid = target.loc[target.fold.eq(fold)]
        eligible = target.loc[target.fold.ne(fold)]
        assert set(valid.spatial_group).isdisjoint(eligible.spatial_group)
        valid_x = target_tensor[target.fold.to_numpy() == fold]
        y = valid.age_class.to_numpy(dtype=int)
        for repetition in range(CONFIG['repetitions']):
            seed = 10042 + 100 * fold + repetition
            ids = members.loc[members.fold.eq(fold) & members.repetition.eq(repetition)
                              & members.samples_per_class.eq(25), 'location_id']
            assert len(ids) == 100 and ids.is_unique
            sample = eligible.set_index('location_id', drop=False).loc[ids]
            assert sample.age_class.value_counts().reindex(CLASSES).eq(25).all()
            assert set(sample.location_id).isdisjoint(valid.location_id)
            train_x = torch.tensor(preprocessing.transform(sample[features]), dtype=torch.float32)
            train_y = torch.tensor(sample.age_class.to_numpy(dtype=np.int64)-1)
            p_rf = aligned(reference.loc[reference.method.eq('original_blend')], fold, repetition, valid)
            old = baseline.loc[baseline.fold.eq(fold) & baseline.repetition.eq(repetition)].set_index('location_id').loc[valid.location_id]
            np.testing.assert_array_equal(old.y_true, y)
            np.testing.assert_array_equal(old.y_pred, CLASSES[p_rf.argmax(1)])
            candidates = dict(source_only=source_p[target.fold.to_numpy() == fold], original_blend=p_rf,
                              raw_mlp=aligned(reference.loc[reference.method.eq('raw_mlp')], fold, repetition, valid))
            for method in ['frozen_encoder', 'fine_tuned', 'scratch', 'scratch_matched_lr']:
                model, history, delta = adapt(source_model, train_x, train_y, method, seed)
                candidates[method] = probabilities(model, valid_x)
                histories.extend(dict(method=method, fold=fold, repetition=repetition, **h) for h in history)
                audits.append(dict(method=method, fold=fold, repetition=repetition, n_labels=100,
                                   overlap_locations=0, overlap_spatial_groups=0, encoder_squared_change=delta))
                if fold == 0 and repetition == 0:
                    torch.save(dict(state_dict=model.state_dict(), config=CONFIG, features=features), OUT / f'{method}_fold0_rep0.pt')
            candidates['rf_fine_tuned_blend'] = (p_rf + candidates['fine_tuned']) / 2
            for method, p in candidates.items():
                pred = CLASSES[p.argmax(1)]
                frame = pd.DataFrame(dict(location_id=valid.location_id.to_numpy(), spatial_group=valid.spatial_group.to_numpy(),
                    y_true=y, y_pred=pred, method=method, fold=fold, repetition=repetition,
                    **{f'p{c}': p[:, i] for i, c in enumerate(CLASSES)}))
                checked_probabilities(frame, y)
                predictions.append(frame)
                rows.append(dict(method=method, fold=fold, repetition=repetition, **metrics(y, p)))
                precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=CLASSES, zero_division=0)
                classes.extend(dict(method=method, fold=fold, repetition=repetition, age_class=int(c),
                                    precision=precision[i], recall=recall[i], f1=f1[i], support=int(support[i])) for i, c in enumerate(CLASSES))
        print(f'Amsterdam fold {fold+1}/5 complete; {time.monotonic()-started:.1f}s elapsed', flush=True)
    runs = pd.DataFrame(rows)
    runs.to_csv(OUT / 'run_metrics.csv', index=False)
    folds = runs.groupby(['method', 'fold'])[METRICS].mean()
    folds.to_csv(OUT / 'fold_metrics.csv')
    summary = folds.groupby('method').mean()
    summary['macro_f1_fold_sd'] = folds.groupby('method').macro_f1.std()
    summary['delta_vs_rf'] = summary.macro_f1 - summary.loc['original_blend', 'macro_f1']
    summary['delta_vs_raw_mlp'] = summary.macro_f1 - summary.loc['raw_mlp', 'macro_f1']
    summary.to_csv(OUT / 'summary.csv')
    summary.to_csv(ROOT / 'docs/results/torch_transfer_scores.csv')
    paired = folds.macro_f1.unstack('method')
    paired['fine_tuned_minus_frozen'] = paired.fine_tuned - paired.frozen_encoder
    paired['fine_tuned_minus_scratch'] = paired.fine_tuned - paired.scratch
    paired['fine_tuned_minus_scratch_matched_lr'] = paired.fine_tuned - paired.scratch_matched_lr
    paired['fine_tuned_minus_rf'] = paired.fine_tuned - paired.original_blend
    paired.to_csv(OUT / 'paired_fold_scores.csv')
    pd.concat(predictions, ignore_index=True).to_parquet(OUT / 'predictions.parquet', index=False)
    pd.DataFrame(classes).to_csv(OUT / 'class_metrics.csv', index=False)
    pd.DataFrame(histories).to_csv(OUT / 'target_history.csv', index=False)
    pd.DataFrame(audits).to_csv(OUT / 'split_audit.csv', index=False)
    pd.concat([members.loc[members.samples_per_class.eq(25)]], ignore_index=True).to_csv(OUT / 'adaptation_membership.csv', index=False)
    manifest['elapsed_seconds'] = time.monotonic() - started
    manifest['outputs'] = {p.name: sha(p) for p in OUT.iterdir() if p.is_file() and p.name != 'experiment.json'}
    (OUT / 'experiment.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(summary.to_string(), flush=True)


def verify():
    manifest = json.loads((OUT / 'experiment.json').read_text())
    assert manifest['code_sha256'] == sha(__file__)
    for name, digest in manifest['inputs'].items():
        assert sha(ROOT / name) == digest, f'Changed input: {name}'
    for name, digest in manifest['outputs'].items():
        assert sha(OUT / name) == digest, f'Changed output: {name}'
    target = pd.read_parquet(ROOT / 'data/preprocessed_age_labels/amsterdam_supervised_features.parquet')
    source = pd.read_parquet(ROOT / 'data/preprocessed_age_labels/madrid_supervised_features.parquet')
    membership = pd.read_parquet(OUT / 'source_membership.parquet').set_index('location_id').loc[source.location_id]
    np.testing.assert_array_equal(membership.harmonized_age_class, source_labels(source))
    fit = membership.pretraining_partition.eq('fit')
    assert set(membership.loc[fit, 'spatial_group']).isdisjoint(membership.loc[~fit, 'spatial_group'])
    preprocessing = joblib.load(OUT / 'source_preprocessor.joblib')
    expected = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler()).fit(source.loc[source.fold.ne(CONFIG['source_checkpoint_fold']), manifest['features']])
    for step, attributes in [('simpleimputer', ['statistics_']), ('standardscaler', ['mean_', 'scale_'])]:
        for attr in attributes:
            np.testing.assert_allclose(getattr(preprocessing.named_steps[step], attr), getattr(expected.named_steps[step], attr))
    predictions = pd.read_parquet(OUT / 'predictions.parquet')
    runs = pd.read_csv(OUT / 'run_metrics.csv').set_index(['method', 'fold', 'repetition'])
    assert len(runs) == len(METHODS)*15 and runs.index.is_unique
    assert set(runs.index.get_level_values('method')) == set(METHODS)
    recomputed = []
    for key, frame in predictions.groupby(['method', 'fold', 'repetition']):
        assert frame.location_id.is_unique
        valid = target.loc[target.fold.eq(key[1])].set_index('location_id')
        assert set(frame.location_id) == set(valid.index)
        np.testing.assert_array_equal(frame.y_true, valid.loc[frame.location_id, 'age_class'])
        p = checked_probabilities(frame, frame.y_true)
        result = metrics(frame.y_true.to_numpy(), p)
        for metric, value in result.items():
            np.testing.assert_allclose(value, runs.loc[key, metric], atol=1e-12)
        recomputed.append(dict(method=key[0], fold=key[1], repetition=key[2], **result))
    recomputed = pd.DataFrame(recomputed).groupby(['method', 'fold'])[METRICS].mean()
    summary = pd.read_csv(OUT / 'summary.csv', index_col='method')
    np.testing.assert_allclose(recomputed.groupby('method').mean(), summary[METRICS], atol=1e-12)
    np.testing.assert_allclose(recomputed.groupby('method').macro_f1.std(), summary.macro_f1_fold_sd, atol=1e-12)
    members = pd.read_csv(OUT / 'adaptation_membership.csv')
    original = pd.read_csv(ROOT / 'data/baseline/adaptation_membership.csv')
    for (fold, repetition), sample in members.groupby(['fold', 'repetition']):
        ids = sample.location_id
        old_ids = original.loc[original.fold.eq(fold) & original.repetition.eq(repetition) & original.samples_per_class.eq(25), 'location_id']
        assert set(ids) == set(old_ids) and len(ids) == 100 and ids.is_unique
        train = target.set_index('location_id').loc[ids]
        valid = target.loc[target.fold.eq(fold)]
        assert train.age_class.value_counts().reindex(CLASSES).eq(25).all()
        assert set(train.spatial_group).isdisjoint(valid.spatial_group)
    # Verify frozen versus changed encoders in the actual exported weights.
    source_weights = torch.load(OUT / 'source_weights.pt', weights_only=True)['state_dict']
    for method in ['frozen_encoder', 'fine_tuned', 'scratch', 'scratch_matched_lr']:
        weights = torch.load(OUT / f'{method}_fold0_rep0.pt', weights_only=True)['state_dict']
        same = all(torch.equal(weights[k], v) for k, v in source_weights.items() if k.startswith('encoder.'))
        assert same == (method == 'frozen_encoder')
    print(f'Verified {len(runs)} run scores, {len(predictions):,} probability rows, 15 exact budgets, source preprocessing and encoder weights.')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    verify() if args.verify else run()
