"""PyTorch denoising self-supervision and frozen-encoder MLP, spatial development CV.

Run: uv run python -m scripts.self_supervised_torch
See notebooks/experiments/self_supervised.ipynb for the protocol and discussion.
"""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import platform
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, precision_recall_fscore_support)
from sklearn.model_selection import GroupShuffleSplit
import torch
from torch import nn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/self_supervised_torch'
CLASSES = np.arange(1, 5)
CONFIG = dict(seed=42, folds=5, repetitions=3, labels_per_class=25,
              hidden_layers=[64, 16, 64], mask_fraction=0.2, max_epochs=60,
              patience=8, min_delta=1e-4, ssl_learning_rate=0.001,
              ssl_batch_size=512, ssl_alpha=0.01, classifier_hidden=[32],
              pretraining_cap_per_city=20000, checkpoint_fraction=0.1, head_epochs=300, head_learning_rate=0.001,
              head_weight_decay=0.01, device="cpu")


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


class DenoisingEncoder(TransformerMixin, BaseEstimator):
    """PyTorch denoising autoencoder; checkpoint on training blocks only."""
    def __init__(self, random_state=42):
        self.random_state = random_state

    def fit(self, X, y=None, *, groups=None):
        if groups is None:
            raise ValueError('Spatial groups are required')
        seed_all(self.random_state)
        fit_idx, check_idx = next(GroupShuffleSplit(
            n_splits=1, test_size=CONFIG['checkpoint_fraction'],
            random_state=self.random_state).split(X, groups=groups))
        assert set(np.asarray(groups)[fit_idx]).isdisjoint(np.asarray(groups)[check_idx])
        self.preprocessor_ = Pipeline([
            ('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
            ('scale', StandardScaler())])
        train = torch.from_numpy(self.preprocessor_.fit_transform(X.iloc[fit_idx]).astype(np.float32))
        check = torch.from_numpy(self.preprocessor_.transform(X.iloc[check_idx]).astype(np.float32))
        self.encoder_ = nn.Sequential(nn.Linear(X.shape[1], 64), nn.ReLU(), nn.Linear(64, 16), nn.ReLU())
        self.decoder_ = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, X.shape[1]))
        model = nn.Sequential(self.encoder_, self.decoder_)
        optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['ssl_learning_rate'], weight_decay=CONFIG['ssl_alpha'])
        check_masked = check * (torch.rand_like(check) >= CONFIG['mask_fraction'])
        best, stale = np.inf, 0
        self.history_ = []
        for epoch in range(CONFIG['max_epochs']):
            model.train()
            losses = []
            for idx in torch.randperm(len(train)).split(CONFIG['ssl_batch_size']):
                clean = train[idx]
                corrupted = clean * (torch.rand_like(clean) >= CONFIG['mask_fraction'])
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.mse_loss(model(corrupted), clean)
                if not torch.isfinite(loss):
                    raise ValueError('Non-finite reconstruction loss')
                loss.backward()
                optimizer.step()
                losses.append((loss.item(), len(idx)))
            model.eval()
            with torch.no_grad():
                mse = nn.functional.mse_loss(model(check_masked), check).item()
            self.history_.append(dict(epoch=epoch+1, checkpoint_mse=mse,
                training_loss=sum(v*n for v,n in losses)/len(train)))
            if mse < best - CONFIG['min_delta']:
                best, stale = mse, 0
                best_state = deepcopy(model.state_dict())
                self.best_epoch_ = epoch + 1
            else:
                stale += 1
            if stale >= CONFIG['patience']:
                break
        model.load_state_dict(best_state)
        model.eval()
        self.encoder_.requires_grad_(False)
        self.checkpoint_mse_ = best
        self.checkpoint_zero_mse_ = check.square().mean().item()
        self.n_fit_, self.n_checkpoint_ = len(train), len(check)
        return self

    def transform(self, X):
        x = torch.from_numpy(self.preprocessor_.transform(X).astype(np.float32))
        self.encoder_.eval()
        with torch.no_grad():
            z = self.encoder_(x).numpy()
        assert z.shape[1] == 16 and np.isfinite(z).all()
        return z


class TorchHead(BaseEstimator):
    """Fixed-budget target MLP: raw logits go to cross entropy, labels 0..3."""
    def __init__(self, random_state=42):
        self.random_state = random_state

    def fit(self, X, y):
        seed_all(self.random_state)
        x = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        labels = torch.as_tensor(np.asarray(y, dtype=np.int64)-1)
        assert set(labels.tolist()) == {0, 1, 2, 3}
        self.classes_ = CLASSES.copy()
        self.model_ = nn.Sequential(nn.Linear(x.shape[1], 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 4))
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=CONFIG['head_learning_rate'], weight_decay=CONFIG['head_weight_decay'])
        self.history_ = []
        self.model_.train()
        for epoch in range(CONFIG['head_epochs']):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(self.model_(x), labels)
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
            self.history_.append(loss.item())
        self.model_.eval()
        return self

    def predict_proba(self, X):
        self.model_.eval()
        with torch.no_grad():
            return self.model_(torch.as_tensor(np.asarray(X), dtype=torch.float32)).softmax(dim=1).numpy()


def head(seed):
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
        ('scale', StandardScaler()), ('classifier', TorchHead(random_state=seed))])


def score(y, p):
    pred = CLASSES[p.argmax(axis=1)]
    return dict(macro_f1=f1_score(y, pred, labels=CLASSES, average='macro', zero_division=0),
                weighted_f1=f1_score(y, pred, labels=CLASSES, average='weighted', zero_division=0),
                accuracy=accuracy_score(y, pred), balanced_accuracy=balanced_accuracy_score(y, pred))


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    torch.set_num_threads(2)
    base = ROOT / 'data/preprocessed_age_labels'
    paths = [base / 'amsterdam_supervised_features.parquet',
             base / 'madrid_all_location_features.parquet',
             ROOT / 'data/baseline/adaptation_membership.csv',
             ROOT / 'data/baseline/validation_predictions.parquet',
             ROOT / 'data/baseline/experiment.json']
    features = json.loads(paths[-1].read_text())['feature_names']
    assert len(features) == 54 and not any('year' in f and 'slope' not in f for f in features)
    # Read source features only: age labels and label-derived columns are excluded.
    madrid = pd.read_parquet(paths[1], columns=['location_id', 'px_key', 'py_key'] + features)
    madrid['spatial_group'] = ('Madrid:' + (madrid.px_key // 20).astype(str)
                               + ':' + (madrid.py_key // 20).astype(str))
    amsterdam = pd.read_parquet(paths[0])
    assert amsterdam.location_id.is_unique and madrid.location_id.is_unique
    members = pd.read_csv(paths[2])
    old = pd.read_parquet(paths[3])
    old = old.loc[old.strategy.eq('source_target_blend') & old.samples_per_class.eq(25)]
    manifest = dict(config=CONFIG, features=features,
                    protocol='Frozen denoising encoder + target-only MLP; original spatial folds and exact baseline adaptation samples',
                    independent_test=False, outer_results_previously_used_for_development=True,
                    source_labels_used=False, target_labels_per_run=100,
                    outer_validation_features_used_for_fitting=False,
                    tuning='Fixed architecture; checkpoint by training-domain reconstruction MSE only',
                    inputs={str(p.relative_to(ROOT)): sha(p) for p in paths},
                    code_sha256=sha(__file__),
                    versions=dict(python=platform.python_version(), sklearn=sklearn.__version__,
                                  numpy=np.__version__, pandas=pd.__version__, torch=torch.__version__))
    (OUT / 'experiment.json').write_text(json.dumps(manifest, indent=2) + '\n')
    rows, class_rows, predictions, histories, audits = [], [], [], [], []
    with threadpool_limits(limits=2):
        for fold in range(CONFIG['folds']):
            valid = amsterdam.loc[amsterdam.fold.eq(fold)].copy()
            eligible = amsterdam.loc[amsterdam.fold.ne(fold)].copy()
            assert set(valid.spatial_group).isdisjoint(eligible.spatial_group)
            # Equal city caps prevent source sample count dominating reconstruction.
            pools = [d.sample(n=min(len(d), CONFIG['pretraining_cap_per_city']),
                              random_state=42 + fold) for d in [madrid, eligible]]
            pool = pd.concat(pools, ignore_index=True)
            assert set(pool.location_id).isdisjoint(valid.location_id)
            assert set(pool.spatial_group).isdisjoint(valid.spatial_group)
            pool[['location_id', 'spatial_group']].assign(fold=fold).to_parquet(
                OUT / f'pretraining_membership_fold{fold}.parquet', index=False)
            encoder = DenoisingEncoder(random_state=42 + fold).fit(
                pool[features], groups=pool.spatial_group)
            histories.extend(dict(fold=fold, **h) for h in encoder.history_)
            audits.append(dict(fold=fold, n_pretraining=len(pool), n_validation=len(valid),
                               n_fit=encoder.n_fit_, n_checkpoint=encoder.n_checkpoint_,
                               best_epoch=encoder.best_epoch_, checkpoint_mse=encoder.checkpoint_mse_,
                               checkpoint_zero_mse=encoder.checkpoint_zero_mse_,
                               overlap_locations=0, overlap_spatial_groups=0))
            z_valid = encoder.transform(valid[features])
            for repetition in range(CONFIG['repetitions']):
                seed = 42 + 10000 + 100 * fold + repetition
                ids = members.loc[members.fold.eq(fold) & members.repetition.eq(repetition)
                                  & members.samples_per_class.eq(25), 'location_id']
                sample = eligible.set_index('location_id', drop=False).loc[ids]
                assert len(sample) == 100
                assert sample.age_class.value_counts().reindex(CLASSES).eq(25).all()
                y = valid.age_class.to_numpy(dtype=int)
                z_sample = encoder.transform(sample[features])
                for method, train_x, valid_x in [
                    ('ssl_mlp', z_sample, z_valid),
                    ('raw_mlp', sample[features], valid[features]),
                ]:
                    model = head(seed)
                    model.fit(train_x, sample.age_class)
                    p = model.predict_proba(valid_x)
                    assert np.array_equal(model.classes_, CLASSES)
                    assert np.isfinite(p).all() and np.allclose(p.sum(axis=1), 1)
                    rows.append(dict(method=method, fold=fold, repetition=repetition,
                                     convergence_warnings=0,
                                     **score(y, p)))
                    pred = CLASSES[p.argmax(axis=1)]
                    precision, recall, f1, support = precision_recall_fscore_support(
                        y, pred, labels=CLASSES, zero_division=0)
                    class_rows.extend(dict(method=method, fold=fold, repetition=repetition,
                                           age_class=int(c), precision=precision[i], recall=recall[i],
                                           f1=f1[i], support=int(support[i])) for i, c in enumerate(CLASSES))
                    predictions.append(pd.DataFrame(dict(location_id=valid.location_id.to_numpy(),
                        y_true=y, y_pred=pred, method=method, fold=fold, repetition=repetition,
                        **{f'p{c}': p[:, i] for i, c in enumerate(CLASSES)})))
                    pd.DataFrame({'epoch': np.arange(1, CONFIG['head_epochs']+1),
                                  'loss': model.named_steps['classifier'].history_}).to_csv(
                        OUT / f'{method}_head_fold{fold}_rep{repetition}.csv', index=False)
                    if fold == 0 and repetition == 0:
                        torch.save(dict(encoder=encoder.encoder_.state_dict() if method == 'ssl_mlp' else None,
                                        head=model.named_steps['classifier'].model_.state_dict(),
                                        features=features, config=CONFIG), OUT / f'{method}_weights.pt')
                        joblib.dump(dict(encoder_preprocessor=encoder.preprocessor_ if method == 'ssl_mlp' else None,
                                         head_preprocessor=model[:-1], features=features),
                                    OUT / f'{method}_fold0_rep0.joblib')
                reference = old.loc[old.fold.eq(fold) & old.repetition.eq(repetition)].set_index('location_id').loc[valid.location_id]
                np.testing.assert_array_equal(reference.y_true, y)
                rows.append(dict(method='original_blend', fold=fold, repetition=repetition,
                                 convergence_warnings=0,
                                 macro_f1=f1_score(y, reference.y_pred, labels=CLASSES, average='macro'),
                                 weighted_f1=f1_score(y, reference.y_pred, labels=CLASSES, average='weighted'),
                                 accuracy=accuracy_score(y, reference.y_pred),
                                 balanced_accuracy=balanced_accuracy_score(y, reference.y_pred)))
            print(f'Fold {fold + 1}/5 complete; encoder epoch {encoder.best_epoch_}; '
                  f'elapsed {time.monotonic() - started:.1f}s', flush=True)
    runs = pd.DataFrame(rows)
    runs.to_csv(OUT / 'run_metrics.csv', index=False)
    pd.DataFrame(class_rows).to_csv(OUT / 'class_metrics.csv', index=False)
    pd.concat(predictions, ignore_index=True).to_parquet(OUT / 'predictions.parquet', index=False)
    pd.DataFrame(histories).to_csv(OUT / 'pretraining_history.csv', index=False)
    pd.DataFrame(audits).to_csv(OUT / 'split_audit.csv', index=False)
    metric_names = ['macro_f1', 'weighted_f1', 'accuracy', 'balanced_accuracy']
    folds = runs.groupby(['method', 'fold'])[metric_names].mean().reset_index()
    folds.to_csv(OUT / 'fold_metrics.csv', index=False)
    summary = folds.groupby('method')[metric_names].mean()
    summary['macro_f1_fold_sd'] = folds.groupby('method').macro_f1.std()
    summary.to_csv(OUT / 'summary.csv')
    paired = folds.pivot(index='fold', columns='method', values='macro_f1')
    paired['ssl_minus_raw'] = paired.ssl_mlp - paired.raw_mlp
    paired['ssl_minus_blend'] = paired.ssl_mlp - paired.original_blend
    paired.to_csv(OUT / 'paired_fold_scores.csv')
    manifest['elapsed_seconds'] = time.monotonic() - started
    manifest['outputs'] = {p.name: sha(p) for p in OUT.iterdir() if p.is_file() and p.name != 'experiment.json'}
    (OUT / 'experiment.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(summary.to_string(), flush=True)
    return summary


if __name__ == '__main__':
    run()
