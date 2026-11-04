"""Regularized CatBoost with nested target spatial selection at 25 labels/class.

Run: uv run --extra boosting --extra deep-learning python -m scripts.catboost_experiment
Verify: same command with --verify. Existing transfer artifacts are prerequisites.
"""
from pathlib import Path
import json
import platform
import shutil
import time

from catboost import CatBoostClassifier
import catboost
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
import torch

from scripts.self_supervised_torch import CLASSES, sha
from scripts.torch_transfer import TransferMLP, probabilities, metrics
from scripts.compare_torch import checked_probabilities

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/catboost_experiment'
CONFIG = dict(seed=42, source_depth=4, source_tree_candidates=[150, 300, 600],
    target_depths=[3, 4, 5], target_tree_candidates=[150, 300],
    learning_rate=0.03, l2_leaf_reg=10.0, inner_folds=3,
    folds=5, repetitions=3, labels_per_class=25, source_checkpoint_fold=0,
    source_weight=0.5, neural_weight=0.5, thread_count=2)
METRICS = ['macro_f1', 'weighted_f1', 'accuracy', 'balanced_accuracy', 'severe_error_rate']
METHODS = ['catboost_source_only', 'catboost_target', 'catboost_source_target',
           'catboost_neural_blend', 'original_blend', 'rf_fine_tuned_blend']


def make_model(depth, iterations, seed):
    return CatBoostClassifier(depth=int(depth), iterations=int(iterations),
        learning_rate=CONFIG['learning_rate'], l2_leaf_reg=CONFIG['l2_leaf_reg'],
        loss_function='MultiClass', auto_class_weights='Balanced',
        random_seed=int(seed), thread_count=CONFIG['thread_count'],
        task_type='CPU', allow_writing_files=False, verbose=False,
        use_best_model=False)


def imputer():
    return SimpleImputer(strategy='median', keep_empty_features=True)


def cb_probabilities(model, x, trees=None):
    assert set(model.classes_) == set(CLASSES)
    kwargs = dict(ntree_end=int(trees)) if trees is not None else {}
    p = model.predict_proba(x, **kwargs)
    p = p[:, [list(model.classes_).index(c) for c in CLASSES]]
    assert np.isfinite(p).all() and (p >= 0).all()
    np.testing.assert_allclose(p.sum(1), 1, atol=1e-6)
    return p


def aligned(reference, method, fold, repetition, valid):
    frame = reference.loc[reference.method.eq(method) & reference.fold.eq(fold)
                          & reference.repetition.eq(repetition)].set_index('location_id')
    assert frame.index.is_unique and set(frame.index) == set(valid.location_id)
    return checked_probabilities(frame.loc[valid.location_id], valid.age_class.to_numpy(dtype=int))


def spatial_inner_splits(sample, seed):
    for n in [3, 2]:
        splitter = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed)
        splits = list(splitter.split(sample, sample.age_class, sample.spatial_group))
        if all(set(sample.iloc[train].age_class) == set(CLASSES) for train, _ in splits):
            for train, check in splits:
                assert set(sample.iloc[train].spatial_group).isdisjoint(sample.iloc[check].spatial_group)
            return splits
    raise ValueError('Allocated labels cannot support a spatial inner split with all training classes.')


def train_source(source, features):
    # Original Madrid class boundaries match the RF source task.
    fit = source.fold.ne(CONFIG['source_checkpoint_fold'])
    assert set(source.loc[fit, 'spatial_group']).isdisjoint(source.loc[~fit, 'spatial_group'])
    prep = imputer()
    x = prep.fit_transform(source.loc[fit, features])
    check_x = prep.transform(source.loc[~fit, features])
    model = make_model(CONFIG['source_depth'], max(CONFIG['source_tree_candidates']), CONFIG['seed'])
    model.fit(x, source.loc[fit, 'age_class'])
    choices = []
    for trees in CONFIG['source_tree_candidates']:
        p = cb_probabilities(model, check_x, trees)
        choices.append(dict(iterations=trees, **metrics(source.loc[~fit, 'age_class'].to_numpy(dtype=int), p)))
    scores = pd.DataFrame(choices).sort_values(['macro_f1', 'iterations'], ascending=[False, True])
    selected = int(scores.iloc[0].iterations)
    scores.to_csv(OUT / 'source_checkpoint_scores.csv', index=False)
    source[['location_id', 'spatial_group', 'fold', 'age_class']].assign(
        selection_partition=np.where(fit, 'fit', 'checkpoint'), final_source_fit=True
    ).to_parquet(OUT / 'source_membership.parquet', index=False)
    # Refit on all Madrid locations only after source-only selection.
    final_prep = imputer()
    final_model = make_model(CONFIG['source_depth'], selected, CONFIG['seed'])
    final_model.fit(final_prep.fit_transform(source[features]), source.age_class)
    final_model.save_model(str(OUT / 'source.cbm'))
    joblib.dump(final_prep, OUT / 'source_imputer.joblib')
    print(f'Madrid source selected {selected} trees using its spatial checkpoint; refitted on {len(source):,} source locations.', flush=True)
    return final_model, final_prep, dict(iterations=selected, n_source=len(source),
        n_selection_fit=int(fit.sum()), n_checkpoint=int((~fit).sum()),
        selected_checkpoint_macro_f1=float(scores.iloc[0].macro_f1))


def select_target(sample, features, seed):
    """Select using only the allocated sample. Outer data cannot enter here."""
    rows, inner_predictions, memberships = [], [], []
    for split, (train, check) in enumerate(spatial_inner_splits(sample, seed)):
        train_frame, check_frame = sample.iloc[train], sample.iloc[check]
        prep = imputer()
        x = prep.fit_transform(train_frame[features])
        check_x = prep.transform(check_frame[features])
        for partition, frame in [('fit', train_frame), ('check', check_frame)]:
            memberships.append(frame[['location_id', 'spatial_group', 'age_class']].assign(inner_split=split, partition=partition))
        for depth in CONFIG['target_depths']:
            model = make_model(depth, max(CONFIG['target_tree_candidates']), seed)
            model.fit(x, train_frame.age_class)
            for trees in CONFIG['target_tree_candidates']:
                p = cb_probabilities(model, check_x, trees)
                y = check_frame.age_class.to_numpy(dtype=int)
                rows.append(dict(inner_split=split, depth=depth, iterations=trees, **metrics(y, p)))
                inner_predictions.append(pd.DataFrame(dict(location_id=check_frame.location_id.to_numpy(),
                    inner_split=split, depth=depth, iterations=trees, y_true=y,
                    y_pred=CLASSES[p.argmax(1)], **{f'p{c}': p[:, i] for i, c in enumerate(CLASSES)})))
    scores = pd.DataFrame(rows)
    means = scores.groupby(['depth', 'iterations']).macro_f1.mean().reset_index().sort_values(
        ['macro_f1', 'depth', 'iterations'], ascending=[False, True, True])
    best = means.iloc[0]
    prep = imputer()
    model = make_model(best.depth, best.iterations, seed)
    model.fit(prep.fit_transform(sample[features]), sample.age_class)
    return model, prep, dict(depth=int(best.depth), iterations=int(best.iterations),
        inner_macro_f1=float(best.macro_f1)), scores, pd.concat(inner_predictions), pd.concat(memberships)


def predict_example(frame):
    """Fold-0/repetition-0 native CatBoost + neural bundle, not a final model."""
    meta = json.loads((OUT / 'experiment.json').read_text())
    features = meta['features']
    if frame.empty or not set(features).issubset(frame.columns):
        raise ValueError('Provide nonempty data with the expected 54 predictors.')
    source = CatBoostClassifier().load_model(str(OUT / 'source.cbm'))
    target = CatBoostClassifier().load_model(str(OUT / 'target_fold0_rep0.cbm'))
    source_prep = joblib.load(OUT / 'source_imputer.joblib')
    target_prep = joblib.load(OUT / 'target_imputer_fold0_rep0.joblib')
    neural_prep = joblib.load(OUT / 'neural_preprocessor.joblib')
    checkpoint = torch.load(OUT / 'fine_tuned_fold0_rep0.pt', weights_only=True, map_location='cpu')
    assert checkpoint['features'] == features
    neural = TransferMLP(len(features))
    neural.load_state_dict(checkpoint['state_dict'])
    x = torch.tensor(neural_prep.transform(frame[features]), dtype=torch.float32)
    p_cb = .5*cb_probabilities(source, source_prep.transform(frame[features])) + .5*cb_probabilities(target, target_prep.transform(frame[features]))
    return .5*p_cb + .5*probabilities(neural, x)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    torch.set_num_threads(2)
    paths = [ROOT / p for p in [
        'data/preprocessed_age_labels/madrid_supervised_features.parquet',
        'data/preprocessed_age_labels/amsterdam_supervised_features.parquet',
        'data/baseline/experiment.json', 'data/baseline/adaptation_membership.csv',
        'data/torch_transfer/predictions.parquet', 'data/torch_transfer/experiment.json',
        'data/torch_transfer/fine_tuned_fold0_rep0.pt', 'data/torch_transfer/source_preprocessor.joblib',
        'scripts/torch_transfer.py', 'scripts/self_supervised_torch.py', 'scripts/compare_torch.py']]
    previous = json.loads(paths[5].read_text())
    assert previous['code_sha256'] == sha(paths[8])
    for name, digest in previous['inputs'].items():
        assert sha(ROOT / name) == digest, f'Changed input: {name}'
    for path in [paths[4], paths[6], paths[7]]:
        assert previous['outputs'][path.name] == sha(path)
    features = json.loads(paths[2].read_text())['feature_names']
    assert len(features) == 54 and not set(features).intersection(['age_class', 'weighted_mean_year', 'px_key', 'py_key', 'fold'])
    source, target = pd.read_parquet(paths[0]), pd.read_parquet(paths[1])
    assert source.location_id.is_unique and target.location_id.is_unique
    assert set(source.location_id).isdisjoint(target.location_id)
    members, reference = pd.read_csv(paths[3]), pd.read_parquet(paths[4])
    manifest = dict(config=CONFIG, features=features, independent_test=False,
        outer_folds_previously_used_for_development=True,
        source_labels='Original Madrid bins (1960/1984/2004/2024), matching RF source',
        neural_labels='Earlier harmonized Madrid neural pretraining; unchanged saved fine-tuned predictions',
        target_selection='Three spatial inner folds within each 100-label adaptation sample; fallback to two if required',
        tie_breaker='Highest mean inner macro F1, then smaller depth, then fewer trees',
        blend='Fixed 0.25 Madrid CatBoost + 0.25 Amsterdam CatBoost + 0.50 fine-tuned MLP',
        inputs={str(p.relative_to(ROOT)): sha(p) for p in paths}, code_sha256=sha(__file__),
        versions=dict(catboost=catboost.__version__, python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__))
    (OUT / 'experiment.json').write_text(json.dumps(manifest, indent=2)+'\n')
    cb_source, source_prep, source_details = train_source(source, features)
    manifest['source_selection'] = source_details
    source_p = cb_probabilities(cb_source, source_prep.transform(target[features]))
    rows, predictions, classes, selections, inner_scores, inner_predictions, inner_members = [], [], [], [], [], [], []
    audits = []
    for fold in range(5):
        valid = target.loc[target.fold.eq(fold)]
        eligible = target.loc[target.fold.ne(fold)]
        assert set(valid.spatial_group).isdisjoint(eligible.spatial_group)
        y = valid.age_class.to_numpy(dtype=int)
        for repetition in range(3):
            seed = 10042 + 100*fold + repetition
            ids = members.loc[members.fold.eq(fold) & members.repetition.eq(repetition) & members.samples_per_class.eq(25), 'location_id']
            assert len(ids) == 100 and ids.is_unique
            sample = eligible.set_index('location_id', drop=False).loc[ids].reset_index(drop=True)
            assert sample.age_class.value_counts().reindex(CLASSES).eq(25).all()
            assert set(sample.spatial_group).isdisjoint(valid.spatial_group)
            model, prep, selected, scores, inner_p, inner_m = select_target(sample, features, seed)
            key = dict(fold=fold, repetition=repetition)
            selections.append(dict(**key, **selected))
            inner_scores.append(scores.assign(**key))
            inner_predictions.append(inner_p.assign(**key))
            inner_members.append(inner_m.assign(**key))
            p_target = cb_probabilities(model, prep.transform(valid[features]))
            p_source = source_p[target.fold.to_numpy() == fold]
            p_transfer = .5*p_source+.5*p_target
            p_neural = aligned(reference, 'fine_tuned', fold, repetition, valid)
            candidates = dict(catboost_source_only=p_source, catboost_target=p_target,
                catboost_source_target=p_transfer, catboost_neural_blend=.5*p_transfer+.5*p_neural,
                original_blend=aligned(reference, 'original_blend', fold, repetition, valid),
                rf_fine_tuned_blend=aligned(reference, 'rf_fine_tuned_blend', fold, repetition, valid))
            for method, p in candidates.items():
                pred = CLASSES[p.argmax(1)]
                frame = pd.DataFrame(dict(location_id=valid.location_id.to_numpy(), spatial_group=valid.spatial_group.to_numpy(),
                    method=method, **key, y_true=y, y_pred=pred, **{f'p{c}': p[:, i] for i,c in enumerate(CLASSES)}))
                checked_probabilities(frame, y)
                predictions.append(frame)
                rows.append(dict(method=method, **key, **metrics(y, p)))
                pr, re, f1, support = precision_recall_fscore_support(y, pred, labels=CLASSES, zero_division=0)
                classes.extend(dict(method=method, **key, age_class=int(c), precision=pr[i], recall=re[i],
                                    f1=f1[i], support=int(support[i])) for i,c in enumerate(CLASSES))
            audits.append(dict(**key, target_labels=100, overlap_locations=0, overlap_spatial_groups=0))
            if fold == 0 and repetition == 0:
                model.save_model(str(OUT / 'target_fold0_rep0.cbm'))
                joblib.dump(prep, OUT / 'target_imputer_fold0_rep0.joblib')
                shutil.copyfile(paths[6], OUT / paths[6].name)
                shutil.copyfile(paths[7], OUT / 'neural_preprocessor.joblib')
                np.testing.assert_allclose(predict_example(valid), candidates['catboost_neural_blend'], atol=1e-6)
            print(f'Fold {fold+1}/5 rep {repetition+1}/3: selected depth={selected["depth"]}, trees={selected["iterations"]}; {time.monotonic()-started:.1f}s', flush=True)
    runs = pd.DataFrame(rows)
    folds = runs.groupby(['method', 'fold'])[METRICS].mean()
    summary = folds.groupby('method').mean()
    summary['macro_f1_fold_sd'] = folds.groupby('method').macro_f1.std()
    summary['delta_vs_rf'] = summary.macro_f1-summary.loc['original_blend','macro_f1']
    summary['delta_vs_current_hybrid'] = summary.macro_f1-summary.loc['rf_fine_tuned_blend','macro_f1']
    runs.to_csv(OUT / 'run_metrics.csv', index=False)
    folds.to_csv(OUT / 'fold_metrics.csv')
    summary.to_csv(OUT / 'summary.csv')
    summary.to_csv(ROOT / 'docs/results/catboost_scores.csv')
    pd.concat(predictions, ignore_index=True).to_parquet(OUT / 'predictions.parquet', index=False)
    pd.DataFrame(classes).to_csv(OUT / 'class_metrics.csv', index=False)
    pd.DataFrame(selections).to_csv(OUT / 'selections.csv', index=False)
    pd.concat(inner_scores, ignore_index=True).to_csv(OUT / 'inner_scores.csv', index=False)
    pd.concat(inner_predictions, ignore_index=True).to_parquet(OUT / 'inner_predictions.parquet', index=False)
    pd.concat(inner_members, ignore_index=True).to_parquet(OUT / 'inner_membership.parquet', index=False)
    pd.DataFrame(audits).to_csv(OUT / 'split_audit.csv', index=False)
    members.loc[members.samples_per_class.eq(25)].to_csv(OUT / 'adaptation_membership.csv', index=False)
    paired = folds.macro_f1.unstack('method')
    paired['new_hybrid_minus_current'] = paired.catboost_neural_blend-paired.rf_fine_tuned_blend
    paired.to_csv(OUT / 'paired_fold_scores.csv')
    manifest['elapsed_seconds'] = time.monotonic()-started
    manifest['outputs'] = {p.name: sha(p) for p in OUT.iterdir() if p.is_file() and p.name != 'experiment.json'}
    (OUT / 'experiment.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(summary.to_string(), flush=True)
    return summary


def verify():
    manifest = json.loads((OUT / 'experiment.json').read_text())
    assert manifest['code_sha256'] == sha(__file__)
    for name,digest in manifest['inputs'].items():
        assert sha(ROOT / name) == digest, f'Changed input {name}'
    for name,digest in manifest['outputs'].items():
        assert sha(OUT / name) == digest, f'Changed output {name}'
    runs = pd.read_csv(OUT / 'run_metrics.csv').set_index(['method','fold','repetition'])
    predictions = pd.read_parquet(OUT / 'predictions.parquet')
    target = pd.read_parquet(ROOT / 'data/preprocessed_age_labels/amsterdam_supervised_features.parquet')
    assert len(runs) == 90 and runs.index.is_unique and set(runs.index.get_level_values(0)) == set(METHODS)
    for key,frame in predictions.groupby(['method','fold','repetition']):
        valid = target.loc[target.fold.eq(key[1])].set_index('location_id')
        assert frame.location_id.is_unique and set(frame.location_id) == set(valid.index)
        np.testing.assert_array_equal(frame.y_true, valid.loc[frame.location_id, 'age_class'])
        p = checked_probabilities(frame, frame.y_true)
        for name,value in metrics(frame.y_true.to_numpy(),p).items():
            np.testing.assert_allclose(value, runs.loc[key,name], atol=1e-12)
    folds = runs.reset_index().groupby(['method','fold'])[METRICS].mean()
    summary = pd.read_csv(OUT / 'summary.csv', index_col='method')
    np.testing.assert_allclose(folds.groupby('method').mean(), summary[METRICS], atol=1e-12)
    np.testing.assert_allclose(folds.groupby('method').macro_f1.std(), summary.macro_f1_fold_sd, atol=1e-12)
    inner_p = pd.read_parquet(OUT / 'inner_predictions.parquet')
    inner_scores = pd.read_csv(OUT / 'inner_scores.csv').set_index(['fold','repetition','inner_split','depth','iterations'])
    for key,frame in inner_p.groupby(['fold','repetition','inner_split','depth','iterations']):
        p = checked_probabilities(frame, frame.y_true)
        for name,value in metrics(frame.y_true.to_numpy(),p).items():
            np.testing.assert_allclose(value, inner_scores.loc[key,name], atol=1e-12)
    members = pd.read_csv(OUT / 'adaptation_membership.csv')
    originals = pd.read_csv(ROOT / 'data/baseline/adaptation_membership.csv')
    inners = pd.read_parquet(OUT / 'inner_membership.parquet')
    choices = pd.read_csv(OUT / 'selections.csv').set_index(['fold','repetition'])
    for (fold,rep),sample in members.groupby(['fold','repetition']):
        assert len(sample) == 100 and sample.location_id.is_unique
        old = originals.loc[originals.fold.eq(fold) & originals.repetition.eq(rep) & originals.samples_per_class.eq(25)]
        assert set(sample.location_id) == set(old.location_id)
        train = target.set_index('location_id').loc[sample.location_id]
        assert train.age_class.value_counts().reindex(CLASSES).eq(25).all()
        assert set(train.spatial_group).isdisjoint(target.loc[target.fold.eq(fold),'spatial_group'])
        for _,frame in inners.loc[inners.fold.eq(fold) & inners.repetition.eq(rep)].groupby('inner_split'):
            fit,check = frame.loc[frame.partition.eq('fit')], frame.loc[frame.partition.eq('check')]
            assert set(fit.location_id).isdisjoint(check.location_id)
            assert set(fit.location_id)|set(check.location_id) == set(sample.location_id)
            assert set(fit.spatial_group).isdisjoint(check.spatial_group)
        means = inner_scores.loc[(fold,rep)].reset_index().groupby(['depth','iterations']).macro_f1.mean().reset_index()
        best = means.sort_values(['macro_f1','depth','iterations'],ascending=[False,True,True]).iloc[0]
        chosen = choices.loc[(fold,rep)]
        assert int(chosen.depth) == int(best.depth) and int(chosen.iterations) == int(best.iterations)
        np.testing.assert_allclose(chosen.inner_macro_f1,best.macro_f1,atol=1e-12)
    for _,frame in predictions.groupby(['fold','repetition']):
        parts = {m: s.set_index('location_id').sort_index() for m,s in frame.groupby('method')}
        cols = [f'p{c}' for c in CLASSES]
        np.testing.assert_allclose(parts['catboost_source_target'][cols], .5*parts['catboost_source_only'][cols]+.5*parts['catboost_target'][cols],atol=1e-12)
    valid = target.loc[target.fold.eq(0)]
    frame = predictions.loc[predictions.method.eq('catboost_neural_blend') & predictions.fold.eq(0) & predictions.repetition.eq(0)].set_index('location_id').loc[valid.location_id]
    np.testing.assert_allclose(predict_example(valid),frame[[f'p{c}' for c in CLASSES]],atol=1e-6)
    print(f'Verified {len(runs)} outer runs, {len(predictions):,} probability rows, {len(inner_scores)} inner scores, 15 selections/budgets and exported inference.')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify',action='store_true')
    args = parser.parse_args()
    verify() if args.verify else run()
