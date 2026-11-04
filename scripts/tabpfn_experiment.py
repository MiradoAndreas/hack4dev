"""Local TabPFN-v2 matched spatial evaluation. Built with PriorLabs-TabPFN.

Run: uv run --extra tabpfn --extra deep-learning python -m scripts.tabpfn_experiment
Resume: add --resume. Verify scores: --verify. Fresh-process inference: --reload.
Checkpoint acquisition is separate; see docs/TABPFN.md. No hosted prediction.
"""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/tabpfn_experiment'
os.environ.setdefault('SKB_DATA_DIRECTORY', str(OUT / 'cache/skrub'))
os.environ.setdefault('MPLCONFIGDIR', str(OUT / 'cache/matplotlib'))
os.environ['TABPFN_NO_BROWSER'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

from contextlib import contextmanager
import gc
import hashlib
import importlib.metadata
import json
import platform
import resource
import socket
import time
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, precision_recall_fscore_support
import torch
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

from scripts.self_supervised_torch import CLASSES, sha
from scripts.torch_transfer import metrics
from scripts.compare_torch import checked_probabilities

CONFIG = dict(model_version='v2', n_estimators=4, device='cpu', torch_threads=4,
    fit_mode='fit_with_cache', n_preprocessing_jobs=1, query_batch_size=256,
    folds=5, repetitions=3, labels_per_class=25, rf_weight=0.5,
    seed_base=10042, balance_probabilities=False)
METRICS = ['macro_f1', 'weighted_f1', 'accuracy', 'balanced_accuracy',
           'severe_error_rate', 'log_loss', 'brier']
METHODS = ['tabpfn_v2', 'rf_tabpfn_v2_blend', 'original_blend', 'rf_fine_tuned_blend']


@contextmanager
def no_network():
    """Fail closed on attempted socket connections during model inference."""
    def deny(*args, **kwargs):
        raise RuntimeError('Network access is disabled during this local TabPFN experiment.')
    with patch.object(socket.socket, 'connect', deny), patch.object(socket.socket, 'connect_ex', deny), \
         patch.object(socket, 'create_connection', deny), patch.object(socket, 'getaddrinfo', deny):
        yield


def score(y, p):
    # Native float32 probabilities can become float64 when methods are combined
    # in a Parquet table. Use one numerical convention for probability metrics.
    # Classification metrics always use the original probabilities/argmax.
    prob = np.asarray(p, dtype=np.float64)
    prob = prob / prob.sum(axis=1, keepdims=True)
    return dict(**metrics(y, p), log_loss=log_loss(y, prob, labels=CLASSES),
                brier=float(np.mean(np.sum((prob - (np.asarray(y)[:,None] == CLASSES))**2, axis=1))))


def checkpoint_path():
    info = json.loads((OUT / 'checkpoint.json').read_text())
    path = OUT / 'models' / info['filename']
    assert info['gated'] is False and sha(path) == info['sha256']
    assert info['filename'] == 'tabpfn-v2-classifier.ckpt'
    return path


def make_model(seed):
    return TabPFNClassifier.create_default_for_version(ModelVersion.V2,
        model_path=str(checkpoint_path()), device=CONFIG['device'],
        n_estimators=CONFIG['n_estimators'], random_state=int(seed),
        fit_mode=CONFIG['fit_mode'], n_preprocessing_jobs=CONFIG['n_preprocessing_jobs'],
        balance_probabilities=CONFIG['balance_probabilities'], show_progress_bar=False)


def model_probabilities(model, x, progress=None):
    assert np.array_equal(model.classes_, CLASSES)
    parts = []
    for start in range(0, len(x), CONFIG['query_batch_size']):
        parts.append(model.predict_proba(x[start:start+CONFIG['query_batch_size']]))
        if progress and (start // CONFIG['query_batch_size']) % 4 == 3:
            print(f'{progress}: {min(start+CONFIG["query_batch_size"],len(x))}/{len(x)} locations', flush=True)
    p = np.concatenate(parts)
    assert p.shape == (len(x),4) and np.isfinite(p).all() and (p >= 0).all()
    np.testing.assert_allclose(p.sum(1),1,atol=1e-6)
    return p


def aligned(reference, method, fold, repetition, valid):
    frame = reference.loc[reference.method.eq(method) & reference.fold.eq(fold)
                          & reference.repetition.eq(repetition)].set_index('location_id')
    assert frame.index.is_unique and set(frame.index) == set(valid.location_id)
    return checked_probabilities(frame.loc[valid.location_id], valid.age_class.to_numpy(dtype=int))


def input_paths():
    return [ROOT / p for p in [
        'data/preprocessed_age_labels/amsterdam_supervised_features.parquet',
        'data/baseline/experiment.json', 'data/baseline/adaptation_membership.csv',
        'data/torch_transfer/predictions.parquet', 'data/torch_transfer/experiment.json',
        'scripts/self_supervised_torch.py', 'scripts/torch_transfer.py', 'scripts/compare_torch.py',
        'data/tabpfn_experiment/checkpoint.json', 'data/tabpfn_experiment/models/LICENSE.txt',
        'pyproject.toml', 'uv.lock']]


def run(resume=False):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'runs').mkdir(exist_ok=True)
    started = time.monotonic()
    torch.set_num_threads(CONFIG['torch_threads'])
    checkpoint_path()
    paths = input_paths()
    previous = json.loads(paths[4].read_text())
    assert previous['outputs']['predictions.parquet'] == sha(paths[3])
    assert previous['code_sha256'] == sha(ROOT / 'scripts/torch_transfer.py')
    for name,digest in previous['inputs'].items():
        assert sha(ROOT / name) == digest, f'Changed prerequisite: {name}'
    features = json.loads(paths[1].read_text())['feature_names']
    assert len(features)==54 and not set(features).intersection(['age_class','weighted_mean_year','px_key','py_key','fold'])
    target = pd.read_parquet(paths[0])
    assert target.location_id.is_unique
    assert not np.isinf(target[features].to_numpy()).any()
    members = pd.read_csv(paths[2])
    reference = pd.read_parquet(paths[3])
    checkpoint = json.loads(paths[8].read_text())
    manifest = dict(config=CONFIG, checkpoint=checkpoint, features=features,
        independent_test=False, outer_folds_previously_used_for_development=True,
        target_labels_per_run=100, source_city_labels_in_tabpfn_context=False,
        external_pretraining='Public original TabPFN-v2 synthetic-prior classifier, not v3.5 and not real-data-finetuned v2 alias',
        preprocessing='Raw 54 features; built-in TabPFN preprocessing fitted on the 100-label context only',
        checkpoint_selection='Fixed original v2 for public CPU access; no outer score selection',
        tuning='None; fixed four inference estimators, seeds and 50/50 RF blend; no gradient fine-tuning or calibration',
        network='Local inference under blocked socket connections, offline model path; no hosted client',
        skill=dict(name='tabpfn',repository='magnus919/agent-skills',revision='4f03205616b01360374f458f471791188f448591'),
        inputs={str(p.relative_to(ROOT)):sha(p) for p in paths}, code_sha256=sha(__file__),
        versions={name:importlib.metadata.version(name) for name in ['tabpfn','torch','scikit-learn','numpy','pandas']},
        python=platform.python_version(), license='Prior Labs License 1.1 (May 2025), attribution retained; local research evaluation')
    manifest_path = OUT / 'experiment.json'
    if resume and manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        for key in ['config','inputs','code_sha256','versions']:
            assert existing[key]==manifest[key], f'Resume contract changed: {key}'
    elif list((OUT/'runs').glob('*.parquet')):
        raise FileExistsError('Run artifacts exist; use --resume to preserve and validate them.')
    manifest['status']='running'
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    rows, class_rows, parts, audits, timings = [], [], [], [], []
    for fold in range(5):
        valid = target.loc[target.fold.eq(fold)]
        eligible = target.loc[target.fold.ne(fold)]
        assert set(valid.spatial_group).isdisjoint(eligible.spatial_group)
        y = valid.age_class.to_numpy(dtype=int)
        for repetition in range(3):
            ids = members.loc[members.fold.eq(fold)&members.repetition.eq(repetition)&members.samples_per_class.eq(25),'location_id']
            assert len(ids)==100 and ids.is_unique
            sample = eligible.set_index('location_id',drop=False).loc[ids]
            assert sample.age_class.value_counts().reindex(CLASSES).eq(25).all()
            assert set(sample.spatial_group).isdisjoint(valid.spatial_group)
            seed = CONFIG['seed_base']+100*fold+repetition
            run_path = OUT/'runs'/f'fold{fold}_rep{repetition}.parquet'
            time_path = run_path.with_suffix('.json')
            if resume and run_path.exists() and time_path.exists():
                timing=json.loads(time_path.read_text())
                assert timing['sha256']==sha(run_path) and timing['seed']==seed
                saved=pd.read_parquet(run_path).set_index('location_id')
                assert saved.index.is_unique and set(saved.index)==set(valid.location_id)
                p=checked_probabilities(saved.loc[valid.location_id],y)
                print(f'Reused fold {fold} rep {repetition}',flush=True)
            else:
                fit_started=time.monotonic()
                with no_network():
                    model=make_model(seed)
                    x=sample[features].to_numpy(dtype=np.float32)
                    model.fit(x,sample.age_class.to_numpy(dtype=int))
                    fit_seconds=time.monotonic()-fit_started
                    predict_started=time.monotonic()
                    print(f'Fold {fold+1}/5 rep {repetition+1}/3: context fitted in {fit_seconds:.1f}s; predicting {len(valid)} rows',flush=True)
                    p=model_probabilities(model,valid[features].to_numpy(dtype=np.float32),f'Fold {fold+1} rep {repetition+1}')
                predict_seconds=time.monotonic()-predict_started
                saved=pd.DataFrame(dict(location_id=valid.location_id.to_numpy(),y_true=y,y_pred=CLASSES[p.argmax(1)],
                    **{f'p{c}':p[:,i] for i,c in enumerate(CLASSES)}))
                saved.to_parquet(run_path,index=False)
                timing=dict(fold=fold,repetition=repetition,seed=seed,fit_seconds=fit_seconds,
                    predict_seconds=predict_seconds,n_validation=len(valid),
                    peak_process_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                    actual_n_estimators=int(model.n_estimators_),sha256=sha(run_path))
                assert timing['actual_n_estimators']==4
                time_path.write_text(json.dumps(timing,indent=2)+'\n')
                if fold==0 and repetition==0:
                    np.savez_compressed(OUT/'example_context.npz',X_train=x,
                        y_train=sample.age_class.to_numpy(dtype=int),location_ids=sample.location_id.to_numpy(dtype=str),
                        X_probe=valid[features].head(CONFIG['query_batch_size']).to_numpy(dtype=np.float32),p_probe=p[:CONFIG['query_batch_size']],
                        probe_ids=valid.location_id.head(CONFIG['query_batch_size']).to_numpy(dtype=str))
                    # Same-context repeated prediction checks within-process caching.
                    with no_network():
                        np.testing.assert_allclose(model_probabilities(model,valid[features].head(CONFIG['query_batch_size']).to_numpy(dtype=np.float32)),p[:CONFIG['query_batch_size']],atol=1e-6)
                del model
                gc.collect()
            timings.append(timing)
            p_rf=aligned(reference,'original_blend',fold,repetition,valid)
            candidates=dict(tabpfn_v2=p,rf_tabpfn_v2_blend=.5*p+.5*p_rf,original_blend=p_rf,
                rf_fine_tuned_blend=aligned(reference,'rf_fine_tuned_blend',fold,repetition,valid))
            for method,prob in candidates.items():
                pred=CLASSES[prob.argmax(1)]
                frame=pd.DataFrame(dict(location_id=valid.location_id.to_numpy(),spatial_group=valid.spatial_group.to_numpy(),
                    method=method,fold=fold,repetition=repetition,y_true=y,y_pred=pred,
                    **{f'p{c}':prob[:,i] for i,c in enumerate(CLASSES)}))
                checked_probabilities(frame,y)
                parts.append(frame)
                rows.append(dict(method=method,fold=fold,repetition=repetition,**score(y,prob)))
                pr,re,f1,support=precision_recall_fscore_support(y,pred,labels=CLASSES,zero_division=0)
                class_rows.extend(dict(method=method,fold=fold,repetition=repetition,age_class=int(c),
                    precision=pr[i],recall=re[i],f1=f1[i],support=int(support[i])) for i,c in enumerate(CLASSES))
            audits.append(dict(fold=fold,repetition=repetition,seed=seed,target_labels=100,
                overlap_locations=0,overlap_spatial_groups=0))
            # Persist completed run metrics for progress; never select on these scores.
            pd.DataFrame(rows).to_csv(OUT/'metrics_checkpoint.csv',index=False)
            print(f'Completed fold {fold+1}/5 rep {repetition+1}/3; elapsed {time.monotonic()-started:.1f}s',flush=True)
    runs=pd.DataFrame(rows)
    folds=runs.groupby(['method','fold'])[METRICS].mean()
    summary=folds.groupby('method').mean()
    summary['macro_f1_fold_sd']=folds.groupby('method').macro_f1.std()
    summary['delta_vs_rf']=summary.macro_f1-summary.loc['original_blend','macro_f1']
    summary['delta_vs_current_hybrid']=summary.macro_f1-summary.loc['rf_fine_tuned_blend','macro_f1']
    runs.to_csv(OUT/'run_metrics.csv',index=False)
    folds.to_csv(OUT/'fold_metrics.csv')
    summary.to_csv(OUT/'summary.csv')
    summary.to_csv(ROOT/'docs/results/tabpfn_scores.csv')
    pd.concat(parts,ignore_index=True).to_parquet(OUT/'predictions.parquet',index=False)
    pd.DataFrame(class_rows).to_csv(OUT/'class_metrics.csv',index=False)
    pd.DataFrame(audits).to_csv(OUT/'split_audit.csv',index=False)
    pd.DataFrame(timings).to_csv(OUT/'timings.csv',index=False)
    members.loc[members.samples_per_class.eq(25)].to_csv(OUT/'adaptation_membership.csv',index=False)
    paired=folds.macro_f1.unstack('method')
    paired['tabpfn_blend_minus_current']=paired.rf_tabpfn_v2_blend-paired.rf_fine_tuned_blend
    paired.to_csv(OUT/'paired_fold_scores.csv')
    manifest['status']='complete'
    manifest['elapsed_seconds_this_invocation']=time.monotonic()-started
    manifest['total_context_fit_predict_seconds']=float(sum(t['fit_seconds']+t['predict_seconds'] for t in timings))
    manifest['peak_process_rss_mib']=max(t['peak_process_rss_mib'] for t in timings)
    manifest['outputs']={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*')
        if p.is_file() and not any(part in ['cache','models'] for part in p.relative_to(OUT).parts)
        and p.name!='experiment.json'}
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    print(summary.to_string(),flush=True)
    return summary


def recompute_probability_metrics():
    """Resolve native/mixed-dtype rounding without redoing any predictions.

    Original prediction code is retained and its complete inference procedure
    must be unchanged. Primary scores must remain identical to the original run.
    """
    import ast
    import shutil
    manifest_path = OUT / 'experiment.json'
    manifest = json.loads(manifest_path.read_text())
    assert manifest['status'] == 'complete'
    archived = OUT / 'prediction_source.py'
    assert manifest['code_sha256'] == sha(archived)
    for name,digest in manifest['inputs'].items():
        assert sha(ROOT/name) == digest, f'Changed input: {name}'
    for name,digest in manifest['outputs'].items():
        assert sha(OUT/name) == digest, f'Changed output: {name}'
    original_source, current_source = archived.read_text(), Path(__file__).read_text()
    def definitions(text):
        return {node.name: ast.get_source_segment(text,node) for node in ast.parse(text).body
                if isinstance(node,ast.FunctionDef)}
    before,after = definitions(original_source),definitions(current_source)
    for name in ['no_network','checkpoint_path','make_model','model_probabilities','aligned','input_paths','run']:
        assert before[name] == after[name], f'Prediction procedure changed: {name}'
    frame = pd.read_parquet(OUT / 'predictions.parquet')
    old_runs = pd.read_csv(OUT / 'run_metrics.csv').set_index(['method','fold','repetition'])
    rows=[]
    for key,part in frame.groupby(['method','fold','repetition']):
        prob = checked_probabilities(part,part.y_true)
        result = score(part.y_true.to_numpy(),prob)
        for name in ['macro_f1','weighted_f1','accuracy','balanced_accuracy','severe_error_rate']:
            np.testing.assert_allclose(result[name],old_runs.loc[key,name],rtol=0,atol=1e-12)
        rows.append(dict(method=key[0],fold=key[1],repetition=key[2],**result))
    shutil.copyfile(OUT/'run_metrics.csv', OUT/'metrics_before_float64.csv')
    runs=pd.DataFrame(rows)
    folds=runs.groupby(['method','fold'])[METRICS].mean()
    summary=folds.groupby('method').mean()
    summary['macro_f1_fold_sd']=folds.groupby('method').macro_f1.std()
    summary['delta_vs_rf']=summary.macro_f1-summary.loc['original_blend','macro_f1']
    summary['delta_vs_current_hybrid']=summary.macro_f1-summary.loc['rf_fine_tuned_blend','macro_f1']
    runs.to_csv(OUT/'run_metrics.csv',index=False)
    runs.to_csv(OUT/'metrics_checkpoint.csv',index=False)
    folds.to_csv(OUT/'fold_metrics.csv')
    summary.to_csv(OUT/'summary.csv')
    summary.to_csv(ROOT/'docs/results/tabpfn_scores.csv')
    manifest['prediction_code_sha256']=manifest['code_sha256']
    manifest['prediction_code_file']='prediction_source.py'
    manifest['code_sha256']=sha(__file__)
    manifest['probability_metric_precision']='Float64 per-row normalization for log loss/Brier only; raw stored probabilities and all classification scores unchanged'
    manifest['outputs']={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*')
        if p.is_file() and not any(part in ['cache','models'] for part in p.relative_to(OUT).parts)
        and p.name!='experiment.json'}
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    print('Recomputed probability diagnostics in float64. All classification metrics and predictions are unchanged.')


def verify(reload=False):
    torch.set_num_threads(CONFIG['torch_threads'])
    checkpoint_path()
    manifest=json.loads((OUT/'experiment.json').read_text())
    assert manifest['status']=='complete' and manifest['code_sha256']==sha(__file__)
    for name,digest in manifest['inputs'].items():
        assert sha(ROOT/name)==digest,f'Changed input: {name}'
    for name,digest in manifest['outputs'].items():
        assert sha(OUT/name)==digest,f'Changed output: {name}'
    predictions=pd.read_parquet(OUT/'predictions.parquet')
    runs=pd.read_csv(OUT/'run_metrics.csv').set_index(['method','fold','repetition'])
    assert len(runs)==60 and runs.index.is_unique and set(runs.index.get_level_values(0))==set(METHODS)
    target=pd.read_parquet(ROOT/'data/preprocessed_age_labels/amsterdam_supervised_features.parquet')
    for key,frame in predictions.groupby(['method','fold','repetition']):
        valid=target.loc[target.fold.eq(key[1])].set_index('location_id')
        assert frame.location_id.is_unique and set(frame.location_id)==set(valid.index)
        np.testing.assert_array_equal(frame.y_true,valid.loc[frame.location_id,'age_class'])
        prob=checked_probabilities(frame,frame.y_true)
        for name,value in score(frame.y_true.to_numpy(),prob).items():
            np.testing.assert_allclose(value,runs.loc[key,name],atol=1e-12)
    folds=runs.reset_index().groupby(['method','fold'])[METRICS].mean()
    summary=pd.read_csv(OUT/'summary.csv',index_col='method')
    np.testing.assert_allclose(folds.groupby('method').mean(),summary[METRICS],atol=1e-12)
    np.testing.assert_allclose(folds.groupby('method').macro_f1.std(),summary.macro_f1_fold_sd,atol=1e-12)
    members=pd.read_csv(OUT/'adaptation_membership.csv')
    original=pd.read_csv(ROOT/'data/baseline/adaptation_membership.csv')
    for (fold,rep),sample in members.groupby(['fold','repetition']):
        assert len(sample)==100 and sample.location_id.is_unique
        ids=original.loc[original.fold.eq(fold)&original.repetition.eq(rep)&original.samples_per_class.eq(25),'location_id']
        assert set(sample.location_id)==set(ids)
        train=target.set_index('location_id').loc[ids]
        assert train.age_class.value_counts().reindex(CLASSES).eq(25).all()
        assert set(train.spatial_group).isdisjoint(target.loc[target.fold.eq(fold),'spatial_group'])
    for _,frame in predictions.groupby(['fold','repetition']):
        parts={m:s.set_index('location_id').sort_index() for m,s in frame.groupby('method')}
        cols=[f'p{c}' for c in CLASSES]
        np.testing.assert_allclose(parts['rf_tabpfn_v2_blend'][cols],.5*parts['tabpfn_v2'][cols]+.5*parts['original_blend'][cols],atol=1e-12)
    if reload:
        data=np.load(OUT/'example_context.npz',allow_pickle=False)
        expected=target.set_index('location_id').loc[data['location_ids']]
        np.testing.assert_allclose(expected[manifest['features']].to_numpy(dtype=np.float32),data['X_train'],equal_nan=True)
        np.testing.assert_array_equal(expected.age_class,data['y_train'])
        with no_network():
            model=make_model(CONFIG['seed_base'])
            model.fit(data['X_train'],data['y_train'])
            actual=model_probabilities(model,data['X_probe'])
        np.testing.assert_allclose(actual,data['p_probe'],atol=1e-6)
        print('Fresh-process checkpoint + labeled-context reconstruction matches 256 held-out probabilities at the fixed query-batch shape.',flush=True)
    print(f'Verified {len(runs)} run scores, {len(predictions):,} probability rows, 15 exact budgets, spatial exclusions and fixed blending.',flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--verify',action='store_true')
    parser.add_argument('--reload',action='store_true')
    parser.add_argument('--recompute-metrics',action='store_true')
    args=parser.parse_args()
    if args.recompute_metrics:
        recompute_probability_metrics()
    else:
        verify(reload=args.reload) if args.verify or args.reload else run(resume=args.resume)
