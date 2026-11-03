"""Few-shot adapters for a frozen Madrid forest, with spatial inner selection.

Only the supplied labeled adaptation sample is used by fit_adaptation. The outer
validation data is deliberately absent from its signature. Artifacts contain
ordinary sklearn pipelines and metadata, so no custom estimator is pickled.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from threadpoolctl import threadpool_limits

CLASSES = np.array([1, 2, 3, 4])
C_VALUES = (0.1, 1.0, 10.0)
# The ordering is the tie-breaker: prefer the original baseline on tied scores.
RECIPES = [
    {'name': 'fixed_rf_blend', 'c': None, 'source': .5, 'target': .5, 'leaf': 0.},
    {'name': 'target_only', 'c': None, 'source': 0., 'target': 1., 'leaf': 0.},
]
for c in C_VALUES:
    RECIPES.extend([
        {'name': f'leaf_c{c:g}', 'c': c, 'source': 0., 'target': 0., 'leaf': 1.},
        {'name': f'leaf_source_c{c:g}', 'c': c, 'source': .25, 'target': 0., 'leaf': .75},
        {'name': f'leaf_rf_source_c{c:g}', 'c': c, 'source': .25, 'target': .25, 'leaf': .5},
    ])


def probabilities(model, x):
    """Align class columns; incomplete training classes are forbidden."""
    if set(model.classes_) != set(CLASSES):
        raise ValueError('Adapter training requires all four classes.')
    result = model.predict_proba(x)[:, [list(model.classes_).index(c) for c in CLASSES]]
    if not np.isfinite(result).all() or not np.allclose(result.sum(axis=1), 1):
        raise ValueError('Invalid class probabilities.')
    return result


def source_representation(source_model, raw_features):
    """Apply frozen source transformations; fit nothing on target locations."""
    imputed = source_model.named_steps['imputer'].transform(raw_features)
    leaves = source_model.named_steps['classifier'].apply(imputed).astype(np.int32)
    return leaves, probabilities(source_model, raw_features)


def target_forest(seed, rf_parameters):
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
        ('classifier', RandomForestClassifier(random_state=seed, **rf_parameters)),
    ])


def fit_leaf_adapters(leaves, labels, c_values=C_VALUES):
    """Fit leaf vocabulary and regularized heads using training locations only.

    Unknown leaves in a validation pixel are ignored. A pixel is never used to
    extend the vocabulary outside the supplied adaptation-training sample.
    """
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True, dtype=np.float64)
    encoded = encoder.fit_transform(leaves)
    adapters = {}
    with threadpool_limits(limits=1):
        for c in c_values:
            head = LogisticRegression(C=c, max_iter=400, tol=1e-4,
                                      solver='lbfgs', class_weight='balanced')
            head.fit(encoded, labels)
            if int(head.n_iter_.max()) >= head.max_iter:
                raise RuntimeError('Leaf logistic adapter did not converge.')
            adapters[c] = Pipeline([('onehot', encoder), ('classifier', head)])
    return adapters


def blend(recipe, source, target, leaf=None):
    result = recipe['source'] * source + recipe['target'] * target
    if recipe['leaf']:
        if leaf is None:
            raise ValueError('This recipe requires leaf probabilities.')
        result = result + recipe['leaf'] * leaf
    if not np.allclose(result.sum(axis=1), 1):
        raise ValueError('Blend weights must sum to one.')
    return result


def inner_splits(labels, groups, seed):
    """Fall back to two groups if a three-way split loses a training class."""
    for n in (3, 2):
        if len(np.unique(groups)) < n:
            continue
        splitter = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed)
        splits = list(splitter.split(np.zeros((len(labels), 1)), labels, groups))
        if all(set(labels[train]) == set(CLASSES) for train, _ in splits):
            for train, valid in splits:
                assert set(groups[train]).isdisjoint(groups[valid])
            return splits
    return []


def fit_adaptation(sample, features, leaves, source_probabilities, rf_parameters,
                   seed, tie_tolerance=.005):
    """Select an adapter without access to outer evaluation rows or labels.

    Parameters are arrays of the labeled adaptation sample only, in the same
    row order. Return a fitted artifact and all inner-fold selection scores.
    """
    if len(sample) != len(leaves) or len(sample) != len(source_probabilities):
        raise ValueError('Misaligned adaptation representations.')
    if not sample.location_id.is_unique:
        raise ValueError('Adaptation locations must be unique.')
    labels = sample.age_class.to_numpy(dtype=int)
    groups = sample.spatial_group.to_numpy()
    if set(labels) != set(CLASSES):
        raise ValueError('All four classes are required.')
    splits = inner_splits(labels, groups, seed)
    records = []
    for split, (train, valid) in enumerate(splits):
        target = target_forest(seed, rf_parameters)
        target.fit(sample.iloc[train][features], labels[train])
        target_prob = probabilities(target, sample.iloc[valid][features])
        heads = fit_leaf_adapters(leaves[train], labels[train])
        with threadpool_limits(limits=1):
            leaf_probs = {c: probabilities(head, leaves[valid]) for c, head in heads.items()}
        for recipe in RECIPES:
            proba = blend(recipe, source_probabilities[valid], target_prob,
                          leaf_probs.get(recipe['c']))
            score = f1_score(labels[valid], CLASSES[proba.argmax(axis=1)],
                             labels=CLASSES, average='macro', zero_division=0)
            records.append({'candidate': recipe['name'], 'inner_fold': split,
                            'macro_f1': score, 'n_inner_train': len(train),
                            'n_inner_validation': len(valid)})
    scores = pd.DataFrame(records)
    selected = RECIPES[0]
    if len(scores):
        means = scores.groupby('candidate').macro_f1.mean()
        best = means.max()
        # Require a small minimum gain over an earlier/less flexible candidate.
        selected = next(r for r in RECIPES if means[r['name']] >= best - tie_tolerance)
    # Refit target components on the entire, still budget-limited sample.
    target = target_forest(seed, rf_parameters)
    target.fit(sample[features], labels)
    heads = fit_leaf_adapters(leaves, labels, (selected['c'],)) if selected['leaf'] else {}
    artifact = {
        'recipe': dict(selected), 'target_pipeline': target,
        'leaf_pipeline': heads.get(selected['c']), 'feature_names': list(features),
        'class_order': CLASSES.copy(), 'adaptation_location_ids': sample.location_id.tolist(),
        'seed': seed, 'inner_split_count': len(splits), 'tie_tolerance': tie_tolerance,
        'selection_used_outer_labels': False,
    }
    return artifact, scores


def predict_adaptation(artifact, source_model, feature_table):
    """Inference uses no target labels; only the stored ordered feature schema."""
    x = feature_table[artifact['feature_names']]
    leaves, source_prob = source_representation(source_model, x)
    target_prob = probabilities(artifact['target_pipeline'], x)
    with threadpool_limits(limits=1):
        leaf_prob = (probabilities(artifact['leaf_pipeline'], leaves)
                     if artifact['leaf_pipeline'] is not None else None)
    proba = blend(artifact['recipe'], source_prob, target_prob, leaf_prob)
    return CLASSES[proba.argmax(axis=1)], proba
