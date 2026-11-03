"""Inductive source-feature moment alignment using permitted target samples only."""
import numpy as np
from sklearn.preprocessing import StandardScaler
from scripts.transfer_adaptation import probabilities, CLASSES


def fit_source_reference(source_model, source_features, source_labels):
    """Match the uniform per-class sampling distribution of the few-shot task."""
    values = source_model.named_steps['imputer'].transform(source_features)
    labels = np.asarray(source_labels)
    weights = np.zeros(len(labels), dtype=float)
    for c in CLASSES:
        mask = labels == c
        weights[mask] = 1. / mask.sum()
    scaler = StandardScaler().fit(values, sample_weight=weights)
    return {'mean': scaler.mean_, 'variance': scaler.var_, 'n_source': len(labels)}


def fit_target_moments(source_model, adaptation_features):
    """Fit no statistics outside this budget-limited feature table."""
    values = source_model.named_steps['imputer'].transform(adaptation_features)
    scaler = StandardScaler().fit(values)
    return {'mean': scaler.mean_, 'variance': scaler.var_, 'n_target': len(values)}


def aligned_source_probabilities(source_model, features, reference, target, prior_count=0.):
    """Map target features to source units, with an optional fixed shrinkage prior.

    prior_count=0 is ordinary per-feature mean/std alignment. prior_count=40
    shrinks noisy target moments toward source moments, useful when n is small.
    A fixed scale floor guards against division by a tiny estimated variance.
    All parameters come from source training or adaptation training, never eval.
    """
    values = source_model.named_steps['imputer'].transform(features)
    strength = target['n_target'] / (target['n_target'] + prior_count)
    source_variance = np.maximum(reference['variance'], 1e-12)
    target_mean = strength * target['mean'] + (1 - strength) * reference['mean']
    target_variance = strength * target['variance'] + (1 - strength) * source_variance
    target_variance = np.maximum(target_variance, .01 * source_variance)
    aligned = (values - target_mean) * np.sqrt(source_variance / target_variance) + reference['mean']
    return probabilities(source_model.named_steps['classifier'], aligned)
