"""Small scientific invariants for leakage, ordinal validity and metadata gates."""
import numpy as np
import pandas as pd

from scripts.audit_acquisitions import physical_indices
from scripts.improve_25 import split_training, nearest, predict


def check():
    group = np.repeat(np.arange(25), 4)
    sample = pd.DataFrame({'age_class': np.tile([1, 2, 3, 4], 25), 'spatial_group': group,
                           'px_key': (group // 5) * 10., 'py_key': (group % 5) * 10.})
    # Distinct groups can be spatial neighbors; the buffer must handle this.
    sample.loc[group == 1, ['px_key', 'py_key']] = [.1, .1]
    splits = split_training(sample, 42, 2)
    for train, valid in splits:
        assert nearest(sample.iloc[train], sample.iloc[valid]).min() > 2
        assert set(sample.iloc[train].spatial_group).isdisjoint(sample.iloc[valid].spatial_group)
    assert len(np.concatenate([v for _, v in splits])) == len(sample)

    class FixedBinary:
        classes_ = np.array([0, 1])

        def __init__(self, p):
            self.p = p

        def predict_proba(self, x):
            return np.tile([1-self.p, self.p], (len(x), 1))

    artifact = {'method': 'ordinal_rf', 'features': ['x'], 'models': [FixedBinary(.8), FixedBinary(.2), FixedBinary(.7)]}
    result = predict(artifact, pd.DataFrame({'x': [0, 1]}), None)
    np.testing.assert_allclose(result, np.tile([.5, 0, .2, .3], (2, 1)))
    bands = pd.DataFrame({'NIR': [5., 2.], 'Red': [3., 2.]})
    try:
        physical_indices(bands)
    except ValueError:
        pass
    else:
        raise AssertionError('Unconfirmed physical units must be rejected')
    ndvi = physical_indices(bands, scale=.1, offset=-.2, confirmed=True).NDVI
    assert np.isclose(ndvi.iloc[0], .5) and np.isnan(ndvi.iloc[1])
    print('PASS: spatial neighbor exclusion, OOF coverage, monotone ordinal probabilities and physical-unit gate.')


if __name__ == '__main__':
    check()
