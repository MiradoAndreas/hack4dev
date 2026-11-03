"""Bounded raw-acquisition audit; never infer units or alter QA from values."""
import hashlib
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from scripts.improve_25 import ROOT, OUT, BANDS


def physical_indices(bands, *, scale=None, offset=None, confirmed=False):
    """NDVI after explicitly confirmed scale/offset, not guessed from ranges.

    Caller must supply coherent band observations and independently verified
    metadata. No missing values are filled and near-zero denominators stay NaN.
    Formula source: USGS Landsat Normalized Difference Vegetation Index.
    """
    if not confirmed or scale is None or offset is None:
        raise ValueError('Reflectance scale and offset require data-owner confirmation')
    nir = bands['NIR'] * scale + offset
    red = bands['Red'] * scale + offset
    den = nir + red
    return pd.DataFrame({'NDVI': (nir - red).divide(den.where(den.abs() > 1e-12))})


def audit():
    OUT.mkdir(exist_ok=True)
    quality, samples, composition = [], [], []
    rng = np.random.default_rng(42)
    for city, filename in [('Madrid', 'madrid_train.parquet'), ('Amsterdam', 'amsterdam_data.parquet')]:
        file = pq.ParquetFile(ROOT / 'data' / filename)
        cols = ['year'] + [f'{x}_{s}' for s in [1, 2, 3] for x in [*BANDS, 'qa_valid', 'QA_RADSAT', 'doy', 'dataset']]
        counts = np.zeros((3, 6), dtype=np.int64)
        for batch in file.iter_batches(batch_size=32_768, columns=cols):
            frame = batch.to_pandas()
            selected = np.zeros(len(frame), dtype=bool)
            in_period = frame.year.between(1984, 2024).to_numpy()
            for slot in [1, 2, 3]:
                qa = frame[f'qa_valid_{slot}'].fillna(False).to_numpy(dtype=bool)
                values = frame[[f'{b}_{slot}' for b in BANDS]].to_numpy(dtype=float)
                complete = np.isfinite(values).all(axis=1)
                chosen = ~selected & qa & complete & in_period
                counts[slot-1] += [len(frame), qa.sum(), (qa & complete).sum(), chosen.sum(),
                                  (qa & frame[f'QA_RADSAT_{slot}'].fillna(0).ne(0).to_numpy()).sum(),
                                  (chosen & ~frame[f'doy_{slot}'].between(1, 366).to_numpy()).sum()]
                selected |= chosen
                category = frame.loc[chosen, f'dataset_{slot}'].astype(str)
                vc = category.value_counts()
                for label, count in vc.items():
                    token = hashlib.sha256(label.encode()).hexdigest()[:12]
                    composition.append({'city': city, 'slot': slot, 'dataset_token': token, 'count': int(count)})
                # A reproducible ~1% sample across all batches, clearly not a full quantile audit.
                keep = chosen & (rng.random(len(frame)) < .01)
                sample = pd.DataFrame(values[keep], columns=BANDS)
                sample['city'], sample['slot'] = city, slot
                sample['epoch'] = (frame.loc[keep, 'year'].to_numpy() // 10) * 10
                samples.append(sample)
        for slot, row in enumerate(counts, 1):
            quality.append(dict(zip(['rows', 'qa_valid', 'qa_complete', 'selected', 'qa_valid_with_nonzero_radsat', 'selected_invalid_doy'], map(int, row)), city=city, slot=slot))
    pd.DataFrame(quality).to_csv(OUT / 'acquisition_quality.csv', index=False)
    pd.DataFrame(composition).groupby(['city', 'slot', 'dataset_token'], as_index=False)['count'].sum().to_csv(OUT / 'acquisition_composition.csv', index=False)
    sample = pd.concat(samples, ignore_index=True)
    summaries = []
    for (city, epoch), group in sample.groupby(['city', 'epoch']):
        for band in BANDS:
            q = group[band].quantile([.01, .25, .5, .75, .99])
            summaries.append({'city': city, 'epoch': int(epoch), 'band': band, 'sample_n': len(group),
                              **{f'q{int(k*100)}': v for k, v in q.items()}})
    pd.DataFrame(summaries).to_csv(OUT / 'acquisition_sample_quantiles.csv', index=False)
    (OUT / 'metadata_confirmation.json').write_text(json.dumps({
        'reflectance_units_confirmed': False, 'scale_offset_confirmed': False,
        'qa_bit_definitions_confirmed': False, 'building_footprints_available': False,
        'age_bin_definitions_confirmed': False, 'new_independent_labels_available': False,
        'quantile_scope': 'seeded 1% selected-slot sample across all raw batches; not full-data quantiles',
        'qa_scope': 'all raw rows; follows supplied qa_valid flags without reinterpreting bits',
        'action': 'Request data-owner documentation; do not automatically rescale, relabel or enable physical indices'
    }, indent=2) + '\n')
    print('Acquisition audit complete; physical units and label semantics remain unconfirmed.')


if __name__ == '__main__':
    audit()
