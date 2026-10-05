#!/usr/bin/env python3
"""Fit the frozen Cross ERP-template candidate from the audited cache; no scoring."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
from eeg_comp.data import load_cache
from eeg_comp.classical import RECIPES, extract
from eeg_comp.erp_template import ERPTemplateCovariance
from scripts.run_classical import cache_provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a fresh reference output directory')
    started = time.monotonic()
    arrays, meta = load_cache(ROOT/'artifacts', 'cross_subject')
    provenance, source_digest = cache_provenance(ROOT/'artifacts/cross_subject', arrays)
    X = arrays['X']
    # Independently extract directly from audited raw epochs, not a feature cache.
    features = np.concatenate([extract(X[i:i+128], RECIPES['erp_pre'])
                               for i in range(0, len(X), 128)]).reshape(len(meta), 8, 50)
    source = np.flatnonzero(meta.split.eq('train').to_numpy())
    query = np.flatnonzero(meta.split.eq('test').to_numpy())
    if (len(source) != 1795 or len(query) != 360 or meta.iloc[source].subject.nunique() != 17
            or set(meta.iloc[source].subject) & set(meta.iloc[query].subject)):
        raise ValueError('Invalid final source/target participant boundary')
    model = ERPTemplateCovariance(C=.1, shrink=.1).fit(features[source], meta.iloc[source].y.to_numpy())
    test = meta.iloc[query].copy()
    test['p_move'] = model.predict_proba(features[query])[:, 1]
    test['prediction'] = np.where(test.p_move >= .5, 'move', 'rest')
    test = test.sort_values('test_order')
    args.output.mkdir(parents=True)
    test.to_csv(args.output/'test_predictions.csv', index=False)
    model_path = args.output/'erp_template_model.joblib'
    joblib.dump({'model': model, 'competition': 'cross_subject', 'category': 'ML',
                 'training_epoch_indices': source, 'test_epoch_indices': query}, model_path)
    replayed = joblib.load(model_path)['model'].predict_proba(features[query])[:, 1]
    np.testing.assert_array_equal(replayed, test.set_index('epoch_index').loc[query, 'p_move'])
    record = {'competition': 'cross_subject', 'category': 'ML',
              'config': {'recipe': 'erp_pre', 'C': .1, 'shrink': .1, 'threshold': .5,
                         'weighting': 'empirical; no sample weights'},
              'source_cache': provenance, 'source_digest': source_digest,
              'training_subjects': sorted(meta.iloc[source].subject.unique()),
              'training_epoch_indices': source.tolist(), 'test_epoch_indices': query.tolist(),
              'source_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in [Path(__file__), ROOT/'eeg_comp/data.py',
                                             ROOT/'eeg_comp/classical.py', ROOT/'eeg_comp/erp_template.py']},
              'artifact_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in [model_path, args.output/'test_predictions.csv']},
              'saved_model_replay_exact': True, 'elapsed_seconds': time.monotonic()-started,
              'status': 'Full source refit reference; no validation scoring or upload performed'}
    (args.output/'manifest.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({'rows': len(test), 'elapsed_seconds': record['elapsed_seconds'],
                      'saved_model_replay_exact': True, 'output': str(args.output)}))


if __name__ == '__main__':
    main()
