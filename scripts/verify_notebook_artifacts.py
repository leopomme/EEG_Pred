#!/usr/bin/env python3
"""Check completed notebook outputs against their manifest and original audit.

This verifies recorded input hashes; it does not reread the raw EEG recordings.
Run the notebook again to establish identity of newly changed raw inputs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eeg_comp.submission import build_submission


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify(result, competition):
    result = Path(result)
    manifest = json.loads((result / 'manifest.json').read_text())
    if manifest['competition'] != competition:
        raise ValueError('Wrong competition manifest')
    execution = json.loads((result / 'local_execution.json').read_text())
    if execution['returncode'] != 0:
        raise ValueError('Notebook did not complete successfully')
    notebook = Path(execution['notebook'])
    if sha256(notebook) != execution['notebook_sha256']:
        raise ValueError('Current notebook differs from the executed notebook')
    for name, expected in manifest['artifacts_sha256'].items():
        if sha256(result / name) != expected:
            raise ValueError('Output hash differs: ' + name)
    for name, expected in manifest['source_sha256'].items():
        source = result / 'embedded_source' / name
        if not source.exists():
            source = result / 'embedded_source' / 'eeg_comp' / name
        if sha256(source) != expected:
            raise ValueError('Embedded source hash differs: ' + name)
    audit = json.loads((ROOT / 'artifacts' / competition / 'audit.json').read_text())
    expected_inputs = {row['file']: row['sha256'] for row in audit['files']}
    recorded_inputs = {row['file']: row['sha256'] for row in manifest['raw_inputs']}
    if expected_inputs != recorded_inputs or len(recorded_inputs) != len(manifest['raw_inputs']):
        raise ValueError('Notebook recorded raw inputs differ from the original audit')
    metadata = pd.read_csv(result / 'metadata.csv')
    audited_meta = pd.read_csv(ROOT / 'artifacts' / competition / 'metadata.csv')
    pd.testing.assert_frame_equal(metadata, audited_meta)
    predictions = pd.read_csv(result / 'test_predictions.csv')
    expected_submission = build_submission(predictions, metadata, competition)
    pd.testing.assert_frame_equal(pd.read_csv(result / 'submission.csv'), expected_submission)
    fitted = []
    for fit in manifest['fits']:
        indices = np.asarray(fit['training_epoch_indices'], dtype=int)
        if len(np.unique(indices)) != len(indices):
            raise ValueError('Duplicate source rows in a final fit')
        rows = metadata.iloc[indices]
        own = (metadata.subject.eq(fit['participant']) if competition == 'within_subject'
               else pd.Series(True, index=metadata.index))
        expected = np.flatnonzero(metadata.split.eq('train') & own)
        if not np.array_equal(indices, expected):
            raise ValueError('Final fit violates its allowed training boundary')
        if sorted(rows.subject.unique()) != fit['training_participants']:
            raise ValueError('Reported source participants differ from fitted rows')
        fitted.extend(indices.tolist())
    if sorted(fitted) != np.flatnonzero(metadata.split.eq('train')).tolist():
        raise ValueError('Final fits omit or duplicate labeled source rows')
    report = {'verified': True, 'competition': competition, 'category': manifest['category'],
              'notebook_sha256': execution['notebook_sha256'],
              'manifest_sha256': sha256(result / 'manifest.json'),
              'recorded_input_hashes_match_audit': True, 'input_files': len(recorded_inputs),
              'metadata_exact': True, 'metadata_rows': len(metadata),
              'submission_rows': len(expected_submission), 'fit_count': len(manifest['fits']),
              'training_boundaries_verified': True, 'output_and_embedded_source_hashes_verified': True,
              'scope': 'Local execution and saved artifacts; does not establish Kaggle execution or cross-device identity'}
    (result / 'validation_summary.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--competition', required=True, choices=['within_subject', 'cross_subject'])
    parser.add_argument('--result', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.result, args.competition), indent=2))
