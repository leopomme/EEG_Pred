#!/usr/bin/env python3
"""Evaluate a fixed ERP + filter-bank CSP feature classifier, without final fitting."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd

from eeg_comp.data import load_cache
from eeg_comp.joint_ml import JointERPCSP
from eeg_comp.validation import make_splits, summarize
from scripts.run_classical import cache_provenance, feature_cache, sha256_file


ROOT = Path(__file__).resolve().parents[1]
SEED = 20261003


def evaluate(meta, erp, covariances, competition, protocol, directory,
             source_manifest, source_digest, feature_signatures):
    directory.mkdir()
    model_directory = directory / 'models'
    model_directory.mkdir()
    if competition == 'cross_subject':
        folds, mode = set(range(5)), 'run3'
        people = sorted(meta.loc[meta.split.eq('train'), 'subject'].unique())
        excluded_people = set(np.array_split(np.asarray(people), 6)[-1])
    else:
        folds, mode, excluded_people = None, protocol, set()
    splits = list(make_splits(meta, competition, folds=folds, seed=SEED, mode=mode))
    records, split_records = [], []
    started = time.monotonic()
    for name, source, query in splits:
        source_people = set(meta.iloc[source].subject)
        query_people = set(meta.iloc[query].subject)
        if np.intersect1d(source, query).size:
            raise ValueError('Source/query rows overlap')
        if not meta.iloc[np.r_[source, query]].split.eq('train').all():
            raise ValueError('Evaluation must use allowed labeled training rows only')
        if competition == 'cross_subject':
            if source_people & query_people or (source_people | query_people) & excluded_people:
                raise ValueError('Cross participant or development boundary violated')
        elif len(source_people | query_people) != 1:
            raise ValueError('Within model crossed a participant boundary')
        model = JointERPCSP().fit(erp[source], covariances[source], meta.iloc[source].y.to_numpy())
        probability = model.predict_proba(erp[query], covariances[query])[:, 1]
        frame = meta.iloc[query].copy()
        frame['fold'], frame['p_move'] = name, probability
        frame['prediction'] = np.where(probability >= .5, 'move', 'rest')
        records.append(frame)
        split_record = {'fold': name, 'protocol': protocol,
                        'source_epoch_indices': source.tolist(), 'query_epoch_indices': query.tolist(),
                        'source_participants': sorted(source_people), 'query_participants': sorted(query_people),
                        'source_n': len(source), 'query_n': len(query)}
        checkpoint_path = model_directory / (name + '.joblib')
        joblib.dump({'model': model, 'competition': competition, 'protocol': protocol,
                     'split': split_record, 'model_config': model.config(),
                     'source_digest': source_digest, 'source_cache': source_manifest,
                     'feature_signatures': feature_signatures}, checkpoint_path)
        saved = joblib.load(checkpoint_path)
        if (saved['competition'] != competition or saved['protocol'] != protocol
                or saved['split'] != split_record or saved['source_digest'] != source_digest
                or saved['feature_signatures'] != feature_signatures):
            raise ValueError('Reloaded fold checkpoint contract differs from its source fitting')
        replay_probability = saved['model'].predict_proba(erp[query], covariances[query])[:, 1]
        np.testing.assert_allclose(replay_probability, probability, rtol=0, atol=1e-10)
        np.testing.assert_array_equal(replay_probability >= .5, probability >= .5)
        split_record['checkpoint'] = checkpoint_path.relative_to(directory).as_posix()
        split_record['checkpoint_sha256'] = sha256_file(checkpoint_path)
        split_record['checkpoint_replay'] = {
            'same_process': True, 'labels_identical': True, 'probability_atol': 1e-10,
            'max_abs_probability_difference': float(np.max(np.abs(replay_probability - probability)))}
        split_records.append(split_record)
        print(json.dumps({'competition': competition, 'protocol': protocol, 'fold': name,
                          'accuracy': float(np.mean((probability >= .5) == frame.y)),
                          'source_n': len(source), 'query_n': len(query)}), flush=True)
    oof = pd.concat(records).sort_values('epoch_index')
    expected = 1575 if competition == 'cross_subject' else (170 if protocol == 'run3' else 85)
    if len(oof) != expected or oof.epoch_index.duplicated().any():
        raise ValueError('Unexpected or duplicated outer query coverage')
    oof.to_csv(directory / 'oof.csv', index=False)
    (directory / 'splits.json').write_text(json.dumps(split_records, indent=2) + '\n')
    metrics = summarize(oof, meta, seed=SEED)
    metrics.update(protocol=protocol, elapsed_seconds=time.monotonic() - started,
                   checkpoint_replay_complete=True,
                   checkpoint_replay_max_probability_difference=max(
                       record['checkpoint_replay']['max_abs_probability_difference'] for record in split_records),
                   within_selection_note=('Aggregate Within results are descriptive; each participant'
                                          ' must select a final predictor using only their own data.'
                                          if competition == 'within_subject' else None))
    (directory / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    print(json.dumps({'competition': competition, 'protocol': protocol,
                      **{key: metrics[key] for key in ('accuracy', 'target_weighted_accuracy',
                                                        'log_loss', 'mean_domain_auc', 'elapsed_seconds')}}), flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--competition', required=True, choices=['cross_subject', 'within_subject'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a fresh experiment output directory')
    cache_directory = ROOT / 'artifacts' / args.competition
    arrays, meta = load_cache(cache_directory.parent, args.competition)
    source_manifest, source_digest = cache_provenance(cache_directory, arrays)
    feature_values, signatures = {}, {}
    for name in ('erp_pre', 'csp_full'):
        feature_values[name], signatures[name] = feature_cache(
            cache_directory, name, arrays['X'], arrays['valid_mask'], arrays['phase_valid_mask'], source_digest)
    del arrays
    if feature_values['erp_pre'].shape[1:] != (400,) or feature_values['csp_full'].shape[1:] != (6, 8, 8):
        raise ValueError('The prespecified ERP/CSP feature dimensions differ from the audited release')
    protocols = ['development'] if args.competition == 'cross_subject' else ['run3', 'forward']
    args.output.mkdir(parents=True)
    source_paths = ['eeg_comp/joint_ml.py', 'eeg_comp/classical.py', 'eeg_comp/data.py',
                    'eeg_comp/validation.py', 'scripts/run_joint_ml.py', 'scripts/run_classical.py']
    config = {'competition': args.competition, 'category': 'ML', 'model': JointERPCSP().config(),
              'created_utc': datetime.now(timezone.utc).isoformat(), 'seed': SEED,
              'protocols': protocols, 'feature_dimensions': {'erp_pre': 400, 'csp_log_variance': 24},
              'feature_signatures': signatures, 'source_digest': source_digest, 'source_cache': source_manifest,
              'code_sha256': {name: sha256_file(ROOT / name) for name in source_paths},
              'threshold': .5, 'training_weights': 'empirical, all source rows weight1',
              'selection': 'one fixed joint feature classifier; no hyperparameter/feature-weight search',
              'confirmation': 'Cross group5 excluded from all fitting and queries; no confirmation scoring',
              'final_test_fitting': False,
              'versions': {'numpy': np.__version__, 'pandas': pd.__version__},
              'within_policy': 'Each fold fits one participant only; aggregate scores cannot select another person\'s model.'}
    (args.output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    summaries = {}
    for protocol in protocols:
        metrics = evaluate(meta, feature_values['erp_pre'], feature_values['csp_full'], args.competition,
                           protocol, args.output / protocol, source_manifest, source_digest, signatures)
        summaries[protocol] = {key: metrics[key] for key in ('accuracy', 'target_weighted_accuracy',
                                                          'log_loss', 'mean_domain_auc', 'elapsed_seconds')}
    (args.output / 'summary.json').write_text(json.dumps(summaries, indent=2) + '\n')


if __name__ == '__main__':
    main()
