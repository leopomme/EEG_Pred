#!/usr/bin/env python3
"""Strict paired development comparison for saved ERP+CSP predictions.

This reads saved OOF predictions and provenance only: no fitting, confirmation
scoring, final test predictions, or leaderboard inference. Historical ml_screen
runs lack saved split manifests; their exact query assignments are checked and
their source splits are explicitly reconstructed from the frozen configuration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from eeg_comp.validation import binary_metrics, make_splits


ROOT = Path(__file__).resolve().parents[1]
SEED = 20261003
RECIPES = ('erp_pre', 'csp_full')
PREDICTION_COLUMNS = {'p_move', 'prediction'}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read_frame(path, competition, protocol):
    frame = pd.read_csv(path, keep_default_na=False).sort_values('epoch_index').reset_index(drop=True)
    required = {'epoch_index', 'competition', 'split', 'subject', 'run', 'trial_index',
                'file', 'test_order', 'label', 'y', 'fold', 'p_move'}
    if not required.issubset(frame.columns):
        raise ValueError(f'Incomplete OOF metadata: {path}')
    expected = 1575 if competition == 'cross_subject' else (170 if protocol == 'run3' else 85)
    if (len(frame) != expected or frame.epoch_index.duplicated().any()
            or not frame.competition.eq(competition).all() or not frame.split.eq('train').all()
            or not frame.y.isin([0, 1]).all() or not frame.test_order.eq(-1).all()
            or not np.isfinite(frame.p_move).all() or not frame.p_move.between(0, 1).all()):
        raise ValueError(f'Invalid OOF coverage, values or visibility: {path}')
    if not np.array_equal(frame.label.map({'rest': 0, 'move': 1}), frame.y):
        raise ValueError(f'OOF labels disagree with their binary mapping: {path}')
    if competition == 'cross_subject':
        if frame.subject.nunique() != 15 or set(frame.subject) & {'S019', 'S020'}:
            raise ValueError(f'Cross OOF includes confirmation or has wrong cohort: {path}')
    elif frame.subject.nunique() != 17 or not frame.run.eq(3).all():
        raise ValueError(f'Within OOF is not all17 own-person feedback queries: {path}')
    if 'prediction' in frame and not np.array_equal(
            frame.prediction, np.where(frame.p_move >= .5, 'move', 'rest')):
        raise ValueError(f'Saved labels disagree with thresholded probabilities: {path}')
    return frame


def assert_same_metadata(candidate, baseline):
    new_columns = set(candidate.columns) - PREDICTION_COLUMNS
    old_columns = set(baseline.columns) - PREDICTION_COLUMNS
    if new_columns != old_columns:
        raise ValueError('Candidate and baseline have different metadata columns')
    columns = sorted(new_columns)
    pd.testing.assert_frame_equal(candidate[columns], baseline[columns], check_dtype=False,
                                  check_exact=True, obj='Paired OOF metadata, labels and folds')


def validate_provenance(candidate_root, candidate, baseline_root, competition, protocol):
    config_path = candidate_root / 'config.json'
    baseline_config_path = baseline_root / 'config.json'
    config = json.loads(config_path.read_text())
    baseline_config = json.loads(baseline_config_path.read_text())
    expected_mode = 'run3' if protocol == 'development' else protocol
    if (config.get('competition') != competition or config.get('seed') != SEED
            or config.get('model', {}).get('C') != .1 or config.get('final_test_fitting') is not False
            or config.get('training_weights') != 'empirical, all source rows weight1'):
        raise ValueError('Joint candidate differs from the fixed experiment configuration')
    if (baseline_config.get('competition') != competition or baseline_config.get('validation') != expected_mode
            or baseline_config.get('seed') != SEED or baseline_config.get('C') != .1
            or baseline_config.get('weighting') != 'empirical'
            or baseline_config.get('support_weight') != 1. or baseline_config.get('permutation') is not None):
        raise ValueError('Baseline differs from its frozen competition/protocol/configuration')
    if competition == 'cross_subject':
        if set(map(int, baseline_config.get('folds', '').split(','))) != set(range(5)):
            raise ValueError('Frozen Cross baseline folds differ from development groups0..4')
    elif baseline_config.get('folds') is not None:
        raise ValueError('Frozen Within baseline did not evaluate all person-specific folds')
    metadata_path = ROOT / 'artifacts' / competition / 'metadata.csv'
    metadata_digest = digest(metadata_path)
    if (config.get('source_cache', {}).get('metadata_sha256') != metadata_digest
            or baseline_config.get('metadata_sha256') != metadata_digest):
        raise ValueError('Historical/current/candidate metadata hashes differ')
    epochs_path = metadata_path.with_name('epochs.npz')
    if config.get('source_cache', {}).get('epochs_sha256') != digest(epochs_path):
        raise ValueError('Candidate signal cache differs from the current recorded cache')
    # Cross splitting needs IDs/runs only. Do not read the reserved label columns.
    use_columns = (lambda name: name not in {'y', 'label'}) if competition == 'cross_subject' else None
    meta = pd.read_csv(metadata_path, keep_default_na=False, usecols=use_columns)
    if not np.array_equal(meta.epoch_index, np.arange(len(meta))) or not meta.competition.eq(competition).all():
        raise ValueError('Current metadata row/competition contract differs')
    expected_splits = list(make_splits(meta, competition,
                                      folds=set(range(5)) if competition == 'cross_subject' else None,
                                      seed=SEED, mode=expected_mode))
    expected_query = np.sort(np.concatenate([query for _, _, query in expected_splits]))
    if not np.array_equal(candidate.epoch_index, expected_query):
        raise ValueError('Candidate coverage differs from the exact prespecified queries')
    splits_path = candidate_root / protocol / 'splits.json'
    recorded = json.loads(splits_path.read_text())
    records = {record['fold']: record for record in recorded}
    if len(records) != len(recorded) or set(records) != {name for name, _, _ in expected_splits}:
        raise ValueError('Candidate split manifest is incomplete or duplicated')
    used_indices, max_replay_delta = set(), 0.
    for name, source, query in expected_splits:
        record = records[name]
        if (record['source_epoch_indices'] != source.tolist()
                or record['query_epoch_indices'] != query.tolist()
                or record['protocol'] != protocol):
            raise ValueError('Candidate split source/query rows differ from the fixed protocol')
        if not np.array_equal(candidate.loc[candidate.fold.eq(name), 'epoch_index'], np.sort(query)):
            raise ValueError('Candidate OOF fold assignments differ from the manifest')
        source_people = set(meta.iloc[source].subject)
        query_people = set(meta.iloc[query].subject)
        if (set(record['source_participants']) != source_people
                or set(record['query_participants']) != query_people
                or np.intersect1d(source, query).size):
            raise ValueError('Candidate split participant/row boundaries differ')
        if competition == 'cross_subject':
            if source_people & query_people or (source_people | query_people) & {'S019', 'S020'}:
                raise ValueError('Candidate development split uses confirmation or leaks participants')
        elif len(source_people | query_people) != 1:
            raise ValueError('Candidate Within split crosses participants')
        replay = record.get('checkpoint_replay', {})
        delta = replay.get('max_abs_probability_difference')
        if (replay.get('same_process') is not True or replay.get('labels_identical') is not True
                or replay.get('probability_atol') != 1e-10 or delta is None
                or not np.isfinite(delta) or not 0 <= delta <= 1e-10):
            raise ValueError('Candidate fold lacks a successful strict checkpoint replay')
        checkpoint = candidate_root / protocol / record['checkpoint']
        if digest(checkpoint) != record.get('checkpoint_sha256'):
            raise ValueError('Candidate checkpoint hash differs from its replayed artifact')
        max_replay_delta = max(max_replay_delta, float(delta))
        used_indices.update(source.tolist())
        used_indices.update(query.tolist())
    baseline_splits_path = baseline_root / 'splits.json'
    if baseline_splits_path.exists():
        old_records = json.loads(baseline_splits_path.read_text())
        old_by_name = {record['name']: record for record in old_records}
        if len(old_by_name) != len(old_records) or set(old_by_name) != set(records):
            raise ValueError('Historical baseline split manifest has different coverage')
        for name, source, query in expected_splits:
            if (old_by_name[name]['support_epoch_indices'] != source.tolist()
                    or old_by_name[name]['query_epoch_indices'] != query.tolist()):
                raise ValueError('Historical baseline source/query split differs')
        baseline_split_status = 'Historical saved source/query manifest checked exactly'
    else:
        baseline_split_status = ('Historical source manifest absent; source rows reconstructed from frozen'
                                 ' configuration and hashed metadata, query assignments checked exactly in OOF')
    feature_audit = {}
    selected = np.asarray(sorted(used_indices), dtype=int)
    feature_directory = metadata_path.parent / 'classical_features'
    for recipe in RECIPES:
        baseline_metrics_path = baseline_root / (recipe + '_metrics.json')
        old_signature = json.loads(baseline_metrics_path.read_text())['feature_signature']
        new_signature = config['feature_signatures'][recipe]
        old_path = feature_directory / f'{recipe}_{old_signature[:12]}.npy'
        new_path = feature_directory / f'{recipe}_{new_signature[:12]}.npy'
        old_array, new_array = (np.load(path, mmap_mode='r', allow_pickle=False) for path in (old_path, new_path))
        # Only the allowed source/query rows are read, including no Cross confirmation rows.
        if old_array.shape != new_array.shape or not np.array_equal(old_array[selected], new_array[selected]):
            raise ValueError(f'Historical/current {recipe} source/query features differ')
        feature_audit[recipe] = {'historical_signature': old_signature, 'candidate_signature': new_signature,
                                 'source_query_features_bitwise_equal': True, 'checked_rows': len(selected),
                                 'historical_file_sha256': digest(old_path), 'candidate_file_sha256': digest(new_path)}
    test_weights = meta.loc[meta.split.eq('test'), 'run'].value_counts(normalize=True).to_dict()
    return test_weights, {
        'same_oof_metadata_labels_folds': True, 'exact_query_coverage': True,
        'source_query_boundary_checked': True, 'baseline_split_status': baseline_split_status,
        'candidate_checkpoint_replay_max_probability_difference': max_replay_delta,
        'current_metadata_sha256': metadata_digest, 'candidate_splits_sha256': digest(splits_path),
        'candidate_config_sha256': digest(config_path), 'baseline_config_sha256': digest(baseline_config_path),
        'feature_audit': feature_audit,
        'historical_provenance_note': ('Legacy ml_screen lacks a historical signal-cache digest and source-split manifest.'
                                     ' Current feature equality and reconstructed splits do not restore those historical records.'
                                     if 'source_digest' not in baseline_config else None)}, meta


def compare(candidate, baseline, weights, draws):
    assert_same_metadata(candidate, baseline)
    new_correct = ((candidate.p_move >= .5) == candidate.y).to_numpy()
    old_correct = ((baseline.p_move >= .5) == baseline.y).to_numpy()
    pairs = candidate[['subject', 'run', 'fold']].copy()
    pairs['candidate'], pairs['baseline'] = new_correct.astype(float), old_correct.astype(float)
    pairs['delta'] = pairs.candidate - pairs.baseline
    domains = pairs.groupby(['subject', 'run'])[['candidate', 'baseline', 'delta']].mean()
    per_person, conditional_deltas = {}, []
    for subject in sorted(candidate.subject.unique()):
        domain = domains.loc[subject]
        coverage = sum(weights.get(int(run), 0) for run in domain.index)
        if coverage <= 0:
            raise ValueError('Participant has no overlap with the target run inventory')
        values = {name: float(sum(weights.get(int(run), 0) * domain.loc[run, name]
                                  for run in domain.index) / coverage)
                  for name in ('candidate', 'baseline', 'delta')}
        conditional_deltas.append(values['delta'])
        per_person[subject] = {'available_runs': [int(run) for run in domain.index],
                              'target_run_coverage': float(coverage),
                              'candidate_conditional_target_accuracy': values['candidate'],
                              'baseline_conditional_target_accuracy': values['baseline'],
                              'conditional_accuracy_delta': values['delta']}
    participant_delta = np.asarray(conditional_deltas)
    rng = np.random.default_rng(SEED)
    bootstrap = rng.choice(participant_delta, (draws, len(participant_delta)), replace=True).mean(axis=1)
    per_run = {}
    for run in sorted(candidate.run.unique()):
        new = binary_metrics(candidate.loc[candidate.run.eq(run), 'y'],
                             candidate.loc[candidate.run.eq(run), 'p_move'])
        old = binary_metrics(baseline.loc[baseline.run.eq(run), 'y'],
                             baseline.loc[baseline.run.eq(run), 'p_move'])
        per_run[str(run)] = {'candidate': new, 'baseline': old,
                             'accuracy_delta': new['accuracy'] - old['accuracy'],
                             'log_loss_delta': new['log_loss'] - old['log_loss']}
    missing_runs = sorted(set(weights) - set(candidate.run.unique()))
    coverage = float(sum(weight for run, weight in weights.items() if str(run) in per_run))
    new_target = sum(weight * per_run[str(run)]['candidate']['accuracy']
                     for run, weight in weights.items() if str(run) in per_run)
    old_target = sum(weight * per_run[str(run)]['baseline']['accuracy']
                     for run, weight in weights.items() if str(run) in per_run)
    new_metrics, old_metrics = binary_metrics(candidate.y, candidate.p_move), binary_metrics(baseline.y, baseline.p_move)
    return {'candidate': new_metrics, 'baseline': old_metrics,
            'pooled_accuracy_delta': new_metrics['accuracy'] - old_metrics['accuracy'],
            'log_loss_delta': new_metrics['log_loss'] - old_metrics['log_loss'],
            'target_run_weights': weights, 'missing_target_runs': missing_runs,
            'target_run_coverage': coverage,
            'candidate_target_run_weighted_accuracy': new_target if not missing_runs else None,
            'baseline_target_run_weighted_accuracy': old_target if not missing_runs else None,
            'target_run_weighted_accuracy_delta': new_target - old_target if not missing_runs else None,
            'candidate_conditional_target_run_weighted_accuracy': new_target / coverage if coverage else None,
            'baseline_conditional_target_run_weighted_accuracy': old_target / coverage if coverage else None,
            'participant_conditional_macro_accuracy_delta': float(participant_delta.mean()),
            'participant_conditional_paired_bootstrap_95ci': np.quantile(bootstrap, [.025, .975]).tolist(),
            'participant_bootstrap_estimand': 'Mean of participant-specific target-run-weighted accuracy deltas, conditional on each participant\'s available runs.',
            'participants_improved': int(np.sum(participant_delta > 1e-12)),
            'participants_unchanged': int(np.sum(np.abs(participant_delta) <= 1e-12)),
            'participants_worsened': int(np.sum(participant_delta < -1e-12)),
            'per_participant': per_person, 'per_run': per_run,
            'per_fold_pooled_accuracy_delta': pairs.groupby('fold').delta.mean().to_dict(),
            'per_participant_run_accuracy_delta': {f'{subject}/run{run}': float(row.delta)
                                                 for (subject, run), row in domains.iterrows()},
            'changed_wrong_to_correct': int(np.sum(new_correct & ~old_correct)),
            'changed_correct_to_wrong': int(np.sum(~new_correct & old_correct))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--competition', choices=['both', 'cross_subject', 'within_subject'], default='both')
    parser.add_argument('--cross-root', type=Path, default=ROOT / 'results/cross_subject/joint_ml_v2')
    parser.add_argument('--within-root', type=Path, default=ROOT / 'results/within_subject/joint_ml_v2')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--bootstrap-draws', type=int, default=20000)
    args = parser.parse_args()
    if args.bootstrap_draws < 1000:
        parser.error('Use at least1000 descriptive bootstrap draws')
    output = args.output or ROOT / 'results' / f'joint_ml_v2_{args.competition}_paired.json'
    if output.exists():
        raise FileExistsError('Choose a fresh comparison output file')
    specs = [('cross_subject', 'development', args.cross_root, ROOT / 'results/cross_subject/ml_screen'),
             ('within_subject', 'run3', args.within_root, ROOT / 'results/within_subject/ml_screen'),
             ('within_subject', 'forward', args.within_root, ROOT / 'results/within_subject/ml_forward')]
    results, input_hashes = {}, {}
    for competition, protocol, candidate_root, baseline_root in specs:
        if args.competition != 'both' and competition != args.competition:
            continue
        candidate_path = candidate_root / protocol / 'oof.csv'
        candidate = read_frame(candidate_path, competition, protocol)
        baselines = {recipe: read_frame(baseline_root / (recipe + '_oof.csv'), competition, protocol)
                     for recipe in RECIPES}
        for baseline in baselines.values():
            assert_same_metadata(candidate, baseline)
        weights, audit, _ = validate_provenance(candidate_root, candidate, baseline_root, competition, protocol)
        values = {recipe: compare(candidate, baseline, weights, args.bootstrap_draws)
                  for recipe, baseline in baselines.items()}
        key = f'{competition}/{protocol}'
        results[key] = {'audit': audit, 'comparisons': values}
        for path in [candidate_path] + [baseline_root / (recipe + '_oof.csv') for recipe in RECIPES]:
            input_hashes[str(path)] = digest(path)
        for recipe, value in values.items():
            print(json.dumps({'comparison': key, 'baseline': recipe,
                              'candidate_target_accuracy': value['candidate_target_run_weighted_accuracy'],
                              'baseline_target_accuracy': value['baseline_target_run_weighted_accuracy'],
                              'target_accuracy_delta': value['target_run_weighted_accuracy_delta'],
                              'participant_conditional_delta': value['participant_conditional_macro_accuracy_delta'],
                              'participant_descriptive_95ci': value['participant_conditional_paired_bootstrap_95ci'],
                              'log_loss_delta': value['log_loss_delta']}), flush=True)
    record = {'script_sha256': digest(Path(__file__)), 'input_sha256': input_hashes,
              'bootstrap_seed': SEED, 'bootstrap_draws': args.bootstrap_draws,
              'estimand_notes': [
                  'Primary target accuracy weights pooled per-run accuracies by the supplied test run inventory.',
                  'The participant conditional macro is a different estimand: each person is weighted equally after renormalizing target weights over their available runs; S007 lacks run2 in Cross.',
                  'The paired bootstrap resamples participant deltas and is descriptive because fitted training sets overlap, cohorts are small and development scores guide research.',
                  'Within aggregate results describe fixed independent models; they cannot select a participant\'s predictor using other participants\' outcomes.',
                  'Forward Within queries are the lastfive of the ten supplied run3 prefix trials; they do not reproduce the later40 hidden trials.',
                  'No confirmation scores, hidden labels or public/private leaderboard results enter this comparison.'],
              'results': results}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + '\n')
    print(f'Wrote {output}', flush=True)


if __name__ == '__main__':
    main()
