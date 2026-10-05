#!/usr/bin/env python3
"""Paired development-only comparisons; Within summaries cannot select others' models."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eeg_comp.validation import summarize


KEYS = ['epoch_index', 'competition', 'subject', 'run', 'file', 'trial_index', 'y']
MODES = ['phase_stable', 'phase_baseline', 'cue_baseline']
V2_SOURCES = ['eeg_comp/neural.py', 'eeg_comp/neural_v2.py', 'eeg_comp/data.py',
              'eeg_comp/validation.py', 'scripts/run_neural.py', 'scripts/run_neural_v2.py']
NATIVE_SOURCES = ['eeg_comp/neural.py', 'scripts/run_neural.py']


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read(path, competition, meta=None):
    frame = pd.read_csv(path).sort_values('epoch_index').reset_index(drop=True)
    n = 1575 if competition == 'cross_subject' else 170
    if (not set(KEYS + ['split', 'p_move']) <= set(frame.columns)
            or len(frame) != n or frame.epoch_index.duplicated().any()
            or not frame.competition.eq(competition).all()
            or not frame.split.eq('train').all()
            or not np.isfinite(frame.p_move).all() or not frame.p_move.between(0, 1).all()):
        raise ValueError(f'Incomplete or invalid development OOF: {path}')
    if competition == 'cross_subject' and set(frame.subject) & {'S019', 'S020'}:
        raise ValueError('Used confirmation must not enter development comparisons')
    if meta is not None:
        expected = meta.loc[meta.split.eq('train')]
        expected = (expected.loc[~expected.subject.isin(['S019', 'S020'])]
                    if competition == 'cross_subject' else expected.loc[expected.run.eq(3)])
        expected = expected.sort_values('epoch_index').reset_index(drop=True)
        pd.testing.assert_frame_equal(frame[KEYS], expected[KEYS])
    return frame


def require_hashes(recorded, actual, description):
    if not isinstance(recorded, dict) or any(recorded.get(name) != value for name, value in actual.items()):
        raise ValueError(f'{description} differs from current artifacts/source files')


def validate_v2(path, competition, mode, cache_hashes, source_hashes):
    config = json.loads((path.parent / 'config.json').read_text())
    result = json.loads((path.parent / 'metrics.json').read_text())
    arguments = config.get('arguments', {})
    model = config.get('model', {})
    task_samples = 500 if mode == 'cue_baseline' else 1200
    if (config.get('competition') != competition or arguments.get('competition') != competition
            or model.get('mode') != mode or arguments.get('mode') != mode
            or model.get('task_samples') != task_samples or model.get('baseline_samples') != 500
            or config.get('raw_crop_samples') != [250, 750 + task_samples]
            or config.get('category') != 'end-to-end DL' or config.get('external_data') is not False
            or config.get('threshold') != .5 or not result.get('complete_requested_validation')):
        raise ValueError(f'Experiment mode/configuration/completion mismatch: {path.parent}')
    require_hashes(config.get('cache_sha256'), cache_hashes, f'{mode} cache provenance')
    require_hashes(config.get('source_sha256'), source_hashes, f'{mode} source provenance')
    return config


def validate_native(path, competition, cache_hashes, source_hashes):
    config = json.loads((path.parent / 'config.json').read_text())
    result = json.loads((path.parent / 'metrics.json').read_text())
    arguments = config.get('arguments', {})
    cache = config.get('cache', {})
    if (arguments.get('competition') != competition or arguments.get('window') != 'task0_48'
            or cache.get('crop_samples') != [750, 1950]
            or config.get('category') != 'end-to-end DL' or config.get('inference_mode') != 'training'
            or arguments.get('protocol_balanced_loss') is not False
            or not result.get('complete_requested_validation')):
        raise ValueError(f'Matched native configuration/completion mismatch: {path.parent}')
    recorded = {name: cache.get(key) for name, key in [('epochs.npz', 'epochs_sha256'),
                ('metadata.csv', 'metadata_sha256'), ('audit.json', 'audit_sha256')]}
    require_hashes(recorded, cache_hashes, 'Matched native cache provenance')
    require_hashes(config.get('implementation_sha256'), source_hashes, 'Matched native source provenance')
    return config


def check_matched_policy(configs, native=None):
    """A matched reference must share the declared experimental conditions."""
    reference = configs['phase_stable']
    conditions = ['seed', 'max_epochs', 'patience', 'threads', 'device']
    for mode, config in configs.items():
        if any(config['arguments'].get(key) != reference['arguments'].get(key) for key in conditions):
            raise ValueError(f'{mode} changed a shared experimental condition')
        if (config.get('optimizer') != reference.get('optimizer')
                or config.get('torch_version') != reference.get('torch_version')
                or config.get('gpu') != reference.get('gpu')
                or config.get('runtime') != reference.get('runtime')):
            raise ValueError(f'{mode} differs in optimizer/device/backend policy')
    if native is None:
        return
    native_optimizer = {'name': 'AdamW', 'lr': native['arguments'].get('learning_rate'),
                        'weight_decay': native['arguments'].get('weight_decay'),
                        'batch_size': native['arguments'].get('batch_size')}
    if (any(native['arguments'].get(key) != reference['arguments'].get(key) for key in conditions)
            or native_optimizer != reference.get('optimizer')
            or native.get('model') != reference['model'].get('phase_config')
            or native.get('torch_version') != reference.get('torch_version')
            or native.get('gpu') != reference.get('gpu') or native.get('runtime') != reference.get('runtime')):
        raise ValueError('Native reference is not matched in model, optimizer, seed or device/backend policy')


def paired(candidate, baseline):
    pd.testing.assert_frame_equal(candidate[KEYS], baseline[KEYS])
    change = candidate[['subject', 'run']].copy()
    change['accuracy_delta'] = ((candidate.p_move >= .5) == candidate.y).astype(float) - (
        (baseline.p_move >= .5) == baseline.y).astype(float)
    per_domain = change.groupby(['subject', 'run']).accuracy_delta.mean()
    per_person = per_domain.groupby('subject').mean()
    values = per_person.to_numpy()
    rng = np.random.default_rng(20261005)
    boot = rng.choice(values, (20000, len(values)), replace=True).mean(axis=1)
    return {'pooled_accuracy_delta': float(change.accuracy_delta.mean()),
            'per_run_accuracy_delta': {str(k): float(v) for k, v in change.groupby('run').accuracy_delta.mean().items()},
            'participant_conditional_run_macro_delta': float(values.mean()),
            'paired_participant_bootstrap_95ci': np.quantile(boot, [.025, .975]).tolist(),
            'participants_improved': int(np.sum(values > 1e-12)),
            'participants_worsened': int(np.sum(values < -1e-12)),
            'participants_unchanged': int(np.sum(np.abs(values) <= 1e-12)),
            'per_participant_delta': per_person.to_dict(),
            'uncertainty': 'Descriptive participant bootstrap; shared source folds and model exploration not accounted for.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--competition', required=True, choices=['within_subject', 'cross_subject'])
    args = parser.parse_args()
    root = ROOT / 'results' / args.competition
    cache_root = ROOT / 'artifacts' / args.competition
    meta = pd.read_csv(cache_root / 'metadata.csv')
    cache_hashes = {name: digest(cache_root / name) for name in ['epochs.npz', 'metadata.csv', 'audit.json']}
    source_hashes = {name: digest(ROOT / name) for name in V2_SOURCES}
    baseline_path = root / ('dl_full_combined' if args.competition == 'cross_subject' else 'dl_pilot_full') / 'oof.csv'
    ml_path = root / 'ml_screen' / ('erp_pre_oof.csv' if args.competition == 'cross_subject' else 'csp_full_oof.csv')
    original, ml = read(baseline_path, args.competition, meta), read(ml_path, args.competition, meta)
    frames = {'historical_phase': original, 'classical_reference': ml}
    paths = {'historical_phase': baseline_path, 'classical_reference': ml_path}
    configs = {}
    for mode in MODES:
        p = root / ('v2_' + mode) / 'oof.csv'
        frames[mode], paths[mode] = read(p, args.competition, meta), p
        configs[mode] = validate_v2(p, args.competition, mode, cache_hashes, source_hashes)
    native_path = root / 'v2_native_reference' / 'oof.csv'
    native_config = None
    phase_reference = 'historical_phase'
    if native_path.exists():
        native_config = validate_native(native_path, args.competition, cache_hashes,
                                        {name: source_hashes[name] for name in NATIVE_SOURCES})
        frames['matched_native_phase'], paths['matched_native_phase'] = read(native_path, args.competition, meta), native_path
        phase_reference = 'matched_native_phase'
    check_matched_policy(configs, native_config)
    report = {'competition': args.competition, 'split': 'development only',
              'phase_reference': phase_reference,
              'phase_reference_note': ('Primary phase comparisons use the native rerun on the matched '
                  'device/backend and declared training policy; historical results remain descriptive.'
                  if native_config is not None else 'Matched native rerun is absent; primary phase '
                  'comparisons use the historical reference with unverified historical raw/source identity.'),
              'within_policy': 'Aggregate results are descriptive; no pooled participant selection or tuning',
              'input_sha256': {name: digest(p) for name, p in paths.items()},
              'current_cache_sha256': cache_hashes,
              'verified_source_sha256': source_hashes,
              'configuration_sha256': {name: digest(paths[name].parent / 'config.json')
                                      for name in MODES + (['matched_native_phase'] if native_config is not None else [])},
              'historical_provenance_limit': ('Historical phase raw-cache and implementation hashes were '
                  'not recorded in the original runs. Current canonical metadata and paired query identities '
                  'are verified, but historical raw/source snapshot identity cannot be established. '
                  'Current files cannot retrospectively establish those missing hashes.'),
              'metrics': {name: summarize(frame, meta) for name, frame in frames.items()},
              'paired': {name: {'versus_phase_reference': paired(frames[name], frames[phase_reference]),
                                'versus_historical_phase': paired(frames[name], original),
                                'versus_classical_reference': paired(frames[name], ml)} for name in MODES}}
    if native_config is not None:
        report['matched_native_versus_historical'] = paired(frames['matched_native_phase'], original)
        for name in MODES:
            report['paired'][name]['versus_matched_native_phase'] = paired(frames[name], frames['matched_native_phase'])
    report['paired']['phase_baseline']['versus_stable_phase'] = paired(frames['phase_baseline'], frames['phase_stable'])
    output = root / 'v2_comparison.json'
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({name: {'accuracy': m['accuracy'], 'target_weighted_accuracy': m['target_weighted_accuracy'],
                            'mean_domain_auc': m['mean_domain_auc'], 'log_loss': m['log_loss']}
                      for name, m in report['metrics'].items()}, indent=2))


if __name__ == '__main__':
    main()
