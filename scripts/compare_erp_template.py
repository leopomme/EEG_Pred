#!/usr/bin/env python3
"""Compare already-saved development OOF predictions without opening confirmation.

Reproduce from the repository root:
  myenv/bin/python scripts/compare_erp_template.py

The default output is results/cross_subject/erp_template/paired_comparison.json.
Only development OOF rows and development-indexed feature-cache rows are read.
No model fitting, raw metadata loading, or confirmation evaluation is performed.
"""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results/cross_subject'
FEATURES = ROOT / 'artifacts/cross_subject/classical_features'
RESERVED = {'S019', 'S020'}
SEED = 20261003


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_oof(path):
    frame = pd.read_csv(path).sort_values('epoch_index').reset_index(drop=True)
    if (len(frame) != 1575 or frame.epoch_index.duplicated().any()
            or set(frame.subject) & RESERVED or frame.subject.nunique() != 15
            or not frame.split.eq('train').all()
            or not np.isfinite(frame.p_move).all()
            or not frame.p_move.between(0, 1).all()):
        raise ValueError(f'Invalid development-only OOF data: {path}')
    return frame


def metrics(frame):
    probability = np.clip(frame.p_move.to_numpy(), 1e-15, 1-1e-15)
    y = frame.y.to_numpy()
    return {
        'n': len(frame),
        'accuracy': float(np.mean((probability >= .5) == y)),
        'log_loss': float(np.mean(-y*np.log(probability)-(1-y)*np.log1p(-probability))),
        'auc': float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else None,
    }


def comparison(candidate, baseline, draws):
    if not candidate.drop(columns='p_move').equals(baseline.drop(columns='p_move')):
        raise ValueError('Candidate and comparator do not have identical OOF metadata/folds')
    candidate_correct = ((candidate.p_move >= .5) == candidate.y).to_numpy()
    baseline_correct = ((baseline.p_move >= .5) == baseline.y).to_numpy()
    pairs = candidate[['subject', 'run', 'fold']].copy()
    pairs['candidate'] = candidate_correct.astype(float)
    pairs['baseline'] = baseline_correct.astype(float)
    pairs['delta'] = pairs.candidate-pairs.baseline
    domains = pairs.groupby(['subject', 'run'])[['candidate', 'baseline', 'delta']].mean()
    participant = domains.groupby('subject').mean()
    delta = participant.delta.to_numpy()
    rng = np.random.default_rng(SEED)
    bootstrap = rng.choice(delta, (draws, len(delta)), replace=True).mean(axis=1)
    per_run = {}
    for run in sorted(candidate.run.unique()):
        new = metrics(candidate[candidate.run.eq(run)])
        old = metrics(baseline[baseline.run.eq(run)])
        per_run[str(run)] = {
            'candidate': new, 'baseline': old,
            'accuracy_delta': new['accuracy']-old['accuracy'],
            'log_loss_delta': new['log_loss']-old['log_loss'],
        }
    # This exact sign diagnostic is not a formal confirmation test: folds share
    # fitted source people, and these data have already guided model development.
    signs = np.array([[(j >> i & 1)*2-1 for i in range(len(delta))]
                      for j in range(1 << len(delta))])
    sign_means = (signs*delta).mean(axis=1)
    per_person = {}
    for subject, row in participant.iterrows():
        per_person[subject] = {
            'available_runs': [int(run) for run in domains.loc[subject].index],
            'candidate_conditional_run_macro_accuracy': float(row.candidate),
            'baseline_conditional_run_macro_accuracy': float(row.baseline),
            'accuracy_delta': float(row.delta),
        }
    new_metrics, old_metrics = metrics(candidate), metrics(baseline)
    return {
        'candidate': new_metrics,
        'baseline': old_metrics,
        'pooled_accuracy_delta': new_metrics['accuracy']-old_metrics['accuracy'],
        'candidate_target_run_weighted_accuracy': float(np.mean([
            value['candidate']['accuracy'] for value in per_run.values()])),
        'baseline_target_run_weighted_accuracy': float(np.mean([
            value['baseline']['accuracy'] for value in per_run.values()])),
        'target_run_weighted_accuracy_delta': float(np.mean([
            value['accuracy_delta'] for value in per_run.values()])),
        'per_run': per_run,
        'per_participant': per_person,
        'participant_conditional_macro_accuracy_delta': float(delta.mean()),
        'participant_paired_bootstrap_95ci': np.quantile(bootstrap, [.025, .975]).tolist(),
        'participants_improved': int(np.sum(delta > 1e-12)),
        'participants_unchanged': int(np.sum(np.abs(delta) <= 1e-12)),
        'participants_worsened': int(np.sum(delta < -1e-12)),
        'per_fold_pooled_accuracy_delta': pairs.groupby('fold').delta.mean().to_dict(),
        'per_participant_run_accuracy_delta': {
            f'{subject}/run{run}': float(row.delta)
            for (subject, run), row in domains.iterrows()},
        'trials_changed_wrong_to_correct': int(np.sum(candidate_correct & ~baseline_correct)),
        'trials_changed_correct_to_wrong': int(np.sum(~candidate_correct & baseline_correct)),
        'participant_sign_randomization_two_sided_p_descriptive': float(
            np.mean(np.abs(sign_means) >= abs(delta.mean())-1e-14)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=RESULTS/'erp_template/paired_comparison.json')
    parser.add_argument('--bootstrap-draws', type=int, default=20000)
    args = parser.parse_args()
    if args.bootstrap_draws < 1000:
        parser.error('Use at least 1000 bootstrap draws')
    paths = {
        'template': RESULTS/'erp_template/oof.csv',
        'erp_pre': RESULTS/'ml_screen/erp_pre_oof.csv',
        'erp_pre_centered': RESULTS/'ml_cue/erp_pre_centered_oof.csv',
        'erp_pre_unitnorm': RESULTS/'ml_cue/erp_pre_unitnorm_oof.csv',
    }
    frames = {name: read_oof(path) for name, path in paths.items()}
    candidate = frames['template']
    allowed = dict(zip(candidate.epoch_index, candidate.subject))
    splits_path = RESULTS/'erp_template/splits.json'
    splits = json.loads(splits_path.read_text())
    for split in splits:
        source, query = set(split['source_indices']), set(split['query_indices'])
        if source & query or source | query != set(allowed):
            raise ValueError('Fold source/query coverage is not exactly the development set')
        if {allowed[i] for i in source} & {allowed[i] for i in query}:
            raise ValueError('Source/query participant leakage')
        if query != set(candidate.loc[candidate.fold.eq(split['fold']), 'epoch_index']):
            raise ValueError('Split manifest and OOF query rows disagree')
    old_signature = json.loads((RESULTS/'ml_screen/erp_pre_metrics.json').read_text())['feature_signature']
    new_signature = json.loads((RESULTS/'erp_template/config.json').read_text())['feature_signature']
    feature_paths = [FEATURES/f'erp_pre_{signature[:12]}.npy'
                     for signature in (old_signature, new_signature)]
    # mmap accesses only selected development rows; reserved rows stay unread.
    feature_arrays = [np.load(path, mmap_mode='r') for path in feature_paths]
    indices = candidate.epoch_index.to_numpy()
    features_identical = np.array_equal(feature_arrays[0][indices], feature_arrays[1][indices])
    if not features_identical:
        raise ValueError('Historical and current ERP inputs differ on development rows')
    result = {
        'generation': {
            'command': '/rds/general/user/lh5218/home/anaconda3/envs/myenv/bin/python scripts/compare_erp_template.py',
            'script_sha256': digest(Path(__file__)),
            'input_sha256': {str(path.relative_to(ROOT)): digest(path) for path in paths.values()},
            'splits_sha256': digest(splits_path),
            'bootstrap_seed': SEED, 'bootstrap_draws': args.bootstrap_draws,
        },
        'audit': {
            'n_development_queries': len(candidate),
            'development_subjects': sorted(candidate.subject.unique()),
            'same_oof_metadata_and_folds': True,
            'source_query_participants_disjoint': True,
            'confirmation_excluded_from_every_source_query_set': True,
            'historical_current_erp_features_bitwise_equal_on_development_rows': features_identical,
            'historical_current_feature_signatures': [old_signature, new_signature],
        },
        'estimand_notes': [
            'Target run weighted accuracy averages pooled run1/run2/run3 accuracy equally, matching the known Cross test run mixture.',
            'Participant conditional macro first averages each person over available runs, then averages 15 people; S007 lacks run2, so this is explicitly conditional.',
            'The paired bootstrap resamples the 15 participant accuracy differences, preserving each participant\'s observed runs.',
            'Confidence intervals and exact sign randomization are descriptive: shared training folds, few people, and prior development model comparisons preclude confirmatory interpretation.',
            'The ERP-centered comparator differs in clipping order as well as centering: it centers before clipping, whereas template covariance centers already clipped ERP bins.',
        ],
        'comparisons': {name: comparison(candidate, frame, args.bootstrap_draws)
                        for name, frame in frames.items() if name != 'template'},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(f'Wrote {args.output}')
    for name, value in result['comparisons'].items():
        print(name, 'target delta', value['target_run_weighted_accuracy_delta'],
              'conditional participant delta', value['participant_conditional_macro_accuracy_delta'],
              'paired CI', value['participant_paired_bootstrap_95ci'])


if __name__ == '__main__':
    main()
