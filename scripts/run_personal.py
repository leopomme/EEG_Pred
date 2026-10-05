#!/usr/bin/env python3
"""Evaluate fixed personal intercept adaptation and nested family selection."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold

from eeg_comp.classical import ClassicalModel, RECIPES
from eeg_comp.data import load_cache
from eeg_comp.personal import FAMILIES, OFFSET_STRENGTH, REGULARIZATION_C, PersonalSources, model_logits
from eeg_comp.validation import summarize
from scripts.run_classical import PROJECT_ROOT, cache_provenance, feature_cache, sha256_file


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261003)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a new personal experiment output directory')
    arrays, meta = load_cache('artifacts', 'within_subject')
    cache_dir = Path('artifacts/within_subject')
    manifest, digest = cache_provenance(cache_dir, arrays)
    features, signatures = {}, {}
    for family in FAMILIES:
        print('Features', family, flush=True)
        features[family], signatures[family] = feature_cache(
            cache_dir, family, arrays['X'], arrays['valid_mask'], arrays['phase_valid_mask'], digest)
    del arrays
    args.output.mkdir(parents=True, exist_ok=False)
    model_dir = args.output / 'models'
    model_dir.mkdir()
    code_files = ['eeg_comp/personal.py', 'scripts/run_personal.py', 'eeg_comp/classical.py',
                  'eeg_comp/data.py', 'eeg_comp/validation.py', 'scripts/run_classical.py']
    config = {
        'competition': 'within_subject', 'category': 'ML', 'args': {**vars(args), 'output': str(args.output)},
        'created_utc': datetime.now(timezone.utc).isoformat(), 'source_digest': digest,
        'cache_provenance': manifest, 'feature_signatures': signatures,
        'code_sha256': {name: sha256_file(PROJECT_ROOT / name) for name in code_files},
        'families': list(FAMILIES), 'C': REGULARIZATION_C, 'offset_strength': OFFSET_STRENGTH,
        'offset_objective': 'sum(logaddexp(0, z+b) - y*(z+b)) + strength*b*b/2',
        'selection_score': '0.5 * mean held calibration-run logloss + 0.5 * support LOO offset logloss',
        'tie_break': list(FAMILIES), 'split_seed': args.seed, 'bootstrap_seed': args.seed,
        'model_seed': None, 'model_seed_note': 'Deterministic lbfgs logistic models',
        'scope': 'Each fit, offset and family selection uses only this participant; source models use runs 1/2 only.',
        'information_budget': 'Inductive trial-local features, own labeled support; no target batch fitting',
        'selection_note': 'No aggregate score chooses a family, hyperparameter, prior or threshold for another participant.',
    }
    write_json(args.output / 'config.json', config)
    records = {name: [] for family in FAMILIES for name in [family + '_source', family + '_offset']}
    records.update(selector=[], selector_source=[])
    provenance, final_predictions = [], []
    people = sorted(meta.loc[meta.split.eq('train'), 'subject'].unique())
    for person in people:
        print('Person', person, flush=True)
        own = meta.subject.eq(person)
        target = np.flatnonzero((own & meta.split.eq('train') & meta.run.eq(3)).to_numpy())
        test = np.flatnonzero((own & meta.split.eq('test')).to_numpy())
        sources = PersonalSources(meta, features, person)
        person_record = {'subject': person, 'source_indices': sources.source_indices.tolist(),
                         'calibration_splits': [{'name': name, 'train_indices': train.tolist(), 'query_indices': query.tolist()}
                                                for name, train, query in sources.calibration_splits],
                         'calibration_fold_losses': sources.calibration_fold_losses, 'outer_folds': []}
        splitter = StratifiedKFold(2, shuffle=True, random_state=args.seed)
        for fold, (s, q) in enumerate(splitter.split(target, meta.iloc[target].y)):
            support, query = target[s], target[q]
            if set(support) & set(query) or set(sources.source_indices) & set(target):
                raise ValueError('Support/query/source overlap')
            selected, offsets, scores = sources.adapt(support)
            query_logits = sources.logits(query)
            row = meta.iloc[query].copy()
            row['fold'] = f'{person}_fold{fold}'
            for family in FAMILIES:
                for suffix, offset in [('source', 0.), ('offset', offsets[family])]:
                    result = row.copy()
                    result['family'], result['offset'] = family, offset
                    result['p_move'] = expit(query_logits[family] + offset)
                    records[family + '_' + suffix].append(result)
            for method, offset in [('selector', offsets[selected]), ('selector_source', 0.)]:
                result = row.copy()
                result['family'], result['offset'] = selected, offset
                result['p_move'] = expit(query_logits[selected] + offset)
                records[method].append(result)
            person_record['outer_folds'].append({'fold': fold, 'support_indices': support.tolist(),
                                                  'query_indices': query.tolist(), 'selected_family': selected,
                                                  'offsets': offsets, 'selection_scores': scores})
        # Final selection sees all own labeled run3, never its 40 test labels.
        selected, offsets, scores = sources.adapt(target)
        final_model = ClassicalModel(RECIPES[selected], REGULARIZATION_C).fit(
            features[selected][sources.source_indices], meta.iloc[sources.source_indices].y.to_numpy())
        predicted = meta.iloc[test].copy()
        predicted['p_move'] = expit(model_logits(final_model, features[selected][test]) + offsets[selected])
        predicted['prediction'] = np.where(predicted.p_move >= .5, 'move', 'rest')
        predicted['family'], predicted['offset'] = selected, offsets[selected]
        final_predictions.append(predicted)
        final_record = {'selected_family': selected, 'offset': offsets[selected],
                        'offset_support_indices': target.tolist(), 'test_indices': test.tolist(),
                        'selection_scores': scores, 'model_config': final_model.config()}
        person_record['final'] = final_record
        joblib.dump({'model': final_model, 'subject': person, 'competition': 'within_subject',
                     'source_indices': sources.source_indices, 'source_digest': digest,
                     'feature_signature': signatures[selected], 'experiment_config': '../config.json',
                     **final_record}, model_dir / (person + '.joblib'))
        provenance.append(person_record)
        # Persist completed people if a later scheduler interruption occurs.
        write_json(args.output / 'provenance.json', provenance)
    summary = {}
    tables = {}
    for name, parts in records.items():
        oof = pd.concat(parts, ignore_index=True).sort_values('epoch_index')
        if oof.epoch_index.duplicated().any():
            raise ValueError('Repeated out-of-fold trial')
        oof.to_csv(args.output / (name + '_oof.csv'), index=False)
        metrics = summarize(oof, meta, args.seed)
        metrics['selection_policy'] = config['selection_score'] if name.startswith('selector') else 'Fixed family; no family selection'
        write_json(args.output / (name + '_metrics.json'), metrics)
        summary[name] = {key: metrics[key] for key in ['accuracy', 'log_loss', 'mean_domain_auc', 'subject_bootstrap_95ci']}
        tables[name] = oof
    comparisons = {}
    for family in (*FAMILIES, 'selector'):
        source = tables[family + '_source']
        adapted = tables['selector' if family == 'selector' else family + '_offset']
        before = source.assign(correct=(source.p_move >= .5) == source.y).groupby('subject').correct.mean()
        after = adapted.assign(correct=(adapted.p_move >= .5) == adapted.y).groupby('subject').correct.mean()
        delta = (after - before).to_numpy()
        rng = np.random.default_rng(args.seed)
        boot = rng.choice(delta, (2000, len(delta)), replace=True).mean(axis=1)
        comparisons[family] = {'participant_mean_accuracy_change': float(delta.mean()),
                               'participant_bootstrap_95ci': np.quantile(boot, [.025, .975]).tolist(),
                               'per_subject_accuracy_change': (after - before).to_dict()}
    predictions = pd.concat(final_predictions, ignore_index=True).sort_values('test_order')
    if (not np.array_equal(predictions.test_order, np.arange(meta.split.eq('test').sum())) or
            not np.isfinite(predictions.p_move).all()):
        raise ValueError('Missing or unordered final predictions')
    predictions.to_csv(args.output / 'personal_selector_test_predictions.csv', index=False)
    write_json(args.output / 'summary.json', summary)
    write_json(args.output / 'paired_offset_comparisons.json', comparisons)
    write_json(args.output / 'selection_counts.json', {
        'outer_folds': dict(Counter(fold['selected_family'] for row in provenance for fold in row['outer_folds'])),
        'final_people': dict(Counter(row['final']['selected_family'] for row in provenance))})
    print(json.dumps({'summary': summary, 'test_predictions': len(predictions)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
