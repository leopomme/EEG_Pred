#!/usr/bin/env python3
"""Paired, fixed ERP comparison on alternate development groups or confirmation.

The alternate grouping and all model settings were committed before its results.
Confirmation is an explicit separate mode; it never runs by default.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eeg_comp.classical import ClassicalModel, RECIPES
from eeg_comp.data import load_cache
from eeg_comp.erp_template import ERPTemplateCovariance
from eeg_comp.validation import summarize
from scripts.run_classical import cache_provenance, feature_cache, sha256_file


def comparison_splits(meta, confirmation=False):
    visible = meta.split.eq('train').to_numpy()
    people = sorted(meta.loc[visible, 'subject'].unique())
    if len(people) != 17:
        raise ValueError('Expected the audited 17 source people')
    reserved = np.array_split(np.asarray(people), 6)[-1]
    development = np.asarray([person for person in people if person not in reserved])
    if confirmation:
        groups = [reserved]
    else:
        groups = np.array_split(np.random.default_rng(20261004).permutation(development), 5)
    for i, group in enumerate(groups):
        source = development if confirmation else np.setdiff1d(development, group)
        train = np.flatnonzero(visible & meta.subject.isin(source).to_numpy())
        query = np.flatnonzero(visible & meta.subject.isin(group).to_numpy())
        if set(meta.iloc[train].subject) & set(meta.iloc[query].subject):
            raise ValueError('Participant leakage')
        if not confirmation and set(meta.iloc[np.r_[train, query]].subject) & set(reserved):
            raise ValueError('Reserved people entered development')
        yield ('confirmation' if confirmation else f'alternate_group{i}'), train, query


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--confirmation', action='store_true',
                        help='Explicitly open S019/S020 once, with frozen models')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    arrays, meta = load_cache('artifacts', 'cross_subject')
    directory = Path('artifacts/cross_subject')
    manifest, digest = cache_provenance(directory, arrays)
    erp, signature = feature_cache(directory, 'erp_pre', arrays['X'],
                                   arrays['valid_mask'], arrays['phase_valid_mask'], digest)
    waveforms = np.asarray(erp).reshape(len(meta), 8, 50)
    del arrays
    args.output.mkdir(parents=True)
    config = {
        'competition': 'cross_subject', 'category': 'ML',
        'mode': 'one-time confirmation' if args.confirmation else 'alternate grouped development',
        'split_seed': 20261004, 'confirmation_people': ['S019', 'S020'],
        'recipes': {'linear_erp': {'name': 'erp_pre', 'C': .1},
                    'erp_template': {'C': .1, 'covariance_shrinkage': .1}},
        'source_cache': manifest, 'feature_signature': signature,
        'selection': 'Fixed comparison; no parameter search, blending or threshold tuning',
        'code_sha256': {str(p): sha256_file(p) for p in [Path(__file__),
            Path('eeg_comp/erp_template.py'), Path('eeg_comp/classical.py')]},
    }
    (args.output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    records = {name: [] for name in config['recipes']}
    splits = []
    started = time.monotonic()
    for name, train, query in comparison_splits(meta, args.confirmation):
        labels = meta.iloc[train].y.to_numpy()
        linear = ClassicalModel(RECIPES['erp_pre'], C=.1).fit(erp[train], labels)
        template = ERPTemplateCovariance().fit(waveforms[train], labels)
        for recipe, prob in [('linear_erp', linear.predict_proba(erp[query])[:, 1]),
                             ('erp_template', template.predict_proba(waveforms[query])[:, 1])]:
            frame = meta.iloc[query].copy()
            frame['fold'], frame['p_move'] = name, prob
            records[recipe].append(frame)
        splits.append({'fold': name, 'source_indices': train.tolist(), 'query_indices': query.tolist()})
        print(name, 'completed', flush=True)
    for recipe, frames in records.items():
        oof = pd.concat(frames).sort_values('epoch_index')
        expected = 220 if args.confirmation else 1575
        if len(oof) != expected or oof.epoch_index.duplicated().any():
            raise ValueError('Unexpected held-out coverage')
        oof.to_csv(args.output / f'{recipe}_oof.csv', index=False)
        metrics = summarize(oof, meta)
        metrics['elapsed_seconds'] = time.monotonic() - started
        (args.output / f'{recipe}_metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
        print(recipe, json.dumps({key: metrics[key] for key in
              ['accuracy', 'target_weighted_accuracy', 'log_loss', 'mean_domain_auc']}), flush=True)
    (args.output / 'splits.json').write_text(json.dumps(splits, indent=2) + '\n')


if __name__ == '__main__':
    main()
