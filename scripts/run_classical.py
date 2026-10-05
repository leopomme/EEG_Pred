#!/usr/bin/env python3
"""Run a prespecified classical comparison with isolated competition caches."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
from eeg_comp.classical import RECIPES, ClassicalModel, CUE, FS, extract
from eeg_comp.data import LABELS, load_cache
from eeg_comp.validation import make_splits, sample_weights, summarize


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def cache_provenance(cache_dir, arrays):
    """Fingerprint raw samples, masks and the validated interpretation of them."""
    contract = {'version': int(arrays['cache_version']), 'fs': int(arrays['fs']),
                'cue_index': int(arrays['cue_index']),
                'channels': arrays['channels'].tolist(),
                'epoch_shape': list(arrays['X'].shape[1:]),
                'epoch_dtype': str(arrays['X'].dtype), 'label_mapping': LABELS}
    manifest = {'metadata_sha256': sha256_file(cache_dir / 'metadata.csv'),
                'epochs_sha256': sha256_file(cache_dir / 'epochs.npz'),
                'contract': contract}
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return manifest, digest


def feature_cache(cache_dir, recipe_name, X, masks, phase_masks, source_digest):
    recipe = RECIPES[recipe_name]
    feature_dir = cache_dir / 'classical_features'
    feature_dir.mkdir(exist_ok=True)
    signature = hashlib.sha256((source_digest + json.dumps(asdict(recipe),sort_keys=True) +
                               sha256_file(PROJECT_ROOT / 'eeg_comp/classical.py')).encode()).hexdigest()
    path = feature_dir / (recipe_name + '_' + signature[:12] + '.npy')
    if path.exists():
        return np.load(path, mmap_mode='r'), signature
    windows = [recipe.window]
    if recipe.relative or recipe.kind in ('erp', 'paired'):
        windows.append((-2., 0.))
    for lo, hi in windows:
        a,b = CUE+round(lo*FS), CUE+round(hi*FS)
        if not (masks[:,a:b] & phase_masks[:,a:b]).all():
            raise ValueError(f'{recipe_name}: unsupported phase window {lo,hi}; inspect data audit')
    features = []
    for start in range(0, len(X), 128):
        features.append(extract(X[start:start+128], recipe))
    features = np.concatenate(features)
    if not np.isfinite(features).all():
        raise ValueError('Nonfinite extracted features')
    # Atomic cache replacement avoids partial reads if a job is interrupted.
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temp.open('wb') as f:
        np.save(f, features)
    temp.replace(path)
    return features, signature


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--competition', required=True, choices=['within_subject','cross_subject'])
    p.add_argument('--recipes', default='erp_early,erp_pre,erp_late,power_pre,power_late,power_full,relpower_full,tangent_full,csp_full,paired_cov_full,erp_baseline,power_baseline')
    p.add_argument('--folds', default=None)
    p.add_argument('--validation', choices=['run3','forward','calibration'], default='run3')
    p.add_argument('--weighting', choices=['empirical','target'], default='empirical')
    p.add_argument('--support-weight', type=float, default=1.)
    p.add_argument('--C', type=float, default=.1)
    p.add_argument('--seed', type=int, default=20261003)
    p.add_argument('--permutation', type=int, help='Permute source training labels within person/run for a null control')
    p.add_argument('--fit-final', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    recipes = args.recipes.split(',')
    if any(name not in RECIPES for name in recipes):
        p.error('Unknown feature recipe')
    if len(set(recipes)) != len(recipes):
        p.error('Duplicate recipes would overwrite experiment artifacts')
    if args.fit_final and args.permutation is not None:
        p.error('Cannot fit final predictions under permuted labels')
    if args.output.exists():
        raise FileExistsError('Experiment output already exists; choose a fresh directory')
    cache_dir = Path('artifacts') / args.competition
    arrays, meta = load_cache(cache_dir.parent, args.competition)
    X, y = arrays['X'], arrays['y']
    masks, phase_masks = arrays['valid_mask'], arrays['phase_valid_mask']
    manifest, digest = cache_provenance(cache_dir, arrays)
    folds = set(map(int,args.folds.split(','))) if args.folds else (set(range(5)) if args.competition == 'cross_subject' else None)
    splits = list(make_splits(meta,args.competition,folds,args.seed,args.validation))
    if not splits:
        raise ValueError('No validation splits selected')
    args.output.mkdir(parents=True, exist_ok=False)
    config = {**vars(args), 'output':str(args.output),
              'created_utc':datetime.now(timezone.utc).isoformat(),
              'metadata_sha256':manifest['metadata_sha256'],
              'source_digest':digest, 'cache_provenance':manifest,
              'seeds':{'split':args.seed, 'bootstrap':args.seed,
                       'permutation':args.permutation,
                       'model':{name:20261003 if getattr(RECIPES[name], 'classifier', 'logistic') == 'rbf' else None
                                for name in recipes}},
              'model_seed_note':'LogisticRegression lbfgs is deterministic; RBF SVC probability fitting uses seed 20261003.',
              'model_configs':{name:ClassicalModel(RECIPES[name],args.C).config() for name in recipes},
              'split_provenance':'splits.json',
              'category':'ML', 'information_budget':'inductive, trial-local features',
              'confirmation_policy':('Group5 excluded entirely from development folds0..4'
                                     if args.competition == 'cross_subject' else 'Each participant fitted separately'),
              'label_mapping':{'rest':0,'move':1},
              'code_sha256':{f:sha256_file(PROJECT_ROOT / f) for f in
                             ['eeg_comp/classical.py','eeg_comp/data.py','eeg_comp/validation.py','scripts/run_classical.py']}}
    (args.output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    split_records = [{'name':name, 'support_epoch_indices':support.tolist(),
                      'query_epoch_indices':query.tolist(),
                      'support_subjects':sorted(meta.iloc[support].subject.unique().tolist()),
                      'query_subjects':sorted(meta.iloc[query].subject.unique().tolist())}
                     for name,support,query in splits]
    (args.output/'splits.json').write_text(json.dumps(split_records,indent=2)+'\n')
    summary = {}
    for name in recipes:
        started = time.time()
        print('Features:',args.competition,name,flush=True)
        features, signature = feature_cache(cache_dir,name,X,masks,phase_masks,digest)
        records = []
        for split, support, query in splits:
            labels = y[support].copy()
            if args.permutation is not None:
                rng = np.random.default_rng(args.permutation)
                subset = meta.iloc[support].reset_index(drop=True)
                for _,g in subset.groupby(['subject','run']):
                    labels[g.index] = rng.permutation(labels[g.index])
            model = ClassicalModel(RECIPES[name],args.C)
            w = sample_weights(meta.iloc[support],args.competition,args.weighting,args.support_weight)
            model.fit(features[support],labels,w)
            prob = model.predict_proba(features[query])[:,1]
            result = meta.iloc[query].copy()
            result['fold'] = split
            result['p_move'] = prob
            records.append(result)
        oof = pd.concat(records,ignore_index=True)
        oof.to_csv(args.output/(name+'_oof.csv'),index=False)
        metrics = summarize(oof,meta,args.seed)
        metrics.update(feature_signature=signature,elapsed_seconds=time.time()-started)
        (args.output/(name+'_metrics.json')).write_text(json.dumps(metrics,indent=2)+'\n')
        summary[name] = {k:metrics[k] for k in ['accuracy','target_weighted_accuracy',
                                             'conditional_target_weighted_accuracy',
                                             'target_run_coverage','mean_domain_auc',
                                             'subject_bootstrap_95ci',
                                             'subject_conditional_target_bootstrap_95ci',
                                             'subject_macro_empirical_bootstrap_95ci','elapsed_seconds']}
        print(json.dumps({'recipe':name,**summary[name]}),flush=True)
        if args.fit_final:
            final_predict(args,meta,y,features,name,signature,digest)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


def final_predict(args,meta,y,features,name,feature_signature,source_digest):
    model_dir = args.output/(name+'_models')
    model_dir.mkdir(exist_ok=True)
    test = meta[meta.split.eq('test')].sort_values('test_order').copy()
    test['p_move'] = np.nan
    people = sorted(test.subject.unique()) if args.competition == 'within_subject' else [None]
    for person in people:
        allowed = np.ones(len(meta),dtype=bool) if person is None else meta.subject.eq(person).to_numpy()
        support = np.flatnonzero(allowed & meta.split.eq('train').to_numpy())
        query = np.flatnonzero(allowed & meta.split.eq('test').to_numpy())
        model = ClassicalModel(RECIPES[name],args.C)
        model.fit(features[support],y[support],sample_weights(meta.iloc[support],args.competition,args.weighting,args.support_weight))
        test.loc[query,'p_move'] = model.predict_proba(features[query])[:,1]
        joblib.dump({'model':model,'competition':args.competition,'train_indices':support,
                     'test_indices':query,'recipe':name,'model_config':model.config(),
                     'feature_signature':feature_signature,'source_digest':source_digest,
                     'experiment_config':'../config.json',
                     'model_seed':20261003 if getattr(RECIPES[name], 'classifier', 'logistic') == 'rbf' else None},
                    model_dir/(str(person or 'all_sources')+'.joblib'))
    if test.p_move.isna().any():
        raise ValueError('Missing test predictions')
    test['prediction'] = np.where(test.p_move >= .5,'move','rest')
    test.to_csv(args.output/(name+'_test_predictions.csv'),index=False)
    # ID origin must be verified against the official sample before making a CSV.
    print('Saved ordered test predictions; submission ID mapping is a separate validated step.',flush=True)


if __name__ == '__main__':
    main()
