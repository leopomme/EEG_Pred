"""Trial-level splits and reporting shared by the classical experiments.

No training information is exchanged across competitions or between people
in the within-subject task. Fixed thresholds are 0.5 throughout.
"""
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, log_loss
from sklearn.model_selection import StratifiedKFold


def make_splits(meta, competition, folds=None, seed=20261003, mode='run3'):
    train_mask = meta['split'].eq('train').to_numpy()
    subjects = sorted(meta.loc[train_mask, 'subject'].unique())
    if competition == 'cross_subject':
        groups = np.array_split(np.array(subjects), 6)
        confirmation = groups[-1]
        for fold, heldout in enumerate(groups):
            if folds is not None and fold not in folds:
                continue
            query = np.flatnonzero(train_mask & meta.subject.isin(heldout).to_numpy())
            excluded = heldout if fold == 5 else np.r_[heldout, confirmation]
            support = np.flatnonzero(train_mask & ~meta.subject.isin(excluded).to_numpy())
            assert not set(meta.iloc[support].subject) & set(meta.iloc[query].subject)
            yield f'group{fold}', support, query
    elif competition == 'within_subject':
        for person in subjects:
            own = train_mask & meta.subject.eq(person).to_numpy()
            calibration = np.flatnonzero(own & meta.run.ne(3).to_numpy())
            target = np.flatnonzero(own & meta.run.eq(3).to_numpy())
            if mode == 'calibration':
                support = np.flatnonzero(own & meta.run.eq(1).to_numpy())
                query = np.flatnonzero(own & meta.run.eq(2).to_numpy())
                if len(support) and len(query):
                    yield person + '_run1to2', support, query
                continue
            if mode == 'forward':
                cut = len(target) // 2
                yield person + '_forward', np.r_[calibration, target[:cut]], target[cut:]
                continue
            splitter = StratifiedKFold(2, shuffle=True, random_state=seed)
            for fold, (s, q) in enumerate(splitter.split(target, meta.iloc[target].y)):
                if folds is not None and fold not in folds:
                    continue
                support, query = np.r_[calibration, target[s]], target[q]
                assert set(meta.iloc[support].subject) == {person}
                yield f'{person}_fold{fold}', support, query
    else:
        raise ValueError(competition)


def sample_weights(meta, competition, weighting='empirical', support_weight=1.):
    weights = np.ones(len(meta), dtype=np.float64)
    if competition == 'cross_subject' and weighting == 'target':
        # Equal person/run strata approximate verified 1/3 run mixture; class
        # proportions are never imposed on a target file.
        strata = meta.subject.astype(str) + '/' + meta.run.astype(str)
        counts = strata.value_counts()
        weights = (1. / strata.map(counts)).to_numpy()
    if competition == 'within_subject':
        weights[meta.run.eq(3)] *= support_weight
    return weights / weights.mean()


def binary_metrics(y, p):
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    return {'n': len(y), 'accuracy': float(accuracy_score(y, p >= .5)),
            'balanced_accuracy': float(balanced_accuracy_score(y, p >= .5)),
            'auc': float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
            'log_loss': float(log_loss(y, np.c_[1-p, p], labels=[0, 1])),
            'brier': float(np.mean((p-y)**2))}


def summarize(oof, meta, seed=20261003):
    result = binary_metrics(oof.y, oof.p_move)
    result['per_run'] = {str(k): binary_metrics(g.y, g.p_move) for k, g in oof.groupby('run')}
    result['per_subject'] = {str(k): binary_metrics(g.y, g.p_move) for k, g in oof.groupby('subject')}
    test_weights = meta.loc[meta.split.eq('test'), 'run'].value_counts(normalize=True).to_dict()
    result['target_run_weights'] = test_weights
    result['target_weighted_accuracy'] = float(sum(test_weights.get(int(k), 0)*v['accuracy']
                                                   for k,v in result['per_run'].items()))
    # Within-domain ranking is distinguishable from cross-person score offsets.
    domain_metrics = {f'{s}/run{r}': binary_metrics(g.y, g.p_move)
                      for (s,r),g in oof.groupby(['subject','run'])}
    result['per_subject_run'] = domain_metrics
    aucs = [v['auc'] for v in domain_metrics.values() if v['auc'] is not None]
    result['mean_domain_auc'] = float(np.mean(aucs)) if aucs else None
    people = sorted(oof.subject.unique())
    values = []
    for person in people:
        p = oof[oof.subject.eq(person)]
        per_run = p.groupby('run').apply(lambda g: float(np.mean((g.p_move >= .5) == g.y)))
        available = sum(test_weights.get(int(r), 0) for r in per_run.index)
        values.append(sum(test_weights.get(int(r),0)*acc for r,acc in per_run.items())/available
                      if available else float(np.mean((p.p_move >= .5) == p.y)))
    rng = np.random.default_rng(seed)
    boot = np.mean(rng.choice(values, (2000, len(values)), replace=True), axis=1)
    result['subject_macro_target_accuracy'] = float(np.mean(values))
    result['subject_bootstrap_95ci'] = np.quantile(boot, [.025, .975]).tolist()
    result['uncertainty_note'] = 'Descriptive participant bootstrap; few people and shared training folds limit precision.'
    return result
