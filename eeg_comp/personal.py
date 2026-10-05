"""Fixed, participant-only source models and shrunk run-3 intercept adaptation."""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit

from eeg_comp.classical import ClassicalModel, RECIPES


FAMILIES = ('erp_pre', 'power_full', 'tangent_full', 'csp_full')
REGULARIZATION_C = .1
OFFSET_STRENGTH = 2.
CALIBRATION_SCORE_WEIGHT = .5


def log_loss_from_logits(logits, labels):
    logits, labels = np.asarray(logits, dtype=float), np.asarray(labels)
    if logits.ndim != 1 or labels.shape != logits.shape or not len(labels):
        raise ValueError('Expected nonempty aligned vectors')
    if not np.isfinite(logits).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError('Expected finite logits and binary labels')
    return float(np.mean(np.logaddexp(0., logits) - labels * logits))


def fit_offset(logits, labels, strength=OFFSET_STRENGTH):
    """Minimize summed binary loss + strength * offset**2 / 2.

    The positive prior gives a unique finite solution even with one class.
    Empty support returns the prior mode and is useful for size-one LOO.
    """
    logits, labels = np.asarray(logits, dtype=float), np.asarray(labels)
    if (logits.ndim != 1 or labels.shape != logits.shape or
            not np.isfinite(strength) or strength <= 0):
        raise ValueError('Invalid offset inputs')
    if not np.isfinite(logits).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError('Expected finite logits and binary support labels')
    if not len(labels):
        return 0.
    bound = max(1., len(labels) / strength)
    gradient = lambda value: float(np.sum(expit(logits + value) - labels) + strength * value)
    return float(brentq(gradient, -bound, bound, xtol=1e-12))


def leave_one_out_offset(logits, labels):
    """Each support trial is predicted using an offset fitted to other labels."""
    logits, labels = np.asarray(logits, dtype=float), np.asarray(labels)
    log_loss_from_logits(logits, labels)  # Validate before indexing.
    offsets = np.array([fit_offset(np.delete(logits, i), np.delete(labels, i))
                        for i in range(len(labels))])
    return logits + offsets, offsets


def select_family(calibration_losses, support_logits, support_labels):
    """A fixed four-family decision, using only this participant's support."""
    scores = {}
    for family in FAMILIES:
        loo_logits, offsets = leave_one_out_offset(support_logits[family], support_labels)
        support_loss = log_loss_from_logits(loo_logits, support_labels)
        calibration_loss = float(calibration_losses[family])
        if not np.isfinite(calibration_loss):
            raise ValueError('Nonfinite held-calibration score')
        scores[family] = {
            'calibration_log_loss': calibration_loss,
            'support_loo_log_loss': support_loss,
            'selection_score': CALIBRATION_SCORE_WEIGHT * calibration_loss +
                               (1 - CALIBRATION_SCORE_WEIGHT) * support_loss,
            'support_loo_offsets': offsets.tolist(),
        }
    # Python min preserves the declared family order on exact ties.
    selected = min(FAMILIES, key=lambda name: scores[name]['selection_score'])
    return selected, scores


def calibration_splits(meta, source_indices):
    """Own run1->run2 and run2->run1; missing runs use contiguous half blocks."""
    source_indices = np.asarray(source_indices, dtype=int)
    source = meta.iloc[source_indices]
    if (not len(source) or source.subject.nunique() != 1 or
            not source.split.eq('train').all() or not source.run.isin([1, 2]).all()):
        raise ValueError('Calibration must contain only one participant\'s labeled runs 1/2')
    first = source_indices[source.run.eq(1).to_numpy()]
    second = source_indices[source.run.eq(2).to_numpy()]
    if len(first) and len(second):
        pairs = [('run1_to_run2', first, second), ('run2_to_run1', second, first)]
    else:
        source_indices = np.sort(source_indices)
        middle = len(source_indices) // 2
        first, second = source_indices[:middle], source_indices[middle:]
        pairs = [('block_first_to_second', first, second), ('block_second_to_first', second, first)]
    for name, train, query in pairs:
        if not len(query) or len(np.unique(meta.iloc[train].y)) != 2:
            raise ValueError(f'Both classes required in calibration training block {name}')
        yield name, train, query


def model_logits(model, features):
    """Exact binary logistic scores, avoiding numerical probability clipping."""
    return model.classifier_.decision_function(model.scaler_.transform(model._represent(features)))


class PersonalSources:
    """One participant's four source-only models and source validation scores."""
    def __init__(self, meta, features, subject):
        self.meta, self.features, self.subject = meta, features, subject
        if 'competition' in meta and not meta.competition.eq('within_subject').all():
            raise ValueError('Personal sources require the isolated within-subject cache')
        if any(len(features[family]) != len(meta) for family in FAMILIES):
            raise ValueError('Features must align with metadata rows')
        self.source_indices = np.flatnonzero((meta.subject.eq(subject) &
                                               meta.split.eq('train') & meta.run.isin([1, 2])).to_numpy())
        self.calibration_splits = list(calibration_splits(meta, self.source_indices))
        self.models, self.calibration_losses, self.calibration_fold_losses = {}, {}, {}
        for family in FAMILIES:
            fold_losses = []
            for name, train, query in self.calibration_splits:
                model = ClassicalModel(RECIPES[family], REGULARIZATION_C)
                model.fit(features[family][train], meta.iloc[train].y.to_numpy())
                fold_losses.append({'name': name, 'n': len(query),
                                    'log_loss': log_loss_from_logits(
                                        model_logits(model, features[family][query]), meta.iloc[query].y.to_numpy())})
            # Equal held-run/block weights: do not let an incomplete run change
            # the prespecified 1->2 / 2->1 contrast's importance.
            self.calibration_losses[family] = float(np.mean([row['log_loss'] for row in fold_losses]))
            self.calibration_fold_losses[family] = fold_losses
            self.models[family] = ClassicalModel(RECIPES[family], REGULARIZATION_C).fit(
                features[family][self.source_indices], meta.iloc[self.source_indices].y.to_numpy())

    def logits(self, indices):
        indices = np.asarray(indices, dtype=int)
        if not self.meta.iloc[indices].subject.eq(self.subject).all():
            raise ValueError('Cross-participant prediction request')
        return {family: model_logits(self.models[family], self.features[family][indices])
                for family in FAMILIES}

    def adapt(self, support_indices):
        support_indices = np.asarray(support_indices, dtype=int)
        if len(np.unique(support_indices)) != len(support_indices):
            raise ValueError('Repeated support would invalidate leave-one-out selection')
        support = self.meta.iloc[support_indices]
        if (not len(support) or not support.subject.eq(self.subject).all() or
                not support.split.eq('train').all() or not support.run.eq(3).all()):
            raise ValueError('Offsets and selection require own labeled run-3 support')
        logits = self.logits(support_indices)
        labels = support.y.to_numpy()
        selected, scores = select_family(self.calibration_losses, logits, labels)
        offsets = {family: fit_offset(logits[family], labels) for family in FAMILIES}
        return selected, offsets, scores
