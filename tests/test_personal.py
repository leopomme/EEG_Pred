import unittest

import numpy as np
import pandas as pd
from scipy.special import expit

from eeg_comp.personal import (FAMILIES, PersonalSources, calibration_splits,
                               fit_offset, leave_one_out_offset,
                               log_loss_from_logits, select_family)


class OffsetTests(unittest.TestCase):
    def test_prior_is_finite_with_single_class_and_shrinks_to_zero(self):
        logits = np.array([-2., -.3, 0., .4, 1.])
        for value in (0, 1):
            labels = np.full(len(logits), value)
            offset = fit_offset(logits, labels, strength=2.)
            self.assertTrue(np.isfinite(offset))
            self.assertAlmostEqual(float(np.sum(expit(logits + offset) - labels)) + 2 * offset, 0.)
            self.assertLess(abs(fit_offset(logits, labels, strength=20.)), abs(offset))
            self.assertLess(log_loss_from_logits(logits + offset, labels),
                            log_loss_from_logits(logits, labels))
        self.assertEqual(fit_offset([], []), 0.)
        for strength in (0., -1., np.nan, np.inf):
            with self.assertRaises(ValueError):
                fit_offset(logits, np.ones(5), strength=strength)

    def test_loo_offset_never_uses_its_own_label(self):
        logits, labels = np.array([-.8, .2, .4, 1.3]), np.array([0, 1, 0, 1])
        predictions, offsets = leave_one_out_offset(logits, labels)
        for index in range(len(labels)):
            changed = labels.copy()
            changed[index] = 1 - changed[index]
            changed_predictions, changed_offsets = leave_one_out_offset(logits, changed)
            self.assertEqual(offsets[index], changed_offsets[index])
            self.assertEqual(predictions[index], changed_predictions[index])
        np.testing.assert_array_equal(leave_one_out_offset([.7], [1])[0], [.7])

    def test_selection_has_fixed_equal_weights_and_tie_order(self):
        labels = np.array([0, 1, 0, 1])
        logits = {name: np.array([-.2, .4, .1, .7]) for name in FAMILIES}
        losses = dict.fromkeys(FAMILIES, .6)
        selected, scores = select_family(losses, logits, labels)
        self.assertEqual(selected, FAMILIES[0])
        for score in scores.values():
            self.assertAlmostEqual(score['selection_score'],
                                   .5 * .6 + .5 * score['support_loo_log_loss'])
        losses['power_full'] = .4
        self.assertEqual(select_family(losses, logits, labels)[0], 'power_full')


def fixture():
    rows = []
    for person in ('S001', 'S002'):
        for run in (1, 2, 3):
            for trial in range(8):
                rows.append(dict(subject=person, run=run, trial_index=trial, split='train',
                                 y=trial % 2, competition='within_subject'))
        for trial in range(4):
            rows.append(dict(subject=person, run=3, trial_index=trial, split='test',
                             y=-1, competition='within_subject'))
    meta = pd.DataFrame(rows)
    rng = np.random.default_rng(70)
    features = {name: rng.normal(size=(len(meta), 6)) for name in ('erp_pre', 'power_full')}
    for name in ('tangent_full', 'csp_full'):
        x = rng.normal(size=(len(meta), 2, 4, 4))
        features[name] = x @ x.swapaxes(-1, -2) + np.eye(4)
    return meta, features


class IsolationTests(unittest.TestCase):
    def test_model_selection_and_probabilities_ignore_query_labels_and_other_people(self):
        meta, features = fixture()
        first = PersonalSources(meta, features, 'S001')
        support, query = np.arange(16, 20), np.arange(20, 28)
        selected, offsets, scores = first.adapt(support)
        before = first.logits(query)
        changed_meta = meta.copy()
        changed_meta.loc[query, 'y'] = -99  # Query labels cannot enter any fit or selection.
        other = changed_meta.subject.eq('S002')
        changed_meta.loc[other, 'y'] = -99
        changed_features = {name: values.copy() for name, values in features.items()}
        for values in changed_features.values():
            values[other] = np.nan  # Any cross-person fitting now also fails numerically.
        second = PersonalSources(changed_meta, changed_features, 'S001')
        self.assertEqual(second.adapt(support), (selected, offsets, scores))
        self.assertEqual(second.calibration_losses, first.calibration_losses)
        after = second.logits(query)
        for family in FAMILIES:
            np.testing.assert_array_equal(before[family], after[family])
            np.testing.assert_array_equal(first.models[family].scaler_.mean_,
                                          second.models[family].scaler_.mean_)

    def test_source_fits_ignore_all_run3_labels_and_test_features(self):
        meta, features = fixture()
        first = PersonalSources(meta, features, 'S001')
        modified = meta.copy()
        modified.loc[modified.run.eq(3), 'y'] = -99
        modified_features = {name: values.copy() for name, values in features.items()}
        for values in modified_features.values():
            values[meta.split.eq('test')] = np.nan
        second = PersonalSources(modified, modified_features, 'S001')
        self.assertEqual(first.calibration_losses, second.calibration_losses)
        for family in FAMILIES:
            np.testing.assert_array_equal(first.models[family].classifier_.coef_,
                                          second.models[family].classifier_.coef_)

    def test_rejects_foreign_support_test_support_and_repeated_support(self):
        meta, features = fixture()
        sources = PersonalSources(meta, features, 'S001')
        for support in ([44, 45], [24, 25], [0, 1], [16, 16], []):
            with self.subTest(support=support), self.assertRaises(ValueError):
                sources.adapt(support)
        with self.assertRaises(ValueError):
            sources.logits([28])
        with self.assertRaises(ValueError):
            PersonalSources(meta.assign(competition='cross_subject'), features, 'S001')

    def test_calibration_splits_use_own_source_only_with_single_run_fallback(self):
        meta, _ = fixture()
        for _, train, query in calibration_splits(meta, np.arange(16)):
            self.assertFalse(set(train) & set(query))
            self.assertEqual(set(meta.iloc[train].run), {3 - meta.iloc[query].run.iloc[0]})
        pairs = list(calibration_splits(meta, np.arange(8)))
        self.assertEqual(pairs[0][0], 'block_first_to_second')
        np.testing.assert_array_equal(pairs[0][1], np.arange(4))
        np.testing.assert_array_equal(pairs[0][2], np.arange(4, 8))
        with self.assertRaises(ValueError):
            list(calibration_splits(meta, np.arange(24)))


if __name__ == '__main__':
    unittest.main()
