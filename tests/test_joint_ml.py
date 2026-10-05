"""Behavioral checks for a source-fitted, single-classifier ERP/CSP model."""
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from eeg_comp.classical import ClassicalModel, RECIPES
from eeg_comp.joint_ml import JointERPCSP
from scripts.compare_joint_ml import assert_same_metadata, compare


def covariances(rng, n, bands=2, channels=4):
    raw = rng.normal(size=(n, bands, channels, channels))
    return raw @ raw.swapaxes(-1, -2) + .2 * np.eye(channels)


class JointModelTests(unittest.TestCase):
    def test_csp_transform_exactly_matches_existing_source_fitted_formula(self):
        rng = np.random.default_rng(22)
        erp = rng.normal(size=(60, 400))
        c = covariances(rng, 60, bands=6, channels=8).astype(np.float32)
        labels = np.arange(48) % 2
        weights = rng.uniform(.5, 2., size=48)
        joint = JointERPCSP().fit(erp[:48], c[:48], labels, weights)
        baseline = ClassicalModel(RECIPES['csp_full']).fit(c[:48], labels, weights)
        np.testing.assert_allclose(joint.spatial_, baseline.spatial_, atol=1e-12, rtol=0)
        np.testing.assert_allclose(joint.transform(erp[48:], c[48:])[:, 400:],
                                   baseline._represent(c[48:]), atol=1e-12, rtol=0)
        self.assertEqual(joint.transform(erp[48:], c[48:]).shape, (12, 424))

    def test_other_queries_and_order_do_not_change_prediction_or_fitted_state(self):
        rng = np.random.default_rng(39)
        erp = rng.normal(size=(48, 16))
        c = covariances(rng, 48)
        original_erp, original_cov = erp.copy(), c.copy()
        model = JointERPCSP().fit(erp[:40], c[:40], np.arange(40) % 2)
        spatial = model.spatial_.copy()
        mean, scale = model.scaler_.mean_.copy(), model.scaler_.scale_.copy()
        coefficient = model.classifier_.coef_.copy()
        p = model.predict_proba(erp[40:], c[40:])
        np.testing.assert_allclose(p[:1], model.predict_proba(erp[40:41], c[40:41]), atol=1e-12)
        order = np.array([7, 2, 4, 0, 1, 6, 5, 3])
        np.testing.assert_allclose(p[order], model.predict_proba(erp[40:][order], c[40:][order]), atol=1e-12)
        altered_erp, altered_cov = erp[40:].copy(), c[40:].copy()
        altered_erp[1:] *= 1000
        altered_cov[1:] *= 1000
        np.testing.assert_allclose(p[:1], model.predict_proba(altered_erp, altered_cov)[:1], atol=1e-12)
        for before, after in [(spatial, model.spatial_), (mean, model.scaler_.mean_),
                              (scale, model.scaler_.scale_), (coefficient, model.classifier_.coef_),
                              (original_erp, erp), (original_cov, c)]:
            np.testing.assert_array_equal(before, after)

    def test_query_values_cannot_affect_source_fitting_and_saved_model_replays(self):
        rng = np.random.default_rng(14)
        erp = rng.normal(size=(40, 12))
        c = covariances(rng, 40)
        labels = np.arange(32) % 2
        fitted = JointERPCSP().fit(erp[:32], c[:32], labels)
        erp_changed, c_changed = erp.copy(), c.copy()
        erp_changed[32:] = 1e6
        c_changed[32:] *= 1e6
        repeat = JointERPCSP().fit(erp_changed[:32], c_changed[:32], labels)
        np.testing.assert_array_equal(fitted.spatial_, repeat.spatial_)
        np.testing.assert_array_equal(fitted.scaler_.mean_, repeat.scaler_.mean_)
        np.testing.assert_array_equal(fitted.classifier_.coef_, repeat.classifier_.coef_)
        expected = fitted.predict_proba(erp[32:], c[32:])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'source_model.joblib'
            joblib.dump(fitted, path)
            reloaded = joblib.load(path)
            np.testing.assert_array_equal(expected, reloaded.predict_proba(erp[32:], c[32:]))

    def test_two_blocks_resolve_a_task_that_either_block_alone_cannot(self):
        rng = np.random.default_rng(20261005)
        bits = np.tile(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), (160, 1))
        rng.shuffle(bits)
        labels = (bits.sum(axis=1) > 0).astype(int)
        erp = bits[:, :1] + rng.normal(scale=.01, size=(len(bits), 1))
        c = np.zeros((len(bits), 1, 2, 2))
        sign = 2 * bits[:, 1] - 1
        c[:, 0, 0, 0] = np.exp(sign)
        c[:, 0, 1, 1] = np.exp(-sign)
        source, query = np.arange(480), np.arange(480, len(bits))
        model = JointERPCSP().fit(erp[source], c[source], labels[source])
        p = model.predict_proba(erp[query], c[query])[:, 1]
        joint_accuracy = float(np.mean((p >= .5) == labels[query]))
        branch_accuracies = []
        for train_values, query_values in [(erp[source], erp[query]),
                                            (model._csp_features(c[source]), model._csp_features(c[query]))]:
            scaler = StandardScaler().fit(train_values)
            branch = LogisticRegression(C=.1, max_iter=1000).fit(scaler.transform(train_values), labels[source])
            branch_accuracies.append(float(np.mean(branch.predict(scaler.transform(query_values)) == labels[query])))
        self.assertGreaterEqual(joint_accuracy, .98)
        self.assertLess(max(branch_accuracies), .82)
        self.assertGreater(joint_accuracy - max(branch_accuracies), .15)

    def test_invalid_labels_features_and_weights_fail_before_prediction(self):
        rng = np.random.default_rng(3)
        erp, c = rng.normal(size=(8, 3)), covariances(rng, 8)
        labels = np.arange(8) % 2
        with self.assertRaisesRegex(ValueError, 'binary'):
            JointERPCSP().fit(erp, c, np.zeros(8))
        with self.assertRaisesRegex(ValueError, 'weights'):
            JointERPCSP().fit(erp, c, labels, np.zeros(8))
        bad = erp.copy()
        bad[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'finite'):
            JointERPCSP().fit(bad, c, labels)
        fitted = JointERPCSP().fit(erp, c, labels)
        with self.assertRaisesRegex(ValueError, 'dimensions'):
            fitted.predict_proba(erp[:, :2], c)


class PairedComparisonTests(unittest.TestCase):
    def frames(self):
        labels = np.arange(12) % 2
        baseline = pd.DataFrame({'epoch_index': np.arange(12), 'subject': ['S001'] * 8 + ['S002'] * 4,
                                 'run': [1] * 4 + [2] * 2 + [3] * 2 + [1] * 2 + [3] * 2,
                                 'fold': ['group0'] * 12, 'y': labels,
                                 'file': ['source.csv'] * 12, 'p_move': np.where(labels == 1, .9, .1)})
        candidate = baseline.copy()
        candidate.loc[:3, 'p_move'] = 1 - candidate.loc[:3, 'p_move']
        candidate['prediction'] = np.where(candidate.p_move >= .5, 'move', 'rest')
        return candidate, baseline

    def test_target_pooled_run_score_is_distinct_from_participant_conditional_mean(self):
        candidate, baseline = self.frames()
        result = compare(candidate, baseline, {1: 1/3, 2: 1/3, 3: 1/3}, 1000)
        self.assertAlmostEqual(result['target_run_weighted_accuracy_delta'], -2/9)
        self.assertAlmostEqual(result['participant_conditional_macro_accuracy_delta'], -1/6)
        self.assertAlmostEqual(result['per_participant']['S002']['target_run_coverage'], 2/3)
        self.assertEqual(result['changed_wrong_to_correct'], 0)
        self.assertEqual(result['changed_correct_to_wrong'], 4)
        self.assertEqual(result['missing_target_runs'], [])
        identical = compare(baseline, baseline, {1: 1/3, 2: 1/3, 3: 1/3}, 1000)
        self.assertEqual(identical['participant_conditional_paired_bootstrap_95ci'], [0., 0.])

    def test_metadata_or_fold_change_is_rejected_before_pairing(self):
        candidate, baseline = self.frames()
        assert_same_metadata(candidate, baseline)
        candidate.loc[0, 'fold'] = 'other_group'
        with self.assertRaises(AssertionError):
            assert_same_metadata(candidate, baseline)


if __name__ == '__main__':
    unittest.main()
