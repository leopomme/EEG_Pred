import unittest
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
from eeg_comp.classical import RECIPES, extract, ClassicalModel, sym_function
from eeg_comp.data import CHANNELS, CUE_INDEX, EPOCH_SAMPLES, FS, load_cache
from eeg_comp.validation import make_splits, summarize
from scripts.run_classical import cache_provenance, feature_cache, main


class RepresentationTests(unittest.TestCase):
    def test_negative_control_never_sees_post_cue(self):
        rng = np.random.default_rng(52)
        x = rng.normal(size=(6, 8, 2000)).astype('float32')
        changed = x.copy()
        changed[:, :, 750:] *= 1e6
        for name in ['erp_baseline', 'power_baseline']:
            np.testing.assert_array_equal(extract(x,RECIPES[name]),extract(changed,RECIPES[name]))

    def test_covariance_and_supervised_models_are_finite_on_degenerate_channel(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(16,8,2000)).astype('float32')
        x[:,7] = 0
        y = np.arange(16)%2
        for name in ['tangent_full','csp_full','paired_cov_full']:
            features = extract(x,RECIPES[name])
            model = ClassicalModel(RECIPES[name]).fit(features[:12],y[:12])
            p = model.predict_proba(features[12:])
            self.assertTrue(np.isfinite(p).all())
            np.testing.assert_allclose(p.sum(1),1.)
            state = model.scaler_.mean_.copy()
            _ = model.predict_proba(features[12:]*10)
            np.testing.assert_array_equal(model.scaler_.mean_,state)

    def test_matrix_invsqrt_is_correct(self):
        rng = np.random.default_rng(6)
        x = rng.normal(size=(2,4,4)); c = x@x.swapaxes(-1,-2) + np.eye(4)
        w = sym_function(c,lambda x:x**-.5)
        np.testing.assert_allclose(w@c@w,np.broadcast_to(np.eye(4),(2,4,4)),atol=1e-12)


class SplitTests(unittest.TestCase):
    def meta(self):
        rows=[]
        for i in range(17):
            for run in [1,2,3]:
                for j in range(10):
                    rows.append(dict(subject=f'S{i+1:03}',run=run,y=j%2,split='train'))
        rows.append(dict(subject='S999',run=3,y=-1,split='test'))
        return pd.DataFrame(rows)

    def test_confirmation_people_never_in_development(self):
        meta=self.meta()
        for name,train,query in make_splits(meta,'cross_subject',set(range(5))):
            self.assertFalse(set(meta.iloc[train].subject)&set(meta.iloc[query].subject))
            self.assertFalse({'S016','S017','S999'}&set(meta.iloc[np.r_[train,query]].subject))

    def test_single_person_fitting_and_query_disjoint(self):
        meta=self.meta();queries=[]
        for _,train,query in make_splits(meta,'within_subject'):
            self.assertEqual(len(set(meta.iloc[np.r_[train,query]].subject)),1)
            self.assertFalse(set(train)&set(query))
            self.assertTrue(meta.iloc[query].run.eq(3).all())
            queries.extend(query)
        self.assertEqual(len(queries),170)
        self.assertEqual(len(set(queries)),170)


class ReportingTests(unittest.TestCase):
    def test_calibration_accuracy_is_not_zero_target_accuracy(self):
        oof = pd.DataFrame(dict(subject=['S001'] * 2, run=[2, 2],
                                y=[0, 1], p_move=[.1, .9]))
        meta = pd.DataFrame(dict(split=['test'] * 2, run=[3, 3]))
        result = summarize(oof, meta)
        self.assertEqual(result['accuracy'], 1.)
        self.assertIsNone(result['target_weighted_accuracy'])
        self.assertIsNone(result['conditional_target_weighted_accuracy'])
        self.assertIsNone(result['subject_macro_target_accuracy'])
        self.assertIsNone(result['subject_bootstrap_95ci'])
        self.assertEqual(result['missing_target_runs'], [3])
        self.assertEqual(result['subject_macro_empirical_accuracy'], 1.)
        self.assertEqual(result['subject_macro_empirical_bootstrap_95ci'], [1., 1.])

    def test_partial_target_coverage_is_explicitly_conditional(self):
        oof = pd.DataFrame(dict(subject=['S001'] * 4, run=[1, 1, 3, 3],
                                y=[0, 1, 0, 1], p_move=[.1, .9, .9, .1]))
        meta = pd.DataFrame(dict(split=['test'] * 4, run=[1, 2, 3, 3]))
        result = summarize(oof, meta)
        self.assertIsNone(result['target_weighted_accuracy'])
        self.assertEqual(result['target_run_coverage'], .75)
        self.assertAlmostEqual(result['conditional_target_weighted_accuracy'], 1 / 3)
        self.assertIsNone(result['subject_macro_target_accuracy'])
        self.assertAlmostEqual(result['subject_macro_conditional_target_accuracy'], 1 / 3)
        np.testing.assert_allclose(result['subject_conditional_target_bootstrap_95ci'], [1/3, 1/3])

    def test_full_target_coverage_keeps_original_weighted_accuracy(self):
        oof = pd.DataFrame(dict(subject=['S001'] * 6, run=[1, 1, 2, 2, 3, 3],
                                y=[0, 1] * 3, p_move=[.1, .9, .1, .9, .9, .1]))
        meta = pd.DataFrame(dict(split=['test'] * 4, run=[1, 2, 3, 3]))
        result = summarize(oof, meta)
        self.assertEqual(result['target_run_coverage'], 1.)
        self.assertEqual(result['target_weighted_accuracy'], .5)
        self.assertEqual(result['conditional_target_weighted_accuracy'], .5)
        self.assertEqual(result['subject_macro_target_accuracy'], .5)
        self.assertEqual(result['subject_bootstrap_95ci'], [.5, .5])

    def test_subject_missing_a_run_is_not_reported_as_full_target_macro(self):
        oof = pd.DataFrame(dict(subject=['S001'] * 4 + ['S002'] * 2,
                                run=[1, 1, 2, 2, 1, 1], y=[0, 1] * 3,
                                p_move=[.1, .9] * 3))
        meta = pd.DataFrame(dict(split=['test'] * 2, run=[1, 2]))
        result = summarize(oof, meta)
        self.assertEqual(result['target_weighted_accuracy'], 1.)
        self.assertIsNone(result['subject_macro_target_accuracy'])
        self.assertIsNone(result['subject_bootstrap_95ci'])
        self.assertEqual(result['subject_macro_conditional_target_accuracy'], 1.)


class CacheProvenanceTests(unittest.TestCase):
    def write_fixture(self, root):
        cache_dir = root / 'artifacts' / 'within_subject'
        cache_dir.mkdir(parents=True)
        rng = np.random.default_rng(71)
        arrays = {'X':rng.normal(size=(8, len(CHANNELS), EPOCH_SAMPLES)).astype('float32'),
                  'y':np.array([0, 1] * 3 + [-1, -1], dtype=np.int8),
                  'valid_mask':np.ones((8, EPOCH_SAMPLES), dtype=bool),
                  'phase_valid_mask':np.ones((8, EPOCH_SAMPLES), dtype=bool),
                  'fs':np.array(FS), 'cue_index':np.array(CUE_INDEX),
                  'channels':np.array(CHANNELS), 'cache_version':np.array(1)}
        np.savez(cache_dir / 'epochs.npz', **arrays)
        meta = pd.DataFrame({'epoch_index':np.arange(8), 'competition':['within_subject'] * 8,
                             'subject':['S001'] * 8, 'run':[3] * 8,
                             'split':['train'] * 6 + ['test'] * 2, 'y':arrays['y'],
                             'label':['rest', 'move'] * 3 + ['', ''],
                             'test_order':[-1] * 6 + [0, 1]})
        meta.to_csv(cache_dir / 'metadata.csv', index=False)
        return cache_dir, arrays

    def invoke(self, directory, *extra):
        previous = Path.cwd()
        try:
            os.chdir(directory)
            with patch('sys.argv', ['run_classical.py', '--competition', 'within_subject',
                                    '--recipes', 'power_pre', '--output', 'results/test', *extra]), redirect_stdout(io.StringIO()):
                main()
        finally:
            os.chdir(previous)

    def test_changed_eeg_with_unchanged_metadata_invalidates_features(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_dir, arrays = self.write_fixture(Path(directory))
            first_manifest, first_digest = cache_provenance(cache_dir, arrays)
            first_features, first_signature = feature_cache(cache_dir, 'power_pre', arrays['X'],
                                                            arrays['valid_mask'], arrays['phase_valid_mask'], first_digest)
            arrays['X'][:, :, CUE_INDEX:] *= 2
            np.savez(cache_dir / 'epochs.npz', **arrays)
            second_manifest, second_digest = cache_provenance(cache_dir, arrays)
            second_features, second_signature = feature_cache(cache_dir, 'power_pre', arrays['X'],
                                                              arrays['valid_mask'], arrays['phase_valid_mask'], second_digest)
            self.assertEqual(first_manifest['metadata_sha256'], second_manifest['metadata_sha256'])
            self.assertNotEqual(first_manifest['epochs_sha256'], second_manifest['epochs_sha256'])
            self.assertNotEqual(first_signature, second_signature)
            np.testing.assert_allclose(second_features - first_features, np.log(4), atol=1e-6)
            self.assertEqual(len(list((cache_dir / 'classical_features').glob('*.npy'))), 2)

    def test_main_rejects_invalid_epoch_contracts(self):
        for key, value in [('fs', np.array(512)), ('cue_index', np.array(700)),
                           ('channels', np.array(CHANNELS[::-1])),
                           ('X', np.zeros((8, 8, 1999), dtype=np.float32)),
                           ('y', np.array([1, 0] * 3 + [-1, -1], dtype=np.int8))]:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                cache_dir, arrays = self.write_fixture(root)
                arrays[key] = value
                np.savez(cache_dir / 'epochs.npz', **arrays)
                with self.assertRaises(ValueError):
                    self.invoke(root)
                self.assertFalse((root / 'results/test').exists())

    def test_main_records_exact_splits_and_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_fixture(root)
            self.invoke(root, '--fit-final', '--seed', '17')
            output = root / 'results/test'
            config = json.loads((output / 'config.json').read_text())
            self.assertEqual(config['seeds']['split'], 17)
            self.assertEqual(config['seeds']['model']['power_pre'], None)
            self.assertEqual(config['model_configs']['power_pre']['C'], .1)
            self.assertEqual(config['cache_provenance']['contract']['fs'], FS)
            _, meta = load_cache(root / 'artifacts', 'within_subject')
            expected = list(make_splits(meta, 'within_subject', seed=17))
            recorded = json.loads((output / 'splits.json').read_text())
            for entry, (name, support, query) in zip(recorded, expected):
                self.assertEqual(entry['name'], name)
                self.assertEqual(entry['support_epoch_indices'], support.tolist())
                self.assertEqual(entry['query_epoch_indices'], query.tolist())
            before = {path.name:path.read_bytes() for path in output.glob('*') if path.is_file()}
            with self.assertRaises(FileExistsError):
                self.invoke(root)
            self.assertEqual(before, {path.name:path.read_bytes() for path in output.glob('*') if path.is_file()})
            # Also protect partial or unrelated output directories without config.json.
            (output / 'config.json').unlink()
            with self.assertRaises(FileExistsError):
                self.invoke(root)


if __name__ == '__main__':
    unittest.main()
