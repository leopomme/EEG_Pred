import unittest
import numpy as np
import pandas as pd
from eeg_comp.classical import RECIPES, extract, ClassicalModel, sym_function
from eeg_comp.validation import make_splits


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


if __name__ == '__main__':
    unittest.main()
