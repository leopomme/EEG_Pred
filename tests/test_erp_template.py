import unittest
import numpy as np
from eeg_comp.erp_template import ERPTemplateCovariance

class TemplateTests(unittest.TestCase):
    def test_queries_do_not_change_templates_or_each_other(self):
        rng=np.random.default_rng(30)
        train=rng.normal(size=(40,4,50));labels=np.arange(40)%2
        x=rng.normal(size=(8,4,50))
        model=ERPTemplateCovariance().fit(train,labels)
        templates=model.templates_.copy();reference=model.reference_.copy()
        p=model.predict_proba(x)
        np.testing.assert_allclose(p[:1],model.predict_proba(x[:1]),atol=1e-12)
        changed=x.copy();changed[1:]*=100
        np.testing.assert_allclose(p[:1],model.predict_proba(changed)[:1],atol=1e-12)
        np.testing.assert_array_equal(templates,model.templates_)
        np.testing.assert_array_equal(reference,model.reference_)

    def test_degenerate_channels_regularized_and_class_signal_learned(self):
        rng=np.random.default_rng(3);y=np.arange(40)%2
        x=rng.normal(scale=.1,size=(40,4,50));x[:,3]=0
        x[:,0]+=np.where(y[:,None]==1,1.,-1.)*np.sin(np.arange(50)/8.)
        model=ERPTemplateCovariance().fit(x[:32],y[:32])
        self.assertTrue((np.linalg.eigvalsh(model._covariances(x))>0).all())
        p=model.predict_proba(x[32:])[:,1]
        self.assertTrue(np.isfinite(p).all())
        self.assertGreaterEqual(np.mean((p>=.5)==y[32:]),.875)

if __name__=='__main__':unittest.main()
