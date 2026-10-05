"""Inductive ERP-template covariance classifier, with no neural dependencies.

Templates, covariance reference and scaler are fitted on source-fold data only.
One query trial never changes another query's features or predictions.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from eeg_comp.classical import covariance, sym_function, vectorize


class ERPTemplateCovariance:
    def __init__(self, C=.1, shrink=.1):
        self.C, self.shrink = C, shrink

    def _covariances(self, x):
        x=np.asarray(x,dtype=np.float64)
        if x.ndim!=3 or x.shape[1:]!=self.shape_ or not np.isfinite(x).all():
            raise ValueError('Expected finite aligned ERP waveforms')
        templates=np.broadcast_to(self.templates_[None],(len(x),)+self.templates_.shape)
        augmented=np.concatenate([templates,x],axis=1)
        cov=covariance(augmented,self.shrink)
        # Remove joint scale; retain relative template/trial coupling.
        return cov / np.maximum(np.trace(cov,axis1=-2,axis2=-1)[...,None,None],1e-12)

    def _features(self, c):
        centered=self.reference_[None]@c@self.reference_[None]
        return vectorize(sym_function(centered,np.log))

    def fit(self,x,y):
        x,y=np.asarray(x,dtype=np.float64),np.asarray(y)
        if x.ndim!=3 or len(x)!=len(y) or not np.isfinite(x).all() or set(np.unique(y))!={0,1}:
            raise ValueError('Finite source waveforms and both source classes are required')
        self.shape_=x.shape[1:]
        self.templates_=np.concatenate([x[y==label].mean(axis=0) for label in (0,1)],axis=0)
        c=self._covariances(x)
        self.reference_=sym_function(c.mean(axis=0),lambda v:v**-.5)
        f=self._features(c)
        self.scaler_=StandardScaler().fit(f)
        self.classifier_=LogisticRegression(C=self.C,solver='lbfgs',max_iter=1000)
        self.classifier_.fit(self.scaler_.transform(f),y)
        return self

    def predict_proba(self,x):
        return self.classifier_.predict_proba(self.scaler_.transform(self._features(self._covariances(x))))
