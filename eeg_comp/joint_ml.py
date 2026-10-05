"""One pure-ML classifier on cue ERP and source-fitted filter-bank CSP features.

The inputs are the unchanged trial-local ``erp_pre`` and ``csp_full`` extracts.
Only source rows fit CSP, feature scaling and the final logistic classifier.
There is no probability blending, neural dependency, or query-batch adaptation.
"""
from __future__ import annotations

import numpy as np
from scipy import linalg
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class JointERPCSP:
    """Fit source CSP filters, then concatenate ERP and log-CSP variances."""

    def __init__(self, C: float = .1):
        if not np.isfinite(C) or C <= 0:
            raise ValueError('C must be finite and positive')
        self.C = float(C)

    @staticmethod
    def _inputs(erp, covariances):
        erp = np.asarray(erp, dtype=np.float64)
        covariances = np.asarray(covariances, dtype=np.float64)
        if (erp.ndim != 2 or covariances.ndim != 4 or len(erp) != len(covariances)
                or not len(erp) or erp.shape[1] < 1 or covariances.shape[1] < 1
                or covariances.shape[2] < 2
                or covariances.shape[2] != covariances.shape[3]):
            raise ValueError('Expected aligned ERP rows and band/channel covariance matrices')
        if not np.isfinite(erp).all() or not np.isfinite(covariances).all():
            raise ValueError('Inputs must be finite')
        trace = np.trace(covariances, axis1=-2, axis2=-1)
        if np.any(trace <= 0):
            raise ValueError('Covariance traces must be positive')
        return erp, covariances

    def _csp_features(self, covariances):
        # Identical to ClassicalModel's csp_full representation.
        trace = np.maximum(np.trace(covariances, axis1=-2, axis2=-1), 1e-12)
        normalized = covariances / trace[..., None, None]
        values = np.einsum('bik,nbij,bjk->nbk', self.spatial_, normalized, self.spatial_)
        return np.log(np.maximum(values, 1e-12)).reshape(len(covariances), -1)

    def transform(self, erp, covariances):
        if not hasattr(self, 'spatial_'):
            raise ValueError('Model must be fitted before transformation')
        erp, covariances = self._inputs(erp, covariances)
        if erp.shape[1] != self.erp_features_ or covariances.shape[1:] != self.covariance_shape_:
            raise ValueError('Query feature dimensions differ from source fitting')
        return np.concatenate([erp, self._csp_features(covariances)], axis=1)

    def fit(self, erp, covariances, y, sample_weight=None):
        erp, covariances = self._inputs(erp, covariances)
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(erp) or set(np.unique(y)) != {0, 1}:
            raise ValueError('Source rows require both binary classes')
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)
            if (sample_weight.shape != y.shape or not np.isfinite(sample_weight).all()
                    or np.any(sample_weight <= 0)):
                raise ValueError('Sample weights must be aligned, finite and positive')
        self.erp_features_ = erp.shape[1]
        self.covariance_shape_ = covariances.shape[1:]
        trace = np.maximum(np.trace(covariances, axis1=-2, axis2=-1), 1e-12)
        normalized = covariances / trace[..., None, None]
        class_cov = []
        for label in (0, 1):
            weights = None if sample_weight is None else sample_weight[y == label]
            class_cov.append(np.average(normalized[y == label], axis=0, weights=weights))
        spatial = []
        for a, b in zip(*class_cov):
            _, vectors = linalg.eigh(a, a + b)
            k = min(2, len(a) // 2)
            spatial.append(vectors[:, np.r_[0:k, len(a)-k:len(a)]])
        self.spatial_ = np.stack(spatial)
        values = self.transform(erp, covariances)
        self.scaler_ = StandardScaler().fit(values, sample_weight=sample_weight)
        self.classifier_ = LogisticRegression(C=self.C, max_iter=1000, solver='lbfgs')
        self.classifier_.fit(self.scaler_.transform(values), y, sample_weight=sample_weight)
        return self

    def predict_proba(self, erp, covariances):
        return self.classifier_.predict_proba(self.scaler_.transform(self.transform(erp, covariances)))

    def config(self):
        return {'model': 'JointERPCSP', 'C': self.C, 'erp_recipe': 'erp_pre',
                'covariance_recipe': 'csp_full', 'spatial_filters_per_band': 'two smallest + two largest',
                'classifier': 'one StandardScaler + LogisticRegression(lbfgs,max_iter=1000)',
                'information_budget': 'source-only fitting; independent trial-local query prediction'}
