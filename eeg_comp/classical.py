"""Classical EEG representations. No neural dependencies or test-batch fitting."""
from dataclasses import dataclass, asdict
import numpy as np
from scipy import linalg, signal
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

CHANNELS = ('Fz', 'C3', 'Cz', 'C4', 'PO7', 'Pz', 'PO8', 'Oz')
BANDS = ((1., 4.), (4., 8.), (8., 13.), (13., 20.), (20., 30.), (30., 45.))
FS, CUE = 250, 750


@dataclass(frozen=True)
class Recipe:
    kind: str
    window: tuple = (0., 4.8)
    channels: tuple = tuple(range(8))
    bands: tuple = BANDS
    relative: bool = False


# Prespecified, interpretable contrasts, not an automatic model search.
RECIPES = {
    'erp_early': Recipe('erp', (0., .8)),
    'erp_pre': Recipe('erp', (0., 2.)),
    'erp_late': Recipe('erp', (2., 4.)),
    'erp_early_central': Recipe('erp', (0., .8), (1, 2, 3)),
    'erp_early_posterior': Recipe('erp', (0., .8), (4, 5, 6, 7)),
    'erp_baseline': Recipe('erp', (-2., 0.)),
    'power_pre': Recipe('power', (0., 2.)),
    'power_late': Recipe('power', (2., 4.)),
    'power_full': Recipe('power'),
    'power_baseline': Recipe('power', (-2., 0.)),
    'relpower_full': Recipe('power', relative=True),
    'tangent_full': Recipe('tangent'),
    'tangent_central': Recipe('tangent', channels=(1, 2, 3)),
    'tangent_posterior': Recipe('tangent', channels=(4, 5, 6, 7)),
    'csp_full': Recipe('csp'),
    'paired_cov_full': Recipe('paired'),
}


def crop(X, window, channels=tuple(range(8))):
    a, b = [CUE + round(t * FS) for t in window]
    if a < 0 or b > X.shape[-1] or b <= a:
        raise ValueError(f'Invalid crop {window} for {X.shape}')
    result = np.asarray(X[:, channels, a:b], dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite data in required crop; inspect the data audit')
    return result


def filtered(x, band):
    # Filter EACH crop independently, including the baseline negative control.
    # Neither a neighboring trial nor later experimental phase can leak in.
    centered = x - x.mean(axis=-1, keepdims=True)
    sos = signal.butter(4, band, btype='bandpass', fs=FS, output='sos')
    return signal.sosfiltfilt(sos, centered, axis=-1)


def covariance(x, shrink=.05):
    x = x - x.mean(axis=-1, keepdims=True)
    c = x @ x.swapaxes(-1, -2) / max(x.shape[-1] - 1, 1)
    n = x.shape[-2]
    scale = np.maximum(np.trace(c, axis1=-2, axis2=-1) / n, 1e-12)
    return (1. - shrink) * c + shrink * scale[..., None, None] * np.eye(n)


def sym_function(c, function):
    vals, vecs = np.linalg.eigh(c)
    vals = np.maximum(vals, 1e-12)
    return (vecs * function(vals)[..., None, :]) @ vecs.swapaxes(-1, -2)


def vectorize(c):
    p = c.shape[-1]
    i, j = np.triu_indices(p)
    return c[..., i, j] * np.where(i == j, 1., np.sqrt(2.))


def extract(X, recipe):
    """Return trial-local features/covariances; supervised fitting happens later."""
    x = crop(X, recipe.window, recipe.channels)
    if recipe.kind == 'erp':
        # Runtime trial-local scaling; no pooled target statistics.
        baseline = crop(X, (-2., 0.), recipe.channels)
        offset = np.median(baseline, axis=-1, keepdims=True)
        scale = np.maximum(np.std(baseline, axis=-1, keepdims=True), 1e-6)
        if recipe.window[1] <= 0:
            offset = np.median(x, axis=-1, keepdims=True)
            scale = np.maximum(np.std(x, axis=-1, keepdims=True), 1e-6)
        x = (x - offset) / scale
        sos = signal.butter(4, 20., btype='lowpass', fs=FS, output='sos')
        x = signal.sosfiltfilt(sos, x, axis=-1)
        # 40-ms averages retain cue-locked shape with modest dimensionality.
        n = x.shape[-1] // 10
        x = x[..., :n * 10].reshape(*x.shape[:-1], n, 10).mean(-1)
        return np.clip(x, -20., 20.).reshape(len(x), -1).astype('float32')
    covs = np.stack([covariance(filtered(x, band)) for band in recipe.bands], axis=1)
    if recipe.kind in ('tangent', 'csp'):
        return covs.astype('float32')
    power = np.maximum(np.diagonal(covs, axis1=-2, axis2=-1), 1e-12)
    if recipe.kind == 'power' and not recipe.relative:
        return np.log(power).reshape(len(x), -1).astype('float32')
    base = crop(X, (-2., 0.), recipe.channels)
    base_cov = np.stack([covariance(filtered(base, band)) for band in recipe.bands], axis=1)
    if recipe.kind == 'power':
        base_power = np.maximum(np.diagonal(base_cov, axis1=-2, axis2=-1), 1e-12)
        return np.log(power / base_power).reshape(len(x), -1).astype('float32')
    if recipe.kind == 'paired':
        b = sym_function(base_cov, lambda v: v ** -.5)
        generalized = b @ covs @ b
        eig = np.linalg.eigvalsh(generalized)
        ratios = np.log(power / np.maximum(np.diagonal(base_cov, axis1=-2, axis2=-1), 1e-12))
        return np.concatenate([np.log(np.maximum(eig, 1e-12)).reshape(len(x), -1),
                               ratios.reshape(len(x), -1)], axis=1).astype('float32')
    raise ValueError(recipe)


class ClassicalModel:
    """All covariance references/CSP/scaling fitted within the training fold."""
    def __init__(self, recipe, C=.1):
        self.recipe, self.C = recipe, C

    def _represent(self, features):
        c = np.asarray(features, dtype=np.float64)
        if self.recipe.kind == 'tangent':
            trace = np.maximum(np.trace(c, axis1=-2, axis2=-1), 1e-12)
            normalized = c / trace[..., None, None]
            whitened = self.reference_[None] @ normalized @ self.reference_[None]
            values = vectorize(sym_function(whitened, np.log)).reshape(len(c), -1)
            return np.concatenate([values, np.log(trace)], axis=1)
        if self.recipe.kind == 'csp':
            trace = np.maximum(np.trace(c, axis1=-2, axis2=-1), 1e-12)
            c = c / trace[..., None, None]
            values = np.einsum('bik,nbij,bjk->nbk', self.spatial_, c, self.spatial_)
            return np.log(np.maximum(values, 1e-12)).reshape(len(c), -1)
        return c

    def fit(self, features, y, sample_weight=None):
        if self.recipe.kind in ('tangent', 'csp'):
            c = np.asarray(features, dtype=np.float64)
            c = c / np.maximum(np.trace(c, axis1=-2, axis2=-1)[..., None, None], 1e-12)
            if self.recipe.kind == 'tangent':
                reference = np.average(c, axis=0, weights=sample_weight)
                self.reference_ = sym_function(reference, lambda v: v ** -.5)
            else:
                class_cov = []
                for label in (0, 1):
                    w = None if sample_weight is None else sample_weight[y == label]
                    class_cov.append(np.average(c[y == label], axis=0, weights=w))
                spatial = []
                for a, b in zip(*class_cov):
                    _, v = linalg.eigh(a, a + b)
                    k = min(2, len(a) // 2)
                    spatial.append(v[:, np.r_[0:k, len(a)-k:len(a)]])
                self.spatial_ = np.stack(spatial)
        x = self._represent(features)
        self.scaler_ = StandardScaler().fit(x, sample_weight=sample_weight)
        x = self.scaler_.transform(x)
        self.classifier_ = LogisticRegression(C=self.C, max_iter=1000, solver='lbfgs')
        self.classifier_.fit(x, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, features):
        return self.classifier_.predict_proba(self.scaler_.transform(self._represent(features)))

    def config(self):
        return {'recipe': asdict(self.recipe), 'C': self.C}
