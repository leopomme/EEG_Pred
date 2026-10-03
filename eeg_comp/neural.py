"""Small independently implemented end-to-end EEG classifier.

Input is raw, named-channel EEG with its time axis intact. Normalization is
computed independently for each trial and channel at each forward pass.
There is no fitted preprocessing or target-batch state.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class NeuralConfig:
    channels: int = 8
    temporal_filters: int = 8
    spatial_multiplier: int = 2
    temporal_kernel: int = 63
    separable_kernel: int = 15
    phase_bins: int = 8
    dropout: float = 0.35
    normalization_eps: float = 1e-5

    def __post_init__(self) -> None:
        for name in ("channels", "temporal_filters", "spatial_multiplier", "phase_bins"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if any(value < 1 or value % 2 != 1 for value in (self.temporal_kernel, self.separable_kernel)):
            raise ValueError("Convolution kernels must be positive and odd")
        if not 0 <= self.dropout < 1 or self.normalization_eps <= 0:
            raise ValueError("Invalid dropout or normalization epsilon")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


class TrialChannelStandardize(nn.Module):
    """Normalize without mixing people, trials, or time windows."""

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        centered = x - x.mean(dim=-1, keepdim=True)
        # Clamp before sqrt: sqrt(0) followed by a clamp has a NaN derivative
        # on flat channels, even though its forward output is finite.
        scale = centered.square().mean(dim=-1, keepdim=True).clamp_min(self.eps**2).sqrt()
        return centered / scale


class PhaseConvNet(nn.Module):
    """Temporal/spatial convolutions with phase-sensitive learned summaries.

Eight ordered temporal bins preserve coarse cue timing. Both the mean and
standard deviation of the *learned* representations enter the classifier;
no handcrafted band powers, covariance matrices, or external features enter.
Group normalization has no running or cross-trial statistics.
"""

    def __init__(self, config: NeuralConfig | None = None):
        super().__init__()
        self.config = config or NeuralConfig()
        cfg = self.config
        width = cfg.temporal_filters * cfg.spatial_multiplier
        self.standardize = TrialChannelStandardize(cfg.normalization_eps)
        self.temporal = nn.Conv2d(
            1, cfg.temporal_filters, (1, cfg.temporal_kernel),
            padding=(0, cfg.temporal_kernel // 2), bias=False,
        )
        self.spatial = nn.Conv2d(
            cfg.temporal_filters, width, (cfg.channels, 1),
            groups=cfg.temporal_filters, bias=False,
        )
        self.norm1 = nn.GroupNorm(cfg.temporal_filters, width)
        self.depthwise = nn.Conv2d(
            width, width, (1, cfg.separable_kernel),
            padding=(0, cfg.separable_kernel // 2), groups=width, bias=False,
        )
        self.pointwise = nn.Conv2d(width, width, 1, bias=False)
        self.norm2 = nn.GroupNorm(cfg.temporal_filters, width)
        self.dropout = nn.Dropout(cfg.dropout)
        self.classifier = nn.Linear(width * cfg.phase_bins * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != self.config.channels:
            raise ValueError(f"Expected (trials,{self.config.channels},samples), got {tuple(x.shape)}")
        if x.shape[-1] < 16 * self.config.phase_bins:
            raise ValueError("Window too short for two pooling layers and the phase bins")
        x = self.standardize(x).unsqueeze(1)
        x = self.dropout(F.avg_pool2d(F.elu(self.norm1(self.spatial(self.temporal(x)))), (1, 4)))
        x = self.dropout(F.avg_pool2d(F.elu(self.norm2(self.pointwise(self.depthwise(x)))), (1, 4)))
        x = x.squeeze(2)
        bins = torch.tensor_split(x, self.config.phase_bins, dim=-1)
        means = torch.stack([part.mean(dim=-1) for part in bins], dim=-1)
        second = torch.stack([part.square().mean(dim=-1) for part in bins], dim=-1)
        deviations = (second - means.square()).clamp_min(1e-6).sqrt()
        return self.classifier(torch.cat([means, deviations], dim=1).flatten(1)).squeeze(1)

    def configuration(self) -> dict:
        return asdict(self.config)


def binary_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    """Metrics only; this function never fits a predictor or threshold."""
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    if y.ndim != 1 or p.shape != y.shape:
        raise ValueError("Expected matching one-dimensional labels and probabilities")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid binary labels or probabilities")
    if len(y) == 0:
        return {"n": 0}
    pred = p >= 0.5
    clipped = p.clip(1e-7, 1 - 1e-7)
    counts = [int((y == label).sum()) for label in (0, 1)]
    recalls = [float((pred[y == label] == label).mean()) for label in (0, 1) if counts[label]]
    # Tied-score AUC, calculated without model-fitting dependencies.
    auc = None
    if all(counts):
        _, inverse = np.unique(p, return_inverse=True)
        pos = np.bincount(inverse, weights=y)
        neg = np.bincount(inverse, weights=1 - y)
        lower_neg = np.cumsum(neg) - neg
        auc = float(np.sum(pos * (lower_neg + 0.5 * neg)) / (counts[0] * counts[1]))
    return {
        "n": int(len(y)), "rest_n": counts[0], "move_n": counts[1],
        "accuracy": float(np.mean(pred == y)), "balanced_accuracy": float(np.mean(recalls)),
        "log_loss": float(np.mean(-y * np.log(clipped) - (1-y) * np.log1p(-clipped))),
        "brier": float(np.mean((p-y)**2)), "auc": auc,
    }
