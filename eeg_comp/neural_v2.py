"""Raw EEG models that test which cue-relative information normalization keeps.

Every transform is recomputed inside the forward pass for one trial and
channel at a time. There are no fitted preprocessing statistics, batch
statistics, or handcrafted EEG features. The historical phase model remains
unchanged in ``neural.py``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math

import torch
from torch import nn

from eeg_comp.neural import NeuralConfig, PhaseConvNet


@dataclass(frozen=True)
class NeuralV2Config:
    mode: str = "phase_stable"
    task_samples: int = 1200
    baseline_samples: int = 500
    phase_config: NeuralConfig = field(default_factory=NeuralConfig)
    dropout: float = 0.35
    cue_filters_per_channel: int = 2
    cue_mix_channels: int = 16
    cue_kernel: int = 41
    cue_bins: int = 50

    def __post_init__(self) -> None:
        if self.mode not in {"phase_stable", "phase_baseline", "cue_baseline"}:
            raise ValueError("Unknown raw EEG normalization/model mode")
        if isinstance(self.task_samples, bool) or not isinstance(self.task_samples, int) or self.task_samples not in {500, 1200}:
            raise ValueError("task_samples must be 500 (2 s) or 1200 (4.8 s)")
        for name in ("baseline_samples", "cue_filters_per_channel", "cue_mix_channels", "cue_bins"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.phase_config, NeuralConfig) or self.phase_config.channels != 8:
            raise ValueError("phase_config must be a NeuralConfig with eight channels")
        if not math.isfinite(self.phase_config.normalization_eps):
            raise ValueError("Normalization epsilon must be finite")
        if not math.isfinite(self.phase_config.dropout):
            raise ValueError("Phase dropout must be finite")
        if self.task_samples < 16 * self.phase_config.phase_bins:
            raise ValueError("Task window too short for the configured phase bins")
        if isinstance(self.cue_kernel, bool) or not isinstance(self.cue_kernel, int) or self.cue_kernel < 1 or self.cue_kernel % 2 != 1:
            raise ValueError("cue_kernel must be a positive odd integer")
        if self.cue_bins > (self.task_samples + 9) // 10:
            raise ValueError("cue_bins cannot exceed the learned temporal sequence length")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")


class OrderedAdaptiveMean(nn.Module):
    """Adaptive temporal means with a deterministic CUDA backward path.

    PyTorch's adaptive average pooling CUDA backward is unavailable under
    deterministic algorithms. Explicit contiguous slice means implement the
    same floor/ceiling bin boundaries without an atomic scatter reduction.
    """

    def __init__(self, bins: int):
        super().__init__()
        if isinstance(bins, bool) or not isinstance(bins, int) or bins < 1:
            raise ValueError("bins must be a positive integer")
        self.bins = bins

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] < 1:
            raise ValueError("Expected a nonempty (trials,channels,samples) sequence")
        length = x.shape[-1]
        if length == self.bins:
            return x
        return torch.stack([
            x[..., (index * length) // self.bins:
              ((index + 1) * length + self.bins - 1) // self.bins].mean(dim=-1)
            for index in range(self.bins)
        ], dim=-1)


class RawEEGV2(nn.Module):
    """Classifier accepting baseline followed by task as one uniform raw window.

    ``phase_stable`` reproduces task-only z-scoring, first subtracting a
    task sample so the reduction does not sum a large recording DC offset.
    ``phase_baseline`` instead centers and scales the task from its own
    pre-cue baseline, then uses the original phase network. That network's
    group normalization can attenuate some amplitude information, so
    ``cue_baseline`` also tests a learned temporal model without group
    normalization or a second task z-score.
    """

    def __init__(self, config: NeuralV2Config | None = None):
        super().__init__()
        self.config = config or NeuralV2Config()
        cfg = self.config
        if cfg.mode.startswith("phase_"):
            self.network = PhaseConvNet(cfg.phase_config)
            self.network.standardize = nn.Identity()
        else:
            width = 8 * cfg.cue_filters_per_channel
            # The prespecified 500-sample cue crop gives exactly 50 temporal
            # positions, so its 50-bin pool is mathematically the identity.
            learned_samples = (cfg.task_samples + 9) // 10
            pool = nn.Identity() if learned_samples == cfg.cue_bins else OrderedAdaptiveMean(cfg.cue_bins)
            self.network = nn.Sequential(
                nn.Conv1d(8, width, cfg.cue_kernel, stride=10,
                          padding=cfg.cue_kernel // 2, groups=8, bias=False),
                nn.Conv1d(width, cfg.cue_mix_channels, 1),
                nn.ELU(),
                nn.Dropout(cfg.dropout),
                pool,
                nn.Flatten(1),
                nn.Linear(cfg.cue_mix_channels * cfg.cue_bins, 1),
            )

    def normalize_task(self, x: torch.Tensor) -> torch.Tensor:
        """Return the task crop using independent runtime normalization.

        Subtracting a representable sample before the mean is algebraically
        redundant, but substantially improves float32 accuracy for recording
        offsets much larger than the cue response. It cannot recover detail
        already lost when the raw recording was quantized.
        """
        cfg = self.config
        expected_samples = cfg.baseline_samples + cfg.task_samples
        if x.ndim != 3 or x.shape[1] != 8 or x.shape[2] != expected_samples:
            raise ValueError(f"Expected (trials,8,{expected_samples}), got {tuple(x.shape)}")
        if not x.is_floating_point():
            raise ValueError("Raw EEG must use a floating-point tensor")
        if not bool(torch.isfinite(x).all()):
            raise ValueError("Raw EEG contains non-finite samples")
        # Promote low-precision input for safe variance and epsilon arithmetic.
        # The persisted raw cache is float32; this also keeps future autocast
        # from reducing the precision of normalization statistics.
        if x.dtype in {torch.float16, torch.bfloat16}:
            x = x.float()
        task = x[:, :, cfg.baseline_samples:]
        if cfg.mode == "phase_stable":
            shifted_task = task - task[:, :, :1]
            center = shifted_task.mean(dim=-1, keepdim=True)
            centered_task = shifted_task - center
            variance = centered_task.square().mean(dim=-1, keepdim=True)
        else:
            shifted = x - x[:, :, :1]
            baseline = shifted[:, :, :cfg.baseline_samples]
            center = baseline.mean(dim=-1, keepdim=True)
            centered_task = shifted[:, :, cfg.baseline_samples:] - center
            variance = (baseline - center).square().mean(dim=-1, keepdim=True)
        # Clamp before sqrt to keep the derivative finite on flat channels.
        scale = variance.clamp_min(cfg.phase_config.normalization_eps ** 2).sqrt()
        return centered_task / scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.network(self.normalize_task(x))
        return logits.squeeze(-1) if self.config.mode == "cue_baseline" else logits

    def configuration(self) -> dict:
        return asdict(self.config)
