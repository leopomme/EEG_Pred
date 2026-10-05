"""Checks for raw EEG information retention and stable numerical reductions."""
import io
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from eeg_comp.neural import NeuralConfig, TrialChannelStandardize
from eeg_comp.neural_v2 import NeuralV2Config, OrderedAdaptiveMean, RawEEGV2


torch.set_num_threads(1)


class RawEEGV2Tests(unittest.TestCase):
    def test_ordered_pool_matches_adaptive_outputs_and_gradients(self):
        torch.manual_seed(41)
        for samples, bins in [(50, 50), (120, 50), (121, 50), (3, 5), (256, 8)]:
            x = torch.randn(2, 3, samples, dtype=torch.float64, requires_grad=True)
            reference_x = x.detach().clone().requires_grad_(True)
            actual = OrderedAdaptiveMean(bins)(x)
            reference = F.adaptive_avg_pool1d(reference_x, bins)
            torch.testing.assert_close(actual, reference, rtol=1e-12, atol=1e-12)
            weights = torch.randn_like(actual)
            (actual * weights).sum().backward()
            (reference * weights).sum().backward()
            torch.testing.assert_close(x.grad, reference_x.grad, rtol=1e-12, atol=1e-12)
        model = RawEEGV2(NeuralV2Config(mode="cue_baseline", task_samples=500))
        self.assertIsInstance(model.network[4], nn.Identity)
        general = RawEEGV2(NeuralV2Config(mode="cue_baseline", task_samples=1200))
        self.assertIsInstance(general.network[4], OrderedAdaptiveMean)

    def test_large_dc_normalization_matches_quantized_float64_reference(self):
        torch.manual_seed(4)
        x = 1_000_000 + torch.randn(3, 8, 1000)
        task64 = x[:, :, 500:].double()
        centered64 = task64 - task64.mean(dim=-1, keepdim=True)
        reference = (centered64 / centered64.square().mean(dim=-1, keepdim=True).sqrt()).float()
        stable = RawEEGV2(NeuralV2Config(task_samples=500)).normalize_task(x)
        native = TrialChannelStandardize()(x[:, :, 500:])
        torch.testing.assert_close(stable, reference, rtol=1e-6, atol=1e-6)
        self.assertGreater(float((native - reference).abs().max()), 1e-3)
        # Baseline normalization needs the same stable reduction, but its
        # scale and mean must come entirely from the baseline.
        baseline64 = x[:, :, :500].double()
        mean64 = baseline64.mean(dim=-1, keepdim=True)
        scale64 = (baseline64 - mean64).square().mean(dim=-1, keepdim=True).sqrt()
        baseline_reference = ((task64 - mean64) / scale64).float()
        for mode in ("phase_baseline", "cue_baseline"):
            actual = RawEEGV2(NeuralV2Config(mode=mode, task_samples=500)).normalize_task(x)
            torch.testing.assert_close(actual, baseline_reference, rtol=1e-6, atol=1e-6)

    def test_baseline_keeps_task_offset_and_amplitude_while_phase_erases_them(self):
        wave = (torch.arange(500).remainder(2) * 2 - 1).float()
        x = torch.cat([wave, wave]).reshape(1, 1, 1000).expand(2, 8, -1).clone()
        offset = x.clone()
        offset[:, :, 500:] += 3
        amplitude = x.clone()
        amplitude[:, :, 500:] *= 4
        stable = RawEEGV2(NeuralV2Config(task_samples=500))
        torch.testing.assert_close(stable.normalize_task(x), stable.normalize_task(offset), rtol=0, atol=0)
        torch.testing.assert_close(stable.normalize_task(x), stable.normalize_task(amplitude), rtol=0, atol=0)
        for mode in ("phase_baseline", "cue_baseline"):
            model = RawEEGV2(NeuralV2Config(mode=mode, task_samples=500))
            original = model.normalize_task(x)
            torch.testing.assert_close(model.normalize_task(offset), original + 3, rtol=0, atol=0)
            torch.testing.assert_close(model.normalize_task(amplitude), original * 4, rtol=0, atol=0)

    def test_common_trial_channel_offsets_and_positive_scales_cancel(self):
        torch.manual_seed(12)
        x = torch.randn(3, 8, 1000)
        scales = torch.rand(3, 8, 1) + 0.5
        offsets = 20 * torch.randn(3, 8, 1)
        for mode in ("phase_stable", "phase_baseline", "cue_baseline"):
            model = RawEEGV2(NeuralV2Config(mode=mode, task_samples=500))
            torch.testing.assert_close(model.normalize_task(x), model.normalize_task(x * scales + offsets),
                                       rtol=2e-5, atol=2e-5)

    def test_predictions_are_independent_of_trial_order_and_other_trials(self):
        torch.manual_seed(17)
        x = torch.randn(3, 8, 1700)
        for mode in ("phase_stable", "phase_baseline", "cue_baseline"):
            model = RawEEGV2(NeuralV2Config(mode=mode)).eval()
            with torch.no_grad():
                batched = model(x)
                single = torch.cat([model(row[None]) for row in x])
                shuffled = model(x[[2, 0, 1]])
            self.assertEqual(tuple(batched.shape), (3,))
            torch.testing.assert_close(batched, single, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(shuffled, batched[[2, 0, 1]], rtol=1e-5, atol=1e-6)
            self.assertFalse(any("running_mean" in key for key in model.state_dict()))

    def test_flat_channels_have_finite_forward_and_backward(self):
        for mode in ("phase_stable", "phase_baseline", "cue_baseline"):
            x = torch.ones(2, 8, 1000, requires_grad=True)
            model = RawEEGV2(NeuralV2Config(mode=mode, task_samples=500)).eval()
            logits = model(x)
            self.assertTrue(torch.isfinite(logits).all())
            logits.square().mean().backward()
            self.assertTrue(torch.isfinite(x.grad).all())
            for parameter in model.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_phase_architecture_parameter_count_and_exact_serialized_reload(self):
        torch.manual_seed(19)
        x = torch.randn(2, 8, 1000)
        for mode in ("phase_stable", "phase_baseline", "cue_baseline"):
            model = RawEEGV2(NeuralV2Config(mode=mode, task_samples=500)).eval()
            if mode.startswith("phase_"):
                self.assertEqual(sum(p.numel() for p in model.parameters()), 1449)
                self.assertIsInstance(model.network.standardize, nn.Identity)
            else:
                self.assertFalse(any(isinstance(layer, (nn.GroupNorm, nn.BatchNorm1d))
                                     for layer in model.modules()))
            stream = io.BytesIO()
            torch.save({"state_dict": model.state_dict(), "configuration": model.configuration()}, stream)
            stream.seek(0)
            payload = torch.load(stream, map_location="cpu", weights_only=True)
            config = payload["configuration"]
            config["phase_config"] = NeuralConfig(**config["phase_config"])
            restored = RawEEGV2(NeuralV2Config(**config)).eval()
            restored.load_state_dict(payload["state_dict"], strict=True)
            with torch.no_grad():
                self.assertTrue(torch.equal(model(x), restored(x)))

    def test_cue_model_learns_offset_signal_erased_by_task_standardization(self):
        torch.manual_seed(23)
        labels = torch.arange(24).remainder(2).float()
        wave = (torch.arange(500).remainder(2) * 2 - 1).float()
        baseline = wave[None, None].expand(24, 8, -1)
        task = baseline + (4 * labels - 2)[:, None, None]
        x = torch.cat([baseline, task], dim=-1)
        stable = RawEEGV2(NeuralV2Config(task_samples=500)).eval()
        normalized = stable.normalize_task(x)
        # Both class waveforms are literally identical after task-only
        # normalization, so no classifier on that representation can separate
        # these balanced labels.
        self.assertTrue(torch.equal(normalized[0], normalized[1]))
        with torch.no_grad():
            phase_logits = stable(x)
        self.assertEqual(float(((phase_logits >= 0) == labels.bool()).float().mean()), 0.5)
        model = RawEEGV2(NeuralV2Config(mode="cue_baseline", task_samples=500, dropout=0))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        initial = float(F.binary_cross_entropy_with_logits(model(x), labels))
        for _ in range(30):
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(model(x), labels)
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                for p in model.parameters()))
            optimizer.step()
        model.eval()
        with torch.no_grad():
            final_logits = model(x)
            final = float(F.binary_cross_entropy_with_logits(final_logits, labels))
        self.assertLess(final, initial * 0.4)
        self.assertEqual(float(((final_logits >= 0) == labels.bool()).float().mean()), 1.0)

    def test_invalid_raw_input_and_configuration_fail_explicitly(self):
        model = RawEEGV2(NeuralV2Config(task_samples=500))
        for bad in (torch.ones(8, 1000), torch.ones(1, 7, 1000), torch.ones(1, 8, 999),
                    torch.ones(1, 8, 1000, dtype=torch.int64), torch.full((1, 8, 1000), float("nan")),
                    torch.full((1, 8, 1000), float("inf"))):
            with self.assertRaises(ValueError):
                model(bad)
        for fields in ({"mode": "unknown"}, {"task_samples": 750}, {"task_samples": 500.0}, {"baseline_samples": 0},
                       {"cue_kernel": 40}, {"cue_bins": 121}, {"dropout": 1},
                       {"dropout": float("nan")}, {"phase_config": NeuralConfig(channels=4)},
                       {"phase_config": NeuralConfig(normalization_eps=float("inf"))},
                       {"phase_config": NeuralConfig(normalization_eps=float("nan"))}):
            with self.assertRaises(ValueError):
                NeuralV2Config(**fields)


if __name__ == "__main__":
    unittest.main()
