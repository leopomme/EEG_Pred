"""Behavioral checks for the neural information boundary and optimizer path."""
import argparse
import tempfile
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from eeg_comp.neural import NeuralConfig, PhaseConvNet, TrialChannelStandardize, binary_metrics
from scripts.run_neural import (CHANNELS, checkpoint, inference_from_checkpoints,
                                load_cache, make_folds, predict, weighted_protocol_loss)


torch.set_num_threads(1)


def synthetic_metadata(competition="cross_subject", n_people=17):
    rows, labels = [], []
    for person in range(1, n_people + 1):
        for run, count in ((1, 6), (2, 6), (3, 10)):
            for trial in range(count):
                label = trial % 2
                rows.append({"epoch_index": len(rows), "competition": competition,
                             "split": "train", "subject": f"S{person:03d}", "run": run,
                             "label": "move" if label else "rest", "test_order": -1})
                labels.append(label)
    return pd.DataFrame(rows), np.asarray(labels)


class NetworkTests(unittest.TestCase):
    def test_runtime_standardization_invariance_and_flat_channel_gradients(self):
        normalizer = TrialChannelStandardize()
        torch.manual_seed(31)
        x = torch.randn(3, 8, 256)
        original = normalizer(x)
        shifted = normalizer(x * 3 + 42)
        torch.testing.assert_close(original, shifted, rtol=2e-5, atol=2e-5)
        flat = torch.ones(2, 8, 256, requires_grad=True)
        normalized = normalizer(flat)
        self.assertEqual(float(normalized.abs().max()), 0)
        normalized.sum().backward()
        self.assertTrue(torch.isfinite(flat.grad).all())

    def test_predictions_do_not_depend_on_other_trials_or_order(self):
        torch.manual_seed(9)
        model = PhaseConvNet().eval()
        x = torch.randn(3, 8, 256)
        with torch.no_grad():
            batch = model(x)
            solo = torch.cat([model(row[None]) for row in x])
            shuffled = model(x[[2, 0, 1]])
        torch.testing.assert_close(batch, solo, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(shuffled, batch[[2, 0, 1]], rtol=1e-5, atol=1e-6)
        self.assertFalse(any("running_mean" in key for key in model.state_dict()))

    def test_tiny_phase_task_learns_with_finite_gradients(self):
        torch.manual_seed(11)
        config = NeuralConfig(temporal_filters=2, spatial_multiplier=1, temporal_kernel=15,
                              separable_kernel=7, phase_bins=4, dropout=0)
        model = PhaseConvNet(config)
        y = torch.arange(8).remainder(2).float()
        template = torch.sin(torch.linspace(0, 3 * np.pi, 256))
        x = (2 * y[:, None, None] - 1) * template[None, None, :].expand(8, 8, -1)
        x = x + 0.05 * torch.randn_like(x)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
        initial = float(F.binary_cross_entropy_with_logits(model(x), y))
        for _ in range(30):
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(model(x), y)
            loss.backward()
            self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                                for parameter in model.parameters()))
            optimizer.step()
        final = float(F.binary_cross_entropy_with_logits(model(x), y))
        self.assertLess(final, initial * 0.4)
        self.assertEqual(float(((model(x) >= 0) == y.bool()).float().mean()), 1.0)

    def test_auc_ties_and_missing_classes(self):
        self.assertEqual(binary_metrics(np.array([0, 1]), np.array([0.5, 0.5]))["auc"], 0.5)
        self.assertEqual(binary_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))["auc"], 1)
        self.assertIsNone(binary_metrics(np.zeros(2), np.array([0.1, 0.2]))["auc"])
        with self.assertRaises(ValueError):
            binary_metrics(np.array([0, -1]), np.array([0.1, 0.2]))


class SplitTests(unittest.TestCase):
    def test_cross_source_people_and_confirmation_are_isolated(self):
        meta, y = synthetic_metadata()
        folds = make_folds(meta, y, "cross_subject", 20261003)
        self.assertEqual([len(meta.iloc[fold.query].subject.unique()) for fold in folds], [3, 3, 3, 3, 3, 2])
        confirmation = set(meta.iloc[folds[-1].query].subject)
        all_query = np.concatenate([fold.query for fold in folds])
        np.testing.assert_array_equal(np.sort(all_query), np.arange(len(meta)))
        for fold in folds[:-1]:
            self.assertFalse(set(meta.iloc[fold.train].subject) & confirmation)
            self.assertFalse(set(meta.iloc[fold.inner_train].subject) & set(meta.iloc[fold.inner_valid].subject))
            self.assertFalse(set(meta.iloc[fold.query].subject) & set(meta.iloc[fold.train].subject))
        self.assertEqual(len(meta.iloc[folds[-1].train].subject.unique()), 15)

    def test_within_only_own_person_and_untouched_run3_query(self):
        meta, y = synthetic_metadata("within_subject")
        folds = make_folds(meta, y, "within_subject", 20261003)
        self.assertEqual(len(folds), 34)
        for fold in folds:
            joined = np.concatenate([fold.train, fold.query])
            self.assertEqual(set(meta.iloc[joined].subject), {fold.subject})
            self.assertEqual(set(meta.iloc[fold.query].run), {3})
            self.assertEqual(set(meta.iloc[fold.inner_valid].run), {2})
            self.assertFalse(np.intersect1d(fold.train, fold.query).size)
            self.assertEqual(int((meta.iloc[fold.train].run == 3).sum()), 5)
        # Another person's labels must not change S001's support/query or seed.
        changed = y.copy()
        changed[meta.subject != "S001"] = 1 - changed[meta.subject != "S001"]
        repeat = make_folds(meta, changed, "within_subject", 20261003, subjects=["S001"])
        for original, isolated in zip(folds[:2], repeat):
            self.assertEqual(original.seed, isolated.seed)
            for part in ("train", "query", "inner_train", "inner_valid"):
                np.testing.assert_array_equal(getattr(original, part), getattr(isolated, part))

    def test_within_missing_calibration_run_stays_person_specific(self):
        meta, y = synthetic_metadata("within_subject", 1)
        keep = meta.run != 2
        meta, y = meta.loc[keep].reset_index(drop=True), y[keep]
        for fold in make_folds(meta, y, "within_subject", 20261003):
            self.assertEqual(set(meta.iloc[fold.inner_valid].run), {1})
            # Three examples per class split as two for training and one for
            # validation, so each calibration half remains class-balanced.
            self.assertEqual(len(fold.inner_valid), 2)
            np.testing.assert_array_equal(np.bincount(y[fold.inner_valid]), [1, 1])
            self.assertFalse(np.intersect1d(fold.query, fold.inner_train).size)

    def test_protocol_selection_loss_uses_declared_run_weights(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.1, 0.1, 0.5, 0.5])
        runs = np.array([1, 1, 3, 3])
        expected = (-np.log(0.9) - np.log(0.5)) / 2
        self.assertAlmostEqual(weighted_protocol_loss(y, p, runs, {1: 0.5, 3: 0.5}), expected)


class CacheAndInferenceTests(unittest.TestCase):
    def make_cache(self, directory):
        meta, _ = synthetic_metadata(n_people=1)
        meta = meta.iloc[:2].copy()
        for i in range(2):
            meta.loc[len(meta)] = [len(meta), "cross_subject", "test", "T001", 1, "", i]
        meta.to_csv(directory / "metadata.csv", index=False)
        rng = np.random.default_rng(2)
        arrays = {"X": rng.standard_normal((4, 8, 2000)).astype(np.float32),
                  "valid_mask": np.ones((4, 2000), bool), "phase_valid_mask": np.ones((4, 2000), bool),
                  "fs": np.array(250), "cue_index": np.array(750),
                  "channels": np.asarray(CHANNELS), "y": np.array([0, 1, -1, -1])}
        np.savez(directory / "epochs.npz", **arrays)
        return arrays

    def test_cache_contract_rejects_wrong_competition_and_invalid_crop(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            arrays = self.make_cache(directory)
            x, y, _, _ = load_cache(directory, "cross_subject", "task0_2")
            self.assertEqual(x.shape, (4, 8, 500))
            np.testing.assert_array_equal(y, [0, 1, -1, -1])
            with self.assertRaisesRegex(ValueError, "competition"):
                load_cache(directory, "within_subject", "task0_2")
            arrays["phase_valid_mask"][3, 900] = False
            np.savez(directory / "epochs.npz", **arrays)
            with self.assertRaisesRegex(ValueError, "no automatic padding"):
                load_cache(directory, "cross_subject", "task0_2")

    def test_saved_final_model_reproduces_inductive_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self.make_cache(directory)
            x, y, meta, info = load_cache(directory, "cross_subject", "task0_2")
            experiment = directory / "experiment"
            (experiment / "final").mkdir(parents=True)
            output = directory / "inference"
            output.mkdir()
            args = argparse.Namespace(competition="cross_subject", window="task0_2",
                                      inference_from=experiment, output=output, batch_size=3)
            model = PhaseConvNet().eval()
            checkpoint(experiment / "final/all_sources.pt", model, args, epochs=1,
                       seed=3, train=np.array([0, 1]), cache_info=info)
            expected = predict(model, x, np.array([2, 3]), torch.device("cpu"), 1)[1]
            inference_from_checkpoints(args, x, meta, info, torch.device("cpu"))
            result = pd.read_csv(output / "test_predictions.csv")
            np.testing.assert_allclose(result.p_move, expected, rtol=1e-6, atol=1e-6)
            changed_signals = dict(info, epochs_sha256="different")
            with self.assertRaisesRegex(ValueError, "epochs_sha256"):
                inference_from_checkpoints(args, x, meta, changed_signals, torch.device("cpu"))
            info["metadata_sha256"] = "different"
            with self.assertRaisesRegex(ValueError, "cache contract"):
                inference_from_checkpoints(args, x, meta, info, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
