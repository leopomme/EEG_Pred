import unittest
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from eeg_comp.data import CHANNELS, CUE_INDEX, Trial, audit_competition, extract_epochs, load_cache, parse_markers


class MarkerTests(unittest.TestCase):
    def test_spelling_and_extra_stop_preserve_labels_and_order(self):
        markers = ["trial_start", None, "rest", None, "cue_stop", "trail_start",
                   "move", "cue_stop", "exp_stop", "cue_stop"]
        trials, report = parse_markers(markers, "train")
        self.assertEqual(trials, [Trial(0, 2, 4, "rest"), Trial(5, 6, 7, "move")])
        self.assertEqual(report["orphan_stop_rows"], [9])

    def test_missing_stop_is_not_silently_repaired(self):
        with self.assertRaisesRegex(ValueError, "Missing cue_stop"):
            parse_markers(["trial_start", "rest", "trail_start", "move", "cue_stop"], "train")

    def test_split_marker_mismatch_fails(self):
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            parse_markers(["trial_start", "move", "cue_stop"], "test")

    def test_unclosed_cue_and_unknown_marker_fail(self):
        for markers in [["trial_start", "cue_start"], ["trial_start", "oops", "cue_start", "cue_stop"]]:
            with self.assertRaises(ValueError):
                parse_markers(markers, "test")

    def test_experiment_restart_inside_trial_fails(self):
        with self.assertRaisesRegex(ValueError, "restarted"):
            parse_markers(["trial_start", "rest", "exp_start", "cue_stop"], "train")


class EpochTests(unittest.TestCase):
    def frame(self, n=2200):
        frame = pd.DataFrame({"time": np.arange(n) / 250.0,
                              **{channel: np.arange(n, dtype=float) + j * 10000
                                 for j, channel in enumerate(CHANNELS)},
                              "Marker_val": [None] * n})
        return frame

    def test_named_channels_and_native_phase_boundaries(self):
        frame = self.frame()
        # Deliberately reorder columns; positional loading would mix channels.
        frame = frame[list(reversed(frame.columns))]
        X, valid, phase, rows, timing = extract_epochs(frame, [Trial(103, 850, 2092, "rest")])
        self.assertEqual(X.shape, (1, 8, 2000))
        np.testing.assert_array_equal(X[0, :, CUE_INDEX], 850 + np.arange(8) * 10000)
        self.assertTrue(valid.all())
        self.assertFalse(phase[0, :3].any())
        self.assertFalse(phase[0, -8:].any())
        self.assertTrue(phase[0, 3:-8].all())
        self.assertEqual(rows[0]["native_cue_samples"], 1242)
        self.assertTrue(rows[0]["task_4p8s_valid"])
        self.assertFalse(rows[0]["task_5s_valid"])

    def test_missing_file_support_is_nan_and_explicit(self):
        X, valid, phase, rows, _ = extract_epochs(self.frame(), [Trial(0, 700, 1950, "")])
        self.assertFalse(valid[0, :50].any())
        self.assertTrue(np.isnan(X[0, :, :50]).all())
        self.assertTrue(valid[0, 50:].all())

    def test_never_cross_timestamp_gap_or_nonfinite_signal(self):
        frame = self.frame()
        frame.loc[600:, "time"] += 2
        frame.loc[950, "Oz"] = np.nan
        X, valid, phase, rows, timing = extract_epochs(frame, [Trial(100, 850, 2100, "move")])
        self.assertEqual(timing["discontinuity_rows"], [600])
        self.assertFalse(valid[0, :500].any())
        self.assertFalse(valid[0, 850])
        self.assertTrue(np.isnan(X[0, :, 850]).all())
        self.assertTrue(valid[0, 500:850].all())

    def test_native_clock_is_preserved_without_resampling(self):
        frame = self.frame()
        frame["time"] = 53704 + np.arange(len(frame)) * 0.00399569876
        X, valid, phase, rows, timing = extract_epochs(frame, [Trial(100, 850, 2100, "rest")])
        np.testing.assert_array_equal(X[0, 0], np.arange(100, 2100))
        self.assertAlmostEqual(timing["median_dt_seconds"], 0.00399569876, places=10)
        self.assertAlmostEqual(rows[0]["cue_seconds"], 1250 * 0.00399569876)
        self.assertTrue(valid.all())

    def test_out_of_range_boundary_fails(self):
        with self.assertRaisesRegex(ValueError, "boundaries"):
            extract_epochs(self.frame(), [Trial(100, 850, 2200, "rest")])

    def test_cache_order_and_label_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            competition = root / "data" / "within_subject"
            for folder in ["Training set", "Single subject test set"]:
                (competition / folder).mkdir(parents=True)
            for folder, filename, cue in [
                ("Training set", "S001-001_eeg.csv", "move"),
                ("Single subject test set", "S002_test.csv", "cue_start"),
                ("Single subject test set", "S001_test.csv", "cue_start"),
            ]:
                frame = self.frame()
                frame.loc[100, "Marker_val"] = "trial_start"
                frame.loc[850, "Marker_val"] = cue
                frame.loc[2100, "Marker_val"] = "cue_stop"
                frame.to_csv(competition / folder / filename, index=False)
            report = audit_competition(root / "data", root / "cache", "within_subject")
            arrays, metadata = load_cache(root / "cache", "within_subject")
            np.testing.assert_array_equal(arrays["y"], [1, -1, -1])
            self.assertEqual(metadata.loc[metadata.split == "test", "subject"].tolist(), ["S001", "S002"])
            self.assertEqual(report["test_trials"], 2)
            self.assertEqual(metadata.test_order.tolist(), [-1, 0, 1])
            metadata.loc[0, "label"] = "rest"
            metadata.to_csv(root / "cache" / "within_subject" / "metadata.csv", index=False)
            with self.assertRaisesRegex(ValueError, "label encoding"):
                load_cache(root / "cache", "within_subject")


if __name__ == "__main__":
    unittest.main()
