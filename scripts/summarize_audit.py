#!/usr/bin/env python3
"""Validate one epoch cache and add label-free trial quality/timing evidence."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eeg_comp.data import CHANNELS, COMPETITIONS, CUE_INDEX, load_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--competition", required=True, choices=COMPETITIONS)
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args()
    root = args.artifact_root / args.competition
    arrays, metadata = load_cache(args.artifact_root, args.competition)
    X = arrays["X"]
    if not np.array_equal(np.isfinite(X).all(axis=1), arrays["valid_mask"]):
        raise ValueError("Cache finiteness does not match recording mask")
    report = json.loads((root / "audit.json").read_text())
    records, timing_records, flatlines = [], [], []
    for row in metadata.itertuples(index=False):
        base = X[row.epoch_index, :, CUE_INDEX - 500:CUE_INDEX].astype(np.float64)
        task = X[row.epoch_index, :, CUE_INDEX:CUE_INDEX + 1200].astype(np.float64)
        for c, channel in enumerate(CHANNELS):
            records.append({"epoch_index": row.epoch_index, "split": row.split,
                            "subject": row.subject, "run": row.run, "trial_index": row.trial_index,
                            "channel": channel, "baseline_std": float(np.std(base[c])),
                            "task_std": float(np.std(task[c])),
                            "task_peak_to_peak": float(np.ptp(task[c])),
                            "task_max_abs_step": float(np.max(np.abs(np.diff(task[c]))))})
    for f in report["files"]:
        m = metadata[metadata.file == f["file"]]
        delta = np.diff(m.cue_time)
        timing_records.append({"file": f["file"], "split": f["split"],
                               "first_fixation_after_file_start_seconds": float(m.trial_start_time.iloc[0] - f["first_time"]),
                               "last_cue_stop_to_file_end_seconds": float(f["last_time"] - m.cue_stop_time.iloc[-1]),
                               "cue_interval_min_seconds": float(delta.min()) if len(delta) else None,
                               "cue_interval_max_seconds": float(delta.max()) if len(delta) else None,
                               "cue_intervals_exceeding_15s": int((delta > 15).sum())})
        if max(c["longest_constant_samples"] for c in f["channels"].values()) >= 8:
            frame = pd.read_csv(args.data_root / args.competition / f["file"])
            eeg = frame.loc[:, CHANNELS].to_numpy(dtype=np.float64)
            for c, channel in enumerate(CHANNELS):
                same = np.diff(eeg[:, c]) == 0
                edge = np.diff(np.r_[False, same, False].astype(int))
                for start, stop in zip(np.flatnonzero(edge == 1), np.flatnonzero(edge == -1)):
                    if stop - start + 1 >= 8:
                        overlap = m[(m.trial_start_row <= stop) & (m.cue_stop_row > start)]
                        flatlines.append({"file": f["file"], "channel": channel, "start_row": int(start),
                                          "length_samples": int(stop - start + 1),
                                          "trial_indices_overlapping": overlap.trial_index.tolist()})
    quality = pd.DataFrame(records)
    quality.to_csv(root / "trial_quality.csv", index=False)
    summary = {"competition": args.competition, "verified_cache_alignment": True,
               "quality_windows_nominal_seconds": {"baseline": [-2, 0], "task": [0, 4.8]},
               "quality_units": "Unverified numeric CSV units; amplitude offsets are removed for std/range/steps",
               "file_marker_timing": timing_records,
               "constant_segments_at_least_8_samples": flatlines,
               "largest_task_peak_to_peak": quality.nlargest(16, "task_peak_to_peak").to_dict(orient="records"),
               "largest_task_std": quality.nlargest(16, "task_std").to_dict(orient="records")}
    (root / "quality_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Verified {args.competition}: {len(metadata)} cache rows; {len(quality)} trial/channel quality records", flush=True)


if __name__ == "__main__":
    main()
