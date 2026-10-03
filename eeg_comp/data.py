"""Marker-led loading with an explicit native-sample epoch contract.

Only named EEG columns enter X. Time, marker spellings and file metadata are
audit/segmentation information, never predictor columns. Each competition is
read independently; no data matching across competitions is implemented.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

CHANNELS = ("Fz", "C3", "Cz", "C4", "PO7", "Pz", "PO8", "Oz")
FS = 250
CUE_INDEX = 750
EPOCH_SAMPLES = 2000
LABELS = {"rest": 0, "move": 1}
COMPETITIONS = ("within_subject", "cross_subject")
START_MARKERS = {"trial_start", "trail_start"}
KNOWN_MARKERS = START_MARKERS | {"rest", "move", "cue_start", "cue_stop", "exp_start", "exp_stop"}


@dataclass(frozen=True)
class Trial:
    trial_start: int
    cue: int
    stop: int
    label: str


def parse_markers(markers, split: str) -> tuple[list[Trial], dict]:
    """Pair each cue with its own fixation start and first subsequent cue_stop.

    Extra stop markers outside a trial are reported, not converted into trials.
    Missing boundaries, multiple cues, mixed labeled/test cues and unknown
    marker types are rejected rather than silently losing or relabeling trials.
    """
    if split not in {"train", "test"}:
        raise ValueError(f"Invalid split: {split}")
    trials: list[Trial] = []
    counts: Counter = Counter()
    orphan_stops, unused_starts = [], []
    start = cue = None
    label = ""
    for row, value in enumerate(markers):
        if pd.isna(value):
            continue
        marker = str(value).strip().lower()
        if not marker:
            continue
        counts[marker] += 1
        if marker not in KNOWN_MARKERS:
            raise ValueError(f"Unknown marker {marker!r} at row {row}")
        if marker in START_MARKERS:
            if cue is not None:
                raise ValueError(f"Missing cue_stop before trial start at row {row}")
            if start is not None:
                unused_starts.append(start)
            start = row
        elif marker in {"rest", "move", "cue_start"}:
            if (split == "train") != (marker in LABELS):
                raise ValueError(f"Unexpected {marker} in {split} at row {row}")
            if cue is not None:
                raise ValueError(f"Multiple cues before cue_stop at row {row}")
            if start is None:
                raise ValueError(f"Cue without trial start at row {row}")
            cue, label = row, marker if split == "train" else ""
        elif marker == "cue_stop":
            if cue is None:
                orphan_stops.append(row)
            else:
                trials.append(Trial(start, cue, row, label))
                start = cue = None
        elif marker == "exp_stop":
            if cue is not None:
                raise ValueError(f"Experiment ended before cue_stop at row {row}")
            if start is not None:
                unused_starts.append(start)
                start = None
        elif marker == "exp_start" and (start is not None or cue is not None):
            raise ValueError(f"Experiment restarted inside a trial at row {row}")
    if cue is not None:
        raise ValueError("File ended with an unclosed cue")
    if start is not None:
        unused_starts.append(start)
    if not trials:
        raise ValueError("No complete trials found")
    return trials, {"marker_counts": dict(counts), "orphan_stop_rows": orphan_stops,
                    "unused_start_rows": unused_starts}


def identify_file(path: Path, competition: str, split: str) -> tuple[str, int]:
    if competition not in COMPETITIONS:
        raise ValueError(competition)
    patterns = {
        "train": r"(S\d{3})-(\d{3})_eeg\.csv",
        "cross_subject": r"(S\d{3})_test_(\d{3})_eeg\.csv",
        "within_subject": r"(S\d{3})_test\.csv",
    }
    match = re.fullmatch(patterns["train" if split == "train" else competition], path.name)
    if match is None:
        raise ValueError(f"Unexpected EEG file name: {path.name}")
    return match[1], int(match[2]) if len(match.groups()) == 2 else 3


def extract_epochs(frame: pd.DataFrame, trials: list[Trial]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict], dict]:
    """Extract [-750,1250) native rows around cue; no temporal resampling.

    X contains real recorded samples outside the exact fixation/task bounds,
    flagged separately by phase_valid_mask. Missing/nonfinite samples and
    samples across a timestamp discontinuity from the cue are NaN, with
    valid_mask=False. A discontinuity is <=0 or >1.5 * median positive dt.
    Phase boundaries are half-open: baseline starts at trial_start and task
    ends immediately before cue_stop. Full native phase lengths are retained.
    """
    required = {"time", "Marker_val", *CHANNELS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    time = frame["time"].to_numpy(dtype=np.float64)
    eeg = frame.loc[:, CHANNELS].to_numpy(dtype=np.float32)
    if len(time) < 2 or not np.isfinite(time).all():
        raise ValueError("Recording must have at least two finite timestamps")
    dt = np.diff(time)
    positive = dt[dt > 0]
    if not len(positive):
        raise ValueError("Recording timestamps never increase")
    median_dt = float(np.median(positive))
    if not 0.0035 < median_dt < 0.0045:
        raise ValueError(f"Unexpected sampling interval: {median_dt}")
    discontinuity = (dt <= 0) | (dt > 1.5 * median_dt)
    segments = np.r_[0, np.cumsum(discontinuity)]
    finite_rows = np.isfinite(eeg).all(axis=1)
    X = np.full((len(trials), len(CHANNELS), EPOCH_SAMPLES), np.nan, dtype=np.float32)
    valid = np.zeros((len(trials), EPOCH_SAMPLES), dtype=bool)
    phase_valid = valid.copy()
    metadata = []
    offsets = np.arange(-CUE_INDEX, EPOCH_SAMPLES - CUE_INDEX)
    for i, trial in enumerate(trials):
        if not 0 <= trial.trial_start < trial.cue < trial.stop < len(frame):
            raise ValueError(f"Invalid native trial boundaries: {trial}")
        rows = trial.cue + offsets
        inside = (rows >= 0) & (rows < len(frame))
        idx = np.flatnonzero(inside)
        valid[i, idx] = finite_rows[rows[idx]] & (segments[rows[idx]] == segments[trial.cue])
        good = np.flatnonzero(valid[i])
        X[i][:, good] = eeg[rows[good]].T
        phase_valid[i] = valid[i] & (rows >= trial.trial_start) & (rows < trial.stop)
        metadata.append({
            "trial_index": i, "trial_start_row": trial.trial_start,
            "cue_row": trial.cue, "cue_stop_row": trial.stop,
            "trial_start_time": float(time[trial.trial_start]),
            "cue_time": float(time[trial.cue]), "cue_stop_time": float(time[trial.stop]),
            "label": trial.label, "y": LABELS.get(trial.label, -1),
            "baseline_samples": trial.cue - trial.trial_start,
            "native_cue_samples": trial.stop - trial.cue,
            "baseline_seconds": float(time[trial.cue] - time[trial.trial_start]),
            "cue_seconds": float(time[trial.stop] - time[trial.cue]),
            "valid_samples": int(valid[i].sum()),
            "phase_valid_samples": int(phase_valid[i].sum()),
            "baseline_2s_valid": bool(phase_valid[i, CUE_INDEX - 500:CUE_INDEX].all()),
            "task_4p8s_valid": bool(phase_valid[i, CUE_INDEX:CUE_INDEX + 1200].all()),
            "task_5s_valid": bool(phase_valid[i, CUE_INDEX:].all()),
        })
    timing = {
        "median_dt_seconds": median_dt, "effective_fs_hz": 1.0 / median_dt,
        "dt_min_seconds": float(dt.min()), "dt_max_seconds": float(dt.max()),
        "dt_std_seconds": float(dt.std()), "discontinuity_rows": (np.flatnonzero(discontinuity) + 1).tolist(),
        "nonpositive_intervals": int((dt <= 0).sum()),
        "gap_intervals": int((dt > 1.5 * median_dt).sum()),
        "max_abs_dt_deviation_seconds": float(np.max(np.abs(dt - median_dt))),
        "first_time": float(time[0]), "last_time": float(time[-1]),
    }
    return X, valid, phase_valid, metadata, timing


def channel_quality(frame: pd.DataFrame) -> dict:
    """Descriptive quality statistics in file units; no artifact cleaning."""
    result = {}
    for channel in CHANNELS:
        values = frame[channel].to_numpy(dtype=np.float64)
        finite = values[np.isfinite(values)]
        if not len(finite):
            result[channel] = {"finite_fraction": 0.0}
            continue
        diff = np.diff(values)
        equal = diff == 0
        edges = np.diff(np.r_[False, equal, False].astype(int))
        starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        q = np.quantile(finite, [.001, .01, .5, .99, .999])
        result[channel] = {
            "finite_fraction": float(len(finite) / len(values)),
            "min": float(finite.min()), "max": float(finite.max()),
            "q001": float(q[0]), "q01": float(q[1]), "median": float(q[2]),
            "q99": float(q[3]), "q999": float(q[4]), "std": float(finite.std()),
            "max_abs_step": float(np.nanmax(np.abs(diff))),
            "equal_adjacent_fraction": float(equal.mean()),
            "longest_constant_samples": int((stops - starts).max() + 1) if len(starts) else 1,
            "fraction_at_min": float(np.mean(values == finite.min())),
            "fraction_at_max": float(np.mean(values == finite.max())),
        }
    return result


def competition_paths(data_root: Path, competition: str) -> list[tuple[str, Path]]:
    if competition not in COMPETITIONS:
        raise ValueError(competition)
    root = data_root / competition
    test_folder = "Single subject test set" if competition == "within_subject" else "Cross participant test set"
    paths = [(split, p) for split, folder in [("train", "Training set"), ("test", test_folder)]
             for p in sorted((root / folder).glob("*.csv"))]
    if {split for split, _ in paths} != {"train", "test"}:
        raise FileNotFoundError(f"Missing train/test files under {root}")
    return paths


def audit_competition(data_root: Path, output_root: Path, competition: str) -> dict:
    """Read and write exactly one competition; dictionaries never span tasks."""
    paths = competition_paths(data_root, competition)
    arrays, masks, phase_masks, metadata, file_reports = [], [], [], [], []
    file_hashes, epoch_hashes = {}, {}
    duplicate_files, duplicate_epochs = [], []
    test_order = 0
    for split, path in paths:
        subject, run = identify_file(path, competition, split)
        frame = pd.read_csv(path)
        trials, marker_report = parse_markers(frame["Marker_val"], split)
        X, valid, phase_valid, rows, timing = extract_epochs(frame, trials)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in file_hashes:
            duplicate_files.append([file_hashes[digest], path.name])
        file_hashes[digest] = path.name
        for local_index, row in enumerate(rows):
            row.update({"epoch_index": len(metadata), "competition": competition,
                        "split": split, "subject": subject, "run": run,
                        "file": path.relative_to(data_root / competition).as_posix(),
                        "test_order": test_order if split == "test" else -1})
            if split == "test":
                test_order += 1
            # Hash original float64 numeric values, before cache quantization.
            # Compare recorded EEG only within this competition.
            cue_rows = frame.iloc[trials[local_index].cue:trials[local_index].cue + 1000]
            cue_digest = hashlib.sha256(cue_rows.loc[:, CHANNELS].to_numpy(dtype=np.float64).tobytes()).hexdigest()
            reference = {"file": row["file"], "trial_index": row["trial_index"], "split": split}
            if cue_digest in epoch_hashes:
                duplicate_epochs.append([epoch_hashes[cue_digest], reference])
            epoch_hashes[cue_digest] = reference
            metadata.append(row)
        file_reports.append({"file": path.relative_to(data_root / competition).as_posix(),
                             "split": split, "subject": subject, "run": run,
                             "rows": len(frame), "columns": frame.columns.tolist(), "sha256": digest,
                             "n_trials": len(trials), "labels": dict(Counter(t.label for t in trials if t.label)),
                             **marker_report, **timing, "channels": channel_quality(frame),
                             "overlapping_native_trials": [i for i in range(1, len(trials))
                                                           if trials[i].trial_start <= trials[i - 1].stop],
                             "overlapping_cached_epochs": [i for i in range(1, len(trials))
                                                            if trials[i].cue - trials[i - 1].cue < EPOCH_SAMPLES]})
        arrays.append(X); masks.append(valid); phase_masks.append(phase_valid)
        print(f"{competition} {path.name}: {len(trials)} trials", flush=True)
    meta = pd.DataFrame(metadata)
    X = np.concatenate(arrays)
    valid = np.concatenate(masks)
    phase_valid = np.concatenate(phase_masks)
    chronology = []
    for subject in sorted(meta.subject.unique()):
        train3 = meta[(meta.subject == subject) & (meta.split == "train") & (meta.run == 3)]
        test3 = meta[(meta.subject == subject) & (meta.split == "test") & (meta.run == 3)]
        if len(train3) and len(test3):
            train_report = next(r for r in file_reports if r["subject"] == subject and r["run"] == 3 and r["split"] == "train")
            test_report = next(r for r in file_reports if r["subject"] == subject and r["run"] == 3 and r["split"] == "test")
            join_dt = test_report["first_time"] - train_report["last_time"]
            chronology.append({"subject": subject, "train_trials": len(train3), "test_trials": len(test3),
                               "all_labeled_cues_before_test": bool(train3.cue_time.max() < test3.cue_time.min()),
                               "file_join_dt_seconds": join_dt,
                               "files_abut_one_sample": bool(abs(join_dt / train_report["median_dt_seconds"] - 1) < .01),
                               "recording_time_ranges_overlap": bool(test_report["first_time"] <= train_report["last_time"]),
                               "train_internal_discontinuities": len(train_report["discontinuity_rows"]),
                               "test_internal_discontinuities": len(test_report["discontinuity_rows"]),
                               "last_train_cue_time": float(train3.cue_time.max()),
                               "first_test_cue_time": float(test3.cue_time.min()),
                               "sum_observed_trials": len(train3) + len(test3)})
    counts = meta.groupby(["split", "subject", "run"], sort=True).agg(
        trials=("epoch_index", "size"), rest=("label", lambda s: (s == "rest").sum()),
        move=("label", lambda s: (s == "move").sum())).reset_index()
    report = {
        "competition": competition, "cache_version": 1,
        "channels": list(CHANNELS), "nominal_fs": FS,
        "amplitude_units": "Unverified; CSV contains numeric raw device units with large DC offsets",
        "acquisition_reference": "Not established from delivered CSVs",
        "predictor_columns": list(CHANNELS), "files": file_reports,
        "counts": counts.to_dict(orient="records"), "run3_chronology": chronology,
        "exact_duplicate_files_within_competition": duplicate_files,
        "exact_duplicate_4s_epochs_within_competition": duplicate_epochs,
        "train_trials": int((meta.split == "train").sum()), "test_trials": int((meta.split == "test").sum()),
        "trial_lengths": {col: {"min": float(meta[col].min()), "max": float(meta[col].max()),
                                 "median": float(meta[col].median())}
                          for col in ["baseline_samples", "native_cue_samples", "baseline_seconds", "cue_seconds"]},
        "invalid_recorded_samples": int((~valid).sum()),
        "invalid_phase_samples": int((~phase_valid).sum()),
        "invalid_baseline_2s_trials": int((~meta.baseline_2s_valid).sum()),
        "invalid_task_4p8s_trials": int((~meta.task_4p8s_valid).sum()),
        "invalid_task_5s_trials": int((~meta.task_5s_valid).sum()),
        "sample_submission_available": False,
        "test_order_contract": "0-based, lexicographically sorted test filenames then chronological cue order",
        "run_mixture": meta.groupby(["split", "run"]).size().rename("trials").reset_index().to_dict(orient="records"),
        "subjects": {split: sorted(meta.loc[meta.split == split, "subject"].unique().tolist())
                     for split in ["train", "test"]},
    }
    output = output_root / competition
    output.mkdir(parents=True, exist_ok=True)
    # Uncompressed cache favors repeatable short HPC reads over CPU compression.
    np.savez(output / "epochs.npz", X=X, y=meta.y.to_numpy(dtype=np.int8),
             valid_mask=valid, phase_valid_mask=phase_valid, fs=np.array(FS),
             cue_index=np.array(CUE_INDEX), channels=np.array(CHANNELS), cache_version=np.array(1))
    meta.to_csv(output / "metadata.csv", index=False)
    counts.to_csv(output / "counts.csv", index=False)
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def load_cache(artifact_root: str | Path, competition: str):
    """Load one competition and verify row/label/order alignment."""
    if competition not in COMPETITIONS:
        raise ValueError(competition)
    root = Path(artifact_root) / competition
    with np.load(root / "epochs.npz", allow_pickle=False) as cache:
        arrays = {key: cache[key] for key in cache.files}
    if int(arrays["cache_version"]) != 1 or int(arrays["fs"]) != FS or int(arrays["cue_index"]) != CUE_INDEX:
        raise ValueError("Unsupported cache contract")
    if arrays["X"].shape[1:] != (len(CHANNELS), EPOCH_SAMPLES) or arrays["X"].dtype != np.float32:
        raise ValueError("Invalid epoch shape or dtype")
    if tuple(arrays["channels"]) != CHANNELS:
        raise ValueError("Invalid cached channel order")
    for key in ["valid_mask", "phase_valid_mask"]:
        if arrays[key].shape != (len(arrays["X"]), EPOCH_SAMPLES) or arrays[key].dtype != bool:
            raise ValueError(f"Invalid {key}")
    if (arrays["phase_valid_mask"] & ~arrays["valid_mask"]).any():
        raise ValueError("Phase validity extends beyond recording validity")
    metadata = pd.read_csv(root / "metadata.csv", keep_default_na=False)
    if len(metadata) != len(arrays["X"]) or not np.array_equal(metadata.epoch_index, np.arange(len(metadata))):
        raise ValueError("Metadata and epoch cache are not aligned")
    if not (metadata.competition == competition).all() or not np.array_equal(metadata.y, arrays["y"]):
        raise ValueError("Competition or label mismatch")
    if not metadata.split.isin(["train", "test"]).all():
        raise ValueError("Unknown cache split")
    expected_y = metadata.label.map(LABELS).fillna(-1).to_numpy(dtype=np.int8)
    if not np.array_equal(expected_y, arrays["y"]) or not (metadata.loc[metadata.split == "test", "y"] == -1).all():
        raise ValueError("Invalid label encoding")
    if not metadata.loc[metadata.split == "train", "y"].isin([0, 1]).all():
        raise ValueError("Unlabeled training rows")
    test_order = metadata.loc[metadata.split == "test", "test_order"].to_numpy()
    if not np.array_equal(test_order, np.arange(len(test_order))):
        raise ValueError("Invalid test ordering")
    return arrays, metadata
