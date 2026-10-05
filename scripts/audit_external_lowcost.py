#!/usr/bin/env python3
"""Audit uploaded ds003810 metadata/EDF headers without loading EEG samples.

The original dataset is read-only. This makes no epochs, predictions or fitted
parameters. EDF annotation reads are allowed; ``get_data`` is never called.
Outputs must be fresh and outside the dataset tree.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import sys

import mne
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "extraernal_doc_eg_kagle/Datasets/ds003810_lowcost_mi_rest/v1.0.2"
DEFAULT_OUTPUT = ROOT / "artifacts/external/ds003810_audit"
CUE_TO_CODE = {"OVTK_GDF_Right": 7, "OVTK_GDF_Tongue": 9}
CODE_TO_LABEL = {7: "move", 9: "rest"}
TARGET_CHANNELS = ["Fz", "C3", "Cz", "C4", "PO7", "Pz", "PO8", "Oz"]
WINDOWS = {"baseline_minus2_0": (-2.0, 0.0), "task_0_2": (0.0, 2.0),
           "task_0_3": (0.0, 3.0), "task_0_38": (0.0, 3.8),
           "task_0_4": (0.0, 4.0), "task_0_48": (0.0, 4.8)}


def json_read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def edf_header(path: Path) -> dict:
    """Read the EDF fixed/per-signal header only; never sample records."""
    with path.open("rb") as handle:
        fixed = handle.read(256)
        if len(fixed) != 256:
            raise ValueError(f"Truncated EDF header: {path}")
        header_bytes = int(fixed[184:192].decode("ascii").strip())
        signal_count = int(fixed[252:256].decode("ascii").strip())
        if header_bytes != 256 + 256 * signal_count:
            raise ValueError(f"Unexpected EDF header size: {path}")
        per_signal = handle.read(header_bytes - 256)
    if len(per_signal) != header_bytes - 256:
        raise ValueError(f"Truncated EDF signal header: {path}")
    widths = [("label", 16), ("transducer", 80), ("physical_dimension", 8),
              ("physical_minimum", 8), ("physical_maximum", 8),
              ("digital_minimum", 8), ("digital_maximum", 8),
              ("prefiltering", 80), ("samples_per_record", 8), ("reserved", 32)]
    signals = [dict() for _ in range(signal_count)]
    offset = 0
    for field, width in widths:
        for index in range(signal_count):
            start = offset + index * width
            signals[index][field] = per_signal[start:start+width].decode("latin-1").strip()
        offset += signal_count * width
    return {
        "header_bytes": header_bytes, "signal_count_including_annotations": signal_count,
        "header_sha256": hashlib.sha256(fixed + per_signal).hexdigest(),
        "reserved": fixed[192:236].decode("latin-1").strip(),
        "data_records": int(fixed[236:244].decode("ascii").strip()),
        "record_duration_s": float(fixed[244:252].decode("ascii").strip()),
        "signals": signals,
    }


def distribution(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"n": 0, "min": None, "median": None, "max": None}
    if not np.isfinite(array).all():
        raise ValueError("Non-finite duration")
    return {"n": len(array), "min": float(array.min()),
            "median": float(np.median(array)), "max": float(array.max())}


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def audit(data_root: Path, output: Path) -> dict:
    data_root, output = data_root.resolve(), output.resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(data_root)
    if output.exists():
        raise FileExistsError(f"Audit output must be fresh: {output}")
    if output.is_relative_to(data_root):
        raise ValueError("Audit output may not be inside the read-only dataset")

    root_eeg = json_read(data_root / "task-MIvsRest_eeg.json")
    events_dictionary = json_read(data_root / "task-MIvsRest_events.json")
    description = json_read(data_root / "dataset_description.json")
    participants = pd.read_csv(data_root / "participants.tsv", sep="\t")
    participant_ids = sorted(participants["participant_id"].astype(str).tolist())
    shared_channels = pd.read_csv(data_root / "task-MIvsRest_channels.tsv", sep="\t")
    if not {"name", "type", "units"}.issubset(shared_channels.columns):
        raise ValueError("Invalid shared channel table")
    files = sorted(data_root.glob("sub-*/eeg/*_eeg.edf"))
    if not files:
        raise ValueError("No EDF recordings found")
    trial_rows, recording_rows, annotation_rows, headers, inputs = [], [], [], [], []
    errors = []

    # Text/code provenance is complete; waveform hashes are deliberately deferred.
    for path in sorted(data_root.rglob("*")):
        if path.is_file() and path.suffix in {".json", ".tsv", ".md"}:
            inputs.append({"relative_path": path.relative_to(data_root).as_posix(),
                           "bytes": path.stat().st_size, "sha256": sha256_file(path)})

    for path in files:
        match = re.fullmatch(r"(sub-\d+)_task-MIvsRest_run-(\d+)_eeg\.edf", path.name)
        if match is None:
            raise ValueError(f"Unexpected filename: {path.name}")
        subject, run = match.group(1), int(match.group(2))
        if subject not in participant_ids:
            errors.append(f"Unknown participant {subject}: {path.name}")
        stem = path.name[:-len("_eeg.edf")]
        event_path = path.with_name(stem + "_events.tsv")
        json_path = path.with_suffix(".json")
        channels_path = path.with_name(stem + "_channels.tsv")
        eeg_meta = {**root_eeg, **json_read(json_path)}
        channel_table = pd.read_csv(channels_path, sep="\t") if channels_path.exists() else shared_channels
        event_table = pd.read_csv(event_path, sep="\t")
        required = {"onset", "duration", "sample", "value"}
        if not required.issubset(event_table.columns):
            raise ValueError(f"Missing event columns: {event_path}")
        if not np.isfinite(event_table[list(required)].to_numpy(dtype=float)).all():
            raise ValueError(f"Non-finite events: {event_path}")
        if not set(event_table["value"]).issubset(CODE_TO_LABEL):
            raise ValueError(f"Unknown condition codes: {event_path}")
        if not event_table["onset"].is_monotonic_increasing:
            errors.append(f"Unsorted event table: {event_path.name}")
        if not np.equal(event_table["sample"], np.round(event_table["sample"])).all():
            errors.append(f"Non-integer sample index: {event_path.name}")

        header = edf_header(path)
        raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
        if raw.preload:
            raise RuntimeError("Header audit must never preload EEG")
        fs = float(raw.info["sfreq"])
        duration = raw.n_times / fs
        annotation_onsets = raw.annotations.onset.astype(float)
        annotation_descriptions = raw.annotations.description.astype(str)
        annotation_counts = Counter(annotation_descriptions.tolist())
        native_cues = [int(i) for i, value in enumerate(annotation_descriptions) if value in CUE_TO_CODE]
        starts = np.where(annotation_descriptions == "OVTK_GDF_Start_Of_Trial")[0]
        stops = np.where(annotation_descriptions == "OVTK_GDF_End_Of_Trial")[0]
        crosses = np.where(annotation_descriptions == "OVTK_GDF_Cross_On_Screen")[0]
        beeps = np.where(annotation_descriptions == "OVTK_StimulationId_Beep")[0]
        feedback = np.where(annotation_descriptions == "OVTK_GDF_Feedback_Continuous")[0]
        baseline_starts = np.where(annotation_descriptions == "OVTK_StimulationId_BaselineStart")[0]
        baseline_stops = np.where(annotation_descriptions == "OVTK_StimulationId_BaselineStop")[0]

        if len(native_cues) != len(event_table):
            raise ValueError(f"Native cue / TSV row count mismatch: {path.name}")
        if raw.ch_names != channel_table["name"].astype(str).tolist():
            errors.append(f"Header / sidecar channel mismatch: {path.name}")
        if fs != float(eeg_meta["SamplingFrequency"]):
            errors.append(f"Header / sidecar sampling mismatch: {path.name}")
        if abs(duration - float(eeg_meta["RecordingDuration"])) > 1 / fs:
            errors.append(f"Header / sidecar recording duration mismatch: {path.name}")
        if set(raw.get_channel_types()) != {"eeg"}:
            errors.append(f"Unexpected non-EEG signal channel: {path.name}")

        relative = path.relative_to(data_root).as_posix()
        headers.append({"relative_path": relative, **header, "mne_channel_names": raw.ch_names,
                        "mne_channel_types": raw.get_channel_types(),
                        "mne_original_units": raw._orig_units,
                        "mne_highpass_hz": float(raw.info["highpass"]),
                        "mne_lowpass_hz": float(raw.info["lowpass"]),
                        "inherited_eeg_metadata": eeg_meta})
        inputs.append({"relative_path": relative, "bytes": path.stat().st_size,
                       "edf_header_sha256": header["header_sha256"], "full_file_sha256": None})
        for label, count in sorted(annotation_counts.items()):
            annotation_rows.append({"subject": subject, "run": run, "annotation": label, "count": count})

        rows_this_recording = []
        for trial, (index, (_, event)) in enumerate(zip(native_cues, event_table.iterrows())):
            cue = float(annotation_onsets[index])
            previous_starts = starts[annotation_onsets[starts] <= cue]
            subsequent_stops = stops[annotation_onsets[stops] >= cue]
            start = float(annotation_onsets[previous_starts[-1]]) if len(previous_starts) else float("nan")
            stop = float(annotation_onsets[subsequent_stops[0]]) if len(subsequent_stops) else float("nan")
            next_starts = starts[annotation_onsets[starts] > cue]
            # Do not borrow a following trial's end if this trial's marker is absent.
            if len(next_starts) and stop >= float(annotation_onsets[next_starts[0]]):
                stop = float("nan")
            if not np.isfinite(start) or not np.isfinite(stop):
                errors.append(f"Cue has no enclosing trial markers: {path.name}, row {trial}")
            if start >= cue or stop <= cue:
                raise ValueError(f"Invalid trial boundaries: {path.name}, {trial}")
            code = CUE_TO_CODE[annotation_descriptions[index]]
            cue_sample = int(np.rint(cue * fs))
            onset_error = abs(float(event["onset"]) - cue)
            duration_error = abs(float(event["duration"]) - (stop - cue))
            sample_error = int(event["sample"]) - cue_sample
            if int(event["value"]) != code or onset_error > 0.00011 or duration_error > 0.00011 or sample_error:
                errors.append(f"Native annotation / TSV mismatch: {path.name}, row {trial}")
            task_feedback = feedback[(annotation_onsets[feedback] >= cue) & (annotation_onsets[feedback] <= stop + 0.00011)]
            pre_cue_beeps = beeps[(annotation_onsets[beeps] >= start) & (annotation_onsets[beeps] < cue)]
            cross_in_trial = crosses[(annotation_onsets[crosses] >= start) & (annotation_onsets[crosses] < cue)]
            row = {
                "subject": subject, "run": run, "trial_index": trial,
                "paradigm": "execution" if run == 0 else "imagery",
                "value": code, "label": CODE_TO_LABEL[code], "cue_annotation": annotation_descriptions[index],
                "relative_file": relative, "sfreq_hz": fs, "cue_sample": cue_sample,
                "trial_start_s": start, "cue_onset_s": cue, "trial_stop_s": stop,
                "trial_start_marker_present": bool(np.isfinite(start)),
                "trial_end_marker_present": bool(np.isfinite(stop)),
                "pre_cue_s": cue - start, "task_s": stop - cue,
                "task_samples_floor": int(np.floor((stop - cue) * fs)) if np.isfinite(stop) else None,
                "tsv_duration_s": float(event["duration"]),
                "tsv_duration_checked_against_native": bool(np.isfinite(stop)),
                "tsv_onset_abs_error_s": onset_error, "tsv_duration_abs_error_s": duration_error,
                "tsv_sample_error": sample_error, "cross_markers_pre_cue": len(cross_in_trial),
                "beep_to_cue_s": cue - float(annotation_onsets[pre_cue_beeps[-1]]) if len(pre_cue_beeps) else None,
                "feedback_marker_count_task_including_end": len(task_feedback),
                "feedback_first_from_cue_s": float(annotation_onsets[task_feedback[0]]) - cue if len(task_feedback) else None,
                "feedback_at_trial_end": bool(len(task_feedback) and abs(float(annotation_onsets[task_feedback[0]]) - stop) <= 0.00011),
            }
            for name, (left, right) in WINDOWS.items():
                a, b = cue_sample + int(round(left * fs)), cue_sample + int(round(right * fs))
                row[f"{name}_record_valid"] = bool(0 <= a < b <= raw.n_times)
                row[f"{name}_phase_known"] = bool((left >= 0 or np.isfinite(start)) and
                                                  (right <= 0 or np.isfinite(stop)))
                row[f"{name}_phase_valid"] = bool((left >= 0 or cue + left >= start) and
                                                  (right <= 0 or cue + right <= stop))
            rows_this_recording.append(row)
            trial_rows.append(row)

        baseline_pair_durations = []
        for index in baseline_starts:
            following = baseline_stops[annotation_onsets[baseline_stops] >= annotation_onsets[index]]
            if len(following):
                baseline_pair_durations.append(float(annotation_onsets[following[0]] - annotation_onsets[index]))
        recording_rows.append({
            "subject": subject, "run": run, "paradigm": "execution" if run == 0 else "imagery",
            "relative_file": relative, "bytes": path.stat().st_size,
            "sfreq_hz": fs, "n_channels": len(raw.ch_names), "n_samples": raw.n_times,
            "recording_duration_s": duration, "preloaded": False, "trials": len(event_table),
            "rest_trials": int((event_table["value"] == 9).sum()),
            "move_trials": int((event_table["value"] == 7).sum()),
            "baseline_prefix_markers": len(baseline_pair_durations),
            "baseline_prefix_duration_s": baseline_pair_durations[0] if len(baseline_pair_durations) == 1 else None,
            "native_annotations": len(raw.annotations),
        })
        raw.close()

    trials = pd.DataFrame(trial_rows)
    recordings = pd.DataFrame(recording_rows)
    coverage = {(row["subject"], row["run"]) for row in recording_rows}
    expected = {(subject, run) for subject in participant_ids for run in range(5)}
    if coverage != expected:
        errors.append(f"Run coverage mismatch: missing={sorted(expected-coverage)}, unexpected={sorted(coverage-expected)}")
    if len(coverage) != len(files):
        errors.append("Duplicate recording for a subject/run")
    annotation_table = pd.DataFrame(annotation_rows)
    by_run = trials.groupby(["run", "paradigm", "label"]).size().unstack(fill_value=0).reset_index()
    window_summary = {
        name: {"interval_s": [left, right], "record_invalid": int((~trials[f"{name}_record_valid"]).sum()),
               "phase_invalid": int((trials[f"{name}_phase_known"] & ~trials[f"{name}_phase_valid"]).sum()),
               "phase_unknown": int((~trials[f"{name}_phase_known"]).sum()),
               "imagery_phase_invalid": int(((trials["paradigm"] == "imagery") &
                                                ~trials[f"{name}_phase_valid"]).sum())}
        for name, (left, right) in WINDOWS.items()
    }
    feedback_present = trials["feedback_first_from_cue_s"].notna()
    summary = {
        "dataset": "ds003810 / NEMAR on003810 v1.0.2", "data_root": str(data_root),
        "output": str(output), "script_sha256": sha256_file(Path(__file__).resolve()),
        "read_policy": {"eeg_samples_preloaded": False, "epochs_extracted": False,
                        "full_edf_hashes_computed": False, "edf_headers_and_annotations_read": True,
                        "external_data_used_for_fitting": False},
        "versions": {name: importlib.metadata.version(name) for name in ("mne", "numpy", "pandas")},
        "python": sys.version.split()[0], "dataset_description": description,
        "participants": participant_ids, "recordings": len(files), "trials": len(trials),
        "trials_by_run_class": by_run.to_dict(orient="records"),
        "counts_by_paradigm_label": trials.groupby(["paradigm", "label"]).size().to_dict(),
        "channels": shared_channels.to_dict(orient="records"),
        "target_channels_present": [name for name in TARGET_CHANNELS if name in shared_channels["name"].tolist()],
        "target_channels_missing": [name for name in TARGET_CHANNELS if name not in shared_channels["name"].tolist()],
        "sfreqs_hz": sorted(recordings["sfreq_hz"].unique().tolist()),
        "recording_duration_s": distribution(recordings["recording_duration_s"].tolist()),
        "pre_cue_duration_s": distribution(trials["pre_cue_s"].dropna().tolist()),
        "task_duration_s": distribution(trials["task_s"].dropna().tolist()),
        "missing_trial_start_markers": int((~trials["trial_start_marker_present"]).sum()),
        "missing_trial_end_markers": int((~trials["trial_end_marker_present"]).sum()),
        "imagery_cues_have_observed_trial_boundaries": bool(
            trials.loc[trials["paradigm"] == "imagery", ["trial_start_marker_present", "trial_end_marker_present"]].to_numpy().all()),
        "beep_to_cue_s": distribution(trials["beep_to_cue_s"].dropna().tolist()),
        "baseline_prefix_duration_s": distribution(recordings["baseline_prefix_duration_s"].dropna().tolist()),
        "window_checks": window_summary,
        "tsv_annotation_max_onset_error_s": float(trials["tsv_onset_abs_error_s"].max()),
        "tsv_annotation_max_duration_error_s": float(trials["tsv_duration_abs_error_s"].max()),
        "tsv_annotation_max_sample_error": int(trials["tsv_sample_error"].abs().max()),
        "feedback_marker_trial_count": int(feedback_present.sum()),
        "feedback_marker_at_end_trial_count": int(trials["feedback_at_trial_end"].sum()),
        "feedback_from_cue_s": distribution(trials["feedback_first_from_cue_s"].dropna().tolist()),
        "native_annotation_counts": annotation_table.groupby("annotation")["count"].sum().to_dict(),
        "events_dictionary": events_dictionary,
        "edf_physical_dimensions": sorted({signal["physical_dimension"] for header in headers
                                           for signal in header["signals"] if signal["label"] != "EDF Annotations"}),
        "edf_prefiltering": sorted({signal["prefiltering"] for header in headers
                                     for signal in header["signals"] if signal["label"] != "EDF Annotations"}),
        "warnings": [
            "BIDS declares microvolts but native EDF signal dimensions must be checked before physical-unit conversion.",
            "README reports existing 0.5-45 Hz filtering; empty EDF filter fields do not establish broadband raw data.",
            "RUN0 value 7 is executed movement; RUN1-RUN4 value 7 is imagery. The native label 'Right' does not establish a left/right task.",
            "Feedback marker names do not establish that live feedback was presented during the active task.",
            "Header/annotation audit cannot establish waveform quality, EEG amplitude, clipping or EMG absence outside this export.",
            "Header hashes identify header content only; full EDF checksums are required before a later training use.",
        ],
        "errors": errors, "metadata_contract_passed": not errors,
    }
    summary["counts_by_paradigm_label"] = {
        f"{paradigm}/{label}": int(count)
        for (paradigm, label), count in summary["counts_by_paradigm_label"].items()
    }
    output.mkdir(parents=True, exist_ok=False)
    trials.to_csv(output / "trials.csv", index=False)
    recordings.to_csv(output / "recordings.csv", index=False)
    by_run.to_csv(output / "counts_by_run_class.csv", index=False)
    annotation_table.to_csv(output / "annotations_by_recording.csv", index=False)
    write_json(output / "audit.json", summary)
    write_json(output / "headers.json", {"recordings": headers})
    write_json(output / "input_manifest.json", {"data_root": str(data_root), "inputs": inputs})
    print(json.dumps({"output": str(output), "recordings": len(files), "trials": len(trials),
                      "metadata_contract_passed": not errors, "errors": errors}, sort_keys=True), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = audit(args.data_root, args.output)
    if not report["metadata_contract_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
