#!/usr/bin/env python3
"""Fixed neural experiments with competition-specific leakage barriers.

Example (allocated GPU):
 python scripts/run_neural.py --competition cross_subject --window task0_2 \
   --folds 0,1 --output results/cross_subject/neural_task0_2_pilot
CPU execution is deliberately explicit and intended for small smoke tests.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import warnings

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold

from eeg_comp.neural import NeuralConfig, PhaseConvNet, binary_metrics, seed_everything


WINDOWS = {"task0_2": (0.0, 2.0), "task0_48": (0.0, 4.8), "baseline_m2_0": (-2.0, 0.0)}
CHANNELS = ["Fz", "C3", "Cz", "C4", "PO7", "Pz", "PO8", "Oz"]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_provenance() -> dict:
    root = Path(__file__).resolve().parents[1]
    return {name: file_sha256(root / name)
            for name in ("eeg_comp/neural.py", "scripts/run_neural.py")}


def runtime_provenance() -> dict:
    """Record numerical settings used by both training and checkpoint replay."""
    return {"deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cudnn_version": torch.backends.cudnn.version(),
            "cuda_version": torch.version.cuda,
            "cpu_threads": torch.get_num_threads(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}


@dataclass
class Fold:
    name: str
    train: np.ndarray
    inner_train: np.ndarray
    inner_valid: np.ndarray
    query: np.ndarray
    seed: int
    subject: str | None = None
    confirmation: bool = False


def subject_seed(subject: str, seed: int) -> int:
    """Person's seed does not depend on another supplied person's data/order."""
    return (seed + int(hashlib.sha256(subject.encode()).hexdigest()[:8], 16)) % (2**31)


def stratified_halves(indices: np.ndarray, labels: np.ndarray, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    halves: list[list[int]] = [[], []]
    for label in (0, 1):
        selected = np.asarray(indices)[labels[indices] == label].copy()
        if len(selected) < 2:
            raise ValueError(f"Two-fold stratification needs two examples of each class, got {len(selected)} for {label}")
        rng.shuffle(selected)
        for half in (0, 1):
            halves[half].extend(selected[half::2].tolist())
    return [np.asarray(sorted(part), dtype=int) for part in halves]


def make_folds(meta: pd.DataFrame, labels: np.ndarray, competition: str,
               seed: int, fold_ids: list[int] | None = None,
               subjects: list[str] | None = None) -> list[Fold]:
    if competition not in {"cross_subject", "within_subject"}:
        raise ValueError(f"Unknown competition: {competition}")
    is_train = meta["split"].eq("train").to_numpy()
    if len(labels) != len(meta) or not np.isin(labels[is_train], [0, 1]).all():
        raise ValueError("Labeled training rows must have binary labels")
    people = meta["subject"].to_numpy()
    run = meta["run"].to_numpy(dtype=int)
    train_people = sorted(np.unique(people[is_train]))
    folds = []
    if competition == "cross_subject":
        if subjects is not None:
            raise ValueError("--subjects is available only for within_subject; Cross always uses the canonical full source cohort")
        groups = np.array_split(np.asarray(train_people), 6)
        confirmation_people = groups[-1]
        for fold_id, group in enumerate(groups):
            if fold_ids is not None and fold_id not in fold_ids:
                continue
            # A confirmation cohort must be absent from development fitting as
            # well as scoring; otherwise outer-loop model choices can consume it.
            excluded = group if fold_id == 5 else np.union1d(group, confirmation_people)
            source_people = np.asarray([person for person in train_people if person not in excluded])
            if len(group) == 0 or len(source_people) < 3:
                raise ValueError("Cross grouped validation requires enough source participants")
            # This split is fixed from IDs and seed, before reading their labels.
            rng = np.random.default_rng(seed + fold_id)
            inner_people = rng.permutation(source_people)[:min(3, len(source_people)-1)]
            outer_train = np.flatnonzero(is_train & np.isin(people, source_people))
            inner_valid = np.flatnonzero(is_train & np.isin(people, inner_people))
            inner_train = np.setdiff1d(outer_train, inner_valid)
            query = np.flatnonzero(is_train & np.isin(people, group))
            folds.append(Fold(f"group_{fold_id}", outer_train, inner_train, inner_valid,
                              query, seed + 1000 * fold_id, confirmation=fold_id == len(groups)-1))
    else:
        if subjects is not None and set(subjects) - set(train_people):
            raise ValueError(f"Unknown source people: {sorted(set(subjects)-set(train_people))}")
        for person in train_people:
            if subjects is not None and person not in subjects:
                continue
            person_seed = subject_seed(person, seed)
            calibration = np.flatnonzero(is_train & (people == person) & (run != 3))
            feedback = np.flatnonzero(is_train & (people == person) & (run == 3))
            if any(np.sum(labels[feedback] == label) < 2 for label in (0, 1)):
                raise ValueError(f"{person} needs two run-3 labels of each class")
            # Same precommitted split as the independent classical experiments;
            # splitting is performed separately on this person's labels only.
            splitter = StratifiedKFold(2, shuffle=True, random_state=seed)
            halves = [feedback[query] for _, query in splitter.split(feedback, labels[feedback])]
            calibration_runs = sorted(np.unique(run[calibration]))
            if len(calibration_runs) >= 2:
                inner_valid = calibration[run[calibration] == calibration_runs[-1]]
                inner_calibration = np.setdiff1d(calibration, inner_valid)
            else:
                inner_calibration, inner_valid = stratified_halves(calibration, labels, person_seed + 31)
            for half, query in enumerate(halves):
                if fold_ids is not None and half not in fold_ids:
                    continue
                support = halves[1-half]
                train = np.sort(np.concatenate([calibration, support]))
                inner_train = np.sort(np.concatenate([inner_calibration, support]))
                folds.append(Fold(f"{person}_half_{half}", train, inner_train, inner_valid,
                                  query, person_seed + half, subject=person))
    if not folds:
        raise ValueError("No folds requested")
    for fold in folds:
        assert len(fold.query) and len(fold.train) and len(fold.inner_train) and len(fold.inner_valid)
        assert not np.intersect1d(fold.query, fold.train).size
        assert not np.intersect1d(fold.inner_train, fold.inner_valid).size
        assert np.array_equal(np.union1d(fold.inner_train, fold.inner_valid), fold.train)
        if competition == "cross_subject":
            assert not set(people[fold.query]) & set(people[fold.train])
            assert not set(people[fold.inner_train]) & set(people[fold.inner_valid])
        else:
            assert set(people[np.concatenate([fold.train, fold.query])]) == {fold.subject}
    return folds


def load_cache(cache_dir: Path, competition: str, window: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict]:
    metadata_path = cache_dir / "metadata.csv"
    meta = pd.read_csv(metadata_path, keep_default_na=False)
    required = {"epoch_index", "competition", "split", "subject", "run", "label", "test_order"}
    if not required.issubset(meta.columns):
        raise ValueError(f"Missing metadata columns: {sorted(required-set(meta.columns))}")
    if set(meta["competition"]) != {competition}:
        raise ValueError("Cache competition differs from the requested competition")
    if set(meta["split"]) != {"train", "test"} or not meta["run"].isin([1, 2, 3]).all():
        raise ValueError("Expected train/test rows and run IDs 1,2,3")
    if not np.array_equal(meta["epoch_index"], np.arange(len(meta))):
        raise ValueError("Metadata row order does not match epoch_index")
    with np.load(cache_dir / "epochs.npz", allow_pickle=False) as cache:
        fs = int(cache["fs"])
        cue = int(cache["cue_index"])
        if fs != 250 or cue != 750 or cache["channels"].tolist() != CHANNELS:
            raise ValueError("Unexpected sample rate, cue index, or named EEG channel order")
        start, end = [cue + round(value * fs) for value in WINDOWS[window]]
        raw = cache["X"]
        if raw.shape != (len(meta), 8, 2000) or start < 0 or end > raw.shape[-1]:
            raise ValueError(f"Unexpected epoch shape/window: {raw.shape}, [{start},{end})")
        x = np.ascontiguousarray(raw[:, :, start:end], dtype=np.float32)
        masks = [cache[key] for key in ("valid_mask", "phase_valid_mask")]
        if any(mask.shape != (len(meta), 2000) or mask.dtype != np.bool_ for mask in masks):
            raise ValueError("Expected boolean native-sample validity masks")
        valid = masks[0][:, start:end] & masks[1][:, start:end]
        bad = np.flatnonzero(~valid.all(axis=1) | ~np.isfinite(x).all(axis=(1, 2)))
        if len(bad):
            raise ValueError(f"{len(bad)} trials invalid in {window}; no automatic padding/exclusion. First epoch indices: {bad[:20].tolist()}")
        labels = meta["label"].map({"rest": 0, "move": 1, "": -1}).to_numpy()
        if pd.isna(labels).any():
            raise ValueError("Unexpected label mapping")
        labels = labels.astype(np.int64)
        if cache["y"].dtype.kind not in "iu" or not np.array_equal(cache["y"], labels):
            raise ValueError("Cache labels differ from metadata")
    if not np.array_equal(labels >= 0, meta["split"].eq("train")):
        raise ValueError("Train/test label visibility mismatch")
    if not np.array_equal(meta.loc[meta["split"].eq("test"), "test_order"], np.arange((labels < 0).sum())):
        raise ValueError("Test ordering must be complete and contiguous")
    cache_info = {"cache_directory": str(cache_dir.resolve()), "fs": fs, "cue_index": cue,
                  "crop_samples": [start, end], "channels": CHANNELS,
                  "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
                  "epochs_sha256": file_sha256(cache_dir / "epochs.npz"),
                  "audit_sha256": file_sha256(cache_dir / "audit.json") if (cache_dir / "audit.json").exists() else None,
                  "normalization": "runtime per trial per channel; no fitted state",
                  "filtering": "none", "invalid_crop_policy": "fail; no padding or trial removal"}
    return x, labels, meta, cache_info


def protocol_weights(meta: pd.DataFrame) -> dict[int, float]:
    counts = meta.loc[meta["split"].eq("test"), "run"].value_counts()
    if not len(counts):
        raise ValueError("Need test run inventory to report the target protocol mixture")
    return {int(run): float(count / counts.sum()) for run, count in counts.items()}


def weighted_protocol_loss(y: np.ndarray, p: np.ndarray, runs: np.ndarray, weights: dict[int, float]) -> float:
    available = {run: weight for run, weight in weights.items() if np.any(runs == run)}
    if not available:
        # Within-person stopping uses calibration runs, while target is run 3.
        return binary_metrics(y, p)["log_loss"]
    return sum(weight * binary_metrics(y[runs == run], p[runs == run])["log_loss"]
               for run, weight in available.items()) / sum(available.values())


def report_metrics(frame: pd.DataFrame, weights: dict[int, float]) -> dict:
    y = frame["y"].to_numpy()
    p = frame["p_move"].to_numpy()
    result = binary_metrics(y, p)
    result["per_run"] = {str(run): binary_metrics(group["y"], group["p_move"])
                         for run, group in frame.groupby("run")}
    result["per_subject"] = {str(person): binary_metrics(group["y"], group["p_move"])
                             for person, group in frame.groupby("subject")}
    available = {run: weight for run, weight in weights.items() if str(run) in result["per_run"]}
    result["target_run_weights"] = weights
    coverage = sum(available.values())
    weighted = sum(
        weight * result["per_run"][str(run)]["accuracy"] for run, weight in available.items()
    )
    result["missing_target_runs"] = sorted(set(weights) - set(available))
    result["target_run_coverage"] = coverage
    result["target_weighted_accuracy"] = weighted if available and not result["missing_target_runs"] else None
    result["conditional_target_weighted_accuracy"] = weighted / coverage if coverage else None
    return result


def predict(model: PhaseConvNet, x: np.ndarray, indices: np.ndarray, device: torch.device,
            batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    if not len(indices):
        raise ValueError("No trials selected for inference")
    model.eval()
    values = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch = torch.from_numpy(x[indices[start:start+batch_size]]).to(device)
            values.append(model(batch).cpu().numpy())
    logits = np.concatenate(values)
    # Float64 sigmoid remains stable even if training produced large logits.
    probabilities = np.exp(-np.logaddexp(0, -logits.astype(np.float64)))
    if not np.isfinite(probabilities).all():
        raise RuntimeError("Model produced nonfinite predictions")
    return logits, probabilities


def fit(x: np.ndarray, y: np.ndarray, meta: pd.DataFrame, train: np.ndarray,
        valid: np.ndarray | None, *, seed: int, config: NeuralConfig,
        args: argparse.Namespace, device: torch.device, epochs: int,
        weights: dict[int, float], history_path: Path) -> tuple[PhaseConvNet, int]:
    if set(np.unique(y[train])) != {0, 1}:
        raise ValueError("Training fold lacks one class")
    if valid is not None and (np.intersect1d(train, valid).size or not np.isin(y[valid], [0, 1]).all()):
        raise ValueError("Inner validation must use separate labeled trials")
    seed_everything(seed)
    model = PhaseConvNet(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_weights = np.ones(len(train), dtype=np.float32)
    if args.protocol_balanced_loss:
        strata = meta.iloc[train]["subject"].astype(str) + "/" + meta.iloc[train]["run"].astype(str)
        counts = strata.value_counts()
        loss_weights = (1 / strata.map(counts)).to_numpy(dtype=np.float32)
        loss_weights /= loss_weights.mean()
    dataset = TensorDataset(torch.from_numpy(x[train]), torch.from_numpy(y[train]).float(),
                            torch.from_numpy(loss_weights))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                        generator=generator, pin_memory=device.type == "cuda")
    best_loss, best_epoch, stale = float("inf"), epochs, 0
    best_state = None
    with history_path.open("w") as history:
        for epoch in range(1, epochs+1):
            model.train()
            total_loss = 0.0
            for batch, targets, sample_weights in loader:
                batch, targets, sample_weights = [value.to(device) for value in (batch, targets, sample_weights)]
                optimizer.zero_grad(set_to_none=True)
                losses = F.binary_cross_entropy_with_logits(model(batch), targets, reduction="none")
                loss = (losses * sample_weights).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                total_loss += float(loss.detach()) * len(batch)
            record = {"epoch": epoch, "train_loss": total_loss / len(train)}
            if valid is not None:
                _, prob = predict(model, x, valid, device, args.batch_size)
                val_loss = weighted_protocol_loss(y[valid], prob, meta.iloc[valid]["run"].to_numpy(), weights)
                record.update(valid_loss=val_loss, valid_accuracy=float(np.mean((prob >= 0.5) == y[valid])))
                if val_loss < best_loss - 1e-4:
                    best_loss, best_epoch, stale = val_loss, epoch, 0
                    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                else:
                    stale += 1
            history.write(json.dumps(record) + "\n")
            history.flush()
            if epoch == 1 or epoch % 10 == 0:
                print(json.dumps({"training": history_path.stem, **record}), flush=True)
            if valid is not None and stale >= args.patience:
                break
    if valid is not None:
        if best_state is None:
            raise RuntimeError("No finite validation checkpoint")
        model.load_state_dict(best_state)
    return model, best_epoch


def checkpoint(path: Path, model: PhaseConvNet, args: argparse.Namespace, *,
               epochs: int, seed: int, train: np.ndarray, cache_info: dict) -> None:
    torch.save({"model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "model_config": model.configuration(), "competition": args.competition,
                "window": args.window, "epochs": epochs, "seed": seed,
                "train_epoch_indices": train.tolist(), "cache_info": cache_info,
                "implementation_sha256": implementation_provenance(),
                "runtime": runtime_provenance(),
                "label_mapping": {"rest": 0, "move": 1}}, path)


def save_test_predictions(frames: list[pd.DataFrame], output: Path) -> None:
    combined = pd.concat(frames).sort_values("test_order")
    if not np.array_equal(combined["test_order"], np.arange(len(combined))):
        raise ValueError("Test order is incomplete or duplicated")
    combined.to_csv(output / "test_predictions.csv", index=False)
    print(json.dumps({"saved_test_predictions": len(combined),
                      "path": str(output / "test_predictions.csv")}), flush=True)


def inference_from_checkpoints(args: argparse.Namespace, x: np.ndarray, meta: pd.DataFrame,
                               cache_info: dict, device: torch.device) -> None:
    """Reproduce final predictions using saved person-specific/source models."""
    people = [None] if args.competition == "cross_subject" else sorted(meta.loc[meta["split"].eq("test"), "subject"].unique())
    frames = []
    for person in people:
        state = torch.load(args.inference_from / "final" / f"{person or 'all_sources'}.pt",
                           map_location="cpu", weights_only=True)
        if (state["competition"] != args.competition or state["window"] != args.window
                or state["label_mapping"] != {"rest": 0, "move": 1}):
            raise ValueError("Checkpoint competition, window, or label mapping mismatch")
        for key in ("metadata_sha256", "crop_samples", "fs", "cue_index", "channels"):
            if state["cache_info"].get(key) != cache_info[key]:
                raise ValueError(f"Checkpoint cache contract mismatch: {key}")
        if "epochs_sha256" in state["cache_info"]:
            if state["cache_info"]["epochs_sha256"] != cache_info["epochs_sha256"]:
                raise ValueError("Checkpoint cache contract mismatch: epochs_sha256")
        else:
            warnings.warn("Legacy checkpoint has no signal-cache hash; historical raw-cache identity is unverified.")
        saved_implementation = state.get("implementation_sha256", {})
        if saved_implementation.get("eeg_comp/neural.py") not in (None, implementation_provenance()["eeg_comp/neural.py"]):
            raise ValueError("Checkpoint neural implementation hash differs from current source")
        trained = meta.iloc[state["train_epoch_indices"]]
        expected = meta.loc[meta["split"].eq("train") & (True if person is None else meta["subject"].eq(person))]
        if not np.array_equal(trained["epoch_index"], expected["epoch_index"]):
            raise ValueError("Final checkpoint training indices violate the competition/person boundary")
        model = PhaseConvNet(NeuralConfig(**state["model_config"]))
        model.load_state_dict(state["model_state_dict"], strict=True)
        model.to(device)
        selected = meta["split"].eq("test") & (True if person is None else meta["subject"].eq(person))
        indices = np.flatnonzero(selected)
        logits, probability = predict(model, x, indices, device, args.batch_size)
        frame = meta.iloc[indices].copy()
        frame["logit_move"], frame["p_move"] = logits, probability
        frame["prediction"] = np.where(probability >= 0.5, "move", "rest")
        frames.append(frame)
    save_test_predictions(frames, args.output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--competition", required=True, choices=["within_subject", "cross_subject"])
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", choices=WINDOWS, default="task0_2")
    parser.add_argument("--folds", help="Comma-separated canonical outer fold IDs (Cross 0..5; Within 0,1)")
    parser.add_argument("--subjects", help="Within-only person subset, e.g. S001,S002; useful for smoke tests")
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--disable-tf32", action="store_true",
                        help="Explicit numerical contrast: disable CUDA matmul and cuDNN TF32")
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--protocol-balanced-loss", action="store_true",
                        help="Explicit controlled contrast: equal-weight subject/run strata in training")
    parser.add_argument("--predict-test", action="store_true", help="After full validation, refit final models and save test probabilities")
    parser.add_argument("--inference-from", type=Path,
                        help="Reload final/*.pt from a completed experiment; no fitting or validation")
    args = parser.parse_args()
    if min(args.max_epochs, args.patience, args.batch_size, args.threads) < 1:
        parser.error("Epochs, patience, batch size, and threads must be positive")
    args.cache = args.cache or Path("artifacts") / args.competition
    args.folds = [int(value) for value in args.folds.split(",")] if args.folds else None
    args.subjects = args.subjects.split(",") if args.subjects else None
    allowed = set(range(6 if args.competition == "cross_subject" else 2))
    if args.folds is not None and (len(args.folds) != len(set(args.folds)) or set(args.folds) - allowed):
        parser.error(f"Unique fold IDs required from {sorted(allowed)}")
    if args.predict_test and (args.subjects is not None or (args.folds is not None and set(args.folds) != allowed)):
        parser.error("--predict-test requires full validation of all folds and people")
    if args.inference_from and (args.predict_test or args.subjects is not None or args.folds is not None):
        parser.error("--inference-from cannot be combined with --predict-test, --subjects, or --folds")
    if args.folds is None:
        args.folds = sorted(allowed) if args.predict_test else list(range(5 if args.competition == "cross_subject" else 2))
    return args


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("No allocated CUDA GPU. Use PBS GPU resources; --device cpu is for explicit small smoke tests.")
    # Training calls this inside fit(); checkpoint-only inference needs the same
    # deterministic backend policy before its first convolution as well.
    seed_everything(args.seed)
    if args.disable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    x, y, meta, cache_info = load_cache(args.cache, args.competition, args.window)
    weights = protocol_weights(meta)
    folds = [] if args.inference_from else make_folds(meta, y, args.competition, args.seed, args.folds, args.subjects)
    if (args.output / "config.json").exists():
        raise FileExistsError("Output already has config.json; choose a new experiment directory")
    args.output.mkdir(parents=True, exist_ok=True)
    config = NeuralConfig()
    config_record = {"arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                     "model": asdict(config), "cache": cache_info, "torch_version": str(torch.__version__),
                     "implementation_sha256": implementation_provenance(),
                     "runtime": runtime_provenance(),
                     "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                     "category": "end-to-end DL", "test_run_weights": weights,
                     "epoch_selection": "inner source-person validation for Cross; own-person calibration validation for Within",
                     "experiment_status": "fixed researcher-designed baseline; no automatic hyperparameter search",
                     "inference_mode": "saved checkpoint" if args.inference_from else "training",
                     "confirmation_policy": "group5 absent from fitting and scoring of development groups0..4",
                     "confirmation_group": 5 if args.competition == "cross_subject" else None}
    (args.output / "config.json").write_text(json.dumps(config_record, indent=2) + "\n")
    if args.inference_from:
        inference_from_checkpoints(args, x, meta, cache_info, device)
        return
    all_oof = []
    fold_records = []
    experiment_start = time.monotonic()
    for fold in folds:
        fold_dir = args.output / fold.name
        fold_dir.mkdir(exist_ok=True)
        indices = {key: getattr(fold, key).tolist() for key in ("train", "inner_train", "inner_valid", "query")}
        (fold_dir / "split.json").write_text(json.dumps({**indices, "subject": fold.subject,
            "seed": fold.seed, "confirmation": fold.confirmation}, indent=2) + "\n")
        print(json.dumps({"fold": fold.name, "train_n": len(fold.train), "query_n": len(fold.query),
                          "query_people": sorted(meta.iloc[fold.query]["subject"].unique().tolist())}), flush=True)
        _, best_epoch = fit(x, y, meta, fold.inner_train, fold.inner_valid, seed=fold.seed,
            config=config, args=args, device=device, epochs=args.max_epochs, weights=weights,
            history_path=fold_dir / "inner_history.jsonl")
        # Reinitialize and refit all source data, without consulting outer query labels.
        model, _ = fit(x, y, meta, fold.train, None, seed=fold.seed, config=config, args=args,
            device=device, epochs=best_epoch, weights=weights, history_path=fold_dir / "refit_history.jsonl")
        logits, prob = predict(model, x, fold.query, device, args.batch_size)
        frame = meta.iloc[fold.query].copy()
        frame["y"], frame["logit_move"], frame["p_move"] = y[fold.query], logits, prob
        frame["prediction"] = np.where(prob >= 0.5, "move", "rest")
        frame["fold"], frame["confirmation"] = fold.name, fold.confirmation
        frame.to_csv(fold_dir / "oof.csv", index=False)
        checkpoint(fold_dir / "model.pt", model, args, epochs=best_epoch, seed=fold.seed,
                   train=fold.train, cache_info=cache_info)
        score = report_metrics(frame, weights)
        record = {"fold": fold.name, "subject": fold.subject, "selected_epochs": best_epoch,
                  "confirmation": fold.confirmation, "metrics": score}
        fold_records.append(record)
        all_oof.append(frame)
        combined = pd.concat(all_oof).sort_values("epoch_index")
        if combined["epoch_index"].duplicated().any():
            raise RuntimeError("Duplicated out-of-fold trials")
        combined.to_csv(args.output / "oof.csv", index=False)
        summary = {"metrics": report_metrics(combined, weights), "folds": fold_records,
                   "elapsed_seconds": time.monotonic() - experiment_start,
                   "complete_requested_validation": len(fold_records) == len(folds)}
        development = combined.loc[~combined["confirmation"]]
        confirmation = combined.loc[combined["confirmation"]]
        if len(development):
            summary["development_metrics"] = report_metrics(development, weights)
        if len(confirmation):
            summary["confirmation_metrics"] = report_metrics(confirmation, weights)
        (args.output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps({"completed": fold.name, "epochs": best_epoch,
                          "accuracy": score["accuracy"], "target_weighted_accuracy": score["target_weighted_accuracy"]}), flush=True)
    if args.predict_test:
        final_frames = []
        final_dir = args.output / "final"
        final_dir.mkdir(exist_ok=True)
        people = [None] if args.competition == "cross_subject" else sorted(meta.loc[meta["split"].eq("test"), "subject"].unique())
        for person in people:
            person_rows = np.ones(len(meta), dtype=bool) if person is None else meta["subject"].eq(person).to_numpy()
            train = np.flatnonzero(person_rows & meta["split"].eq("train").to_numpy())
            test = np.flatnonzero(person_rows & meta["split"].eq("test").to_numpy())
            chosen = [record["selected_epochs"] for record in fold_records if record["subject"] == person]
            epochs = max(1, round(float(np.median(chosen))))
            seed = args.seed if person is None else subject_seed(person, args.seed)
            name = person or "all_sources"
            model, _ = fit(x, y, meta, train, None, seed=seed, config=config, args=args,
                device=device, epochs=epochs, weights=weights, history_path=final_dir / f"{name}_history.jsonl")
            logits, prob = predict(model, x, test, device, args.batch_size)
            frame = meta.iloc[test].copy()
            frame["logit_move"], frame["p_move"] = logits, prob
            frame["prediction"] = np.where(prob >= 0.5, "move", "rest")
            final_frames.append(frame)
            checkpoint(final_dir / f"{name}.pt", model, args, epochs=epochs, seed=seed, train=train, cache_info=cache_info)
        save_test_predictions(final_frames, args.output)


if __name__ == "__main__":
    main()
