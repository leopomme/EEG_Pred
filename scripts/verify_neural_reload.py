#!/usr/bin/env python3
"""Compare completed neural final predictions after independent model reload.

No fitting or threshold selection is performed. Tolerance allows floating-point
rounding between the original CUDA device and a subsequent CPU execution.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eeg_comp.neural import NeuralConfig, PhaseConvNet
from scripts.run_neural import file_sha256, implementation_provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--reloaded", type=Path, required=True)
    args = parser.parse_args()
    original = pd.read_csv(args.experiment / "test_predictions.csv", keep_default_na=False)
    reloaded = pd.read_csv(args.reloaded / "test_predictions.csv", keep_default_na=False)
    identity = [column for column in original if column not in ("p_move", "logit_move")]
    pd.testing.assert_frame_equal(original[identity], reloaded[identity])
    np.testing.assert_allclose(original.p_move, reloaded.p_move, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(original.logit_move, reloaded.logit_move, rtol=1e-5, atol=1e-5)
    checkpoints = []
    config = json.loads((args.experiment / "config.json").read_text())
    expected = {"all_sources"} if config["arguments"]["competition"] == "cross_subject" else set(original.subject)
    paths = sorted((args.experiment / "final").glob("*.pt"))
    assert {path.stem for path in paths} == expected, "Missing or unexpected final checkpoints"
    for path in paths:
        state = torch.load(path, map_location="cpu", weights_only=True)
        assert state["model_config"] == config["model"], "Checkpoint/config architecture mismatch"
        model = PhaseConvNet(NeuralConfig(**state["model_config"]))
        model.load_state_dict(state["model_state_dict"], strict=True)
        assert all(torch.isfinite(tensor).all() for tensor in model.state_dict().values())
        checkpoints.append({"path": str(path), "sha256": file_sha256(path),
                            "bytes": path.stat().st_size,
                            "parameters": sum(parameter.numel() for parameter in model.parameters()),
                            "has_historical_signal_hash": "epochs_sha256" in state["cache_info"],
                            "has_historical_source_hash": bool(state.get("implementation_sha256"))})
    report = {"verified": True, "n_predictions": len(original),
              "n_checkpoints": len(checkpoints), "metadata_and_labels_exact": True,
              "maximum_probability_difference": float(np.max(np.abs(original.p_move - reloaded.p_move))),
              "maximum_logit_difference": float(np.max(np.abs(original.logit_move - reloaded.logit_move))),
              "probability_tolerance": {"rtol": 1e-6, "atol": 1e-6},
              "current_implementation_sha256": implementation_provenance(),
              "architecture": config["model"],
              "state_shapes": {name: list(tensor.shape) for name, tensor in model.state_dict().items()},
              "checkpoints": checkpoints,
              "provenance_limit": "Successful replay verifies current signals/source reproduce predictions; it does not retroactively prove historical file identity where original hashes are absent."}
    (args.reloaded / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in
                      ("checkpoints", "state_shapes", "architecture", "current_implementation_sha256")}))


if __name__ == "__main__":
    main()
