#!/usr/bin/env python3
"""Build independent raw epoch caches and machine-readable data audits."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eeg_comp.data import COMPETITIONS, audit_competition


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--competition", choices=(*COMPETITIONS, "all"), default="all")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    for competition in COMPETITIONS if args.competition == "all" else [args.competition]:
        report = audit_competition(args.data_root, args.output_root, competition)
        print(f"Completed {competition}: {report['train_trials']} train, {report['test_trials']} test", flush=True)


if __name__ == "__main__":
    main()
