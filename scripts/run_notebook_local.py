#!/usr/bin/env python3
"""Execute generated plain-Python notebook cells locally in a fresh subprocess."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('notebook', type=Path)
    parser.add_argument('--data-root', type=Path, help='Selected challenge folder, not their common parent')
    parser.add_argument('--output', type=Path, help='New isolated output directory')
    parser.add_argument('--recipe')
    parser.add_argument('--C', type=float)
    parser.add_argument('--weighting', choices=['empirical', 'target'])
    parser.add_argument('--support-weight', type=float)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--check-only', action='store_true', help='Compile every code cell without loading/training')
    parser.add_argument('--compare-predictions', type=Path, help='Ordered cluster prediction CSV from matching configuration')
    parser.add_argument('--atol', type=float, default=1e-10)
    args = parser.parse_args()
    notebook = json.loads(args.notebook.read_text())
    codes = []
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        compile(source, f'{args.notebook.name}:cell{i}', 'exec')
        codes.append((i, source))
    if args.check_only:
        print(f'{args.notebook}: {len(codes)} code cells compile; no data read')
        return
    if args.data_root is None or args.output is None:
        parser.error('--data-root and --output are required for execution')
    if args.output.exists():
        parser.error('--output must be a new directory')
    if not args.data_root.is_dir():
        parser.error('--data-root is not a directory')
    if args.threads < 1 or args.atol < 0:
        parser.error('--threads must be positive and --atol nonnegative')
    args.output.mkdir(parents=True)
    env = os.environ.copy()
    for key in list(env):
        if key.startswith('EEG_'):
            del env[key]
    settings = {'DATA_ROOT': str(args.data_root.resolve()), 'OUTPUT_DIR': str(args.output.resolve()),
                'RECIPE': args.recipe, 'C': args.C, 'WEIGHTING': args.weighting,
                'SUPPORT_WEIGHT': args.support_weight, 'SEED': args.seed, 'THREADS': args.threads}
    env.update({'EEG_' + key: str(value) for key, value in settings.items() if value is not None})
    # Compile/exec each cell with the same globals, matching Run All semantics.
    # A fresh interpreter ensures development-package imports cannot shadow embedded code.
    driver = ["namespace = {'__name__': '__main__'}"]
    for i, source in codes:
        driver.append(f"print('Executing cell {i}', flush=True)")
        driver.append(f'exec(compile({source!r}, {args.notebook.name + ":cell" + str(i)!r}, "exec"), namespace)')
    driver_path = args.output / 'executed_cells.py'
    driver_path.write_text('\n'.join(driver) + '\n')
    with (args.output / 'execution.log').open('w') as log:
        result = subprocess.run([sys.executable, '-u', str(driver_path.resolve())],
                                cwd=args.output.resolve(), env=env, stdout=log, stderr=subprocess.STDOUT)
    report = {'notebook': str(args.notebook.resolve()),
              'notebook_sha256': hashlib.sha256(args.notebook.read_bytes()).hexdigest(),
              'returncode': result.returncode, 'settings': settings}
    (args.output / 'local_execution.json').write_text(json.dumps(report, indent=2) + '\n')
    if result.returncode:
        print((args.output / 'execution.log').read_text()[-12000:], file=sys.stderr)
        raise SystemExit(result.returncode)
    if args.compare_predictions:
        import numpy as np
        import pandas as pd
        generated = pd.read_csv(args.output / 'test_predictions.csv').sort_values('test_order')
        reference = pd.read_csv(args.compare_predictions).sort_values('test_order')
        columns = ['competition', 'subject', 'run', 'file', 'trial_index', 'test_order', 'epoch_index']
        if not generated[columns].reset_index(drop=True).equals(reference[columns].reset_index(drop=True)):
            raise ValueError('Reference trial mapping does not match the notebook')
        np.testing.assert_allclose(generated.p_move, reference.p_move, rtol=0, atol=args.atol)
        if not np.array_equal(generated.p_move >= .5, reference.p_move >= .5):
            raise ValueError('Reference labels differ at threshold .5')
        comparison = {'reference': str(args.compare_predictions.resolve()),
                      'reference_sha256': hashlib.sha256(args.compare_predictions.read_bytes()).hexdigest(),
                      'max_abs_probability_difference': float(np.max(np.abs(generated.p_move - reference.p_move))),
                      'atol': args.atol, 'labels_identical': True}
        (args.output / 'prediction_comparison.json').write_text(json.dumps(comparison, indent=2) + '\n')
        print(json.dumps(comparison, indent=2))
    print(f'Notebook completed: {args.output / "submission.csv"}; log: {args.output / "execution.log"}')


if __name__ == '__main__':
    main()
