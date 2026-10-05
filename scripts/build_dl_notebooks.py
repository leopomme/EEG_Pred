#!/usr/bin/env python3
"""Build standalone raw-EEG DL notebooks with full training and checkpoint replay."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from build_notebooks import LOAD_CODE, SPECS, cell

MODULE_PATHS = ('eeg_comp/data.py', 'eeg_comp/neural.py', 'eeg_comp/submission.py',
                'scripts/run_neural.py')


FIT_CODE = '''
# The network receives only raw EEG in [0,4.8), without filtering or features.
a, b = [CUE_INDEX + round(t * FS) for t in WINDOWS[WINDOW]]
if a < 0 or b > X.shape[-1] or not (valid[:, a:b] & phase_valid[:, a:b]).all():
    raise ValueError('Unsupported native EEG task crop')
x = np.ascontiguousarray(X[:, :, a:b], dtype=np.float32)
y = np.asarray(y, dtype=np.int64)
if not np.isfinite(x).all():
    raise ValueError('Nonfinite neural input')
if not np.array_equal(y >= 0, meta.split.eq('train')):
    raise ValueError('Training/test label visibility differs')
config = NeuralConfig()
args = argparse.Namespace(competition=COMPETITION, window=WINDOW,
                         batch_size=64, learning_rate=.001, weight_decay=.01,
                         protocol_balanced_loss=False, patience=10)
weights = protocol_weights(meta)
cache_info = {'raw_inputs': input_files, 'fs': FS, 'cue_index': CUE_INDEX,
              'crop_samples': [a, b], 'channels': list(CHANNELS),
              'metadata_sha256': sha256_file(OUTPUT_DIR / 'metadata.csv'),
              'normalization': 'runtime per trial per channel; no fitted state',
              'filtering': 'none', 'invalid_crop_policy': 'fail; no padding or trial removal'}
final_dir = OUTPUT_DIR / 'final'
final_dir.mkdir(exist_ok=True)
inner_dir = OUTPUT_DIR / 'inner'
inner_dir.mkdir(exist_ok=True)
epoch_records = []
if COMPETITION == 'within_subject':
    # Identical fold definitions and independent seeded stopping fits to run_neural.
    # Outer query labels are never scored or used to select any setting here.
    for fold in make_folds(meta, y, COMPETITION, SEED, fold_ids=[0, 1]):
        _, selected = fit(x, y, meta, fold.inner_train, fold.inner_valid,
                          seed=fold.seed, config=config, args=args, device=device,
                          epochs=MAX_EPOCHS, weights=weights,
                          history_path=inner_dir / (fold.name + '_history.jsonl'))
        epoch_records.append({'fold': fold.name, 'participant': fold.subject,
                              'seed': fold.seed, 'selected_epochs': int(selected),
                              'inner_train_epoch_indices': fold.inner_train.tolist(),
                              'inner_valid_epoch_indices': fold.inner_valid.tolist(),
                              'unused_outer_query_epoch_indices': fold.query.tolist()})
else:
    # Fixed from the five development fits' inner-source stopping epochs only.
    # The reserved confirmation people are included in final fitting, never scored.
    epoch_records = [{'selected_epochs': CROSS_EPOCHS,
                      'selection': 'fixed median of development inner-source stopping epochs',
                      'development_inner_epochs': [19, 13, 19, 13, 37],
                      'confirmation_labels_used_for_selection': False}]
(OUTPUT_DIR / 'epoch_selection.json').write_text(json.dumps(epoch_records, indent=2) + '\\n')

people = sorted(meta.loc[meta.split.eq('test'), 'subject'].unique()) if COMPETITION == 'within_subject' else [None]
frames, fit_manifest, replay_records = [], [], []
for person in people:
    own = np.ones(len(meta), dtype=bool) if person is None else meta.subject.eq(person).to_numpy()
    train = np.flatnonzero(own & meta.split.eq('train').to_numpy())
    query = np.flatnonzero(own & meta.split.eq('test').to_numpy())
    if person is not None and set(meta.iloc[train].subject) != {person}:
        raise ValueError('Within-subject model crossed a participant boundary')
    chosen = [r['selected_epochs'] for r in epoch_records if r.get('participant') == person]
    epochs = max(1, round(float(np.median(chosen))))
    seed = SEED if person is None else subject_seed(person, SEED)
    name = person or 'all_sources'
    model, _ = fit(x, y, meta, train, None, seed=seed, config=config, args=args,
                   device=device, epochs=epochs, weights=weights,
                   history_path=final_dir / (name + '_history.jsonl'))
    logits, probability = predict(model, x, query, device, args.batch_size)
    checkpoint_path = final_dir / (name + '.pt')
    checkpoint(checkpoint_path, model, args, epochs=epochs, seed=seed,
               train=train, cache_info=cache_info)
    # Strict immediate replay on the same numerical backend; no tolerance widening.
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if (saved['competition'] != COMPETITION or saved['window'] != WINDOW
            or saved['label_mapping'] != {'rest': 0, 'move': 1}
            or saved['train_epoch_indices'] != train.tolist()
            or saved['cache_info'] != cache_info):
        raise ValueError('Saved neural checkpoint contract differs')
    replay = PhaseConvNet(NeuralConfig(**saved['model_config'])).to(device)
    replay.load_state_dict(saved['model_state_dict'], strict=True)
    replay_logits, replay_probability = predict(replay, x, query, device, args.batch_size)
    np.testing.assert_allclose(replay_probability, probability, rtol=0, atol=1e-6)
    np.testing.assert_allclose(replay_logits, logits, rtol=0, atol=1e-5)
    np.testing.assert_array_equal(replay_probability >= .5, probability >= .5)
    replay_records.append({'participant': person, 'rows': len(query),
                           'same_device': str(device), 'labels_identical': True,
                           'max_abs_probability_difference': float(np.max(np.abs(replay_probability - probability))),
                           'max_abs_logit_difference': float(np.max(np.abs(replay_logits - logits))),
                           'probability_atol': 1e-6, 'logit_atol': 1e-5})
    frame = meta.iloc[query].copy()
    frame['logit_move'], frame['p_move'] = logits, probability
    frame['prediction'] = np.where(probability >= .5, 'move', 'rest')
    frames.append(frame)
    fit_manifest.append({'participant': person, 'seed': int(seed), 'epochs': epochs,
                         'training_trials': len(train), 'test_trials': len(query),
                         'training_participants': sorted(meta.iloc[train].subject.unique()),
                         'training_epoch_indices': train.tolist(),
                         'test_epoch_indices': query.tolist(),
                         'checkpoint': checkpoint_path.relative_to(OUTPUT_DIR).as_posix(),
                         'checkpoint_sha256': sha256_file(checkpoint_path)})
    print(f'Fitted and replayed {name}: {len(train)} train, {len(query)} test, {epochs} epochs', flush=True)
test = pd.concat(frames).sort_values('test_order')
if not np.array_equal(test.test_order, np.arange(len(test))):
    raise ValueError('Incomplete or duplicated test prediction mapping')
test.to_csv(OUTPUT_DIR / 'test_predictions.csv', index=False)
(OUTPUT_DIR / 'checkpoint_replay.json').write_text(json.dumps(replay_records, indent=2) + '\\n')
'''


EXPORT_CODE = '''
submission = build_submission(test, meta, COMPETITION)
submission.to_csv(OUTPUT_DIR / 'submission.csv', index=False)
reloaded = pd.read_csv(OUTPUT_DIR / 'submission.csv')
if list(reloaded.columns) != ['ID', 'TARGET'] or not np.array_equal(reloaded.ID, np.arange(EXPECTED_TEST_ROWS)):
    raise ValueError('Saved submission has invalid columns or IDs')
if not set(reloaded.TARGET).issubset({'rest', 'move'}):
    raise ValueError('Saved submission has invalid labels')
if any(name == 'eeg_comp.classical' or name.startswith('sklearn.linear_model')
       or name.startswith('sklearn.discriminant_analysis') or name.startswith('sklearn.svm')
       for name in sys.modules):
    raise RuntimeError('Classical estimator module imported in pure DL execution')
artifact_paths = [OUTPUT_DIR / name for name in ['metadata.csv', 'test_predictions.csv',
                  'submission.csv', 'epoch_selection.json', 'checkpoint_replay.json']]
artifact_paths += sorted(final_dir.glob('*')) + sorted(inner_dir.glob('*'))
manifest = {
    'competition': COMPETITION, 'competition_slug': SLUG, 'category': 'DL',
    'notebook': NOTEBOOK_NAME, 'build_id': BUILD_ID, 'source_sha256': SOURCE_HASHES,
    'seed': SEED, 'versions': VERSIONS, 'python_executable': sys.executable,
    'platform': platform.platform(), 'thread_limit': THREADS,
    'device': str(device), 'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
    'runtime': runtime_provenance(), 'model': asdict(config),
    'training': {**vars(args), 'max_inner_epochs': MAX_EPOCHS, 'cross_fixed_epochs': CROSS_EPOCHS},
    'epoch_selection': epoch_records, 'fits': fit_manifest,
    'information_budget': 'allowed labeled EEG for fitting; independent trial-local test inference',
    'confirmation_policy': 'Cross reserved people never scored or used for stopping; included only in final fitting',
    'threshold': .5, 'label_mapping': {'rest': 0, 'move': 1},
    'channels': list(CHANNELS), 'nominal_fs': FS, 'model_window': list(WINDOWS[WINDOW]),
    'normalization': 'runtime per trial per channel; no fitted state', 'filtering': 'none',
    'ordering': 'participant, numeric run, original marker order; zero-based IDs',
    'ordering_source': f'https://www.kaggle.com/competitions/{SLUG}/overview/evaluation',
    'rows': len(submission), 'id_min': 0, 'id_max': len(submission) - 1,
    'label_counts': submission.TARGET.value_counts().to_dict(),
    'raw_data_root': str(data_root), 'raw_inputs': input_files,
    'artifacts_sha256': {path.relative_to(OUTPUT_DIR).as_posix(): sha256_file(path) for path in artifact_paths},
    'checkpoint_replay': replay_records,
    'portability': 'Same-device replay verified. Historical L40S-to-CPU replay changed one thresholded Within prediction; cross-device identity is not assumed.',
    'elapsed_seconds': time.monotonic() - STARTED,
    'execution_context': ('kaggle' if os.environ.get('KAGGLE_KERNEL_RUN_TYPE') else 'local'),
    'status': 'raw-data training, inference and same-device replay completed; no upload performed',
}
(OUTPUT_DIR / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\\n')
print(submission.head())
print(f'Saved {len(submission)} rows to {OUTPUT_DIR / "submission.csv"}')
print(f'Elapsed: {manifest["elapsed_seconds"]:.1f} seconds')
'''


def build_notebook(competition):
    spec = SPECS[competition]
    sources = {name: (ROOT / name).read_text() for name in MODULE_PATHS}
    sources['eeg_comp/__init__.py'] = '"""Embedded pure raw-EEG neural pipeline."""\n'
    sources['scripts/__init__.py'] = '"""Embedded fitting implementation."""\n'
    hashes = {name: hashlib.sha256(source.encode()).hexdigest() for name, source in sources.items()}
    build_id = hashlib.sha256(json.dumps({'competition': competition, 'spec': spec,
                                        'source_hashes': hashes,
                                        'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
                                       sort_keys=True).encode()).hexdigest()
    name = competition + '_dl.ipynb'
    config_code = f'''
import os
from pathlib import Path

COMPETITION = {competition!r}
SLUG = {spec['slug']!r}
TEST_FOLDER = {spec['test_folder']!r}
EXPECTED_TEST_ROWS = {spec['test_rows']}
EXPECTED_TEST_FILES = {spec['test_files']}
EXPECTED_TEST_PEOPLE = {spec['test_people']}
NOTEBOOK_NAME = {name!r}
BUILD_ID = {build_id!r}
DATA_ROOT = os.environ.get('EEG_DATA_ROOT') or None
OUTPUT_DIR = Path(os.environ.get('EEG_OUTPUT_DIR', '/kaggle/working'))
SEED = int(os.environ.get('EEG_SEED', '20261003'))
THREADS = int(os.environ.get('EEG_THREADS', '4'))
DEVICE = 'cuda'
WINDOW = 'task0_48'
MAX_EPOCHS = 60
CROSS_EPOCHS = 19
if THREADS < 1:
    raise ValueError('THREADS must be positive')
for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[variable] = str(THREADS)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
if any((OUTPUT_DIR / name).exists() for name in ['manifest.json', 'submission.csv', 'test_predictions.csv', 'final', 'inner']):
    raise FileExistsError('Use a fresh output directory; existing candidate files are preserved')
'''
    snapshot = f'''
import argparse
import hashlib
import importlib
import json
import platform
import sys
import time
from dataclasses import asdict

STARTED = time.monotonic()
SOURCE_HASHES = {hashes!r}
SOURCES = {sources!r}
source_root = OUTPUT_DIR / 'embedded_source'
source_root.mkdir(parents=True, exist_ok=True)
for filename, source in SOURCES.items():
    if hashlib.sha256(source.encode()).hexdigest() != SOURCE_HASHES[filename]:
        raise ValueError('Embedded source checksum mismatch: ' + filename)
    path = source_root / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
if any(name == 'eeg_comp' or name.startswith('eeg_comp.') or name == 'scripts' or name.startswith('scripts.') for name in sys.modules):
    raise RuntimeError('Restart the kernel before executing this standalone notebook')
sys.path.insert(0, str(source_root.resolve()))
importlib.invalidate_caches()
import numpy as np
import pandas as pd
import sklearn
import torch
from eeg_comp.data import CHANNELS, FS, CUE_INDEX, identify_file, parse_markers, extract_epochs
from eeg_comp.neural import NeuralConfig, PhaseConvNet, seed_everything
from eeg_comp.submission import build_submission
from scripts.run_neural import (WINDOWS, checkpoint, fit, make_folds, predict,
                                protocol_weights, runtime_provenance, subject_seed)
torch.set_num_threads(THREADS)
if DEVICE == 'cuda' and not torch.cuda.is_available():
    raise RuntimeError('Enable a CUDA GPU for the full-retraining notebook')
device = torch.device(DEVICE)
seed_everything(SEED)
# Explicitly preserve the original GPU experiment's numerical policy.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = True
VERSIONS = {{'python': platform.python_version(), 'numpy': np.__version__,
            'pandas': pd.__version__, 'scikit-learn': sklearn.__version__, 'torch': str(torch.__version__)}}
print(json.dumps({{'competition': COMPETITION, 'device': str(device), 'versions': VERSIONS,
                  'runtime': runtime_provenance()}}, indent=2))
'''
    selection_text = (
        'Each participant uses only their own supplied labels. Two independent inner stopping fits use '
        'that participant’s calibration data and disjoint run-3 support halves. The rounded median of '
        'the two stopping epochs determines a fresh fit on all their labels, exactly as in the research '
        'pipeline. No outer-query prediction or score is computed.'
        if competition == 'within_subject' else
        'All 17 labeled source participants enter the final fit for 19 fixed epochs. This is the median '
        'of the five development folds’ inner-source stopping epochs [19,13,19,13,37]. The reserved '
        'confirmation participants are never scored or used to choose training duration; their labels '
        'enter only this final fit. Unlabeled target participants supply no fitted statistics.')
    intro = f'''# {competition.replace('_', ' ').title()} — pure DL, full retraining

Attach **only** `{spec['slug']}` and run all cells from a fresh **GPU** kernel.
This notebook embeds its exact source and trains from the challenge's raw CSVs.
It requires PyTorch, NumPy, pandas and scikit-learn (used only to define Within
stratified splits); no installation, network, external data, cached feature or
pretrained checkpoint is required. This is a reproducible baseline candidate.

The model takes the eight named EEG channels in `[0,4.8)` at nominal 250 Hz.
Markers segment trials/provide training labels; timestamps check continuity;
filenames establish allowed partitions and submission order. The unchanged
PhaseConvNet normalizes each trial/channel at runtime, then learns temporal
and spatial convolutions and classification jointly. There are no classical
features, filtering, target-batch adaptation or forced class counts.

{selection_text}

AdamW uses learning rate .001, weight decay .01, batch 64 and seed 20261003.
Within stopping allows 60 epochs, patience 10, then independently seeded refits.
Deterministic algorithms, cuDNN deterministic mode, cuDNN TF32 enabled and CUDA
matmul TF32 disabled are explicit. Immediate checkpoint replay must preserve
labels and meet probability `1e-6` / logit `1e-5` absolute tolerances.
An earlier L40S-to-CPU replay changed one borderline Within label; identity
across devices or PyTorch versions is not assumed. Run and validate on Kaggle's
actual GPU before submitting.

Outputs include `submission.csv`, ordered `test_predictions.csv`, `metadata.csv`,
source/input/output hashes and numerical settings in `manifest.json`, stopping
provenance, full training histories, final checkpoints and replay checks.
No competition upload is performed. The full CSV-generating notebook must
also execute in Kaggle before submission.
'''
    cells = [cell('markdown', intro), cell('code', config_code),
             cell('markdown', '## Embedded source and numerical settings'), cell('code', snapshot),
             cell('markdown', '## Load this competition from raw recordings'), cell('code', LOAD_CODE),
             cell('markdown', '## Train independent neural models and replay checkpoints'), cell('code', FIT_CODE),
             cell('markdown', '## Export and verify the submission'), cell('code', EXPORT_CODE)]
    notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                                            'language_info': {'name': 'python', 'version': '3.10.16'},
                                            'eeg_competition': competition, 'category': 'DL', 'build_id': build_id},
                'nbformat': 4, 'nbformat_minor': 4}
    for i, value in enumerate(cells):
        if value['cell_type'] == 'code':
            compile(''.join(value['source']), f'{name}:cell{i}', 'exec')
    for filename, source in sources.items():
        for node in ast.walk(ast.parse(source)):
            modules = [item.name for item in node.names] if isinstance(node, ast.Import) else (
                [node.module or ''] if isinstance(node, ast.ImportFrom) else [])
            if any(module.startswith(('eeg_comp.classical', 'sklearn.linear_model', 'sklearn.svm',
                                      'sklearn.discriminant_analysis')) for module in modules):
                raise ValueError('Classical estimator dependency in DL source: ' + filename)
    return notebook


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--competition', choices=list(SPECS))
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'notebooks')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for competition in [args.competition] if args.competition else SPECS:
        path = args.output_dir / (competition + '_dl.ipynb')
        path.write_text(json.dumps(build_notebook(competition), indent=1, ensure_ascii=False) + '\n')
        print(path)


if __name__ == '__main__':
    main()
