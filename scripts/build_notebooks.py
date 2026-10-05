#!/usr/bin/env python3
"""Build independent, self-contained ML notebooks from current source snapshots."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]
MODULES = ('data', 'classical', 'submission')
SPECS = {
    'within_subject': {'slug': 'low-cost-motor-imagery-decoding-for-rehab',
                       'recipe': 'csp_full', 'test_folder': 'Single subject test set',
                       'test_rows': 680, 'test_files': 17, 'test_people': 17},
    'cross_subject': {'slug': 'low-cost-motor-imagery-decoding-for-rehab-cross-subject',
                      'recipe': 'erp_pre', 'test_folder': 'Cross participant test set',
                      'test_rows': 360, 'test_files': 9, 'test_people': 3},
}


def cell(kind, source):
    source = textwrap.dedent(source).strip() + '\n'
    value = {'cell_type': kind, 'metadata': {}, 'source': source.splitlines(keepends=True)}
    if kind == 'code':
        value.update(execution_count=None, outputs=[])
    return value


LOAD_CODE = '''
def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def locate_data_root(explicit_root):
    # Never recursively search /kaggle/input: the other challenge may be attached.
    if explicit_root is not None:
        candidates = [Path(explicit_root)]
    else:
        candidates = [Path('/kaggle/input') / SLUG,
                      Path('/kaggle/input/competitions') / SLUG]
    found = [root.resolve() for root in candidates
             if (root / 'Training set').is_dir() and (root / TEST_FOLDER).is_dir()]
    if len(set(found)) != 1:
        raise FileNotFoundError('Set DATA_ROOT to this challenge folder containing '
                                f'Training set and {TEST_FOLDER!r}; candidates={candidates}')
    root = found[0]
    # Explicit roots must still have the selected challenge's test layout only.
    other_test = ('Cross participant test set' if COMPETITION == 'within_subject'
                  else 'Single subject test set')
    if (root / other_test).exists():
        raise ValueError('Both competition test folders found in one input root')
    return root


def load_raw_competition(root):
    arrays, masks, phase_masks, rows, file_manifest = [], [], [], [], []
    test_order = 0
    for split, folder, expected_files in [('train', 'Training set', 50),
                                          ('test', TEST_FOLDER, EXPECTED_TEST_FILES)]:
        paths = list((root / folder).glob('*.csv'))
        paths.sort(key=lambda path: (*identify_file(path, COMPETITION, split), path.name))
        if len(paths) != expected_files:
            raise ValueError(f'{split}: expected {expected_files} EEG files, found {len(paths)}')
        for path in paths:
            subject, run = identify_file(path, COMPETITION, split)
            frame = pd.read_csv(path)
            trials, marker_report = parse_markers(frame['Marker_val'], split)
            X_part, valid, phase_valid, trial_rows, timing = extract_epochs(frame, trials)
            # Fail closed: no filling, padding, dropping, or substituting unsupported windows.
            for lo, hi in [(-2., 0.), (0., 4.8)]:
                a, b = CUE_INDEX + round(lo * FS), CUE_INDEX + round(hi * FS)
                if not (valid[:, a:b] & phase_valid[:, a:b]).all():
                    raise ValueError(f'{path.name}: unsupported required phase {lo, hi}')
            for row in trial_rows:
                row.update(epoch_index=len(rows), competition=COMPETITION, split=split,
                           subject=subject, run=run, file=path.relative_to(root).as_posix(),
                           test_order=test_order if split == 'test' else -1)
                if split == 'test':
                    test_order += 1
                rows.append(row)
            file_manifest.append({'file': path.relative_to(root).as_posix(), 'split': split,
                                  'subject': subject, 'run': run, 'trials': len(trials),
                                  'sha256': sha256_file(path), **marker_report,
                                  'median_dt_seconds': timing['median_dt_seconds']})
            arrays.append(X_part)
            masks.append(valid)
            phase_masks.append(phase_valid)
            print(f'{COMPETITION}: {path.name}, {len(trials)} trials', flush=True)
    meta = pd.DataFrame(rows)
    X, valid, phase_valid = map(np.concatenate, [arrays, masks, phase_masks])
    train, test = meta[meta.split.eq('train')], meta[meta.split.eq('test')]
    if len(train) != 1795 or len(test) != EXPECTED_TEST_ROWS:
        raise ValueError('Raw trial counts differ from audited competition release')
    if train.subject.nunique() != 17 or test.subject.nunique() != EXPECTED_TEST_PEOPLE:
        raise ValueError('Unexpected participant counts')
    if not set(train.y.unique()) == {0, 1} or not test.y.eq(-1).all():
        raise ValueError('Invalid label visibility')
    if COMPETITION == 'within_subject':
        if set(train.subject) != set(test.subject) or not test.run.eq(3).all():
            raise ValueError('Within-subject participant/run mismatch')
    elif set(train.subject) & set(test.subject):
        raise ValueError('Cross-subject targets overlap training people')
    ordered = test.sort_values(['subject', 'run', 'trial_index'])
    if not np.array_equal(ordered.test_order, np.arange(EXPECTED_TEST_ROWS)):
        raise ValueError('Test order is not participant/session/original epoch order')
    if ordered.duplicated(['file', 'trial_index']).any():
        raise ValueError('Duplicate test trial mapping')
    if not ordered.groupby('file').size().eq(40).all():
        raise ValueError('Expected exactly forty explicitly marked trials per test file')
    return X, valid, phase_valid, meta, file_manifest


data_root = locate_data_root(DATA_ROOT)
X, valid, phase_valid, meta, input_files = load_raw_competition(data_root)
y = meta.y.to_numpy(dtype=np.int8)
meta.to_csv(OUTPUT_DIR / 'metadata.csv', index=False)
print(meta.groupby(['split', 'run']).size())
'''


FIT_CODE = '''
def training_weights(training_meta):
    weights = np.ones(len(training_meta), dtype=np.float64)
    if COMPETITION == 'cross_subject' and WEIGHTING == 'target':
        strata = training_meta.subject.astype(str) + '/' + training_meta.run.astype(str)
        weights = (1. / strata.map(strata.value_counts())).to_numpy()
    if COMPETITION == 'within_subject':
        weights[training_meta.run.eq(3)] *= SUPPORT_WEIGHT
    return weights / weights.mean()


if RECIPE not in RECIPES or WEIGHTING not in {'empirical', 'target'}:
    raise ValueError('Unknown recipe/weighting')
if not np.isfinite(C) or C <= 0 or not np.isfinite(SUPPORT_WEIGHT) or SUPPORT_WEIGHT <= 0:
    raise ValueError('C and SUPPORT_WEIGHT must be finite and positive')
recipe = RECIPES[RECIPE]
a, b = [CUE_INDEX + round(t * FS) for t in recipe.window]
if a < 0 or b > X.shape[-1] or b <= a or not (valid[:, a:b] & phase_valid[:, a:b]).all():
    raise ValueError('Recipe requests samples outside their native experimental phase')
features = np.concatenate([extract(X[start:start + 128], recipe)
                           for start in range(0, len(X), 128)])
if not np.isfinite(features).all():
    raise ValueError('Nonfinite extracted features')
# Features are trial-local. No fitting or statistics use the unlabeled test batch.
test = meta[meta.split.eq('test')].sort_values('test_order').copy()
test['p_move'] = np.nan
people = sorted(test.subject.unique()) if COMPETITION == 'within_subject' else [None]
fit_manifest = []
for person in people:
    own = np.ones(len(meta), dtype=bool) if person is None else meta.subject.eq(person).to_numpy()
    support = np.flatnonzero(own & meta.split.eq('train').to_numpy())
    query = np.flatnonzero(own & meta.split.eq('test').to_numpy())
    if set(y[support]) != {0, 1}:
        raise ValueError('Each fitted classifier requires both classes')
    if person is not None and set(meta.iloc[support].subject) != {person}:
        raise ValueError('Within-subject fit used another participant')
    model = ClassicalModel(recipe, C=C)
    model.fit(features[support], y[support], training_weights(meta.iloc[support]))
    test.loc[query, 'p_move'] = model.predict_proba(features[query])[:, 1]
    fit_manifest.append({'participant': person, 'training_trials': len(support),
                         'test_trials': len(query),
                         'training_participants': sorted(meta.iloc[support].subject.unique()),
                         'training_epoch_indices': support.tolist()})
    print(f'Fitted {person or "all source participants"}: {len(support)} train, {len(query)} test',
          flush=True)
test['prediction'] = np.where(test.p_move >= .5, 'move', 'rest')
test.to_csv(OUTPUT_DIR / 'test_predictions.csv', index=False)
'''


EXPORT_CODE = '''
submission = build_submission(test, meta, COMPETITION)
submission.to_csv(OUTPUT_DIR / 'submission.csv', index=False)
reloaded = pd.read_csv(OUTPUT_DIR / 'submission.csv')
if list(reloaded.columns) != ['ID', 'TARGET'] or not np.array_equal(reloaded.ID, np.arange(EXPECTED_TEST_ROWS)):
    raise ValueError('Saved submission has invalid columns or IDs')
if not set(reloaded.TARGET).issubset({'rest', 'move'}):
    raise ValueError('Saved submission has invalid labels')
neural_imports = sorted(name for name in sys.modules
                        if name.split('.')[0] in {'torch', 'tensorflow', 'keras', 'jax'})
if neural_imports:
    raise RuntimeError(f'Neural dependencies imported in pure ML execution: {neural_imports[:5]}')
manifest = {
    'competition': COMPETITION, 'competition_slug': SLUG, 'category': 'ML',
    'notebook': NOTEBOOK_NAME, 'build_id': BUILD_ID, 'source_sha256': SOURCE_HASHES,
    'seed': SEED, 'versions': VERSIONS, 'python_executable': sys.executable,
    'platform': platform.platform(), 'thread_limit': THREADS,
    'config': {'recipe_name': RECIPE, 'recipe': asdict(recipe), 'C': C,
               'weighting': WEIGHTING, 'support_weight': SUPPORT_WEIGHT},
    'information_budget': 'inductive fit on allowed labeled EEG; trial-local inference',
    'selection': ('one researcher-specified recipe, independently fitted per participant'
                  if COMPETITION == 'within_subject' else 'fixed configured source-trained recipe'),
    'threshold': .5, 'label_mapping': {'rest': 0, 'move': 1},
    'channels': list(CHANNELS), 'nominal_fs': FS,
    'required_phase_windows': [[-2., 0.], [0., 4.8]],
    'ordering': 'participant, numeric run, original marker order; zero-based IDs',
    'ordering_source': f'https://www.kaggle.com/competitions/{SLUG}/overview/evaluation',
    'rows': len(submission), 'id_min': 0, 'id_max': len(submission) - 1,
    'label_counts': submission.TARGET.value_counts().to_dict(),
    'raw_data_root': str(data_root), 'raw_inputs': input_files, 'fits': fit_manifest,
    'artifacts_sha256': {name: sha256_file(OUTPUT_DIR / name)
                        for name in ['metadata.csv', 'test_predictions.csv', 'submission.csv']},
    'elapsed_seconds': time.monotonic() - STARTED,
    'execution_context': ('kaggle' if os.environ.get('KAGGLE_KERNEL_RUN_TYPE') else 'local'),
    'status': 'raw-data training and inference completed; no upload performed',
}
(OUTPUT_DIR / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\\n')
print(submission.head())
print(f'Saved {len(submission)} rows to {OUTPUT_DIR / "submission.csv"}')
print(f'Elapsed: {manifest["elapsed_seconds"]:.1f} seconds')
'''


def build_notebook(competition):
    spec = SPECS[competition]
    sources = {name + '.py': (ROOT / 'eeg_comp' / (name + '.py')).read_text()
               for name in MODULES}
    # The stub avoids future eager imports added to the development package.
    sources['__init__.py'] = '"""Embedded pure classical EEG pipeline."""\n'
    hashes = {name: hashlib.sha256(source.encode()).hexdigest() for name, source in sources.items()}
    build_id = hashlib.sha256(json.dumps({'competition': competition, 'spec': spec,
                                        'source_hashes': hashes,
                                        'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
                                       sort_keys=True).encode()).hexdigest()
    name = competition + '_ml.ipynb'
    config = f'''
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
# Set an explicit folder containing Training set and the selected test folder if needed.
DATA_ROOT = os.environ.get('EEG_DATA_ROOT') or None
OUTPUT_DIR = Path(os.environ.get('EEG_OUTPUT_DIR', '/kaggle/working'))
RECIPE = os.environ.get('EEG_RECIPE', {spec['recipe']!r})
C = float(os.environ.get('EEG_C', '.1'))
WEIGHTING = os.environ.get('EEG_WEIGHTING', 'empirical')
SUPPORT_WEIGHT = float(os.environ.get('EEG_SUPPORT_WEIGHT', '1.'))
SEED = int(os.environ.get('EEG_SEED', '20261003'))
THREADS = int(os.environ.get('EEG_THREADS', '1'))
if THREADS < 1:
    raise ValueError('THREADS must be positive')
for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[variable] = str(THREADS)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
if any((OUTPUT_DIR / name).exists() for name in ['manifest.json', 'submission.csv', 'test_predictions.csv']):
    raise FileExistsError('Use a fresh output directory; existing candidate files are preserved')
'''
    snapshot = f'''
import hashlib
import importlib
import json
import platform
import random
import sys
import time
from dataclasses import asdict

STARTED = time.monotonic()
SOURCE_HASHES = {hashes!r}
SOURCES = {sources!r}
source_root = OUTPUT_DIR / 'embedded_source'
package_root = source_root / 'eeg_comp'
package_root.mkdir(parents=True, exist_ok=True)
for filename, source in SOURCES.items():
    if hashlib.sha256(source.encode()).hexdigest() != SOURCE_HASHES[filename]:
        raise ValueError('Embedded source checksum mismatch: ' + filename)
    (package_root / filename).write_text(source)
if any(name == 'eeg_comp' or name.startswith('eeg_comp.') for name in sys.modules):
    raise RuntimeError('Restart the kernel before executing this standalone notebook')
sys.path.insert(0, str(source_root.resolve()))
importlib.invalidate_caches()
import numpy as np
import pandas as pd
import scipy
import sklearn
from eeg_comp.data import CHANNELS, FS, CUE_INDEX, identify_file, parse_markers, extract_epochs
from eeg_comp.classical import RECIPES, ClassicalModel, extract
from eeg_comp.submission import build_submission
random.seed(SEED)
np.random.seed(SEED)
VERSIONS = {{'python': platform.python_version(), 'numpy': np.__version__,
            'pandas': pd.__version__, 'scipy': scipy.__version__, 'scikit-learn': sklearn.__version__}}
print(json.dumps({{'competition': COMPETITION, 'recipe': RECIPE, 'versions': VERSIONS}}, indent=2))
'''
    introduction = f'''# {competition.replace('_', ' ').title()} — pure ML, full retraining

Attach **only** `{spec['slug']}` and run all cells from a fresh CPU kernel.
This notebook contains its source and trains from the challenge's raw CSV files;
it requires NumPy, SciPy, pandas and scikit-learn, with no package installation,
network access, external dataset, cached features or pretrained checkpoint.
The initial recipe is `{spec['recipe']}`, C=0.1, threshold=0.5 and empirical weights.
It is a conservative candidate, not a claim of final model selection or Kaggle validation.

Only Fz, C3, Cz, C4, PO7, Pz, PO8 and Oz enter the model. Markers segment trials
and provide training labels. Timestamps check sampling continuity. Filenames
and trial indices establish allowed training partitions and submission order.
Recorded units are retained; the acquisition reference is unverified.
The native 250-Hz contract requires `[-2,0)` baseline and `[0,4.8)` task support.
There is no forced class count, target-batch fitting, or neural feature input.

{'Every person gets an independently fitted model using only that person’s training data. The same prespecified recipe is applied to every person; this notebook makes no model choice from pooled participant outcomes.' if competition == 'within_subject' else 'The model fits the source participants only. Target participants are entirely unseen during fitting.'}

Local validated development environment: Python 3.10.16, NumPy 1.26.4,
SciPy 1.15.3, pandas 1.5.3, scikit-learn 1.2.1. Actual versions are recorded
at execution; numerical results can vary across package/BLAS versions.
See the [official evaluation](https://www.kaggle.com/competitions/{spec['slug']}/overview/evaluation)
for the verified `ID,TARGET` contract, IDs 0 through {spec['test_rows'] - 1}.

Outputs: `submission.csv`, ordered `test_predictions.csv`, `metadata.csv`,
`manifest.json` (source/input/output hashes, versions, seeds, fit provenance,
runtime) and the exact embedded source. Full execution in Kaggle remains
required before an eligible competition submission. No upload is performed.
'''
    cells = [cell('markdown', introduction), cell('code', config),
             cell('markdown', '## Embedded source and reproducible environment'), cell('code', snapshot),
             cell('markdown', '## Load and validate this competition from raw recordings'), cell('code', LOAD_CODE),
             cell('markdown', '## Fit the configured classical model and predict'), cell('code', FIT_CODE),
             cell('markdown', '## Export and verify the official submission'), cell('code', EXPORT_CODE)]
    notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python',
                                                         'name': 'python3'},
                                            'language_info': {'name': 'python', 'version': '3.10.16'},
                                            'eeg_competition': competition, 'category': 'ML',
                                            'build_id': build_id},
                'nbformat': 4, 'nbformat_minor': 4}
    for i, value in enumerate(cells):
        if value['cell_type'] == 'code':
            compile(''.join(value['source']), f'{name}:cell{i}', 'exec')
    # Reject accidental neural imports in snapshots, including a future code change.
    for filename, source in sources.items():
        for node in ast.walk(ast.parse(source)):
            names = [item.name for item in node.names] if isinstance(node, ast.Import) else (
                [node.module or ''] if isinstance(node, ast.ImportFrom) else [])
            if any(value.split('.')[0] in {'torch', 'tensorflow', 'keras', 'jax'} for value in names):
                raise ValueError('Neural import in ML source: ' + filename)
    return notebook


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--competition', choices=list(SPECS), help='Default: build both independent notebooks')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'notebooks')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for competition in [args.competition] if args.competition else SPECS:
        path = args.output_dir / (competition + '_ml.ipynb')
        path.write_text(json.dumps(build_notebook(competition), indent=1, ensure_ascii=False) + '\n')
        print(path)


if __name__ == '__main__':
    main()
