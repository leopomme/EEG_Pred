#!/usr/bin/env python3
"""Build the fixed Cross Subject ERP-template ML notebook from exact sources."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_notebooks import LOAD_CODE, SPECS, cell

MODULES = ('data', 'classical', 'erp_template', 'submission')
NAME = 'cross_subject_erp_template_ml.ipynb'


FIT_CODE = '''
if RECIPE != 'erp_pre' or C != .1 or SHRINK != .1:
    raise ValueError('This notebook implements the frozen ERP-template configuration')
recipe = RECIPES[RECIPE]
features = np.concatenate([extract(X[start:start + 128], recipe)
                           for start in range(0, len(X), 128)])
if features.shape != (len(meta), 400) or not np.isfinite(features).all():
    raise ValueError('Expected finite 8-channel by 50-bin ERP features')
waveforms = features.reshape(len(meta), 8, 50)
support = np.flatnonzero(meta.split.eq('train').to_numpy())
query = np.flatnonzero(meta.split.eq('test').to_numpy())
if set(y[support]) != {0, 1} or not np.all(y[query] == -1):
    raise ValueError('Invalid source or target label visibility')
if set(meta.iloc[support].subject) & set(meta.iloc[query].subject):
    raise ValueError('Target participants overlap source participants')
model = ERPTemplateCovariance(C=C, shrink=SHRINK)
model.fit(waveforms[support], y[support])
probability = model.predict_proba(waveforms[query])[:, 1]
test = meta.iloc[query].copy()
test['p_move'] = probability
test = test.sort_values('test_order')
fit_manifest = [{
    'participant': None,
    'training_trials': len(support), 'test_trials': len(query),
    'training_participants': sorted(meta.iloc[support].subject.unique()),
    'training_epoch_indices': support.tolist(),
    'test_epoch_indices': query.tolist(),
    'weighting': 'empirical; no sample weights',
}]
model_path = OUTPUT_DIR / 'erp_template_model.joblib'
joblib.dump({'model': model, 'competition': COMPETITION, 'category': 'ML',
             'build_id': BUILD_ID, 'source_sha256': SOURCE_HASHES,
             'config': {'recipe': RECIPE, 'C': C, 'shrink': SHRINK, 'threshold': .5},
             'training_epoch_indices': support, 'test_epoch_indices': query}, model_path)
reloaded_model = joblib.load(model_path)['model']
replayed = reloaded_model.predict_proba(waveforms[query])[:, 1]
if not np.array_equal(probability, replayed):
    raise ValueError('Immediate saved-model replay probabilities differ')
replay_manifest = {'model_file': model_path.name, 'n': len(query),
                   'probabilities_exactly_equal': True,
                   'labels_identical': bool(np.array_equal(probability >= .5, replayed >= .5)),
                   'max_abs_probability_difference': float(np.max(np.abs(probability-replayed)))}
test['prediction'] = np.where(test.p_move >= .5, 'move', 'rest')
test.to_csv(OUTPUT_DIR / 'test_predictions.csv', index=False)
print(f'Fitted {len(support)} source trials; predicted {len(query)} target trials; saved-model replay exact')
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
    'build_provenance': BUILD_PROVENANCE,
    'seed': SEED, 'versions': VERSIONS, 'python_executable': sys.executable,
    'platform': platform.platform(), 'thread_limit': THREADS,
    'config': {'recipe_name': RECIPE, 'recipe': asdict(recipe), 'C': C,
               'covariance_shrinkage': SHRINK, 'weighting': 'empirical; no sample weights',
               'classifier': 'ERPTemplateCovariance'},
    'information_budget': 'source-only class templates, covariance reference, scaler and classifier; trial-local target inference',
    'selection': 'Frozen development-selected ERP-template model; full retraining on all 17 supplied source participants',
    'threshold': .5, 'label_mapping': {'rest': 0, 'move': 1},
    'channels': list(CHANNELS), 'nominal_fs': FS,
    'required_phase_windows': [[-2., 0.], [0., 4.8]],
    'model_phase_windows': [[-2., 0.], [0., 2.]],
    'ordering': 'participant, numeric run, original marker order; zero-based IDs',
    'ordering_source': f'https://www.kaggle.com/competitions/{SLUG}/overview/evaluation',
    'rows': len(submission), 'id_min': 0, 'id_max': len(submission) - 1,
    'label_counts': submission.TARGET.value_counts().to_dict(),
    'raw_data_root': str(data_root), 'raw_inputs': input_files, 'fits': fit_manifest,
    'saved_model_replay': replay_manifest,
    'artifacts_sha256': {name: sha256_file(OUTPUT_DIR / name)
                        for name in ['metadata.csv', 'test_predictions.csv', 'submission.csv',
                                     'erp_template_model.joblib']},
    'elapsed_seconds': time.monotonic() - STARTED,
    'execution_context': ('kaggle' if os.environ.get('KAGGLE_KERNEL_RUN_TYPE') else 'local'),
    'status': 'raw-data training and inference completed; no upload performed',
}
(OUTPUT_DIR / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\\n')
print(submission.head())
print(f'Saved {len(submission)} rows to {OUTPUT_DIR / "submission.csv"}')
print(f'Elapsed: {manifest["elapsed_seconds"]:.1f} seconds')
'''


def build_notebook():
    spec = SPECS['cross_subject']
    sources = {name+'.py': (ROOT/'eeg_comp'/(name+'.py')).read_text() for name in MODULES}
    sources['__init__.py'] = '"""Embedded pure classical ERP-template pipeline."""\n'
    hashes = {name: hashlib.sha256(source.encode()).hexdigest() for name, source in sources.items()}
    provenance = {
        'generator': 'scripts/build_erp_notebook.py',
        'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'shared_builder_sha256': hashlib.sha256((ROOT/'scripts/build_notebooks.py').read_bytes()).hexdigest(),
        'shared_raw_loader_sha256': hashlib.sha256(LOAD_CODE.encode()).hexdigest(),
        'source_sha256': hashes,
        'competition': 'cross_subject', 'spec': spec,
        'model': {'recipe': 'erp_pre', 'C': .1, 'shrink': .1, 'threshold': .5,
                  'weighting': 'empirical; no sample weights'},
    }
    build_id = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    config = f'''
import os
from pathlib import Path

COMPETITION = 'cross_subject'
SLUG = {spec['slug']!r}
TEST_FOLDER = {spec['test_folder']!r}
EXPECTED_TEST_ROWS = {spec['test_rows']}
EXPECTED_TEST_FILES = {spec['test_files']}
EXPECTED_TEST_PEOPLE = {spec['test_people']}
NOTEBOOK_NAME = {NAME!r}
BUILD_ID = {build_id!r}
BUILD_PROVENANCE = {provenance!r}
DATA_ROOT = os.environ.get('EEG_DATA_ROOT') or None
OUTPUT_DIR = Path(os.environ.get('EEG_OUTPUT_DIR', '/kaggle/working'))
RECIPE, C, SHRINK, SEED = 'erp_pre', .1, .1, 20261003
THREADS = int(os.environ.get('EEG_THREADS', '1'))
if THREADS < 1:
    raise ValueError('THREADS must be positive')
for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[variable] = str(THREADS)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
if any((OUTPUT_DIR / name).exists() for name in ['manifest.json', 'submission.csv',
                                               'test_predictions.csv', 'erp_template_model.joblib']):
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
import joblib
from eeg_comp.data import CHANNELS, FS, CUE_INDEX, identify_file, parse_markers, extract_epochs
from eeg_comp.classical import RECIPES, extract
from eeg_comp.erp_template import ERPTemplateCovariance
from eeg_comp.submission import build_submission
random.seed(SEED)
np.random.seed(SEED)
VERSIONS = {{'python': platform.python_version(), 'numpy': np.__version__,
            'pandas': pd.__version__, 'scipy': scipy.__version__,
            'scikit-learn': sklearn.__version__, 'joblib': joblib.__version__}}
print(json.dumps({{'competition': COMPETITION, 'model': 'ERPTemplateCovariance',
                  'versions': VERSIONS}}, indent=2))
'''
    introduction = f'''# Cross Subject — ERP-template pure ML, full retraining

Attach only `{spec['slug']}` and run all cells in a fresh CPU kernel.
This notebook embeds its complete source and trains from raw competition CSVs.
It uses NumPy, SciPy, pandas, scikit-learn and joblib. No network access, neural
package, external dataset, cached feature file or pretrained checkpoint is required.

The frozen model uses exactly the `erp_pre` representation: eight EEG channels,
trial-local `[-2,0)` baseline scaling, a 20-Hz lowpass over `[0,2)`, 40-ms bins,
and clipping to [-20,20]. Each source-class mean waveform contributes eight
template channels. The two templates and each trial form 24-channel covariance
matrices with 10% isotropic shrinkage and trace normalization. A source-fitted
covariance reference, tangent representation, scaler and logistic classifier
(C=0.1) complete the model. Covariance removes each waveform's temporal mean.
The threshold is 0.5. Training is empirical, with no sample weights.

All 17 supplied source participants contribute to final fitting. Target people
contribute no fitted statistics or templates, and one target trial does not
affect another. This notebook performs no validation scoring or model selection.
It preserves the original linear baseline notebook as a separate candidate.
This is a mixed-evidence research alternative: its development advantage did
not improve the primary run-weighted confirmation score. Linear ERP remains
the project's default Cross Subject ML candidate; no settings were retuned
on confirmation outcomes.

Only Fz, C3, Cz, C4, PO7, Pz, PO8 and Oz enter the model. Markers provide trial
boundaries and source labels; timestamps check sampling continuity. Filenames
and trial order establish submission IDs. Native units and nominal 250-Hz
sampling are retained; the acquisition reference is unverified. The raw loader
checks `[-2,0)` baseline and `[0,4.8)` task support; the model uses `[0,2)`.
There is no target-batch adaptation or forced class count.

Outputs are `submission.csv`, `test_predictions.csv`, `metadata.csv`, a saved
`erp_template_model.joblib`, the exact embedded source and `manifest.json`.
The manifest records source/input/output hashes, build provenance, versions,
fit participants, runtime and exact immediate saved-model replay agreement.
The [official evaluation](https://www.kaggle.com/competitions/{spec['slug']}/overview/evaluation)
specifies `ID,TARGET`, IDs 0 through 359, and lowercase `rest`/`move` labels.
Actual package versions are recorded; numerical results may differ across
BLAS or package versions. Full Kaggle execution remains necessary before an
eligible submission. No upload is performed.
'''
    cells = [cell('markdown', introduction), cell('code', config),
             cell('markdown', '## Embedded source and reproducible environment'), cell('code', snapshot),
             cell('markdown', '## Load and validate raw Cross Subject recordings'), cell('code', LOAD_CODE),
             cell('markdown', '## Fit the frozen ERP-template model and verify saved-model replay'), cell('code', FIT_CODE),
             cell('markdown', '## Export and verify the official submission'), cell('code', EXPORT_CODE)]
    for index, value in enumerate(cells):
        if value['cell_type'] == 'code':
            compile(''.join(value['source']), f'{NAME}:cell{index}', 'exec')
    for filename, source in sources.items():
        for node in ast.walk(ast.parse(source)):
            imports = ([item.name for item in node.names] if isinstance(node, ast.Import)
                       else [node.module or ''] if isinstance(node, ast.ImportFrom) else [])
            if any(name.split('.')[0] in {'torch', 'tensorflow', 'keras', 'jax'} for name in imports):
                raise ValueError('Neural import in ML source: '+filename)
    return {'cells': cells,
            'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                         'language_info': {'name': 'python', 'version': '3.10.16'},
                         'eeg_competition': 'cross_subject', 'category': 'ML',
                         'build_id': build_id, 'build_provenance': provenance},
            'nbformat': 4, 'nbformat_minor': 4}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'notebooks')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir/NAME
    path.write_text(json.dumps(build_notebook(), indent=1, ensure_ascii=False)+'\n')
    print(path)


if __name__ == '__main__':
    main()
