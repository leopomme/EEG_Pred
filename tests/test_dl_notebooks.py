"""Standalone DL snapshots, actual checkpoint replay and label-boundary checks."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('build_dl_notebooks', ROOT / 'scripts/build_dl_notebooks.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class DLNotebookTests(unittest.TestCase):
    def test_exact_source_snapshots_and_production_defaults(self):
        for competition in builder.SPECS:
            notebook = builder.build_notebook(competition)
            self.assertEqual(notebook, builder.build_notebook(competition))
            self.assertEqual(notebook['metadata']['category'], 'DL')
            for i, cell in enumerate(notebook['cells']):
                if cell['cell_type'] == 'code':
                    compile(''.join(cell['source']), f'cell{i}', 'exec')
            config = ast.parse(''.join(notebook['cells'][1]['source']))
            simple = {target.id: ast.literal_eval(node.value)
                      for node in config.body if isinstance(node, ast.Assign)
                      for target in node.targets if isinstance(target, ast.Name)
                      and target.id in {'DEVICE', 'WINDOW', 'MAX_EPOCHS', 'CROSS_EPOCHS'}}
            self.assertEqual(simple, {'DEVICE': 'cuda', 'WINDOW': 'task0_48',
                                      'MAX_EPOCHS': 60, 'CROSS_EPOCHS': 19})
            snapshot = ast.parse(''.join(notebook['cells'][3]['source']))
            assigned = {target.id: ast.literal_eval(node.value)
                        for node in snapshot.body if isinstance(node, ast.Assign)
                        for target in node.targets if isinstance(target, ast.Name)
                        and target.id in {'SOURCES', 'SOURCE_HASHES'}}
            for filename in builder.MODULE_PATHS:
                source = assigned['SOURCES'][filename]
                self.assertEqual(source, (ROOT / filename).read_text())
                self.assertEqual(assigned['SOURCE_HASHES'][filename], hashlib.sha256(source.encode()).hexdigest())
            self.assertNotIn('eeg_comp/classical.py', assigned['SOURCES'])

    def test_actual_cpu_fit_replay_export_and_within_participant_isolation(self):
        # Production stays CUDA/60-inner-epochs/19-Cross-epochs. Only these small,
        # explicitly synthetic subprocesses override the notebook globals.
        for competition in builder.SPECS:
            notebook = builder.build_notebook(competition)
            codes = [''.join(notebook['cells'][i]['source']) for i in [1, 3, 7, 9]]
            loader_defs = ast.Module(body=[node for node in ast.parse(builder.LOAD_CODE).body
                                          if isinstance(node, ast.FunctionDef)], type_ignores=[])
            definitions = ast.unparse(loader_defs)
            smoke = '''
rng = np.random.default_rng(971)
X = rng.normal(size=(56, 8, 2000)).astype('float32')
valid = np.ones((56, 2000), dtype=bool)
phase_valid = valid.copy()
rows = []
for person_id in range(2):
    for i in range(28):
        split = 'train' if i < 24 else 'test'
        subject = f'S{person_id + 1:03}'
        if COMPETITION == 'cross_subject' and split == 'test':
            subject = f'S{person_id + 8:03}'
        rows.append(dict(epoch_index=len(rows), competition=COMPETITION, subject=subject,
                         split=split, run=min(3, 1 + i // 8), y=i % 2 if i < 24 else -1,
                         trial_index=i, file=subject + '.csv', test_order=person_id * 4 + i - 24
                         if split == 'test' else -1))
meta = pd.DataFrame(rows)
y = meta.y.to_numpy(dtype=np.int64)
meta.to_csv(OUTPUT_DIR / 'metadata.csv', index=False)
input_files = [{'synthetic': True, 'seed': 971}]
data_root = OUTPUT_DIR / 'synthetic_data'
EXPECTED_TEST_ROWS = 8
import eeg_comp.submission
eeg_comp.submission.EXPECTED_ROWS[COMPETITION] = 8
if COMPETITION == 'cross_subject':
    def make_folds(*args, **kwargs):
        raise AssertionError('Final Cross notebook must not generate confirmation/development folds')
'''
            checks = '''
assert len(test) == 8 and np.isfinite(test.p_move).all()
assert str(source_root.resolve()) in sys.modules['eeg_comp'].__file__
assert str(source_root.resolve()) in sys.modules['scripts.run_neural'].__file__
assert 'eeg_comp.classical' not in sys.modules
assert len(fit_manifest) == (2 if COMPETITION == 'within_subject' else 1)
assert all(record['labels_identical'] for record in replay_records)
assert all(record['max_abs_probability_difference'] == 0 for record in replay_records)
assert manifest['runtime']['deterministic_algorithms']
assert manifest['runtime']['cudnn_allow_tf32']
assert not manifest['runtime']['cuda_matmul_allow_tf32']
assert manifest['category'] == 'DL' and manifest['rows'] == 8
for fit_record in fit_manifest:
    trained = meta.iloc[fit_record['training_epoch_indices']]
    assert trained.split.eq('train').all()
    if COMPETITION == 'within_subject':
        assert set(trained.subject) == {fit_record['participant']}
        records = [r for r in epoch_records if r['participant'] == fit_record['participant']]
        assert len(records) == 2
        assert fit_record['epochs'] == max(1, round(float(np.median([r['selected_epochs'] for r in records]))))
        for record in records:
            support = record['inner_train_epoch_indices'] + record['inner_valid_epoch_indices']
            assert set(meta.iloc[support].subject) == {fit_record['participant']}
            assert not set(support) & set(record['unused_outer_query_epoch_indices'])
    else:
        assert set(trained.subject) == {'S001', 'S002'}
        assert fit_record['epochs'] == CROSS_EPOCHS
for path, expected_hash in manifest['artifacts_sha256'].items():
    assert sha256_file(OUTPUT_DIR / path) == expected_hash
if COMPETITION == 'within_subject':
    initial = test.copy()
    first_records = [r for r in epoch_records if r['participant'] == 'S002']
    # A different participant's labels must not change the untouched person's
    # stopping choices or predictions, including random initialization order.
    flipped = meta.subject.eq('S001') & meta.split.eq('train')
    y[flipped] = 1 - y[flipped]
    meta.loc[flipped, 'y'] = y[flipped]
    OUTPUT_DIR = OUTPUT_DIR / 'other_person_labels_changed'
    OUTPUT_DIR.mkdir()
    meta.to_csv(OUTPUT_DIR / 'metadata.csv', index=False)
    exec(FIT_SOURCE)
    np.testing.assert_array_equal(initial.loc[initial.subject.eq('S002'), 'p_move'],
                                  test.loc[test.subject.eq('S002'), 'p_move'])
    assert first_records == [r for r in epoch_records if r['participant'] == 'S002']
print('Synthetic embedded DL notebook fit/replay/export passed:', COMPETITION)
'''
            program = '\n'.join([codes[0], "DEVICE = 'cpu'\nMAX_EPOCHS = 1\nCROSS_EPOCHS = 1",
                                 codes[1], definitions, smoke, codes[2], codes[3],
                                 'FIT_SOURCE = ' + repr(codes[2]), checks])
            with tempfile.TemporaryDirectory() as temporary:
                env = {key: value for key, value in os.environ.items() if not key.startswith('EEG_')}
                env.update(EEG_OUTPUT_DIR=temporary, EEG_THREADS='1')
                program_path = Path(temporary) / 'synthetic_driver.py'
                program_path.write_text(program)
                result = subprocess.run([sys.executable, str(program_path)], env=env,
                                        capture_output=True, text=True, timeout=180)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
