"""Checks for self-contained execution and challenge/person isolation."""
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
spec = importlib.util.spec_from_file_location('build_notebooks', ROOT / 'scripts/build_notebooks.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class NotebookTests(unittest.TestCase):
    def test_generated_sources_compile_and_are_exact_snapshots(self):
        for competition in builder.SPECS:
            notebook = builder.build_notebook(competition)
            self.assertEqual(notebook, builder.build_notebook(competition))
            for i, cell in enumerate(notebook['cells']):
                if cell['cell_type'] == 'code':
                    compile(''.join(cell['source']), f'cell{i}', 'exec')
            snapshot = ast.parse(''.join(notebook['cells'][3]['source']))
            assigned = {target.id: ast.literal_eval(node.value)
                        for node in snapshot.body if isinstance(node, ast.Assign)
                        for target in node.targets if isinstance(target, ast.Name)
                        and target.id in {'SOURCES', 'SOURCE_HASHES'}}
            for name in builder.MODULES:
                filename = name + '.py'
                self.assertEqual(assigned['SOURCES'][filename], (ROOT / 'eeg_comp' / filename).read_text())
                self.assertEqual(assigned['SOURCE_HASHES'][filename],
                                 hashlib.sha256(assigned['SOURCES'][filename].encode()).hexdigest())

    def test_input_resolution_rejects_other_challenge_and_combined_root(self):
        definitions = ast.Module(body=[node for node in ast.parse(builder.LOAD_CODE).body
                                      if isinstance(node, ast.FunctionDef)], type_ignores=[])
        for competition, spec in builder.SPECS.items():
            scope = {'Path': Path, 'COMPETITION': competition, 'SLUG': spec['slug'],
                     'TEST_FOLDER': spec['test_folder']}
            exec(compile(definitions, 'loader_functions', 'exec'), scope)
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / 'Training set').mkdir()
                wrong = ('Cross participant test set' if competition == 'within_subject'
                         else 'Single subject test set')
                (root / wrong).mkdir()
                with self.assertRaises(FileNotFoundError):
                    scope['locate_data_root'](root)
                (root / spec['test_folder']).mkdir()
                with self.assertRaises(ValueError):
                    scope['locate_data_root'](root)
                (root / wrong).rmdir()
                self.assertEqual(scope['locate_data_root'](root), root.resolve())

    def test_embedded_modules_fit_without_neural_imports_and_preserve_person_isolation(self):
        # Fresh interpreters catch accidental use of the development package.
        for competition in builder.SPECS:
            notebook = builder.build_notebook(competition)
            code = [''.join(notebook['cells'][i]['source']) for i in [1, 3, 7]]
            smoke = '''
rng = np.random.default_rng(971)
X = rng.normal(size=(32, 8, 2000)).astype('float32')
valid = np.ones((32, 2000), dtype=bool)
phase_valid = valid.copy()
rows = []
for person_id in range(2):
    for i in range(16):
        split = 'train' if i < 12 else 'test'
        subject = f'S{person_id + 1:03}'
        if COMPETITION == 'cross_subject' and split == 'test':
            subject = f'S{person_id + 8:03}'
        rows.append(dict(epoch_index=len(rows), competition=COMPETITION, subject=subject,
                         split=split, run=3 if i >= 8 else 1, y=i % 2 if i < 12 else -1,
                         trial_index=i, file=subject + '.csv', test_order=person_id * 4 + i - 12
                         if split == 'test' else -1))
meta = pd.DataFrame(rows)
y = meta.y.to_numpy(dtype=np.int8)
'''
            checks = '''
assert np.isfinite(test.p_move).all()
assert str(source_root.resolve()) in sys.modules['eeg_comp'].__file__
assert not any(name.split('.')[0] in {'torch', 'tensorflow', 'keras', 'jax'} for name in sys.modules)
assert len(fit_manifest) == (2 if COMPETITION == 'within_subject' else 1)
for fit in fit_manifest:
    if COMPETITION == 'within_subject':
        assert fit['training_participants'] == [fit['participant']]
if COMPETITION == 'within_subject':
    initial = test.copy()
    y[meta.subject.eq('S001') & meta.split.eq('train')] = 1 - y[meta.subject.eq('S001') & meta.split.eq('train')]
    exec(FIT_SOURCE)
    np.testing.assert_array_equal(initial.loc[initial.subject.eq('S002'), 'p_move'],
                                  test.loc[test.subject.eq('S002'), 'p_move'])
print('Synthetic embedded notebook fit passed:', COMPETITION)
'''
            program = '\n'.join([code[0], code[1], smoke, code[2], 'FIT_SOURCE = ' + repr(code[2]), checks])
            with tempfile.TemporaryDirectory() as temporary:
                env = {key: value for key, value in os.environ.items() if not key.startswith('EEG_')}
                env['EEG_OUTPUT_DIR'] = temporary
                result = subprocess.run([sys.executable, '-c', program], env=env,
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
