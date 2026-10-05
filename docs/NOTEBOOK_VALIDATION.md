# Standalone notebook validation

**Updated 5 October 2026:** all four baseline notebooks and the optional Cross
ERP-template notebook have passed full local raw-data execution. Kaggle itself
has not been used. See KAGGLE_RUNBOOK.md for the handoff.

`notebooks/within_subject_ml.ipynb` and `notebooks/cross_subject_ml.ipynb` are
independent, pure classical-ML candidates. Both retrain from raw CSVs in one
session. They embed `eeg_comp/data.py`, `classical.py` and `submission.py` plus
an empty package initializer. No neural library, external data, fitted cluster
checkpoint, epoch cache, or internet connection is required.

The initial defaults are `csp_full` for within-subject and `erp_pre` for
cross-subject, C=0.1, empirical training weights and a 0.5 threshold. These are
researcher-specified fallback recipes, not final model-selection claims. The
within-subject notebook applies its one configured recipe independently to
every person, fitting only that person's training records; it does not select
recipes or hyperparameters from pooled participant outcomes.

## Build and lightweight checks

From the project root, using `myenv`:

```bash
python scripts/build_notebooks.py
python scripts/run_notebook_local.py notebooks/within_subject_ml.ipynb --check-only
python scripts/run_notebook_local.py notebooks/cross_subject_ml.ipynb --check-only
python -m unittest discover -s tests -p test_notebooks.py -v
```

The generator takes source snapshots when invoked, records their SHA-256
hashes, embeds them verbatim, and compiles every code cell. Rebuild after any
change to the embedded modules. Generation is deterministic for unchanged
source/generator files. Tests cover exact snapshots, compilation, rejection of
the wrong/combined challenge root, embedded-source import provenance, absence
of neural imports, synthetic fitting for both defaults, and independence of
one person's predictions from changes to another person's training labels.
All three tests passed on 4 October 2026 (26.96 seconds). Full raw-data local
execution also passed as recorded below; Kaggle execution remains outstanding.

## Completed raw-data validation — 4 October 2026

Both notebook candidates were rebuilt from the current source and executed in
fresh subprocesses on a CPU PBS node. Each job first fitted an independent
reference using the development pipeline and its own competition's audited
epoch cache, then executed every notebook code cell from the raw recordings.
Both PBS jobs finished with exit status 0.

| Competition | Default recipe | PBS job | Raw notebook runtime | Submission rows | Maximum probability difference |
| --- | --- | --- | ---: | ---: | ---: |
| Within subject | `csp_full` | `4266646.pbs-7` | 43.09 s | 680 | 0.0 |
| Cross subject | `erp_pre` | `4266647.pbs-7` | 41.01 s | 360 | 0.0 |

The saved probabilities and thresholded labels exactly match their independent
references. All 25 metadata columns match the corresponding audited metadata
exactly: 2,475 within-subject rows and 2,155 cross-subject rows, including 1,795
labeled training trials in each competition. Raw input hashes match each
competition's audit for all 67 within-subject files and all 59 cross-subject
files. Each candidate uses `rest=0`, `move=1`, with 900 rest and 895 move
training labels. No training row was omitted or duplicated across the fits.

Within-subject makes 17 independent person-specific fits. Its submission has
344 move and 336 rest predictions. Cross-subject makes one fit on the labeled
source participants, with no labeled target input. Its submission has 192 move
and 168 rest predictions. These predicted counts are reported outcomes, with
no balancing or count constraint applied. IDs are exactly 0–679 and 0–359.

The cross-subject reference's validation uses development fold 0 only. The
reserved confirmation group was not evaluated. Like eventual inference, its
separate full training fit uses all allowed labeled source participants; its
purpose here is computational equivalence, not model selection. Within-subject
reference validation uses the established per-person run-3 cross-fitting.

Evidence and outputs:

- `results/within_subject/notebook_validation_01/`
- `results/cross_subject/notebook_validation_01/`
- Independent references: `results/<competition>/notebook_reference_01/`
- Batch scripts: `jobs/notebook_within_validation.pbs` and
  `jobs/notebook_cross_validation.pbs`

Each validation directory contains `submission.csv`, `test_predictions.csv`,
`metadata.csv`, `manifest.json`, the embedded source, `execution.log`,
`executed_cells.py`, `local_execution.json`, `prediction_comparison.json`,
`validation_summary.json`, and `pbs_accounting.txt`. The manifests preserve
input/output/source hashes, package versions, settings and fitted row indices.
Both jobs requested 2 CPUs, 10 GB and 15 minutes; total scheduler-recorded
walltime was 85 seconds per job, including the independent reference fit.
The synthetic tests and successful notebook export also enforce the absence
of PyTorch, TensorFlow, Keras and JAX imports in ML execution.

This establishes reproducible local fallback candidates. It does not establish
that these recipes are the strongest choices or that a Kaggle kernel has run.
No upload, credential access or Kaggle submission was performed.

## Completed DL and template execution — reviewed 5 October 2026

| Notebook | PBS job | Full runtime | Test rows | Replay/reference result |
| --- | --- | ---: | ---: | --- |
| Within DL | 4268144 | 85.15s, L40S | 680 | Exactly reproduces original GPU predictions after full fresh training;17 checkpoint replays exact. |
| Cross DL | 4268144 | 61.95s, L40S | 360 | Source-only19-epoch fit; saved-checkpoint replay exact. |
| Cross ERP-template ML | 4270273 | 39.65s, CPU | 360 | Independent audited-cache reference and saved-model replay both exact. |

All three raw-input manifests match their original audits, all metadata columns
match, and final training indices cover exactly the permitted source records.
`scripts/verify_notebook_artifacts.py` additionally verifies the executed notebook
hash, embedded-source hashes, all manifested output hashes, final training
boundaries, and exact submitted row/label mapping. Its saved reports are in each
result directory's `validation_summary.json`. It compares recorded input hashes;
it does not reread potentially changed raw EEG files after execution.

DL directories: `results/<competition>/dl_notebook_validation_01/`.
Template directory: `results/cross_subject/erp_notebook_validation_01/`;
reference: `results/cross_subject/erp_notebook_reference_01/`.

DL generation is `scripts/build_dl_notebooks.py`; two focused tests passed with
actual synthetic training, export/hash checks, checkpoint replay and participant
isolation. The unchanged architecture/numerical settings are embedded alongside
the original fitting helpers. Only splitting uses sklearn; no classical model
features enter DL. Within epoch selection uses each person's own inner fits;
Cross's19epochs were fixed before confirmation from development inner choices.

Within notebook output is byte-identical to the packaged historical DL CSV.
Cross DL has143 move/217 rest predictions; Within has349 move/331 rest. Template
has150 move/210 rest. Counts are unconstrained outcomes, not inferred class totals.

The separate L40S checkpoint reload also passed exactly. Historical CPU replay
still changes one borderline label, so cross-device equality is not claimed.
TF32-disabled L40S replay changes probabilities by at most1.15e-7 and no labels.
See NEURAL_PRECISION.md. Do not substitute a locally produced CSV for the actual
output of the organizer-required Kaggle execution.

## Full local execution on a CPU PBS node

Use a fresh output directory for every execution. `--data-root` is the selected
competition directory itself (containing `Training set` and its test folder),
not the common `data` parent. A representative batch job is:

```bash
#!/bin/bash
#PBS -N eeg_nb_within
#PBS -l select=1:ncpus=2:mem=10gb
#PBS -l walltime=00:15:00
#PBS -j oe
set -euo pipefail
cd /rds/general/user/lh5218/home/EEG_pred
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/rds/general/user/lh5218/home/anaconda3/envs/myenv/bin/python -u \
  scripts/run_notebook_local.py notebooks/within_subject_ml.ipynb \
  --data-root data/within_subject \
  --output results/within_subject/notebook_validation_02
```

For cross-subject, change the notebook, raw folder, output folder and job name
to `cross_subject`. Both jobs can run separately; each reads only its own raw
recordings. The local runner executes all notebook code cells in a new Python
process, with the same shared namespace as Run All. It saves an execution log,
the exact executed Python cells, notebook hash, and run settings. This validates
the notebook's computation; an actual Kaggle kernel run is a separate step.

Optional flags `--recipe`, `--C`, `--weighting`, `--support-weight`, `--seed`
and `--threads` override notebook defaults and are captured in the manifest.
Use the same configuration for the cluster reference and notebook comparison.
For example, once a matching final-fit cluster candidate exists:

```bash
python scripts/run_notebook_local.py notebooks/within_subject_ml.ipynb \
  --data-root data/within_subject \
  --output results/within_subject/notebook_validation_03 \
  --recipe csp_full --C .1 --weighting empirical --support-weight 1 \
  --compare-predictions results/within_subject/notebook_reference_01/csp_full_test_predictions.csv
```

Comparison asserts exact trial identity/order, probability agreement (absolute
tolerance 1e-10 by default), and identical thresholded labels. It writes the
reference-file hash and largest probability difference to
`prediction_comparison.json`. A matching environment should reproduce the
same results; different scikit-learn/SciPy/BLAS versions can affect numerical
agreement. Inspect differences before relaxing tolerance.

## Kaggle execution contract

Attach the selected competition's raw dataset and use a fresh CPU kernel.
Automatic discovery probes only `/kaggle/input/<selected-competition-slug>` and
`/kaggle/input/competitions/<selected-competition-slug>`. It never searches other
attached datasets. If the actual mount differs, set `DATA_ROOT` in the first
cell to its exact folder. The notebook rejects a root containing both test
folder types, and validates participant separation, release file/trial counts,
40 marked trials per test file, eight named channels, native phase masks,
labels and numeric session/epoch order. It uses no unlabeled prefix as training
data and applies no class-count constraint.

Run All writes `submission.csv` (680 or 360 rows, `ID,TARGET`, IDs starting at
zero), `test_predictions.csv`, `metadata.csv`, and `manifest.json` under
`/kaggle/working`. The manifest records code/input/output hashes, full training
row provenance, versions, seeds, elapsed time, information budget and settings.
The same files exist under the chosen local output directory. Embedded source
is also preserved. Only NumPy, SciPy, pandas and scikit-learn are used; the
validated development versions are documented inside each notebook. There is
no automatic install/downgrade or checkpoint upload.

**Remaining validation:** execute each notebook in Kaggle from a clean state
and retain its output/version/runtime record. Rebuild and repeat the local
equivalence check after changes to embedded source or candidate configuration.
These notebooks do not perform uploads, submissions, or credential access.
