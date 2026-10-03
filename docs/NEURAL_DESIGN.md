# Fixed end-to-end neural baseline

The independent PyTorch implementation in `eeg_comp/neural.py` consumes named
EEG channels with the native time axis. It does not import external model code,
use pretrained weights, fit a classical transform, or mix the competitions.
`scripts/run_neural.py` supplies the split, fitting, checkpoint, and inference
workflow. The classical and data modules do not import this neural module.

## Input and information budget

The audited cache contains `X` with shape `(trials, 8, 2000)`, a 250 Hz sampling
rate, and cue onset at index 750. Metadata fixes competition, participant, run,
visible labels, and test ordering. Named channels are Fz, C3, Cz, C4, PO7, Pz,
PO8, Oz. Cache labels must agree with metadata (`rest=0`, `move=1`, test=-1).

Fixed initial windows are cue-relative `[0,2)` seconds and `[0,4.8)` seconds.
The last avoids the shortest observed cue phases without padding. A separate
`baseline_m2_0` control uses `[-2,0)` seconds with no filtering across cue onset.
Every crop must have finite samples and pass both native-data and phase-boundary
validity masks. Invalid crops fail explicitly; no examples are silently dropped.

Each forward pass centers and scales each channel within each trial. This is
per-trial inductive inference: no other target trial affects its prediction.
There is no filter, rereferencing, target statistics, alignment, label counting,
pseudo-label update, or hand-engineered feature in the network input.
Normalizing each channel removes absolute amplitude and offset; whether that
loses useful class information is an empirical limitation of this baseline.

## Architecture and optimization

The fixed model uses eight learned temporal filters (63 samples), sixteen
grouped spatial filters across all eight channels, and a learned depthwise /
pointwise temporal block. ELU, within-trial GroupNorm, two average pools of four,
and dropout 0.35 supply a compact nonlinear representation. Eight ordered time
bins retain coarse cue timing. Mean and standard deviation of the learned
representations in each bin enter a learned linear binary classifier. These
are runtime summaries of learned features, not a classical feature transform.

AdamW uses learning rate 0.001, weight decay 0.01, batch size 64, a 60-epoch cap,
gradient clipping at norm 5, and patience 10. An inner validation loss chooses
the epoch; a freshly initialized network then fits all permitted fold-training
trials for that many epochs. Outer query labels enter reporting only. Seeds,
deterministic algorithm settings, exact splits, training history, configuration,
checkpoints, and per-trial probabilities are recorded.

`--protocol-balanced-loss` is an explicit optional research contrast that gives
each source participant/run stratum equal total training weight. It is off in
the first fixed baseline. Cross inner stopping weights run losses according to
the audited test run inventory. Within stopping uses only that person's held-out
calibration labels; no pooled person-level performance selects a predictor.

## Competition-specific validation

Cross uses six sorted participant groups from `numpy.array_split`: sizes
3,3,3,3,3,2 for seventeen people. Group5 is a strict confirmation cohort: its
data are absent from development training, inner validation, and scoring for
groups0..4. Each development fold holds out its three query people, selects
three different source people for inner stopping, and refits its twelve source
people. Confirmation group5 trains on all fifteen development people with an
inner split among those fifteen. Query subjects are never used in early stopping.
The CLI defaults to groups0..4; `--folds 5` explicitly opens confirmation.

Within fits each person independently. That person's ten labeled run-3 trials
are split with `StratifiedKFold(2, shuffle=True, random_state=20261003)`. The
support half joins the person's calibration runs; only the other half is scored.
Epoch selection fits calibration run1 plus support and validates on run2; if
one calibration run is missing, a stratified split of that person's available
calibration run is used instead. Final fitting includes all of that person's
available calibration and labeled run-3 data. Cross-person fitting, checkpoint
transfer, and hyperparameter selection are absent. Identical fixed architecture
and optimizer settings are researcher choices, not results pooled over people.

Both validations retain the unresolved late-run extrapolation gap: initial
labeled feedback trials do not reproduce later hidden trials. The two-window
comparison also confounds phase coverage and duration; it is an initial screen,
not proof that feedback itself supplies useful information.

## Execution and artifacts

Use the existing `myenv` environment and an allocated GPU. A pilot evaluates
groups0,1 only:

```bash
python scripts/run_neural.py --competition cross_subject --window task0_2 \
  --folds 0,1 --output results/cross_subject/neural_task0_2_pilot
python scripts/run_neural.py --competition cross_subject --window task0_48 \
  --folds 0,1 --output results/cross_subject/neural_task0_48_pilot
```

Within validation, using a fixed precommitted configuration:

```bash
python scripts/run_neural.py --competition within_subject --window task0_2 \
  --output results/within_subject/neural_task0_2
```

`--subjects S001 --folds 0 --max-epochs 2 --patience 1 --device cpu` restricts
a smoke run. It is not a scientific evaluation. Production defaults require
CUDA; the runner fails if no CUDA device is allocated. Start with a one-hour
single-GPU reservation for both Cross pilots; this is a conservative scheduling
estimate, not a measured runtime. Saved histories and elapsed times support
more accurate sizing after the first allocated run.

`--predict-test` performs all validation folds, including Cross confirmation,
then refits final models using the median selected epochs (separately per
person for Within). Consequently use it only once a configuration is frozen.
It emits probabilities and labels in verified cache order, not a Kaggle CSV:
the official sample-submission contract must still be checked by the packaging
workflow. Existing experiment directories with `config.json` are protected
against accidental overwrite.

Final checkpoint inference is independently repeatable:

```bash
python scripts/run_neural.py --competition cross_subject --window task0_2 \
  --inference-from results/cross_subject/neural_final \
  --output results/cross_subject/neural_final_reloaded
```

Loading checks the competition, crop, channels, metadata SHA-256, label mapping,
and exact allowed final training indices. Checkpoints alone cannot establish
that a different raw-data cache has identical signals: preserve the audited
cache and its per-file provenance. Predictions are always made with eval-mode
normalization and dropout disabled, and are independent of target batch size.

Behavioral verification is `python -m unittest discover -s tests -p
test_neural.py -v`: finite gradients and actual learning on a tiny phase task,
flat-channel gradients, trial/order independence, strict participant splits,
person-specific split invariance, invalid-crop failure, AUC ties, and final
checkpoint reload equivalence. These tests do not establish real-data accuracy
or CUDA performance; the allocated pilot supplies that evidence.
