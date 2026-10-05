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
The default network has **1,449 trainable parameters**, including its
257-parameter classifier. Float32 parameter storage is 5,796 bytes; roughly
10 KB checkpoints are expected after tensor serialization and metadata.

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
CUDA; the runner fails if no CUDA device is allocated. The saved PBS jobs
request 45 minutes on one L40S each: one job runs both Cross pilots sequentially;
the other evaluates and fits final independent Within models using the full
4.8-second window. These original jobs completed; measured results are below.
The subsequent three-fold Cross job requests only ten minutes because the
two-fold full-window pilot took 20.41 seconds of measured validation execution.

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
and exact allowed final training indices. From the 4 October follow-up onward,
new runs also record the SHA-256 of `epochs.npz`, `audit.json`, neural source,
and runner source. Reload rejects signal-cache or neural-source differences
when those hashes are available. The audit digest links to the audit's raw-file
hashes; it does not itself reread the raw CSV files. Original pilots lack these
signal/source digests and remain unchanged; reloading them warns explicitly
that historical signal-cache identity is unverified. Current replay hashes
must not be represented as historical execution hashes. Predictions are always made with eval-mode
normalization and dropout disabled, and are independent of target batch size.

Behavioral verification is `python -m unittest discover -s tests -p
test_neural.py -v`: finite gradients and actual learning on a tiny phase task,
flat-channel gradients, trial/order independence, strict participant splits,
person-specific split invariance, invalid-crop failure, AUC ties, and final
checkpoint reload equivalence. These tests do not establish real-data accuracy
or CUDA performance; the allocated pilot supplies that evidence.

## Completed execution and review on 4 October 2026

All ten neural behavior tests pass in `myenv`, including the new signal-cache
mismatch rejection check. The latest run took 14.54 seconds excluding Python
import overhead. The missing-calibration-run synthetic test previously expected
four validation examples; its class-balanced split of three examples per class
actually gives four training and two validation examples. The corrected test
also verifies one validation example per class. The implementation is unchanged.

Completed `jobs/neural_cross_pilot.pbs` as `4265947.pbs-7` and
`jobs/neural_within_pilot.pbs` as `4265948.pbs-7`, each requesting one L40S,
four CPUs, 12 GB host memory and 45 minutes. Both jobs succeeded on an NVIDIA
L40S reporting 46,068 MiB and driver 580.82.07. Logs are
`logs/neural_cross_pilot.log` and `logs/neural_within_pilot.log`; results go to
`results/cross_subject/dl_pilot_pre`, `results/cross_subject/dl_pilot_full`, and
`results/within_subject/dl_pilot_full` respectively. Cross confirmation group5
remains unopened. Within validation completed all 34 folds (170 query trials),
then produced all seventeen independent final checkpoints and 680 test
probabilities. These are local research artifacts,
not submitted Kaggle entries.

While batch GPUs were fully allocated, the separate four-CPU benchmark
`jobs/neural_cpu_benchmark.pbs` (`4265950.pbs-7`, node `cx3-3-11`) completed
Cross group0 with two inner and two refit epochs in **16.48 seconds** excluding
cache/import startup. This verifies the complete real-data optimization and
checkpoint path on an allocated compute node; it does not verify CUDA. Its
pre-feedback target-weighted accuracy was 65.11% across 330 held-out trials,
but the two-epoch runtime probe is not a selected scientific candidate.
Artifacts are in `results/cross_subject/dl_cpu_benchmark` and the log in
`logs/neural_cpu_benchmark.log`. An unpruned 60+60 epoch fit extrapolates to
about 8.2 minutes per pre-feedback fold on that four-CPU allocation, with
longer windows proportionally slower; early stopping may reduce this. The later
GPU jobs now verify CUDA as well.

### Pilot results and fair comparison

The Cross comparison below uses exactly the same 635 out-of-fold queries from
S001 through S006 in development groups0,1. The run-weighted score weights runs
1,2,3 equally, matching the observed hidden-test run inventory. It is a local
validation score, not a Kaggle score.

| Fixed model/window | Accuracy | Target-run-weighted accuracy | Log loss |
| --- | ---: | ---: | ---: |
| Classical ERP, 0–2 seconds | 69.76% | 67.61% | 0.5744 |
| PhaseConvNet, 0–2 seconds | 63.94% | 64.72% | 0.6687 |
| PhaseConvNet, 0–4.8 seconds | 67.72% | 67.88% | 0.6194 |

Full-window DL improves on the short-window DL on these two groups. Relative to
ERP it is close on the target mixture but has worse pooled accuracy and loss;
this small pilot does not establish a superior predictor. Machine-readable
paired metrics: `results/cross_subject/dl_pilot_paired_metrics.json`.

Cross short/full validation took 22.65/20.41 seconds respectively, measured
after cache loading, with selected epochs 35/1 and 19/13. The combined original
PBS job used 1 minute 45 seconds including startup. Within validation took
33.17 seconds for 34 folds, before final refits; its complete PBS job used
1 minute 11 seconds. Within accuracy was **60.00%** on 170 own-person held-out
run-3 trials. Its fixed neural candidate is currently behind the classical CSP
candidate, but pooled performance is reporting only: another person's score
must not select or tune a person's Within predictor.

### Follow-up and checkpoint verification

`jobs/neural_cross_development.pbs` was submitted as `4266652.pbs-7` for
groups2,3,4, the same full-window configuration and seed, and no confirmation.
The request was reduced to ten minutes from the original conservative 45 after
the pilot runtime became available. Output is
`results/cross_subject/dl_development_full`; initial status was queued.

`jobs/neural_within_reload.pbs` was submitted as `4266653.pbs-7` on four CPUs.
It reloads the 17 original Within final models and compares all 680 ordered
test predictions with the saved GPU output. The verification requires exact
trial metadata and labels, probability agreement to `atol=rtol=1e-6`, and
matching saved architecture/state shapes. Its independent verification report
is `results/within_subject/dl_pilot_full_reloaded/verification.json` when complete.
The tolerance accommodates CPU/GPU floating-point rounding; exact numerical
bit equality is not required. The report states that successful current replay
cannot retrospectively prove historical code/cache identity.

## Updated completion state, 5 October 2026

The groups2,3,4 job completed. Merged development predictions in
`results/cross_subject/dl_full_combined` cover all1575 queries from15 people:
66.70% equal-run-weighted accuracy,68.13% pooled,.5851 log loss,.7479 mean
participant/run AUC. This is the fair comparison against linear ERP69.98%.

The CPU reload **did not pass** the specified equivalence check:679/680 labels
match, one crosses .5, and maximum probability error is.001675. No tolerance
was widened. Float32 centering of large raw DC offsets is a plausible source;
TF32 and backend effects are not isolated. See NEURAL_PRECISION.md. Same-L40S
replay4268135 passed exactly for all680 probabilities and logits. Disabling
TF32 changes probabilities by at most1.15e-7 with no label changes, so it does
not explain the larger CPU discrepancy.

Both standalone DL notebooks now exist and pass focused synthetic training,
participant-isolation, source-snapshot and checkpoint replay tests. They retrain
from raw CSVs and preserve the current architecture and numerical policy.
Within reproduces the original own-person inner-stopping medians and refitting;
Cross uses a fixed19epochs, the median of development inner epochs19,13,19,13,37.
It does not consult confirmation outcomes for this choice. Both require CUDA
by default and verify immediate checkpoint replay on the same device.

Full raw-data notebook execution completed in4268144:85.15s Within and61.95s
Cross on L40S. Within exactly reproduces all680 original probabilities; both
notebooks pass same-device checkpoint replay. Raw-input hashes, metadata and
fit boundaries match the audits. Frozen Cross confirmation then scored65.33%
target-weighted,69.09% pooled, with14epochs chosen on source-inner validation.
These settings were frozen before ML confirmation in CROSS_CONFIRMATION_FREEZE.md.
S019/S020 are now an **already-used ML/DL confirmation cohort**, not an untouched
future set. Keep their scores separate from development metrics. Kaggle execution
remains outstanding; no notebook or CSV has been uploaded.
