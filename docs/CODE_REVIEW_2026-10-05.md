# EEG pipeline review, 5 October 2026

This review inspects the data loader, classical and personal estimators, neural
runner, validation helpers, and standalone notebook generators. It reads
development training histories and performs two tiny synthetic CPU calculations.
It does not train models, inspect hidden labels, reopen confirmation labels, or
change an existing candidate. The historical confirmation results mentioned
below come from the existing freeze document.

The main opportunity is representation and protocol matching. I found no
evidence that a test label or another participant enters a Within fitted model.
The numerical input policy needs a controlled repair, and the neural baseline
currently discards information that the stronger classical methods preserve.
These findings support specific experiments; they do not establish that a
particular change will improve Kaggle accuracy.

## Evidence that the current boundaries work

- `data.py:37–95` rejects mixed train/test marker types, unknown markers and
  incomplete trial boundaries. `extract_epochs` uses eight named EEG channels;
  timestamps and marker strings do not enter the predictors. Recording and phase
  validity are separate, and the runners reject unsupported crops.
- `classical.py:167–194` fits CSP filters, covariance references, feature scaling
  and classifiers on fold training rows. ERP extraction is trial-local. The
  template estimator fits its class templates and covariance reference on source
  rows. Query predictions do not update these objects.
- `validation.py:14–24` and `run_neural.py:108–130` remove both the query people
  and the confirmation people from Cross development fitting. Neural inner
  stopping uses different source people from its outer queries.
- `run_neural.py:132–171` and `personal.py:94–145` keep Within fitting, support,
  selection and adaptation inside the same participant. Neural GroupNorm has no
  target-batch running state. Personal family selection uses own calibration
  losses and own support leave-one-out losses.
- The notebook generators embed source hashes, reject a mixed competition input
  root, reconstruct ordering from native markers, record exact fit rows and
  validate exported IDs. DL performs immediate checkpoint replay. This is useful
  execution evidence, although Kaggle execution remains a separate check.

S019/S020 have already been scored for the frozen ML/DL comparison. They are
now an already-used confirmation cohort, and must not be described as a new
untouched set when assessing subsequent experiments.

## Prioritized findings

### P1: Large-DC float32 centering is numerically unstable

**Numerical defect; accuracy impact and CPU/GPU cause remain unproven.**
`data.py:127` casts CSV EEG values to float32 before storing epochs. Then
`neural.py:57–62` subtracts a float32 time mean from those large values. At
700,000 file units, float32 spacing is 0.0625. Conversion back to float64 in a
classical extractor cannot recover the lost low bits. The neural standardizer
adds a second error by reducing the large offset before subtraction.

The existing saved Within GPU-to-CPU replay changes one of 680 thresholded
predictions, with maximum probability difference 0.001675. Same-L40S replay is
exact, and disabling TF32 changes probabilities by only 1.15e-7. Those results
rule out TF32 as the explanation for this particular discrepancy, but they do
not isolate standardization from downstream convolution/backend differences.

A new small CPU diagnostic uses NumPy seed 20261005, shape `(2,8,1200)`, Gaussian
standard deviation 20, and offset -700000. The reference normalizes the same
already-quantized input in float64:

| Float32 method | Maximum deviation from reference | Maximum residual channel mean |
| --- | ---: | ---: |
| Subtract raw time mean | 0.0043590 | 0.0043238 |
| Subtract first sample, then time mean | 4.77e-7 | 8.90e-8 |

The first-sample shift is algebraically identical in exact arithmetic and uses
no fitted or cross-trial statistics. It addresses reduction cancellation,
**not** the earlier CSV-to-float32 quantization. A separate higher-precision
loader contrast would be required to assess the latter. Preserve historical
checkpoints and retrain a new configuration; do not patch only their inference.
Add an audited-scale offset-invariance check: the current test exercises offset
42, far below the delivered offsets.

### P1: Task/channel normalization removes useful candidate information

**Confirmed invariance; performance hypothesis.** `neural.py:57–62` divides every
task channel by its own task standard deviation. For positive gain `a_c` and
constant shift `b_c`, this makes `z(a_c*x_c+b_c)` approximately equal to `z(x_c)`.
A synthetic test with channel gains 1 through 8 changes the normalized input
by at most 4.77e-7. The model therefore cannot recover independent channel gain,
total within-window channel energy, or a constant task-versus-baseline shift
that was discarded before its first convolution. GroupNorm subsequently removes
another within-group global scale, although relative learned feature structure
can survive it.

By comparison, classical ERP uses the preceding baseline to normalize the task
(`classical.py:103–111`), preserving task departures from that baseline. Tangent
features retain log band trace (`:155–159`); CSP deliberately removes each band's
total power (`:161–164`) while preserving spatial ratios. Thus the neural versus
classical results compare materially different information, as well as models.
The current neural score does not show that raw EEG lacks the classical signal.

Normalization over all 4.8 task seconds also lets later feedback samples set
the scale and offset of early samples. This is legal trial-local inference, but
an early-versus-full-window comparison changes both available time and the
normalization context. Interpret it as a combined contrast, not an isolated
test of feedback information.

### P1: Stopping and evaluation do not reproduce later Within feedback trials

**Validation limitation and optimization hypothesis, not label leakage.**
Within epoch selection validates on run2, while its outer score comes from the
small run3 prefix (`run_neural.py:146–159`). The final hidden trials occur later
in run3. Randomly interleaving five support and five query prefix trials is a
valid own-person cross-fit, but may overstate transfer under within-run drift.
The existing classical forward check already falls from 71.18% to 67.06%; its
85 queries still come from the beginning of run3. The neural runner currently
has no analogous forward mode.

The same number of epochs also means different Adam update counts after refit
(`run_neural.py:474–480`). For example:

| Development fit | Inner rows | Refit rows | Selected epochs | Inner updates to selection | Refit updates |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cross group0 | 940 | 1245 | 19 | 285 | 380 |
| Cross group1 | 990 | 1270 | 13 | 208 | 260 |
| Within S001 half0 | 55 | 105 | 38 | 38 | 76 |
| Within S007 half0 | 31 | 55 | 8 | 8 | 8 |

Counts use batch64 and the saved histories/splits. For S011 half0, the recorded
selection is 28epochs, giving 56 refit updates versus 28 at selection. The
absolute smallest validation loss occurs at epoch29, but its additional gain
is below the declared 1e-4 improvement rule. Conventional
epoch transfer is legitimate; the discrepancy is evidence of a changed
optimization budget, not proof of overfitting. The Cross notebook's fixed
19epochs on all1795 rows gives 551 updates, whereas its development inner
selection histories used fewer updates. Source-inner loss also optimizes
probability calibration, while the leaderboard scores hard accuracy.

### P2: Eight learned phase bins are coarse for the cue-response mechanism

**Architecture hypothesis.** Two pools reduce time by sixteen and then eight
ordered means/standard deviations summarize the learned features
(`neural.py:103–111`). At 4.8seconds, bins span approximately 0.58–0.64seconds;
the 2second model's bins span approximately 0.19–0.26seconds. The successful
linear ERP keeps 40ms bins. The full-window comparison therefore also changes
physical temporal resolution. Learned convolutions can encode shape before
pooling, so this does not prove an information bottleneck, but increasing
capacity indiscriminately is less informative than checking this mechanism.

### P2: Neural partial-coverage reporting differs from classical reporting

**Reporting defect for incomplete run coverage.** `run_neural.py:245–260`
renormalizes available run weights and calls the result
`target_weighted_accuracy` even when target runs are missing. Classical
`validation.py:79–90` correctly labels that quantity conditional and sets full
target accuracy to null. Current full Cross and Within run3 results have complete
coverage and are unaffected. Future calibration-only or restricted-run neural
experiments could produce a misleading score. Use matching full/conditional
coverage fields before adding such experiments.

The classical primary equal-run score is pooled within each run. With incomplete
source runs, it is not exactly an equal-participant/equal-run estimand. Keep the
existing participant-macro and coverage reports visible; do not treat either
bootstrap as an exact private-leaderboard uncertainty interval.

### P2: Feature/window interpretations require edge and power controls

**Method limitations, not demonstrated implementation errors.** Filtering each
crop independently prevents neighboring-phase contamination, but `sosfiltfilt`
uses reflected edges and accesses the full selected crop. A 1Hz lower band on a
2second crop contains only about two cycles, making edge handling influential.
Lowpass ERP averages can capture ocular, muscle, auditory or visual responses;
the current scores establish predictive EEG-channel information, not a motor
cortex mechanism. Near-flat baselines can greatly amplify task ERP before
the fixed clipping at20. Quality/error analysis should determine whether this
concentrates failures before adding artifact rejection.

## Three controlled experiments

1. **Numerical repair first.** Compare historical task/channel normalization
   against first-sample-shifted normalization with identical architecture,
   batches, seeds, data and source splits. First isolate CPU/GPU standardizer
   outputs and then feed identical precomputed normalized tensors into copied
   downstream networks. Next train fresh models and save paired development
   OOF predictions. Record probability replay, gradients, per-run loss/AUC and
   accuracy; distinguish a reliability fix from an accuracy gain. Exclude
   S019/S020 from development and preserve the historical source snapshots.

2. **Give DL the same baseline normalization context as the cue-response
   hypothesis.** After fixing numerical centering, compare task-derived channel
   z-scores against preceding `[-2,0)` baseline-derived channel centering/scaling.
   Both networks receive the same raw `[0,4.8)` time sequence, architecture and
   training budget; only runtime normalization context changes. Baseline
   statistics come from the same single trial, with a declared epsilon, no
   fitting, narrow filters or hand-engineered predictor vector. Assess early
   cue and run3 errors, not only pooled accuracy. A gain would justify a later
   temporal-resolution contrast; no gain would weaken the amplitude/context
   explanation. Check the current category rules before packaging this new
   normalization in a final notebook.

3. **Within refit budget and temporal transfer.** Add a chronological first-five
   support/last-five query evaluation with exactly the same own-person inner
   stopping boundary as the random cross-fit. Compare epoch transfer against
   update-count transfer: `selected_epochs * ceil(inner_n/batch_size)` becomes
   the target refit update budget, with the exact rounding/last-batch policy
   recorded. Keep normalization and all optimizer settings fixed. Report each
   person's paired outcomes and run1→2 behavior. Any final per-person choice
   must use only that person's allowed training/support evidence; a pooled
   improvement cannot select another participant's Within model.

None of these experiments requires external datasets or target adaptation. New
pretraining, auxiliary channels, cross-trial alignment and ensembles would add
different information/legal questions and should be separate research branches.
The first Kaggle submission will test release ordering and local transfer; a
public score alone cannot decide which representation mechanism is correct.

## Documentation correction

The last paragraph of `NEURAL_PRECISION.md` still says same-L40S reproducibility
is pending, despite its earlier completed replay section. Its conclusion should
state that same-L40S replay is exact and CPU portability remains unresolved.
`NEURAL_DESIGN.md` similarly contains historical "confirmation unopened" and
"queued" paragraphs followed by current updates. Preserve historical chronology,
but add an explicit dated status banner so readers cannot mistake the old text
for current task state.
