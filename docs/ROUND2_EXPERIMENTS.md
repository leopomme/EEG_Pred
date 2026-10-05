# Round 2 — cue information and neural normalization

Preregistered 5 October 2026 before real-data results; completed results appended.
The first-place objective requires stronger evidence and models; the current
handed-off notebooks remain the tested first baseline for leaderboard feedback.

## Decisions and fixed contrasts

The code review found that task/channel z-scoring removes independent channel
gains and task-to-baseline level/amplitude changes. Large raw DC offsets also
make the original float32 mean reduction numerically sensitive. The research
review confirms class-specific audiovisual cues in the source protocol, while
our classical experiments show that early cue responses transfer better than
standalone oscillatory features. Test these concrete mechanisms first.

1. **phase_stable**, task 0–4.8 s: the original 1,449-parameter architecture, with
   algebraically equivalent task normalization after subtracting one task sample.
   This isolates large-DC reduction accuracy without adding information.
2. **phase_baseline**, task 0–4.8 s: normalize using the same trial's 2 s pre-cue
   baseline, then the same original phase network. This preserves cue-relative
   level and amplitude before the network's GroupNorm. It tests normalization
   information, while acknowledging GroupNorm can still attenuate it.
3. **cue_baseline**, task 0–2 s: the same baseline normalization plus a 1,729-parameter
   learned depthwise temporal/spatial network and ordered 50-bin readout. No
   GroupNorm or second task z-score removes the cue-relative amplitude. This
   combines a focused cue window and a different architecture; it is not a
   pure architecture ablation against the full-window models.

All models receive raw eight-channel EEG. No fixed lowpass, narrow filter bank,
classical feature transform, external data or pretrained model enters them.
Normalization is a uniform runtime function independent for each trial/channel.
AdamW lr=.001, weight_decay=.01, batch 64, cap 60 epochs, patience 10, clipping 5,
seed 20261003 and threshold .5 match the original optimizer policy. TF32 and
deterministic flags are recorded. These are three researcher-designed comparisons,
not an automatic search over architectures or tuned parameter grids.

## Validation and decision rule

Cross: evaluate the same five participant-held-out development groups, 1,575
queries. Exclude S019/S020 from every fit/stop/query set. Their already-used
confirmation scores will not select these new configurations. Compare paired
OOF probabilities with full-window PhaseConvNet 66.70% and linear ERP 69.98%
equal-run-weighted accuracy, plus participant/run AUC, loss and feedback accuracy.
Promising gains need another seed/source grouping and a shuffled-label control;
weak branches should be dropped instead of retuning indefinitely on these folds.

Within: run the same 34 independent own-person folds on 170 run3 queries.
Every architecture is prespecified before its aggregate results; epoch selection
uses only the individual's source labels. Aggregate Within accuracy is descriptive
and cannot choose one participant's algorithm using other participants' outcomes.
If a candidate is pursued, add own-person forward stress tests and own-person
selection rather than choosing a pooled winner.

The runner writes exact source/cache hashes, inner/outer splits, losses,
optimizer-step budgets, checkpoints and one prediction per query. Each saved
checkpoint is reloaded and checked on the same device. It forbids Cross fold 5
and produces no hidden-test CSVs. Test predictions are packaged only after a
candidate's training/selection policy has been justified.

Eight model tests already pass: stable large-DC reductions against a float64
reference, preserved baseline-relative signals, offset/scale invariance, trial
independence, finite flat-channel gradients, serialization and actual synthetic
cue-signal learning. These establish behavior, not competition performance.

## Execution controls

The full 58-test suite passed on the cluster, and all six two-epoch CPU smoke
routes completed with checkpoint replay and the expected source/query boundaries.
These short runs verify execution and do not select a model.

The first L40S allocations on cx3-20-5 failed before training because
`nvidia-smi` found no visible GPU. The retries pinned to cx3-20-0 remained queued
with a multi-day estimate and were cancelled before execution. The main jobs
now request an available RTX6000 without a host constraint. Each allocation
first reruns the historical full-window PhaseConvNet as `v2_native_reference`,
then runs the three new variants. Comparisons use this matched GPU control and
also retain the original L40S baseline for transparency. Hardware-dependent
rounding must not be mistaken for a modeling improvement. The existing first
submission notebooks and predictions remain intact.

The matched RTX6000 controls completed for both competitions. The phase
contrasts ran, but the cue branch exposed a deterministic-CUDA backward error
in PyTorch's adaptive pooling. Its prespecified temporal sequence already
has exactly 50 bins, making that pooling operation redundant. A numerical
equivalence fix keeps deterministic training enabled. All pre-fix variant
outputs are archived in `v2_pre_poolfix/`; the three variants are rerun into
fresh directories after a real CUDA backward check. Matched native controls
are retained because their implementation is unchanged. No result from the
failed cue fit is used for model selection.

## Follow-ups after this contrast

- Partial-run neural reports now label conditional accuracy and coverage;
  complete-run scores are intact. A regression test covers both cases.
- Test transferring optimizer updates rather than epoch counts, with source-only
  decisions and fixed optimization settings, if stopping/refit mismatch warrants it.
- If raw cue models become competitive, test source-only domain regularization
  and channel reliability before increasing model size.
- Closest uploaded external benchmarks are ds003810 and the five-channel motor
  intention release. Audit labels, event timing, exact montage and eligibility
  before any external training. MIRepNet's checkpoint is authentic, but its
  prescribed preprocessing and target alignment need additional eligibility work.

See CODE_REVIEW_2026-10-05.md and RESEARCH_PACK_REVIEW.md for the source evidence.

## Completed Cross results

All five development groups completed on RTX6000; all three variants passed
deterministic CUDA backward and saved-model replay. The comparator verifies
current cache/source hashes, full canonical query identities, configured modes
and shared optimizer/seed/backend settings. Results are saved in
`results/cross_subject/v2_comparison.json`.

| Model | Equal-run accuracy | Pooled accuracy | Log loss | Mean participant/run AUC |
| --- | ---: | ---: | ---: | ---: |
| Matched native PhaseConvNet | 66.85% | 68.32% | .5889 | .7375 |
| Stable task normalization | 66.85% | 68.32% | .5889 | .7376 |
| Baseline-normalized phase model | 64.81% | 65.33% | .6076 | .6984 |
| Baseline-normalized early cue model | 66.26% | 67.30% | .6019 | .7342 |
| Classical linear ERP reference | 69.98% | 71.87% | .5774 | .7717 |

Stable task normalization changes no Cross decisions versus the matched native
control. It is a numerical safeguard, with no accuracy gain established.
Baseline normalization loses 2.04 primary percentage points; the cue model
loses .59 points. Their participant-conditional paired mean changes are
-2.09 points [-6.98,+2.44] and -.11 points [-3.49,+2.98], respectively. Those
descriptive intervals concern participant means over available runs, whereas
the primary score weights pooled run accuracies. Shared fitted folds and model
exploration limit the uncertainty estimates.

**Do not promote either baseline-conditioned Cross variant.** Neither accuracy,
loss nor domain ranking improves. A one-decision improvement on feedback for
each variant does not justify a new run-aware model. These results weaken the
specific proposed normalization/architecture contrast; they do not establish
that cue information itself is absent. Linear ERP remains stronger locally and
the pure DL handoff remains the unchanged historical PhaseConvNet.

The historical L40S phase score was 66.70%, versus 66.85% for the RTX6000 native
rerun. That difference is not attributed to a model change. The original pilots
lack historical raw/source hashes; current metadata pairing cannot restore those
records. New variants are compared primarily against the recorded matched control.

## Completed Within results

All 34 own-person folds completed for each variant, covering the same 170 run3
prefix queries. All 102 saved variant checkpoints replay exactly. The result
is in `results/within_subject/v2_comparison.json`. Each model's stopping and
fitting used only its participant's permitted source labels. These aggregate
figures describe fixed methods and cannot select another participant's predictor.

| Model | Prefix crossfit accuracy | Log loss | Mean participant/run AUC |
| --- | ---: | ---: | ---: |
| Matched native PhaseConvNet | 61.76% | .6594 | .6004 |
| Stable task normalization | 61.18% | .6594 | .6004 |
| Baseline-normalized phase model | 65.88% | .6322 | .6334 |
| Baseline-normalized early cue model | 61.76% | .6384 | .6363 |
| Classical CSP reference | 71.18% | .5500 | .7975 |

The baseline-normalized phase model gains seven net correct decisions versus
the matched native control: +4.12 points, paired participant interval
[-2.35,+11.18]. Seven people improve, five worsen and five are unchanged.
Probability loss and domain ranking also improve. This supports a cautious
own-person baseline-normalization follow-up, rather than a globally selected
Within winner. Its uncertainty interval includes no advantage; the ten-trial
prefix still does not validate later hidden-feedback trials. Next evaluation
must use each person's own support for selection and include forward stress
tests and another fixed seed.

The early-cue model has no net accuracy improvement and has six people improve,
seven worsen. Drop this configuration rather than scanning its widths/windows.
Stable centering changes one borderline decision against the native control;
it improves arithmetic behavior but establishes no predictive gain. The native
RTX6000 rerun differs from the historical L40S baseline (61.76% versus60.00%);
comparisons use the matched reference without attributing that change to a
new model.

No new variant is packaged for submission from these results. The four tested
first-attempt notebooks remain the handoff. Cross S019/S020 confirmation was
not used again, and no external recordings or pretrained weights entered training.

## Verification and restart

PBS4272749 completed with exit0 in7m59s. All117 variant fold checkpoints replay
exactly; all six experiment directories are complete. The repaired pooling has
forward/gradient parity tests and all three modes pass actual deterministic
CUDA backward. Final expanded suite: **66 tests passed**,128.20s, PBS4273255.
No project jobs remain queued/running at completion of this record.

For a resumed run, read this record, `PROGRESS.md` and the saved comparisons.
Do not relaunch completed output directories or reopen the used confirmation.
