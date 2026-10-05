# Experiment record

## First controlled comparisons (specified before scores)

The user resumed work on 3 October 2026. Additional uploaded datasets and the
research pack are reserved for later. Only the supplied competition recordings
are used in this stage. Competition artifacts and fitted models are separate.

### Data integrity

Marker-led parsing, native-sample epochs, validity masks, channel quality and
run chronology must pass before training. Epochs retain a nominal 250 Hz grid
without resampling. Primary task endpoint is 4.8 seconds to avoid variable
cue-stop boundaries; negative controls filter only pre-cue data.

### Classical phase/representation screen

Question: does prediction rely on early phase-locked EEG, oscillatory power,
or spatial covariance, and does late feedback add useful information?

Prespecified comparisons: ERP 0–0.8, 0–2 and 2–4 s; band power 0–2, 2–4 and
0–4.8 s; baseline-relative full-window power; multiband tangent covariance;
filter-bank CSP; paired task/baseline covariance eigenvalues. Pre-cue ERP and
power are negative controls. Every filter is applied within its own trial and
phase, not across joins. Linear logistic regression has fixed C=0.1, with
scaling, CSP and tangent references fitted on training folds only. No model
receives timestamps, marker spelling, row number or trial order as features.

Cross Subject: six sorted participant groups. Group5 (last two participants)
is completely excluded from development training and evaluation. Groups0–4
are development held-out groups. The best supported branch will be tested
once on group5. Protocol-weighted reporting uses the observed test run mixture.

Single Subject: independent person-only fits, with two stratified folds of
that person's labeled run3 trials and calibration runs in each training set.
All shared initial method settings were specified before results. Aggregate
reporting is descriptive; it will not be used to choose one person's model
using other people's scores. Forward run3 stress tests and own-person
calibration checks will assess chronological sensitivity.

If early ERP wins, invest in cue-sensitive models; if band/covariance wins,
invest in its normalization and protocol shift. If baseline controls are
strong, audit carryover, sequence constraints and filtering before trusting
the task results. Do not make a final candidate from a negative control.

### End-to-end neural screen

Pure PyTorch raw-EEG temporal/spatial convolutions with trial/channel runtime
normalization and ordered temporal pooling. Compare 0–2 and 0–4.8 s. Source
participant inner validation selects epochs for Cross Subject; own-person
calibration validation selects epochs for Single Subject. Outer queries never
select checkpoints. No narrow fixed filter bank or handcrafted classical
feature enters this model. Initial pilot: Cross groups0 and1, then confirmation
or expansion according to results and runtime. ML and DL predictions remain
separate.

## First results and next contrasts (4 October 2026)

Both audits passed. All 1,795 training trials per competition were retained.
Cross has 120 test trials/run; the labeled feedback fraction is 9.47% versus
33.33% at test. Single Subject has a ten-trial labeled prefix followed by a
forty-trial test suffix for every person. See DATA_AUDIT.md for limitations.

| Initial fixed ML model | Cross development, target-run-weighted | Within run3 crossfit, descriptive |
| --- | ---: | ---: |
| ERP 0–2 s | 69.98% | 65.88% |
| ERP 0–0.8 s | 67.14% | 61.76% |
| ERP 2–4 s | 53.43% | 54.71% |
| Power full 4.8 s | 57.17% | 67.06% |
| Tangent covariance full | 57.92% | 68.82% |
| Filter-bank CSP full | 56.63% | 71.18% |
| Pre-cue ERP control | 50.97% | 52.35% |
| Pre-cue power control | 48.13% | 55.88% |

Cross results use 1,575 out-of-fold trials from 15 development people; the two
confirmation people have not been evaluated. Within results use 170 run3
queries. These are local proxies, not leaderboard scores. Participant-level
uncertainty and per-run probabilities are saved in results/<competition>/ml_screen.

Cross early ERP information transfers much better than the initial oscillatory
features. Prespecified follow-up questions now compare: native vs common-average
reference, retaining vs removing each task channel's mean, unit-norm waveform
scaling, linear vs RBF classification, central vs posterior channels, first vs
second cue second, and empirical vs equal source-person/run training weights.
These are focused mechanism tests. RBF uses C=1, gamma=scale and a fixed seed;
the linear contrasts retain C=0.1. No confirmation labels select a setting.
One training-label permutation control will check that the ERP result collapses
under broken label correspondence.

Within diagnostic follow-up tests the original fixed ERP/power/tangent/CSP
recipes on a forward run3 split and run1-to-run2 calibration. Final method
selection, if used, must be computed independently from each person's own
labels inside the validation boundary. No aggregate score is a shared tuning
target for a person's final predictor. The neural full-window baseline was
fixed before neural scores and preserves this separation.

Code review found three infrastructure issues: feature signatures lacked the
raw cache hash; the ML runner bypassed the audited cache loader; and target-run
reporting misleadingly returned zero when a requested target run was absent.
These are being corrected before follow-up jobs. The completed initial runs
had valid input contracts and full target-run coverage, so their reported
primary accuracies are unaffected.

## Completed follow-ups and resumed work (4 October 2026, evening)

The three infrastructure corrections above are complete. Reference, normalization,
RBF and run-weighting comparisons did not establish a meaningful Cross ERP gain:
unit normalization70.11%, CAR69.59%, temporal centering69.64%, RBF66.32%,
person/run weighting69.12%, versus original69.98% (equal run weighting).
The single permuted ERP control scored52.92%. These are all development results.

Within CSP forward validation scored67.06% on85 queries, compared with71.18%
on170 shuffled-prefix queries. The nested person-specific model-family selector
with support-fitted logit offset scored68.24%, versus70.00% without that offset
using the same selected families. Offset improved probability loss, not accuracy.
The paired participant interval for accuracy change is[-4.12,+0.59] points.
No individual predictor may be chosen using pooled results from other people.
See PERSONAL_MODEL.md for the complete information budget and replay evidence.

Cross full-window DL evaluation is complete on the same15 development people:
66.70% target-run-weighted accuracy,68.13% pooled,.585 log loss,.748 mean domain
AUC. The apparently encouraging67.88% pilot was on six people only; the merged
`dl_full_combined` artifacts provide the fair whole-development comparison.
Within full-window DL scored60.00% on170 queries. Both remain research baselines.

The ERP-template branch in ERP_TEMPLATE.md is promising:73.43% target-weighted,
75.11% pooled,.525 log loss,.816 domain AUC. Per-run accuracy gains versus linear
ERP are+2.48/+3.86/+4.00 points. All1575 trial mappings and original folds match;
historical/current feature cache signatures differ, but the selected development
feature matrices are bitwise equal. Templates, reference and scalers fit source
folds only. Paired comparisons are reproducible with `scripts/compare_erp_template.py`.
Ten people improve and five worsen. The conditional participant/run macro gain
is+3.60 points, with a descriptive paired-bootstrap interval[+0.04,+7.64].
An approximate centered-waveform comparator and unit-normalized comparator also
lose on average, but their paired intervals include zero. Shared source folds,
the small cohort, and accumulated model exploration limit inferential claims.

A source-label permutation within participant/run, seed20261004, gives53.16%
target-weighted accuracy and51.43% pooled. This is one negative-control draw,
not a formal significance test. The unchanged models are now compared under a
prespecified alternate grouping of the same15 people. S019/S020 have not yet
entered any development fitting, scoring, early stopping, or this alternate check.

Both standalone ML notebooks passed full raw-data retraining and exactly matched
independent reference fits; metadata and input hashes also match each audit.
DL notebooks subsequently completed full raw-data training in job4268144;
Within exactly reproduces the original680 GPU probabilities. Same-L40S replay
is exact. CPU inference still flips one borderline prediction; the TF32-disabled
GPU contrast is nearly identical (maximum probability difference1.15e-7).
The optional ERP-template notebook also fully passed in job4270273.

The fixed template survived the alternate development groups (73.94% versus
linear69.41%), but failed to improve the primary frozen confirmation score
(74.33% versus76.00%). Linear ERP remains the default Cross ML candidate.
Frozen neural confirmation scored65.33% target-weighted,69.09% pooled. Both
ML/DL configurations were recorded before confirmation; no tuning followed.
S019/S020 are now used confirmation people and must not be presented as fresh.
No Kaggle execution or upload has happened. See PROGRESS.md for live restart state.
