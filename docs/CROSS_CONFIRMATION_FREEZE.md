# Cross Subject first confirmation freeze

Recorded 4 October2026, before the first S019/S020 confirmation scoring.
These people have been absent from every development training, tuning, early
stopping and query set. Final notebook baseline refits may include all supplied
source people; their hidden-test predictions have not selected a configuration.

## Frozen comparisons

- ML reference: `erp_pre`, baseline-standardized0–2s EEG,20Hz lowpass,40ms bins,
  eight channels, empirical training weights, StandardScaler and logistic C=.1.
- ML candidate: exact `ERPTemplateCovariance(C=.1, shrink=.1)` using those same
  binned waveforms; two source class templates, shrinkage covariance, source
  mean tangent reference, StandardScaler and logistic C=.1.
- DL research baseline: unchanged PhaseConvNet/NeuralConfig, raw0–4.8s EEG,
  per-trial/channel normalization, AdamW lr=.001/weight decay=.01, batch64,
  maximum60epochs/patience10, source-participant inner stopping and full-source
  refit. Seed20261003; canonical `run_neural.py --folds 5` split. No template
  features or ML outputs enter this model. The final retraining notebook uses
  fixed19epochs, selected from development inner stopping before confirmation;
  its test predictions must not be selected using confirmation epoch results.

All thresholds remain .5. No blending, counts, calibration, target-batch
statistics or adaptation is permitted in this comparison. Each model fits the
fifteen development people and scores the220 labeled trials from S019/S020.
Report each person/run, equal-run-weighted accuracy, probability loss and AUC.

## Evidence and decision

Template versus reference on original groups:73.43% versus69.98% target-weighted.
On alternate seed20261004 groups:73.94% versus69.41%. All run-specific gains
in the original split were positive; paired person uncertainty is substantial.
One source-label permutation gives53.16%. The compact DL baseline gives66.70%
on the complete original development split and remains a separate DL-category
fallback, not a competitor to blend with ML.

If the template's confirmation evidence supports the development gain, retain
it as the leading Cross ML candidate for Kaggle verification. If it does not,
record the failure or ambiguity and retain the linear reference; do not retune
on these two people. Even a positive result on two people does not demonstrate
private-leaderboard superiority or resolve late-feedback extrapolation.

Future experiments must describe S019/S020 as an already-used confirmation set.
They cannot be silently reused as untouched evidence. Same-category ensemble
eligibility and external datasets remain deferred/unresolved as documented.

## First confirmation outcome

CPU job4268143 completed the frozen ML comparison. The template's primary
target-weighted gain did **not** confirm:74.33% versus76.00% for linear ERP.
Template pooled accuracy was81.36% versus80.00%, and log loss improved from
.5561 to.4856, but mean domain AUC fell from.8076 to.7731.

| Model | Run1 (100 trials) | Run2 (100 trials) | Run3 (20 trials) |
| --- | ---: | ---: | ---: |
| Linear ERP | 87% | 76% | 65% |
| ERP-template | 86% | 82% | 55% |

Both score40% on S019's ten feedback trials. On S020 feedback, template scores
70% versus linear90%. Two changed feedback decisions have a large effect on
the equal-run-weighted estimate. This small cohort leaves substantial uncertainty;
it does not justify tuning the template to these people. The linear ERP remains
the default Cross ML candidate. The template is preserved as an explicitly
mixed-evidence alternative. Neither has been evaluated on the Kaggle leaderboard.

Frozen DL confirmation completed in job4268144, after both DL notebook checks.
Its configuration was recorded above before the ML confirmation was opened.
It scored65.33% target-run-weighted and69.09% pooled accuracy, with .6032
log loss. Run1/2/3 accuracies were69%/72%/55%. Source-inner stopping selected
14epochs without consulting confirmation labels. The final notebook retains
19epochs fixed from development. No configuration was changed using these
results. Both ML and DL confirmation are now used.
