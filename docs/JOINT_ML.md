# Fixed joint cue-response and oscillation experiment

Preregistered 5 October 2026, before execution. The question is whether the
early cue waveform and source-fitted oscillatory spatial variances carry
complementary information. A single classifier receives their concatenated
features; this is not a probability ensemble.

`JointERPCSP` concatenates unchanged `erp_pre` (400 baseline-standardized,
lowpass 20 Hz, 40 ms EEG waveform features from 0–2 seconds) and the 24 log variances
from unchanged `csp_full` (six bands, 0–4.8 seconds). Each source fold alone fits
two smallest/two largest generalized-eigenvalue filters per band, matching the
existing CSP baseline formula. One StandardScaler and logistic classifier with
C=.1 fit the combined 424 features. There is no neural dependency, feature-weight
search, class-count constraint, test-batch adaptation, or final fitting here.

Cross uses the same 1,575 development queries and groups 0..4 as the baselines;
group 5 is excluded from fitting and queries. Within executes the original
own-person twofold run 3 prefix evaluation (170 queries) and the predeclared
first-five→last-five forward stress test (85 queries). Each person's model fits
only their supplied calibration and permitted run 3 support. Aggregate Within
scores are descriptive and cannot select another participant's predictor.

Preserve per-fold models, cache/source hashes, exact splits, OOF probabilities
and run/person metrics. Compare paired predictions against the existing ERP
and CSP OOF files on identical query rows. A Cross improvement needs consistent
domain results and uncertainty assessment before it earns final packaging. A
Within advantage needs own-person support evidence and forward robustness.
No improvement weakens this simple linear-complementarity hypothesis; do not
rescue it by scanning feature weights or using confirmation feedback.

The PBS job executes the two competitions independently on CPUs and writes
`results/<competition>/joint_ml_v2`. Cross results use `development/`; Within
uses `run3/` and `forward/`. This branch has no effect on the first Kaggle
handoff notebooks or CSVs.

## Completed results and decision

CPU job 4272743 completed the fixed experiment. All 56 saved fold models replayed
their query probabilities exactly in the same process (maximum difference 0).
`scripts/compare_joint_ml.py` verifies complete 1,575/170/85 query coverage,
identical baseline metadata/labels/folds, source/query boundaries, checkpoint
hashes and replay evidence. Historical/current ERP and covariance cache features
are bitwise equal on every used source/query row. The old `ml_screen` runs
lack a historical source-split manifest and signal-cache digest: their source
splits are reconstructed from the frozen config and hashed metadata, while
their saved OOF query assignments are checked exactly. This does not restore
missing historical records.

The paired result is recorded in `results/joint_ml_v2_review_paired.json`, with
20,000 participant-bootstrap draws, seed 20261003. Positive deltas favor joint:

| Protocol and frozen reference | Joint primary accuracy | Reference primary accuracy | Primary delta | Participant conditional mean delta [descriptive 95% CI] | Log-loss delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cross development, linear ERP | 66.12% | 69.98% | -3.86 pp | -3.80 pp [-6.80, -0.67] | +.0588 |
| Within run 3 prefix, CSP | 72.35% | 71.18% | +1.18 pp | +1.18 pp [-6.47, +8.82] | +.1143 |
| Within forward prefix, CSP | 69.41% | 67.06% | +2.35 pp | +2.35 pp [-3.53, +8.24] | +.1908 |

The primary Cross score weights pooled run 1/2/3 accuracies equally. Its interval
instead describes the mean of participant-specific paired deltas after each
participant's run weights are renormalized over their available runs; S007
lacks run 2. Those are different estimands. Within each person has equal query
count and only run 3, so the point deltas coincide. All intervals are descriptive:
training folds overlap, cohorts are small, and development data guide research.

Cross loses accuracy on all three runs and worsens probability loss. Five
development people improve, nine worsen and one is unchanged. The simple
joint classifier is therefore rejected as a Cross replacement for linear ERP;
do not rescue it through feature-weight scanning or confirmation tuning.

Within gains only two net correct decisions in each evaluation. The prefix
comparison changes 24 wrong decisions to correct and 22 correct decisions to
wrong; forward changes 12 and 10. Both uncertainty intervals include no advantage,
and log loss worsens materially (.6643 versus .5500 prefix; .8858 versus .6951
forward). Keep the existing CSP baseline as the current handoff. Joint remains
a documented research result, not a globally selected Within winner; any later
participant-specific choice requires only that person's allowed support evidence.
No additional variant or leaderboard inference is justified by this experiment.
