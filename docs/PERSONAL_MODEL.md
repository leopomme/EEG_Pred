# Participant-only source models with run-3 offset calibration

Specified and implemented 4 October 2026. Only the within-subject competition
cache is used. External data and the research pack remain deferred.

## Question and fixed design

Does the small labeled run-3 prefix correct a source model's decision offset
more reliably than fitting another high-dimensional classifier on those few
trials? The competing explanation is a loss of ranking, which a constant
offset cannot repair. Four source families were retained from the original
screen as a fixed menu: `erp_pre`, `power_full`, `tangent_full`, `csp_full`.
Each uses logistic regression with C=0.1. No aggregate result selects a
family, shared prior, hyperparameter, or threshold for another participant.

For each participant separately:

1. Fit each family on own runs 1/2 only. Scaling, CSP and tangent references
   are fitted exclusively on those source trials.
2. Assess source calibration using run1→run2 and run2→run1, equally weighted.
   If only one calibration run exists, use its contiguous halves in both
   directions. All training blocks must contain both labels.
3. Fit one intercept offset on own allowed run-3 support. Its objective is
   summed binary log loss plus `2 * offset² / 2`. This fixed zero-centered
   prior keeps single-class support finite and shrinks toward the person's
   source model. It is not a prior learned from other people.
4. Select the family by `0.5 * held-calibration log loss + 0.5 * support
   leave-one-out offset log loss`. Every LOO support prediction excludes its
   own label from the offset fit. Exact ties follow the listed family order.
5. Evaluate on untouched own run-3 outer query trials. Two stratified folds
   supply ten query predictions per participant. The outer labels do not fit
   source models, offsets, or family selection.

The final predictor repeats the same selection with all ten own run-3 labels,
fits the selected source model on own runs 1/2, and adapts its intercept on
the ten labeled support trials. The decision threshold is 0.5. Predictions
use trial-local features; no target-batch statistics, hard counts, or test
labels enter fitting. This is an ML entry with no neural dependencies.

## Evaluation and decision

The runner retains per-family source and offset OOF predictions, plus the
selected family's predictions with and without its offset. It reports accuracy,
log loss, within-person/run AUC, person-bootstrap uncertainty, and paired
person-wise accuracy changes. Selection is evaluated as a complete nested
procedure; the most successful family across participants is not selected as
the final shared family.

Improved calibration with comparable AUC supports intercept adaptation.
Unchanged ranking and little calibration gain reject that explanation. Any
aggregate result is descriptive and cannot retune another person's policy.
The ten labeled trials are an early run-3 prefix: randomized cross-fitting
does not establish performance on the late forty-trial test suffix. Forward
stress tests and support-size sensitivity remain separate follow-ups.

## Reproducibility and safeguards

`tests/test_personal.py` checks finite prior behavior, LOO label exclusion,
fixed selection weights, held-query-label independence, isolation from another
person's labels/features, source independence from run-3 labels and test
features, legal support membership, and the missing-run fallback.

Run `jobs/personal_within.pbs` on CPU (2 CPUs, 10 GB, 20-minute limit).
Artifacts are written to `results/within_subject/personal_offset/` and existing
output directories are refused. Configuration records raw-cache and code
hashes, feature signatures, seed, fixed policy and information budget.
`provenance.json` records every source, calibration, support and query index,
offset, inner family score and final family choice. The runner serializes
one model per participant and writes exactly 680 ordered test predictions.
Candidate CSVs must pass `scripts/make_submission.py`; no upload is automatic.

## Results

CPU job `4266650.pbs-7` completed successfully on 4 October 2026 in 37 seconds.
All 17 participants completed; there are 170 untouched outer-query predictions
per method and 680 final test predictions. Seven targeted tests passed before
submission. These are local validation results, not Kaggle scores.

| Fixed method | Source accuracy | Offset accuracy | Source log loss | Offset log loss |
| --- | ---: | ---: | ---: | ---: |
| ERP 0–2 s | 61.76% | 61.76% | 0.986 | 0.935 |
| Full-window power | 63.53% | 68.24% | 0.703 | 0.631 |
| Full-window tangent | 64.71% | 65.88% | 0.694 | 0.637 |
| Full-window CSP | 70.00% | 71.18% | 0.633 | 0.573 |
| Nested participant-only selector | 70.00% | 68.24% | 0.628 | 0.587 |

The selector's source control uses the exact same family selected by the
support-based policy, with its fitted offset removed. Its offset accuracy
is 116/170, compared with 119/170 for this source control. The paired
participant-mean accuracy change is −1.76 percentage points, with descriptive
95% participant-bootstrap interval [−4.12, +0.59] points. The selector's absolute
accuracy interval is [61.76%, 74.12%]. Thus the fixed offset improves log loss
but does not establish an accuracy improvement for the complete selection
policy. None of the four fixed-family accuracy-change intervals excludes zero.
The CSP result is descriptive; it does not authorize choosing CSP for everyone
from aggregate supplied-participant outcomes.

Offsets preserve each outer query fold's ranking exactly, as verified from
saved predictions. The selector's mean outer-fold AUC is 0.8162 both before
and after adaptation. Pooled within-person OOF AUC changes slightly because
the two folds have separately fitted offsets; it should not be mistaken for
a within-fold representation change.

Final participant-only selections are 2 ERP, 2 power, 4 tangent and 9 CSP
models. Each serialized model reproduces its saved test probabilities to
absolute error below 3e-15. Source/support/query provenance checks passed for
every participant. See `postrun_verification.json` in the result directory.

`submissions/within_subject_personal_offset_ML.csv` passed the official mapping
validator: 680 rows, IDs 0–679, 375 `rest` and 305 `move` predictions, with its
own manifest. It is a reproducible local candidate and has not been uploaded
or executed in Kaggle. Its legal family-selection procedure is established;
competitive superiority is not. This candidate has not yet been wired into
the existing Kaggle notebook builders.

Next decision: retain this fixed procedure for blocked/forward and support-size
stress tests, which directly probe its limited ten-trial calibration context.
Do not tune a new shared prior or globally promote one family using these
descriptive scores. The lack of clear accuracy benefit favors prioritizing
protocol drift and representation quality over a larger intercept search.
