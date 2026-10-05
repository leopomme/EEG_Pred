# EEG Kaggle project — progress and remaining work

Updated: **5 October 2026 (London)**. Main restart/handoff document.
Original objective: [PROMPT_FOR_CODEX.md](PROMPT_FOR_CODEX.md).

## How far are we from completing the original prompt?

**Preparation and a complete, locally reproducible first set of ML/DL baselines
are finished for both competitions. First-round model development and frozen
Cross confirmation are also finished. The strongest-competitive-solution
objective is not complete.**

The next external milestone is execution in Kaggle and limited leaderboard
validation. Better modeling, especially DL and late feedback, remains research
work. A percentage completion or guaranteed final score would be misleading.

| Stage | Status | Remaining boundary |
| --- | --- | --- |
| Data, environment, cluster, initial rules | Complete | Advanced-method eligibility questions remain documented. |
| Classical/neural baselines and controls | Complete | Local validation, not leaderboard results. |
| First stronger-candidate comparisons | Complete | Personal offset did not improve accuracy; Cross template gain did not confirm on the primary metric. |
| Frozen Cross confirmation | Complete | S019/S020 are now used and no longer an untouched set. |
| Standalone ML/DL artifacts | Locally validated | Four baseline notebooks and an optional ML alternative execute from raw data. |
| Kaggle execution and final selection | User will execute | First leaderboard scores are pending. |
| Round 2 normalization/cue and joint-feature experiments | Complete | No submission replacement established; own-person neural follow-up remains. |

## Completed preparation and boundaries

- Both archives extracted and extracted-file checksums verified; independent
  caches in `artifacts/within_subject/` and `artifacts/cross_subject/`.
- Each competition: 1,795 labeled trials, 50 training files, 17 source people;
  run counts 825/800/170. S006 run1 shortened; S007 run2 absent.
- Within: 680 test trials, 40 per supplied person. Each run3 training release
  is an early ten-trial prefix followed by the test suffix; prefix validation
  does not fully reproduce late-run prediction.
- Cross: 360 test trials, 40 per run for unseen S008/S013/S015. Feedback is
  9.47% of training but 33.33% of test; equal-run weighting is the local proxy.
- Channels, markers, timestamps, quality, ordering and safe windows audited.
  Every trial supports baseline [-2,0) and task [0,4.8); not all support 5 s.
- `myenv` works: Python 3.10.16, NumPy 1.26.4, SciPy 1.15.3, pandas 1.5.3,
  sklearn 1.2.1, PyTorch 2.3.0+cu121. Existing unrelated conflicts left intact.
- Actual jobs verified NVIDIA L40S, 46,068 MiB reported VRAM, driver 580.82.07.
- Each Within model fits/tunes only its own participant's labels. Aggregate
  Within scores are descriptive, not permission for cross-person model choice.
- Cross development uses 15 people. S019/S020 are excluded from development
  fitting, stopping and scoring. Their frozen confirmation is now complete.
  Final refits can use all 17 source people for hidden-test predictions.
- Competitions' data, learned statistics, models and predictions stay separate.
  Initial archive comparisons are not extended or exploited.
- Pure ML/DL remain separate. No forced counts, target-batch fitting,
  pseudo-label updates, neural-derived ML features or ML/DL blending.
- Verified output: `ID,TARGET`, lowercase `rest`/`move`; IDs 0–679 Within,
  0–359 Cross, ordered by participant/session/original epoch.
- The user authorized review of the research pack and uploaded resources on
  5 October. Inventory and primary-source review are now complete; no external
  data or checkpoint has been used for training or current submissions.

Details: [DATA_AUDIT.md](docs/DATA_AUDIT.md), [SETUP_NOTES.md](SETUP_NOTES.md),
[RULES_AND_SUBMISSION.md](docs/RULES_AND_SUBMISSION.md). The setup note's pending
GPU statement is historical and superseded by successful real GPU execution.

## Results obtained

**Local validation only; no Kaggle leaderboard scores.** Cross rows use equal
run weighting. Within rows use held-out run3 queries.

| Model/evaluation | Accuracy |
| --- | ---: |
| Cross linear ERP, original 15-person development | 69.98% |
| Cross ERP unit normalization, same development | 70.11% |
| Cross ERP-template, same development | 73.43% |
| Cross linear ERP / template, alternate development groups | 69.41% / 73.94% |
| Cross template label-permutation control, one draw | 53.16% |
| Cross full-window neural, complete development | 66.70% |
| Cross linear ERP / template, frozen 220-trial confirmation | 76.00% / 74.33% |
| Cross neural, frozen 220-trial confirmation | 65.33% |
| Within fixed CSP, 170 prefix crossfit queries | 71.18% |
| Within fixed CSP, 85 forward queries | 67.06% |
| Within personal selector, without / with offset | 70.00% / 68.24% |
| Within fixed full-window neural, 170 crossfit queries | 60.00% |
| Cross matched RTX6000 neural / stable centering | 66.85% / 66.85% |
| Cross baseline-normalized phase / early-cue neural | 64.81% / 66.26% |
| Within matched RTX6000 / baseline-normalized phase | 61.76% / 65.88% |
| Within stable centering / early-cue neural | 61.18% / 61.76% |
| Cross joint ERP+CSP, same development | 66.12% |
| Within joint ERP+CSP, prefix / forward | 72.35% / 69.41% |

Cross cue-sensitive ERP outperforms the initial power/covariance baselines.
Reference changes, RBF and run weighting did not establish a gain. Template
improvements survived alternate grouping, but varied by participant. The original
paired conditional participant/run macro gain was +3.60 points, descriptive
bootstrap interval [+0.04,+7.64]; shared folds and prior exploration limit it.

The template failed to improve the primary confirmation metric. It improved
pooled accuracy and loss, but feedback accuracy fell from 65% to 55% on only
20 queries. **Linear ERP remains the default Cross ML candidate.** Template
is a mixed-evidence alternative, with no confirmation tuning. DL remains a
separate category baseline needing improvement. Within offset adaptation
improved probability loss but not accuracy; no accuracy-based promotion.

Round 2 does not establish a replacement for the first submission notebooks.
Cross baseline-conditioned neural models and joint classical features lose
accuracy. Within baseline-conditioned phase improves over its matched neural
control by4.12 points, descriptive paired interval[-2.35,+11.18]; it earns an
own-person source-selection/forward follow-up, not pooled model selection.
Within joint features improve only two net decisions in each protocol while
loss worsens. Keep the current handoff. See
[ROUND2_EXPERIMENTS.md](docs/ROUND2_EXPERIMENTS.md) and [JOINT_ML.md](docs/JOINT_ML.md).

Details: [EXPERIMENTS.md](docs/EXPERIMENTS.md),
[ERP_TEMPLATE.md](docs/ERP_TEMPLATE.md), [PERSONAL_MODEL.md](docs/PERSONAL_MODEL.md),
[CROSS_CONFIRMATION_FREEZE.md](docs/CROSS_CONFIRMATION_FREEZE.md).

## Validated notebooks and local candidates

| Notebook | Full local runtime | Evidence |
| --- | ---: | --- |
| `notebooks/within_subject_ml.ipynb` | 43.09 s | 680 predictions exactly match independent reference. |
| `notebooks/cross_subject_ml.ipynb` | 41.01 s | 360 predictions exactly match independent reference. |
| `notebooks/within_subject_dl.ipynb` | 85.15 s, L40S | 680 predictions exactly match original GPU fits; 17 fresh models. |
| `notebooks/cross_subject_dl.ipynb` | 61.95 s, L40S | 360 predictions; 19-epoch source refit and exact same-device reload. |
| `notebooks/cross_subject_erp_template_ml.ipynb` | 39.65 s | Optional alternative; exact independent-reference and saved-model replay. |

All retrain from raw CSVs without external data, downloaded code or attached
checkpoints. Audit metadata, recorded raw-input hashes, training boundaries,
output hashes and submission ordering pass validation. ML checks absence of
neural imports. DL uses raw EEG and uniform per-trial normalization.

Cross DL confirmation used source-inner stopping (14 epochs). Its final notebook
uses 19 epochs fixed from development inner choices, unchanged by confirmation.

Six local research CSVs and manifests are in `submissions/`: Cross linear ERP,
Cross template, Cross PhaseConvNet; Within CSP, Within personal-offset, Within
PhaseConvNet. Personal-offset has no dedicated notebook and is not one of the
four baseline handoff entries. No CSV is uploaded.

See [NOTEBOOK_VALIDATION.md](docs/NOTEBOOK_VALIDATION.md) and
[KAGGLE_RUNBOOK.md](docs/KAGGLE_RUNBOOK.md).

## Neural numerical checks

- Same-L40S reload: exact probabilities, logits and all 680 labels; max error 0.
- Complete Within notebook retraining from raw data also matches exactly.
- TF32-disabled L40S inference: max probability difference 1.15e-7; no label
  changes. This does not explain the much larger CPU discrepancy.
- CPU replay: max difference .001675; one borderline label changes (ID378,
  .500222 GPU versus .499801 CPU). Cross-device identity has not passed;
  tolerances were not widened.
- Large-DC float32 centering is a plausible contributor, supported by a synthetic
  check. Round 2 tests stable centering in separately recorded variants: no Cross
  decisions change against the matched native model, and one Within decision
  changes. This does not establish cross-device identity or isolate the original
  CPU discrepancy; historical notebook models were not silently changed.
- Early pilots lack historical signal/source hashes. New notebooks record full
  provenance, but replay cannot retroactively prove missing historical identity.

See [NEURAL_PRECISION.md](docs/NEURAL_PRECISION.md).

## Latest jobs and exact restart state

**All Round 2 jobs have completed; no project job remains queued/running.**
Inspect `qstat -u lh5218` and saved outputs before resubmitting any job.

- 4272749: complete neural rerun after deterministic pooling repair, both
  competitions on Quadro RTX 6000, cx3-11-8, exit0,7m59s. All three modes pass
  CUDA backward; all117 variant checkpoints replay exactly.
- 4272737/8: matched native controls completed; new variants reached the cue
  branch, then failed before fitting it because adaptive pooling lacks a
  deterministic CUDA backward path. Pre-fix variants are archived and rerun.
- 4272743: fixed ERP+CSP classical experiment completed for Cross development
  and Within prefix/forward validation; all saved fold models replay exactly.
- 4272742: **63 tests passed**,112.07 s; subsequent pooling regression suite
  **nine tests passed**, including matching forward values and gradients.
- 4273255: final expanded suite **66 tests passed**,128.20 s, exit0.
- 4272698: all six two-epoch CPU neural smoke routes completed successfully.
- 4272685:58 tests passed. The new joint model adds five behavioral tests.

The RTX6000 allocations report24,576 MiB and driver580.82.07. The original
L40S submissions remain unchanged. Earlier Round 2 L40S allocations4272677/8
failed before training because their node exposed no GPU; host-pinned retries
4272696/7 were cancelled while queued. No incomplete fit was reused.

Earlier completed baseline jobs:

- 4268135: exact GPU replay plus TF32 contrast.
- 4268144: both DL notebooks and frozen Cross DL confirmation.
- 4270273: independent template refit and full template notebook.

Results: `results/<competition>/dl_notebook_validation_01/`,
`results/cross_subject/erp_notebook_validation_01/`,
`results/cross_subject/dl_confirmation/`, `results/cross_subject/erp_confirmation/`.
Logs are under `logs/`; notebook directories contain manifests, embedded source,
predictions, submissions, execution logs and validation reports.

Assistant usage interruptions did not stop batch jobs. Inspect saved outputs
and `qstat -u lh5218` before resubmitting; do not mistake interruptions for failures.

## Remaining work

- [x] Data/environment/GPU preparation and empirical audit.
- [x] First classical/neural comparisons, controls and personal adaptation.
- [x] Comparable Cross development and frozen ML/DL confirmation.
- [x] Alternate grouped and paired template checks.
- [x] Four pure baseline candidates with standalone raw-training notebooks.
- [x] Full local execution, same-device replay and artifact validation.
- [ ] Actual Kaggle execution; record environment, runtime and generated hashes.
- [ ] Small justified leaderboard comparison and final entry selection.
- [ ] Stronger DL/feedback representation and robustness where development
  evidence supports further work. Used confirmation is not fresh evidence.
- [ ] Resolve numerical portability before claiming cross-device identity.
- [ ] Resolve relevant eligibility questions before advanced adaptation,
  checkpoint-only inference, ensembles or external-model use.

The user will execute notebooks and submit in Kaggle. No organizer messages,
Kaggle uploads/submissions or external-data training experiments have been
performed by the assistant.

**User choice on 5 October: the user will run the notebooks in Kaggle.** Do not
request credentials again. The handoff archive is
`deliverables/kaggle_handoff_2026-10-05.zip`; use the four baseline notebooks
first, preserving each executed version, submission, manifest and any scores.

Use `/rds/general/user/lh5218/home/anaconda3/envs/myenv/bin/python`, PBS for heavy
computation, fresh outputs and strict competition/person boundaries. Update
this file after every substantial result. Do not re-open confirmation for tuning.

## New research and repository work — 5 October

The user will submit the first attempts and wants continued research toward
first place in both competitions. First place is an objective, not a promised
outcome. Detailed new reviews are in
[CODE_REVIEW_2026-10-05.md](docs/CODE_REVIEW_2026-10-05.md) and
[RESEARCH_PACK_REVIEW.md](docs/RESEARCH_PACK_REVIEW.md).

- Full pre-round2 suite: **49 tests passed**, PBS job4272592,147.55s.
- Removed22 finished PBS `.o`/`.e` wrappers; experiment logs/results preserved.
  All PBS scripts now direct future wrapper outputs into ignored `logs/`.
- External datasets, fitted joblib files and generated deliverables explicitly
  ignored in Git. Raw EEG/models remain outside commits.
- Baseline checkpoint committed as **886352e**. Push failed because GitHub
  authentication was unavailable. **The user will push the commits themselves**;
  do not retry authentication or request credentials. Further changes are
  committed locally after validation.
- Three fixed neural contrasts are implemented: stable task centering,
  baseline-referenced full-window normalization, and a compact early-cue model
  preserving cue-relative amplitude. Original validated notebooks are preserved.
  Eight model behavioral tests and all six CPU smoke routes pass. Full runs
  include a matched-device historical control. See
  [ROUND2_EXPERIMENTS.md](docs/ROUND2_EXPERIMENTS.md).
- A single ERP+CSP classifier tests complementary cue and oscillatory features
  without probability blending. Seven behavioral/comparison tests pass, and
  all56 saved fold models replay exactly. See
  [JOINT_ML.md](docs/JOINT_ML.md).
- The research-pack inventory, primary-source review and strict all-file
  ds003810 event/header audit are complete. Its imagery subset contains1,590
  trials; all support baseline[-2,0) and task[0,2), but none supports4.8s.
  One execution trial lacks an observed end marker; native EDF units are blank
  despite sidecar microvolt declarations. See
  [EXTERNAL_LOW_COST_AUDIT.md](docs/EXTERNAL_LOW_COST_AUDIT.md).
  No external signal has entered a competition model.
- No reuse of S019/S020 for tuning or new confirmation is planned. External
  resources must pass task/channel/provenance/eligibility audits before training.
