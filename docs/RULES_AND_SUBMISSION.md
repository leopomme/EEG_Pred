# Rules and submission contract

Verified 3 October 2026 against current official Kaggle pages and public metadata.
The source responses are preserved in [official_snapshot_2026-10-03](official_snapshot_2026-10-03/README.md).
This file separates published requirements from conservative project choices and unresolved questions.

## Confirmed output contract

| Item | Single Subject | Cross Subject |
| --- | --- | --- |
| Competition ID | 143178 | 157329 |
| Slug | `low-cost-motor-imagery-decoding-for-rehab` | `low-cost-motor-imagery-decoding-for-rehab-cross-subject` |
| CSV columns, in order | `ID,TARGET` | `ID,TARGET` |
| IDs | Every integer from **0 through 679** exactly once | Every integer from **0 through 359** exactly once |
| Rows excluding header | 680 | 360 |
| Target values | `rest`, `move`, lowercase | `rest`, `move`, lowercase |
| Row ordering | Ascending participant, then original epoch position in that participant's test file | Ascending participant, then session, then original epoch position in that file |
| Metric | Trial classification accuracy | Trial classification accuracy |
| Daily submissions | 5 | 5 |
| Selected final submissions | At most 2 | At most 2 |
| Team size | At most 5 | At most 5 |
| Public fraction configured | 60% | 59% |
| Final deadline | 19 February 2027, **01:00 Paris** / 00:00 UTC | Same |

The ID origin is explicit in the official evaluation examples. Their example labels are arbitrary and must not be treated as answers. The Cross Subject example explicitly orders S008 session 001 before S008 session 002, then subsequent people/sessions. Preserve row order within each CSV; do not sort trials by label, quality, prediction or a reconstructed original sequence. Sort parsed participant/session numbers explicitly, with filenames as a deterministic tie-breaker. Current zero-padded filenames also support lexical file ordering. [Single evaluation](https://www.kaggle.com/competitions/low-cost-motor-imagery-decoding-for-rehab/overview/evaluation), [Cross evaluation](https://www.kaggle.com/competitions/low-cost-motor-imagery-decoding-for-rehab-cross-subject/overview/evaluation)

Neither supplied archive contains a sample submission. No separate authoritative sample CSV was obtained during this review; the verified contract above comes from the official evaluation text plus metadata. Submission validation must assert contiguous IDs, exact row counts, valid labels, unique trial mappings and the required sort order. A future sample CSV should be checked if one becomes available.

Metadata gives enabled times of 22 September 2026, 13:39:39.600 UTC (Single) and 13:39:54.243 UTC (Cross). Distinct entry/merger deadlines were not returned and remain unverified. The public leaderboard percentages do not identify which trial is public or private. Exact metadata responses are in the snapshot.

## Learning boundaries that affect implementation

Single Subject requires separate participant training without using the other supplied participants' data. Keep fitting, supervised feature construction, hyperparameter selection and run-3 calibration within each person. A fixed researcher-specified algorithm is a safe initial comparison; using pooled supplied-person outcomes to select another person's algorithm is not presumed permitted. [Single overview](https://www.kaggle.com/competitions/low-cost-motor-imagery-decoding-for-rehab/overview)

Cross Subject trains across source people and evaluates unseen people. Keep this competition's fitted statistics, checkpoints and predictions separate. The Single overview prohibits using the other competition's data; both specific rule pages explicitly prohibit cross-referencing the counterpart test set. Existing extraction-stage comparisons between archives are not training permission and must not be extended into record matching or label recovery. [Single rules](https://www.kaggle.com/competitions/low-cost-motor-imagery-decoding-for-rehab/rules), [Cross rules](https://www.kaggle.com/competitions/low-cost-motor-imagery-decoding-for-rehab-cross-subject/rules)

| Category | Published requirement | Initial eligible implementation |
| --- | --- | --- |
| Classical ML | No neural-derived transforms; PyTorch and TensorFlow explicitly excluded. A narrow single-hidden-layer MLP exception exists. | NumPy/SciPy/scikit-learn EEG features and classical estimators; no neural library import or neural feature/checkpoint dependency in the ML notebook. |
| DL | End-to-end neural classification; classical LDA/PCA/ICA/FIR/SVM operations excluded except stated preprocessing exceptions. | Raw EEG or 50 Hz notch plus broad 1–100 Hz preprocessing, with neural temporal/spatial layers. |
| DL normalization | Same preprocessing function for every participant; runtime data-dependent parameters allowed; participant-specific hard-coded constants forbidden. | Uniform per-trial or training-fitted scalar/channel normalization with explicitly documented information access. |
| DL expanded transforms | Time must remain an axis; examples include spectrograms and CWT; compute transforms from raw signals in the submitted notebook within the allowed runtime. | An optional runtime time-frequency branch, kept separate from handcrafted-feature ML. |

These category requirements appear in section 2.10 of both specific rule pages and in their descriptions. The evaluation pages permit choosing how much of a test epoch is used, so timing-window experiments are supported. The broad-filter exception does not automatically permit arbitrary narrow fixed filter banks. [Single rules](https://www.kaggle.com/competitions/low-cost-motor-imagery-decoding-for-rehab/rules), [Cross rules](https://www.kaggle.com/competitions/low-cost-motor-imagery-decoding-for-rehab-cross-subject/rules)

Both overviews allow one ML and one DL final entry. Maintain independently reproducible pure candidates, and do not blend ML and DL outputs. Same-category ensembling is not expressly resolved in the inspected material; keep a single-model fallback while evaluating complementary models locally.

## Kaggle execution and reproducibility

The organizer's 27 September reply explicitly requires the notebook generating the CSV to be submitted and executed in Kaggle. The category can be indicated in its title; the host was developing automatic category declaration. Public metadata currently says `fileBasedSubmissions: true`, but this does not supersede the notebook requirement. The winning-solution rules additionally require training code, inference code and a reproducible environment description. [Organizer clarification](https://www.kaggle.com/competitions/low-cost-motor-imagery-decoding-for-rehab/discussion/743366)

Current **platform** documentation specifies 12 hours for CPU/GPU sessions and complete Save & Run All execution, 9 hours for TPU, and 20 GB persisted under `/kaggle/working`. It lists 4 CPU cores, 30 GB RAM for CPU sessions; P100 or T4×2 GPU sessions with 4 CPU cores and 29 GB RAM. Treat this as a planning reference, not proof of a challenge-specific runtime allowance or the user's available hardware/quota. [Kaggle notebook technical specifications](https://www.kaggle.com/docs/notebooks#technical-specifications)

No numeric challenge-specific runtime, GPU quota, internet policy or explicit permission to attach a cluster-trained checkpoint was found in current overview, data, rules, evaluation, metadata or the organizer thread. Cluster experiments are useful development, but a checkpoint-only final notebook is not yet organizer-cleared. Keep a compact pipeline whose training and inference can run within one Kaggle session. Full retraining in Kaggle is the conservative fallback.

Before any final submission, record Kaggle image/package versions, seeds, attached data/checkpoint provenance, exact training and inference entrypoints, elapsed runtime and output hash. Execute the final notebook from a clean state. Neither a successful cluster job nor a local CSV proves Kaggle reproducibility.

## Other confirmed restrictions

Human labeling or human prediction of hidden validation/test records, multiple-account participation and private code/data sharing outside the team are prohibited. AutoML products and similar automatic model-building tools are prohibited. Use researcher-designed, hypothesis-driven comparisons. Competition data are noncommercial CC BY-NC-SA 4.0. Open-source code used for submissions must satisfy the foundational OSI/commercial-use requirement, subject to actual specific exceptions. Audit source-code and checkpoint licenses separately. [Single rules](https://www.kaggle.com/competitions/low-cost-motor-imagery-decoding-for-rehab/rules), [Cross rules](https://www.kaggle.com/competitions/low-cost-motor-imagery-decoding-for-rehab-cross-subject/rules)

External datasets/models are conditional on accessibility and rights; public datasets must predate the competition start, and purpose-recorded competition datasets are forbidden. Specific rules contain an exception to relicensing qualifying input data/pretrained models, not a blanket exception for noncommercial source code. The newly uploaded research pack and external datasets are deferred at the user's request and were not inspected or used in this review.

## Organizer questions prepared, not sent

1. May final Kaggle notebooks load checkpoints trained on our cluster using only the appropriate competition training data, with all training code and dependencies supplied? Must the final notebook itself retrain? What numeric runtime and accelerator limits apply?
2. In Single Subject, is any common model/hyperparameter choice based on other supplied people's validation results permitted, or must every selection remain participant-specific?
3. Are same-category probability ensembles permitted, and may the two selected final entries ever belong to the same category?
4. For DL, does runtime normalization allow full spatial covariance whitening/Euclidean alignment? Separately, may its parameters use every unlabeled trial of the target participant/run?
5. Are full-target-batch latent normalization, pseudo-label optimization and entropy-gradient adaptation permitted? These use different information and update budgets, so please distinguish them.
6. Does the DL time-preserving-transform exception allow fixed narrow filter banks and anti-aliasing/resampling needed by public pretrained models?
7. If completeness and selection are proved using only the same competition's release, may joint predictions enforce a class count inferred from the original run design?
8. May auxiliary inertial/device channels be predictors if present, and what separates an ordinary controlled parameter sweep from prohibited AutoML?

Until clarified, these are final-submission eligibility gaps rather than reasons to stop safe labeled-data experiments. No organizer messages, rule acceptance, credential use, uploads or leaderboard submissions were performed during this review.
