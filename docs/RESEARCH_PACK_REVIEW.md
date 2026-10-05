# Research pack review and experiment priorities

Reviewed **5 October 2026**. The user now authorizes reviewing the pack and its
partial uploads. This review reads local metadata, archive directories, source
code and selected primary references. It does not train on external data,
decompress the large archives, download datasets, or change the user's pack.
The inventory is a point-in-time snapshot while uploads continue.

The strongest immediate opportunity is to improve our handling of cue timing,
normalization and session shift on the supplied recordings. The pack offers
useful independent benchmarks and a verifiable pretrained candidate, but its
published accuracies cannot establish expected Kaggle performance.

## What is actually uploaded

The directory is named `extraernal_doc_eg_kagle/`, including that spelling.
Its `Datasets/` subtree currently contains **95,567,780,540 bytes** (95.6 GB,
89.0 GiB), including archives, code and weights. `Papers/` is empty; 16 PDFs
are distributed inside dataset folders, including a duplicate Zenodo PDF.
File presence alone is not a checksum or full scientific data audit.

| Resource under `Datasets/` | Observed upload | Readiness and next check |
| --- | --- | --- |
| `ds003810_lowcost_mi_rest/` | 50 EDFs, event/channel sidecars, 10 participant directories, two papers; 77.7 MB | Best first independent binary benchmark. All five named runs per person are present. Verify event meanings and exclude the initial execution run from an imagery-only comparison. |
| `ZENODO - EEG Motor Intention Dataset for Rehabilitation Oriented Brain Computer Interfaces/` | `5_Channels_Dataset.rar`, `Full_Dataset.rar`, metadata spreadsheet, README and two copies of the same PDF; 3.27 GB | Actual data are here, although the shorter suggested folder is empty. Not extracted. Inspect FIF/event structure and confirm which portions are execution, intention or rest before constructing labels. |
| `bci_iv/` | 45 GDFs: B01–B09, three T and two E sessions each; five PDFs; 287.6 MB | Dataset **2b** is present. Dataset 2a's description is present, but its A*.gdf recordings are absent. Evaluation labels are not present as separate label files; do not assume unlabeled E-session annotations are class labels. |
| `openbmi_lee2019_mi/` | 67 MATs: all 54 session-1 people and 13 session-2 people; 40.0 GB | Partial second session. Record subject/session coverage before cross-session evaluation; never interpret the current 13 as a random representative sample. |
| `high_gamma_schirrmeister/` | 14 train and 14 test EDFs; 13.75 GB | Recording coverage appears complete by filenames. Header, event, label, montage and checksum checks remain. |
| `nm000151_same_upper_extremity/` | Paper plus `nm000151_v1.0.3.zip`, 18.34 GB | Archive index readable; 48 original DATs and 46 BDF derivatives, plus nested ZIPs. About **55.8 GB** of outer uncompressed content. Resolve derivative coverage and avoid counting originals/derivatives as separate trials. |
| `nm000141_wairagkar_motor_execution/` | Paper plus `nm000141_v1.0.2.zip`, 1.29 GB | Index readable; 14 MAT originals and 14 BDF derivatives. Archive metadata contradicts itself about imagery versus execution; use the original protocol/events. |
| `nm000158_acute_stroke_mi/` | Paper plus `nm000158_v1.0.3.zip`, 1.09 GB | Index readable; 50 EDF originals and 50 BDF derivatives. Treat paired formats as the same participants/data. |
| `nm000245_cho2017/` | Paper plus `nm000245_v1.0.2.zip`, 17.20 GB | Index readable; 52 MAT originals and 52 BDF derivatives. Original arrays may expose controls omitted by derivatives; audit before using rest labels. |
| `hs_stroke_2026/` | Paper, update notice and analysis repository; 3.1 MB | **No EEG recordings uploaded.** Useful methodology/reference code, not a usable training dataset yet. |
| `MIRepNet-main/` | Source plus one 27.0 MB checkpoint; 28.1 MB total | Upstream weight identity verified below. Preprocessing and category eligibility remain to be resolved. |
| `LaBraM_ft-main/` | Source, logs, `vqnsp.pth` and `labram-base.pth`; 229.8 MB | Weights present in a fork. Their exact upstream identity, training-data provenance and releases remain unverified. Logs are development artifacts, not dataset evidence. |
| `EEGPT-main/`, `CBraMod-main/` | Source code, about 1.5/2.3 MB | No pretrained weights present. Local source licenses are Apache-2.0/MIT respectively; checkpoint licenses require separate checks. |
| `arl-eegmodels-master/` | EEGNet code and an example 85.6 kB H5 | An example weight file is not evidence of a suitable motor-attempt model. Training provenance unverified. Prefer an independently trained architecture. |
| `physionet_eegmmidb/`, `stroke_longitudinal_affected_hand_2026/`, `motor_intention_rehab_5ch_32ch/`, `moabb/` | Empty directories | Await uploads; do not treat placeholders as installed datasets/software. |

Local NEMAR ZIP metadata declares CC-BY-4.0 for nm000141/nm000158/nm000245
and CC0-1.0 for nm000151. Its embedded availability reports concern older
derivative versions and do not replace our own checksum/coverage checks.

## Scientific claims requiring care

### NeuRestore is highly relevant; its participant mapping is not established

The paper uses 8-channel Unicorn EEG at 250 Hz, two mutually exclusive cohorts
of 20 people, and three runs. Offline trials have no feedback; online run 3
introduces feedback two seconds after cue onset. Move uses hand imagery plus
an audio squeeze instruction; rest uses a beach image plus relaxation audio.
Its LCR combines predictions from overlapping windows **within one trial**.
Its preprocessing includes ICA and a reported 150 Hz notch despite 250 Hz
sampling. [Published paper](https://repository.essex.ac.uk/42506/1/Neurestore_A_New_Benchmark_for_Wearable_BCI-Based_Neuromotor_Training_in_Real-World.pdf)

Our recorded run-3 feedback makes the pack's proposed *offline 20 = train 17
+ hidden 3* mapping questionable. It remains an unverified hypothesis, not an
identity/label mapping. Class-specific sensory cues could explain early ERP
transfer; this is an inference, not proof of exclusively visual or motor signal.
The 150 Hz notch is above our 125 Hz Nyquist limit and cannot be copied at
the stated rate. ICA is also outside the competition DL exceptions. No source
participant tables or original labels may be used to recover hidden answers.

### PhysioNet's T0 is rest, but not the same kind of trial as Kaggle rest

The official protocol defines T0 as relaxation/rest and gives T1/T2 meanings
that depend on whether the run is unilateral fist or bilateral hand/foot,
execution or imagery. Eyes-open/closed baseline runs are separate conditions.
[PhysioNet documentation](https://physionet.org/content/eegmmidb/1.0.0/)

The pack's shorthand `T0 = rest` is correct as an annotation description.
It does **not** justify converting every T0 interval into a balanced,
cue-locked Kaggle-style rest trial. T0 can follow motor activity and include
recovery; eyes-closed baseline changes physiology and stimulus state. A future
loader must use run-aware labels, exclude boundary carryover, and distinguish
inter-trial relaxation from explicitly instructed rest. This is a protocol
compatibility decision, not a correction to PhysioNet's code definitions.

### Closest external montage/task candidates

The local ds003810 montage is Pz,Cz,T6,T4,F8,P4,C4,F4,Fz,T5,T3,F7,P3,C3,F3
at 125 Hz: exactly **five** target electrodes overlap (Fz,C3,Cz,C4,Pz).
There are no PO7/PO8/Oz channels. NEMAR confirms 10 people, five runs with an
initial execution run, CC0, and the uploaded v1.0.2 release of 4 June 2026.
[NEMAR record](https://nemar.org/dataset/on003810)

Zenodo confirms 30 full-montage people and a curated 24-person five-channel
subset. Its README specifies ANT Neuro, 500 Hz and average reference.
The record was published 18 December 2025; its API identifies CC-BY-4.0.
[Dataset record](https://zenodo.org/records/17980608),
[record metadata](https://zenodo.org/api/records/17980608)

These two resources are our highest-value transfer/benchmark candidates.
They match rest-versus-motor intent more closely than left-versus-right MI,
but montage, referencing, movement execution, feedback and sensory protocol
still differ. Five channels cannot reconstruct the target's three posterior
electrodes without assumptions. BCI IV-2b is valuable for low-channel,
multi-session method checks, but its three **bipolar** channels are not three
equivalent monopolar target electrodes. [BCI IV specifications](https://www.bbci.de/competition/iv/)

The uploaded HS-Stroke README confirms that repeated same-session trials were
removed and gives 23,382 trials / 93,528 one-second samples. These are dependent
windows, not 93,528 independent trials. Pin a deduplicated data release before
any evaluation. [Maintainer update](https://github.com/KINLAWWW/HS-Stroke/blob/main/dataset/README.md)

## External-data and model eligibility

Section 6 of both saved official rule pages permits external data subject to
public/equal access, rights and other requirements; public datasets must exist
before the competition starts, and purpose-recorded competition data are
prohibited. The project snapshot records 22 September 2026 as the start.
[Single rules](https://www.kaggle.com/competitions/low-cost-motor-imagery-decoding-for-rehab/rules),
[Cross rules](https://www.kaggle.com/competitions/low-cost-motor-imagery-decoding-for-rehab-cross-subject/rules).
See [saved rule text](official_snapshot_2026-10-03/cross_subject_rules.md),
[rules review](RULES_AND_SUBMISSION.md).

Thus external resources are **conditionally allowed**, rather than universally
forbidden or automatically cleared. ds003810 and the Zenodo data pass the
basic date/license screening above. Before actual use, pin the exact release,
record hashes and provenance, check transformations against the ML/DL category,
and preserve both competitions' isolation. In Single Subject, permission for
external models does not establish permission to train/tune using the other
supplied participants. A single eligible participant-specific fallback remains
necessary. No external examples have entered our baselines.

### MIRepNet has the best verified checkpoint provenance

The uploaded `MIRepNet-main/weight/MIRepNet.pth` has SHA-256
`432288958007e344a5a84a9ffe9d0e5e5c0cb616aef86c85522375a3f4da9aaf`,
matching the upstream hash in Braindecode's conversion record. Its released
model uses 45 channels, 1,000 samples at 250 Hz, and a three-class head with
undocumented semantic order. Its normal preprocessing is 8–30 Hz filtering,
channel-template interpolation and Euclidean alignment.
[Conversion/model card](https://huggingface.co/braindecode/mirepnet-pretrained/raw/main/README.md)

The official model repository was created 30 July 2025, last modified that
day, revision `9bac0439c0d3e9ffdb40ca675d61a51b439a446e`; the converted
repository was created 31 August 2026, revision
`857f1e3642976be9f2ca6883e5508c3f7c91d86f`. Both model cards declare MIT.
[Official model metadata](https://huggingface.co/api/models/starself/MIRepNet),
[converted model metadata](https://huggingface.co/api/models/braindecode/mirepnet-pretrained).
The source repository's MIT license and pre-competition announcements are
also verified. [Official implementation](https://github.com/staraink/MIRepNet)

This passes a meaningful identity/date/license screen. It is still **not a
drop-in final pipeline**: narrow fixed filters and covariance whitening are
not organizer-cleared DL exceptions here. The uploaded code's EA collects a
whole loader before whitening, which would use a complete target batch if
applied unchanged to test data. Replace the classifier head and document the
exact input/preprocessing boundary before a controlled experiment. Pretraining
on MI does not establish efficacy for class-specific audiovisual cues.

LaBraM weights have not received that provenance audit. EEGPT and CBraMod are
source-only uploads. Their official repositories exist, but this review does
not establish checkpoint dates/identities or their preprocessing eligibility.
[EEGPT](https://github.com/BINE022/EEGPT),
[CBraMod](https://github.com/wjq-learning/CBraMod).

## Three next experiments, in priority order

These are proposed comparisons, not implemented results. Fix their information
budgets and decision criteria before scoring. Our former S019/S020 confirmation
has already been opened once; it cannot become a fresh selection set by renaming
it. Continue development using the original 15 people and sensitivity checks;
leave the Kaggle test answers and counterpart competition entirely separate.

### 1. Cross Subject: a neural model that preserves early cue waveforms

Current evidence favors early ERP: linear ERP 69.98% development versus full
raw DL 66.70%. Template covariance improved development to 73.43% but did not
beat linear ERP on frozen confirmation. This supports investigating phase
information and robustness, not declaring the template solved.

Compare a compact raw-input temporal/spatial network with a phase-preserving
temporal head against the existing PhaseConvNet under exactly the same folds.
Use 0–2 s and a separate baseline-plus-cue input [-2,2) so the network can
learn changes relative to fixation. Test uniform pre-cue centering/scaling
against existing whole-task standardization, with the same rule for every
person. Avoid copying the classical 20 Hz ERP filter into a DL pipeline; learn
temporal kernels or use only the stated broad-filter exception.

Keep ordered time bins and limit capacity. Tune epochs only on inner source
people. Compare run-3 performance, per-person gains, AUC and log loss as well
as equal-run accuracy; include the existing pre-cue negative control.
EEGNet is relevant because its original study includes evoked potentials and
movement-related cortical potentials, not only MI rhythms.
[EEGNet paper](https://pubmed.ncbi.nlm.nih.gov/29932424/)

### 2. Single Subject: shrinkage and baseline-relative calibration

CSP reaches 71.18% on shuffled labeled-prefix queries but 67.06% on forward
queries; full-window DL reaches 60%. This gap identifies session/time drift
as a better first target than a larger neural model.

Prespecify a small family of regularized CSP/tangent models, including a
central-channel motor-band option and task-versus-fixation scale normalization.
Compare them independently inside each participant using their own calibration
runs and only allowed run-3 support. Use a nested chronological check; do not
choose another person's settings from aggregate outcomes. Start with fixed
shrinkage and a small scientific window/band comparison rather than a wide
search. Measure the incremental value of the ten support labels separately
from changes to feature extraction. Offset calibration already improved loss
without improving accuracy, so do not silently promote it.

Later, ds003810 and the four-session nm000151 recordings can test whether this
design survives independent drift. External benchmark conclusions must not
be confused with a target-participant validation result.

### 3. Cross Subject: generalization using source domains only

Compare one compact backbone trained with balanced source-person sampling
against the same backbone with a modest cross-person, class-conditioned
representation penalty or supervised contrastive term. Fit all parameters and
normalizers exclusively on source people. Treat participant/run as domains
for the training objective; do not feed their numeric IDs as prediction inputs.
Keep the baseline backbone and loss budget fixed so the comparison answers
whether domain regularization helps, rather than whether a larger model helps.

EEG-DG explicitly targets unseen people without target training data, which
fits our conservative Cross information budget. Its large benchmark scores
are on different tasks and are not expected Kaggle scores.
[Author preprint](https://arxiv.org/abs/2311.05415),
[official code](https://github.com/zxchit2022/EEG-DG).
Ordinary EA is unsupervised but still consumes target signals to estimate a
reference; label-free is not the same as target-free.
[EA author preprint](https://arxiv.org/abs/1808.05464)

## Decisions and remaining research work

Proceed with the three supplied-data experiments above before large-scale
external pretraining. Use ds003810 first for independent benchmarking, followed
by the Zenodo five-channel data after an event audit. Keep MIRepNet as the first
pretrained candidate to investigate after resolving preprocessing eligibility.
Do not bulk-upload the external corpora, papers or weights to Git.

Defer ASFA/SSTDA, full-target EA/RPA, pseudo-label adaptation and entropy updates
until their target-access budgets and category eligibility are explicit. A
source-only reference or trial-local pre-cue normalization is a different
experiment from fitting on every target trial; report them separately.
Do not apply LCR across neighboring trials or force class counts inferred from
the original experiment.

The pack was read in full, but this is a **selected primary-source review**,
not a replication of every listed paper. ATCNet/Conformer, REVE/BIOT, detailed
RPA/contrastive-DG implementations and the other recent clinical studies still
need protocol/code/license review before implementation. A source author's
high accuracy, a repository name or a filename is not that review.

Keep an experiment ledger containing immutable configs, input/source hashes,
inner/outer split membership, per-run/person probabilities and paired
comparisons. Record the user's first Kaggle scores alongside these local
proxies, and preserve executed notebook artifacts. Reaching first place remains
an objective to pursue; the current evidence cannot predict or guarantee it.
