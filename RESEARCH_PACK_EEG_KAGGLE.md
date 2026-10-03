# EEG Kaggle Research Pack

**Project:** Low Cost Motor Imagery Decoding for Rehab — Single Subject & Cross Subject  
**Purpose:** External reference pack for ChatGPT Work / Astra / Codex  
**Last checked:** 2026-10-03

> This file is a research map, not permission to use external data in a Kaggle submission.
> Before any external dataset or pretrained checkpoint is used in a final competition pipeline,
> verify the current Kaggle rules and/or organizer clarification.
>
> The most useful role of these resources may be **idea generation, benchmarking and method selection**
> even if external training data are ultimately forbidden.

---

# 1. Target competition context

The Kaggle problem appears to be **REST vs motor attempt / movement-intention decoding from low-density EEG**, with separate:

- **Single Subject**: generalization across sessions/time within the same participant.
- **Cross Subject**: generalization to unseen participants.

A highly likely scientific source is:

**Chowdhury et al. (2026), “Neurestore: A New Benchmark for Wearable BCI-Based Neuromotor Training in Real-World”**  
IEEE Access 14:10064–10080  
DOI: `10.1109/ACCESS.2026.3652957`  
University of Essex repository: https://repository.essex.ac.uk/42506/

The paper reports a wearable 8-channel g.tec Unicorn setup in a real-world/noisy environment, with 20 participants in the offline experiment and another 20 in the online experiment.

**Important working hypothesis to verify from the actual Kaggle files:** the offline 20-participant NeuRestore cohort may correspond to 17 visible/train participants + 3 hidden Cross Subject participants.

---

# 2. External datasets

## Priority legend

- **P0** = download / inspect first
- **P1** = strong relevance
- **P2** = useful benchmark or pretraining source
- **P3** = mainly method-validation / broad representation learning

---

## P0 — Motor Imagery vs Rest — Low-Cost EEG System (OpenNeuro ds003810)

**Why it matters:** Probably the closest open dataset in task structure and hardware philosophy.

- **Subjects:** 10
- **Recordings:** 50
- **Runs:** 5 per participant
- **Task:** dominant-hand kinesthetic grasp imagery vs rest/idle
- **Special detail:** includes an initial real-movement run
- **Channels:** 15 EEG
- **Sampling:** 125 Hz
- **Approx size:** ~69 MB in EEGDash summary
- **Hardware:** low-cost OpenBCI Cyton + Daisy
- **EMG:** dominant-hand EMG recorded for protocol validation
- **License:** CC0
- **OpenNeuro DOI:** `10.18112/openneuro.ds003810.v2.0.2`

**Why useful for this Kaggle project**
- Binary **hand-grasp vs rest** rather than left-vs-right MI.
- Low-cost / low-density context closer to the target.
- Could test whether preprocessing/methods generalize between inexpensive EEG systems.
- Useful for transfer/pretraining experiments if external training is allowed.

**Main mismatch**
- Motor imagery rather than motor attempt.
- 15 channels and different montage.
- Different visual/cue protocol.

**Sources**
- NEMAR: https://nemar.org/dataset/on003810
- EEGDash: https://huggingface.co/datasets/EEGDash/ds003810
- OpenNeuro ID: `ds003810`

**Suggested local folder**
`external_datasets/ds003810_lowcost_mi_rest/`

---

## P0 — EEG Motor Intention Dataset for Rehabilitation-Oriented BCIs

**Why it matters:** Extremely relevant because a curated low-density version already exists.

- **Subjects:** 30 healthy participants in full dataset
- **Full montage:** 32 EEG channels
- **Curated subset:** 24 low-noise participants
- **Low-density subset channels:** **Cz, C3, C4, Fz, Pz**
- **Task family:** upper- and lower-limb motor intention
- **Format:** FIF + experimental metadata spreadsheet
- **Released:** 2025

**Why useful**
- Explicitly rehabilitation-oriented.
- Low-density subset is unusually close to Kaggle’s sparse montage.
- Very useful for testing whether 5 motor-related channels already contain most of the transferable signal.
- Potential source for motor-intention pretraining if rules allow it.

**Source**
https://zenodo.org/records/17980608

**Suggested local folder**
`external_datasets/motor_intention_rehab_5ch_32ch/`

---

## P0 — PhysioNet EEG Motor Movement/Imagery Dataset (EEGMMIDB)

**Why it matters:** Large subject count and contains both **executed** and **imagined** movements plus rest.

- **Subjects:** 109
- **Channels:** 64 EEG
- **Sampling:** 160 Hz
- **Format:** EDF+
- **Annotations:** `T0 = rest`; `T1/T2` movement/imagery event classes depending on run
- **Tasks:** actual and imagined fist movements, unilateral and bilateral variants
- **Repeated runs:** multiple paradigms per participant

**Why useful**
- Provides **actual movement + imagery in the same resource**, allowing study of representations shared between execution and imagery.
- This is valuable because the Kaggle task may be closer to **motor attempt** than classical pure MI.
- Large N is useful for cross-subject representation learning.

**Main mismatch**
- 64-channel montage.
- 160 Hz.
- Different cues/protocol.
- Target labels are not exactly Kaggle’s binary motor-attempt/rest setup.

**Source**
https://physionet.org/content/eegmmidb/1.0.0/

**Suggested local folder**
`external_datasets/physionet_eegmmidb/`

---

## P0 — Motor Imagery of Same Upper Extremity (NEMAR nm000151 / Tavakolan2017)

**Why it matters:** Contains **rest vs hand grasping imagery** explicitly and multiple sessions.

- **Subjects:** 12 healthy
- **Sessions:** 4 per subject
- **Classes:** rest, right-hand grasping, right-elbow flexion
- **Trials:** 20 per class per session
- **Channels:** 32
- **Sampling:** 1000 Hz
- **Trial duration:** 3 s
- **Format:** BCI2000 / BIDS derivative
- **Approx archive size:** 18.3 GB

**Why useful**
- `rest` and **grasping** are directly present.
- Four sessions make it particularly useful for studying **cross-session drift**.
- Good external testbed for Single Subject methods.

**Source**
https://www.nemar.org/dataset/nm000151

**Suggested download**
`nemar dataset download nm000151`

**Suggested local folder**
`external_datasets/nm000151_same_upper_extremity/`

---

## P0 — Acute Stroke Motor Imagery Dataset (NEMAR nm000158)

**Why it matters:** Large clinical cohort close to the ultimate rehab setting.

- **Subjects:** 50 acute stroke patients
- **Stroke stage:** approximately 1–30 days post-stroke
- **Task:** imagined left- vs right-hand grip
- **Channels:** 29 EEG
- **Sampling:** 500 Hz
- **Data:** raw and preprocessed
- **Approx archive size:** 1.1 GB

**Why useful**
- Tests whether methods survive altered physiology and large patient heterogeneity.
- Useful for understanding subject-specific variability.
- Potential transfer-learning source if allowed.

**Main mismatch**
- Left-vs-right imagery rather than movement-vs-rest.
- Clinical population, higher channel count.

**Source**
https://ww2.nemar.org/dataset/nm000158

**Suggested download**
`nemar dataset download nm000158`

**Suggested local folder**
`external_datasets/nm000158_acute_stroke_mi/`

---

## P0 — Longitudinal Stroke “Affected-Hand” / “Sixth-Finger” MI Dataset (Scientific Data, 2026)

**Why it matters:** Longitudinal stroke dataset with repeated stages of rehabilitation.

- **Subjects:** 24 stroke patients
- **Paradigms:** affected-hand motor imagery + sixth-finger motor imagery
- **Stages:** pre-training, post-training, follow-up
- **Data:** raw EEG, preprocessed EEG, clinical information
- **Reported baseline:** CSP+SVM / CSP+LDA around 85–86% for the two MI paradigms

**Why useful**
- Very valuable for studying **longitudinal distribution shift**, which may inform the Single Subject competition.
- Clinical relevance to rehabilitation.
- Potentially useful for learning what remains stable across weeks/stages.

**Main mismatch**
- Not directly rest-vs-motor.
- Stroke pathology may produce different spatial patterns from healthy subjects.

**Paper/source**
https://www.nature.com/articles/s41597-026-07787-y

**Suggested local folder**
`external_datasets/stroke_longitudinal_affected_hand_2026/`

---

## P1 — Wairagkar et al. 2018 Motor Execution Dataset (NEMAR nm000141)

**Why it matters:** Movement intention/execution and rest, aligned around movement onset.

- **Subjects:** 14 healthy, BCI-naive
- **Channels:** 19
- **Sampling:** 1024 Hz
- **Tasks:** right/left index-finger tapping and rest
- **Trials:** current NEMAR metadata reports 1,665 trials
- **Preprocessing:** filtering + ICA artifact removal in distributed derivative
- **Approx archive size:** 1.3 GB

**Why useful**
- Actual motor activity + rest may be closer to motor attempt than classic left/right imagery.
- Could help identify execution-related features transferable to attempted movement.

**Caution**
NEMAR summaries have used both “motor execution” and “motor imagery” wording for derivatives. Inspect original paper/events before assuming exact behavioral condition.

**Source**
https://www.nemar.org/dataset/nm000141

**Suggested download**
`nemar dataset download nm000141`

**Suggested local folder**
`external_datasets/nm000141_wairagkar_motor_execution/`

---

## P1 — OpenBMI / Lee2019_MI

**Why it matters:** Large cross-subject and cross-session benchmark.

- **Subjects:** 54 healthy
- **Sessions:** 2
- **EEG channels:** 62 (66 total channels in MOABB metadata)
- **Sampling:** 1000 Hz
- **Task:** left-hand vs right-hand motor imagery
- **Trials:** ~100 per session per participant
- **Trial window:** ~4 s
- **Dataset DOI:** `10.5524/100542`
- **Paper DOI:** `10.1093/gigascience/giz002`

**Why useful**
- Strong benchmark for cross-subject and cross-session generalization.
- Widely used by recent domain-adaptation papers.
- Good place to test whether a proposed method genuinely generalizes before applying it to Kaggle.

**Main mismatch**
- Left-vs-right MI, not movement-vs-rest.
- Dense montage.

**Sources**
- MOABB: https://moabb.neurotechx.com/docs/generated/moabb.datasets.Lee2019_MI.html
- Dataset DOI: https://doi.org/10.5524/100542

**Suggested local folder**
`external_datasets/openbmi_lee2019_mi/`

---

## P1 — BCI Competition IV Dataset 2b / BNCI2014-004

**Why it matters:** Very low-channel, multi-session EEG.

- **Subjects:** 9
- **EEG channels:** **3 bipolar EEG channels**
- **EOG channels:** 3
- **Sampling:** 250 Hz
- **Classes:** left-hand vs right-hand MI
- **Sessions:** 5
- **Paradigm:** cued motor imagery

**Why useful**
- Excellent testbed for methods that claim to work with **few channels**.
- Strongly relevant to session-to-session shift.
- Particularly useful for validating CSP/Riemannian/alignment methods.

**Source**
https://www.bbci.de/competition/iv/

**Suggested local folder**
`external_datasets/bci_iv_2b_low_channel/`

---

## P1 — BCI Competition IV Dataset 2a / BNCI2014-001

- **Subjects:** 9
- **EEG channels:** 22
- **EOG:** 3
- **Sampling:** 250 Hz
- **Classes:** left hand, right hand, feet, tongue

**Why useful**
- Canonical benchmark used by a huge fraction of MI literature.
- Makes it easier to reproduce published methods and compare implementations.
- Useful for checking whether Codex’s implementation of FBCSP, EEGNet, ATCNet, Conformer, etc. matches expected behavior.

**Source**
https://www.bbci.de/competition/iv/

**Suggested local folder**
`external_datasets/bci_iv_2a/`

---

## P1 — Cho2017 Motor Imagery Dataset (NEMAR nm000245)

- **Subjects:** 52
- **Sessions:** 1
- **Classes:** left-hand / right-hand MI
- **EEG:** 64 EEG + 4 EMG
- **Sampling:** 512 Hz
- **Approx size:** 16.7–17.2 GB
- **Trials:** roughly 100–120 per class depending on subject
- **License:** CC-BY-4.0

**Why useful**
- Large inter-subject cohort.
- Same montage contains many electrodes overlapping/near Kaggle locations: C3/C4/Cz/Fz/Oz/PO7/PO8/Pz.
- Concurrent EMG is useful for studying contamination/movement-related signals.

**Source**
https://www.nemar.org/dataset/nm000245

**Suggested download**
`nemar dataset download nm000245`

**Suggested local folder**
`external_datasets/nm000245_cho2017/`

---

## P1 — HS-Stroke Dataset (2026)

**Important update:** The published paper initially described 57,902 trials, but the maintainers later identified within-session duplicate trials. The **current release is deduplicated**.

- **Subjects:** 14 chronic stroke patients
- **Sessions:** originally described as 278 sessions
- **Task:** left/right upper-limb motor imagery
- **Current deduplicated release:** **23,382 trials**
- **Current repository notes:** 93,528 one-second samples
- **Clinical data:** includes Fugl-Meyer Assessment information
- **Use cases:** cross-trial / cross-session MI classification and clinical score prediction

**Why useful**
- Very strong source for studying repeated-session variability.
- Large number of trials from relatively few patients allows within-person drift analysis.

**Sources**
- Scientific Data: https://www.nature.com/articles/s41597-026-07742-x
- Repository: https://github.com/KINLAWWW/HS-Stroke

**Suggested local folder**
`external_datasets/hs_stroke_2026/`

---

## P2 — Schirrmeister High-Gamma Dataset

- **Subjects:** 14
- **Channels:** 128
- **Sampling:** 500 Hz
- **Classes:** 4
- **Trials/class:** ~120
- **Trials:** MOABB metadata gives 13,440 total windows/trials in its representation
- **Sessions:** 1
- **Runs:** 2

**Why useful**
- Canonical deep EEG dataset.
- Used in many domain adaptation and representation papers.
- Useful for verifying deep-learning implementations.

**Caution**
MOABB currently labels the paradigm as imagery in its summary; consult the original Schirrmeister et al. paper and dataset task documentation when interpreting motor execution vs imagery details.

**Source**
https://moabb.neurotechx.com/docs/generated/moabb.datasets.Schirrmeister2017.html

**Suggested local folder**
`external_datasets/high_gamma_schirrmeister/`

---

## P2 — MOABB Dataset Collection

**What it is:** Benchmark ecosystem rather than one dataset.

MOABB provides standardized access/evaluation for many BCI datasets and supports:
- WithinSession evaluation
- CrossSession evaluation
- CrossSubject evaluation

**Why useful**
- Fastest way for Codex to reproduce classical and deep baselines consistently.
- Useful for validating a method across multiple datasets before trusting it on Kaggle.
- Can reveal which approaches remain strong under cross-subject rather than random trial CV.

**Docs**
https://moabb.neurotechx.com/

**Suggested local folder**
`external_code/moabb/`

---

# 3. Core papers — read first

## P0 — NeuRestore benchmark

**Chowdhury et al. (2026)**  
*Neurestore: A New Benchmark for Wearable BCI-Based Neuromotor Training in Real-World*  
IEEE Access 14:10064–10080  
DOI: `10.1109/ACCESS.2026.3652957`

**Why it matters**
- Likely the closest scientific source for the Kaggle dataset.
- Wearable 8-channel g.tec Unicorn.
- Real-world/noisy environment.
- Contains classical and deep baselines.
- Key source for trial structure, cues, preprocessing, cross-subject performance and LCR post-processing.

**Source**
https://repository.essex.ac.uk/42506/

**Codex should extract**
- exact channels
- exact timing
- class cues
- preprocessing
- CSP+SVM setup
- FBCNet/EEGNet/ShallowConvNet/DeepConvNet settings
- per-subject results
- cross-subject protocol
- LCR method
- any evidence of cue/visual confounds

---

## P0 — FBCSP

**Ang et al. (2012)**  
*Filter Bank Common Spatial Pattern Algorithm on BCI Competition IV Datasets 2a and 2b*  
Frontiers in Neuroscience 6:39  
DOI: `10.3389/fnins.2012.00039`

**Core idea**
Filter bank → CSP per band → feature selection → classifier.

**Why it matters**
- Strong small-data inductive bias.
- Frequency-band selection is especially important because optimal MI bands differ by subject.
- Cheap enough for very large hyperparameter sweeps.

**Source**
https://pmc.ncbi.nlm.nih.gov/articles/PMC3314883/

---

## P0 — Riemannian covariance classification

**Barachant et al. (2012)**  
*Multi-class Brain Computer Interface Classification by Riemannian Geometry*  
IEEE TBME 59(4):920–928.

**Core idea**
Represent each trial by an SPD covariance matrix and classify on the Riemannian manifold.

Methods include:
- Minimum Distance to Riemannian Mean / MDM
- Tangent-space mapping + LDA

**Why it matters**
- Excellent match for few-channel, small-sample EEG.
- Avoids relying only on learned spatial filters.
- Tangent-space methods are often extremely competitive with deep learning.

**Source**
https://alexandre.barachant.org/papers/journals/IEEE-TBME-2012/

---

## P0 — Euclidean Alignment (EA)

**He & Wu (2020)**  
*Transfer Learning for Brain-Computer Interfaces: A Euclidean Space Data Alignment Approach*  
IEEE TBME 67(2):399–410  
DOI: `10.1109/TBME.2019.2913914`

**Core idea**
Unsupervised alignment/whitening to reduce subject/session distribution differences before any downstream feature extractor/classifier.

**Why it matters**
- Directly targets cross-subject variability.
- Computationally cheap.
- Does not require labels from the new target subject.
- Potentially one of the highest-value methods for the Cross Subject competition.

**Source**
https://pubmed.ncbi.nlm.nih.gov/31034407/

---

## P0 — Riemannian Procrustes Analysis (RPA)

**Rodrigues, Jutten & Congedo (2019)**  
*Riemannian Procrustes Analysis: Transfer Learning for Brain-Computer Interfaces*  
IEEE TBME 66(8):2390–2401  
DOI: `10.1109/TBME.2018.2889705`

**Core idea**
Align SPD distributions across sessions/subjects using translation, scaling and rotation on the manifold.

**Why it matters**
- Explicitly designed for inter-session and inter-subject EEG shift.
- Evaluated across 8 public BCI datasets / 243 subjects.

**Source**
https://pubmed.ncbi.nlm.nih.gov/30596565/

---

## P0 — EEGNet

**Lawhern et al. (2018)**  
*EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces*  
Journal of Neural Engineering 15(5):056013  
DOI: `10.1088/1741-2552/aace8c`

**Why it matters**
- Compact and data-efficient.
- Strong reference baseline.
- Useful sanity check before bigger architectures.

**Source**
https://pubmed.ncbi.nlm.nih.gov/29932424/

---

## P0 — FBCNet

**Mane et al.**  
*A Multi-view CNN with Novel Variance Layer for Motor Imagery Brain Computer Interface*

**Core idea**
Frequency filter-bank views → learned spatial filters → temporal variance layer → classifier.

**Why it matters**
- Neurophysiologically motivated.
- Designed specifically for MI.
- Very parameter-efficient.
- Particularly relevant because NeuRestore already uses it as a main baseline.

**Source**
https://pubmed.ncbi.nlm.nih.gov/33018625/

---

## P0 — ATCNet

**Altaheri et al. (2022/2023)**  
*Physics-Informed Attention Temporal Convolutional Network for EEG-Based Motor Imagery Classification*  
IEEE Transactions on Industrial Informatics 19(2):2249–2258  
DOI: `10.1109/TII.2022.3197419`

**Core idea**
Convolutional sliding windows + multi-head attention + temporal convolutional network.

**Reported benchmark**
BCI IV-2a: ~85.38% subject-dependent and ~70.97% subject-independent in the paper.

**Why it matters**
- Explicitly includes subject-independent evaluation.
- Compact enough to test seriously on small EEG.

**Source**
https://ieeexplore.ieee.org/document/9852687/

---

## P1 — EEG Conformer

**Song et al. (2023)**  
*EEG Conformer: Convolutional Transformer for EEG Decoding and Visualization*  
IEEE TNSRE 31:710–719  
DOI: `10.1109/TNSRE.2022.3230250`

**Core idea**
Local convolutional feature extraction + global self-attention.

**Why it matters**
- Strong modern baseline.
- Useful test of whether long-range temporal attention adds value after simpler approaches are established.

**Source**
https://pubmed.ncbi.nlm.nih.gov/37015413/

---

# 4. Cross-subject / domain-generalization papers

## P0 — EEG-DG

**Zhong et al. (2025)**  
*EEG-DG: A Multi-Source Domain Generalization Framework for Motor Imagery EEG Classification*  
IEEE JBHI 29(4):2484–2495  
DOI: `10.1109/JBHI.2024.3431230`

**Key point**
Treats source subjects/domains separately and learns representations intended to generalize to unseen target subjects **without target data during training**.

**Reported results**
- BCI IV-2a: 81.79%
- BCI IV-2b: 87.12%
- OpenBMI inter-session: 78.37%
- OpenBMI inter-subject: 76.94%

**Why it matters**
This problem formulation is very close to the Kaggle Cross Subject track.

**Source**
https://pubmed.ncbi.nlm.nih.gov/39052465/

---

## P0 — Supervised Contrastive Domain Generalization

**Zhi et al. (2025)**  
*Supervised Contrastive Learning-Based Domain Generalization Network for Cross-Subject Motor Decoding*  
IEEE TBME 72(1):401–412  
DOI: `10.1109/TBME.2024.3432934`

**Core ideas**
- cross-domain correlation alignment
- supervised contrastive learning
- domain-agnostic mixup
- class-level alignment

**Why it matters**
Explicitly targets calibration-free cross-subject **MI/ME** decoding.

**Source**
https://pubmed.ncbi.nlm.nih.gov/39046861/

---

## P1 — ASFA / Source-Free Adaptation

**Privacy-Preserving Domain Adaptation for Motor Imagery-Based Brain-Computer Interfaces**

**Core idea**
Augmentation-Based Source-Free Adaptation (ASFA):
- source model training
- uncertainty reduction on target
- consistency regularization
- no need to retain raw source EEG

**Why it matters**
Very relevant if unlabeled target-subject test data can legally be used for adaptation.

**Source**
https://pubmed.ncbi.nlm.nih.gov/35439124/

---

## P1 — SSTDA

**Unsupervised Domain Adaptation With Synchronized Self-Training for Cross-Domain Motor Imagery Recognition** (2025)

**Core idea**
Target unlabeled EEG + source labeled EEG → easy-to-hard synchronized self-training in shared latent space.

**Reported inter-subject results**
- BCI IV-2a: ~64.43%
- High-Gamma: ~80.40%

**Why it matters**
Useful if Kaggle allows transductive/test-time use of the complete unlabeled target distribution.

**Source**
https://pubmed.ncbi.nlm.nih.gov/40031262/

---

## P1 — Cross-dataset Multi-Source Domain Adaptation

**Miao et al. (2024)**  
*Multi-source deep domain adaptation ensemble framework for cross-dataset motor imagery EEG transfer learning*  
Physiological Measurement 45(5)  
DOI: `10.1088/1361-6579/ad4e95`

**Core idea**
Use multiple external datasets as different source domains, with pretraining, domain adaptation and ensemble.

**Why it matters**
If external training is permitted, this directly supports the idea of treating PhysioNet/OpenBMI/etc. as distinct domains instead of pooling everything blindly.

**Source**
https://pubmed.ncbi.nlm.nih.gov/38772402/

---

## P1 — MSD-DDA (2026)

**Multi-Source Discriminant Dynamic Domain Adaptation for Cross-Subject Motor Imagery EEG Recognition**

**Core idea**
Dynamic matching of global domain and local subdomain differences; multi-source transfer.

**Why it matters**
Recent approach directly focused on subject-to-subject variability.

**Source**
https://pubmed.ncbi.nlm.nih.gov/40966137/

---

## P1 — Domain Generalization via Latent Distribution Exploration (2025)

**Neurocomputing 614, 128889**  
DOI: `10.1016/j.neucom.2024.128889`

**Why it matters**
Calibration-free MI domain-generalization approach; potentially useful for ideas beyond simple domain adversarial training.

**Source**
https://www.sciencedirect.com/science/article/abs/pii/S0925231224016606

---

# 5. Foundation / pretrained EEG models

## P0 if external pretrained models are legal — MIRepNet

**Liu et al. (2026)**  
*MIRepNet: A pipeline and pre-trained model for EEG-based motor imagery classification*  
Knowledge-Based Systems 343:115966  
DOI: `10.1016/j.knosys.2026.115966`

**Why especially interesting**
- MI-specific rather than generic EEG pretraining.
- Uses a channel-template approach to unify different headsets.
- Designed to generalize to novel subjects, headsets and classes.
- Authors report strong few-shot behavior.
- Paper explicitly notes that generic EEG foundation models can perform surprisingly poorly in few-shot MI settings.

**Paper**
https://doi.org/10.1016/j.knosys.2026.115966

**Official code**
https://github.com/staraink/MIRepNet

**Converted pretrained checkpoint**
https://huggingface.co/braindecode/mirepnet-pretrained

---

## P1 — REVE (NeurIPS 2025)

**REVE: A Foundation Model for EEG — Adapting to Any Setup with Large-Scale Pretraining on 25,000 Subjects**

- **Pretraining:** >60,000 hours
- **Datasets:** 92
- **Subjects:** ~25,000
- **Key feature:** positional encoding designed for **arbitrary electrode arrangements**
- **Downstream tasks include:** motor imagery

**Why useful**
The arbitrary-electrode setup is particularly attractive for an 8-channel Kaggle montage.

**Paper**
https://papers.neurips.cc/paper_files/paper/2025/hash/20a917f77773ac0fa8bea2bdd6606b66-Abstract-Conference.html

---

## P1 — EEGPT (NeurIPS 2024)

**EEGPT: Pretrained Transformer for Universal and Reliable Representation of EEG Signals**

- ~10M parameter pretrained transformer
- mask-based dual self-supervised learning
- spatio-temporal representation alignment
- official checkpoint example uses 58 channels, 256 Hz, 4-s windows

**Why useful**
Worth testing as a generic pretrained EEG representation, but channel-montage mismatch is a major consideration.

**Code**
https://github.com/BINE022/EEGPT

---

## P1 — CBraMod (ICLR 2025)

**CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding**

**Core idea**
Masked pretraining with separate spatial/temporal criss-cross modeling.

**Why useful**
Another high-quality foundation baseline with public pretrained checkpoint/code.

**Code**
https://github.com/wjq-learning/cbramod

---

## P2 — BIOT (NeurIPS 2023)

**Yang, Westover & Sun (2023)**  
*BIOT: Biosignal Transformer for Cross-data Learning in the Wild*

**Core idea**
Flexible encoder designed for biosignals with:
- mismatched channel sets
- varying sample lengths
- heterogeneous datasets
- missing channels

**Why useful**
The conceptual approach to channel mismatch may be more valuable than the pretrained weights themselves.

**Paper**
https://proceedings.neurips.cc/paper_files/paper/2023/hash/f6b30f3e2dd9cb53bbf2024402d02295-Abstract.html

---

## P2 — LaBraM

**Large Brain Model for generic EEG representations**

- Pretrained on roughly 2,500 h from ~20 datasets.
- Channel-patch tokenization and neural-spectrum prediction.

**Why useful**
General EEG representation baseline; less MI-specific than MIRepNet.

**Reference implementation**
https://github.com/medonimario/LaBraM_ft

---

# 6. Method families Codex should be aware of

This is not a list of methods that must all be implemented.

## Classical / signal processing
- PSD / bandpower
- ERD/ERS
- CSP
- regularized CSP
- FBCSP
- stationary CSP
- shrinkage covariance
- Riemannian covariance
- MDM
- tangent-space logistic regression / LDA / SVM
- Euclidean Alignment
- Riemannian Alignment / RPA
- CORAL
- subject-specific spectral normalization

## Deep learning
- EEGNet
- ShallowConvNet
- DeepConvNet
- FBCNet
- ATCNet
- EEG Conformer
- TCN variants
- spectral-spatial CNNs
- multi-band/multi-view networks
- contrastive representations
- domain-generalized networks
- mixture-of-experts

## Adaptation / transfer
- Euclidean Alignment
- CORAL
- MMD
- DANN
- supervised contrastive DG
- source-free adaptation
- self-training
- pseudo-labeling
- entropy minimization
- test-time batch-stat adaptation
- subject weighting / source-subject selection
- multi-source domain adaptation

---

# 7. Especially interesting research hypotheses

These are hypotheses, not established truths.

## H1 — Execution→imagery→attempt transfer may be better than generic MI pretraining

The Kaggle target may be motor attempt rather than pure MI.

Therefore, external datasets containing **actual executed movement vs rest** may sometimes be more informative than large left-vs-right MI datasets.

PhysioNet is particularly valuable because executed and imagined tasks coexist in the same dataset.

---

## H2 — Low-density pretraining may matter more than maximal dataset size

Instead of training on all 64/128 channels and later discarding most inputs, deliberately create external low-density montages overlapping:

- Fz
- C3
- Cz
- C4
- Pz
- Oz / POz
- PO7
- PO8

when available.

This may reduce the domain gap to the target headset.

---

## H3 — Treat every dataset and every subject as a separate domain

Do not blindly concatenate external EEG.

Potential hierarchy:

`dataset → subject → session → run`

Possible objective:
learn features predictive of motor state while minimizing information about dataset/subject/session identity.

---

## H4 — External datasets may still be valuable even if forbidden for final training

They can be used to answer methodological questions:

- Which methods survive a 3–8 channel montage?
- Which alignment methods work best cross-session?
- Which models actually generalize cross-subject?
- Does EA consistently improve low-density EEG?
- Are cue-related classifiers fragile across datasets?
- Which augmentations improve true subject-held-out accuracy?

This is useful even if **zero external samples** enter the final Kaggle model.

---

# 8. Suggested download order

## Download first
1. Kaggle Single Subject data
2. Kaggle Cross Subject data
3. NeuRestore paper
4. ds003810 Low-Cost MI vs Rest
5. Motor Intention Rehabilitation dataset (especially 5-channel subset)
6. PhysioNet EEGMMIDB
7. nm000151 same-upper-extremity rest/grasp
8. nm000158 acute stroke
9. longitudinal affected-hand stroke 2026
10. BCI IV-2b
11. OpenBMI

## Download second
12. Cho2017
13. HS-Stroke current deduplicated release
14. High-Gamma
15. BCI IV-2a

## Models/code after rule clarification
16. MIRepNet
17. REVE
18. EEGPT
19. CBraMod
20. BIOT / LaBraM

---

# 9. Suggested directory structure

```text
EEG_pred/
├── kaggle_data/
│   ├── single_subject/
│   └── cross_subject/
│
├── external_datasets/
│   ├── ds003810_lowcost_mi_rest/
│   ├── motor_intention_rehab_5ch_32ch/
│   ├── physionet_eegmmidb/
│   ├── nm000151_same_upper_extremity/
│   ├── nm000158_acute_stroke_mi/
│   ├── stroke_longitudinal_affected_hand_2026/
│   ├── nm000141_wairagkar_motor_execution/
│   ├── openbmi_lee2019_mi/
│   ├── bci_iv_2b_low_channel/
│   ├── bci_iv_2a/
│   ├── nm000245_cho2017/
│   ├── hs_stroke_2026/
│   └── high_gamma_schirrmeister/
│
├── papers/
│   ├── 00_target_neurestore/
│   ├── 01_signal_processing/
│   ├── 02_riemannian_alignment/
│   ├── 03_deep_learning/
│   ├── 04_domain_generalization/
│   ├── 05_test_time_adaptation/
│   └── 06_foundation_models/
│
├── external_code/
│   ├── moabb/
│   ├── mirepnet/
│   ├── eegpt/
│   └── cbramod/
│
└── docs/
    └── RESEARCH_PACK.md
```

---

# 10. Recommended paper filenames

Use filenames that remain searchable by Codex.

```text
papers/
  00_target_neurestore/
    2026_Chowdhury_Neurestore.pdf

  01_signal_processing/
    2012_Ang_FBCSP.pdf
    2012_Barachant_RiemannianGeometry.pdf

  02_riemannian_alignment/
    2019_Rodrigues_RiemannianProcrustesAnalysis.pdf
    2020_HeWu_EuclideanAlignment.pdf

  03_deep_learning/
    2018_Lawhern_EEGNet.pdf
    2020_Mane_FBCNet.pdf
    2023_Altaheri_ATCNet.pdf
    2023_Song_EEGConformer.pdf

  04_domain_generalization/
    2025_Zhong_EEG-DG.pdf
    2025_Zhi_SupervisedContrastiveDG.pdf
    2024_Miao_MultiSourceCrossDatasetDA.pdf
    2025_LatentDistributionExploration_DG.pdf
    2026_MSD-DDA.pdf

  05_test_time_adaptation/
    ASFA_SourceFreeAdaptation.pdf
    2025_SSTDA_SynchronizedSelfTraining.pdf

  06_foundation_models/
    2026_Liu_MIRepNet.pdf
    2025_REVE.pdf
    2024_EEGPT.pdf
    2025_CBraMod.pdf
    2023_BIOT.pdf
    2024_LaBraM.pdf
```

---

# 11. What Codex should NOT assume

1. **External training data legality is not established** merely because datasets are publicly available.
2. A paper reporting high accuracy does not mean its validation matches Kaggle.
3. Trial-wise random CV is not evidence of cross-subject generalization.
4. Left-vs-right MI performance does not automatically transfer to rest-vs-motor-attempt.
5. More channels in external datasets do not automatically help an 8-channel target.
6. Aggressive artifact removal may remove useful label-correlated information present in the Kaggle experiment.
7. Foundation models should not be assumed superior to Riemannian/CSP approaches on a small dataset.
8. Resampling from 160 Hz to 250 Hz does not create information.
9. Similar electrode names across datasets do not guarantee equivalent spatial sampling.
10. Dataset metadata and current releases can change; use current source documentation.

---

# 12. Minimal reading order for an autonomous agent

If time/context is limited, read in this order:

1. NeuRestore
2. Kaggle rules / organizer discussions
3. FBCSP
4. Barachant Riemannian geometry
5. Euclidean Alignment
6. Riemannian Procrustes Analysis
7. FBCNet
8. EEGNet
9. ATCNet
10. EEG-DG
11. Supervised Contrastive DG
12. MIRepNet
13. REVE
14. SSTDA / ASFA

Then inspect the external datasets by relevance:

1. ds003810
2. Motor Intention 5-channel subset
3. PhysioNet EEGMMIDB
4. nm000151
5. BCI IV-2b
6. OpenBMI
7. stroke longitudinal datasets

---

# 13. Source URLs summary

## Datasets
- Low-cost MI/rest: https://nemar.org/dataset/on003810
- Motor intention rehab: https://zenodo.org/records/17980608
- PhysioNet EEGMMIDB: https://physionet.org/content/eegmmidb/1.0.0/
- Same-upper-extremity MI: https://www.nemar.org/dataset/nm000151
- Acute stroke MI: https://ww2.nemar.org/dataset/nm000158
- Longitudinal stroke affected-hand: https://www.nature.com/articles/s41597-026-07787-y
- Wairagkar: https://www.nemar.org/dataset/nm000141
- OpenBMI/MOABB: https://moabb.neurotechx.com/docs/generated/moabb.datasets.Lee2019_MI.html
- BCI IV: https://www.bbci.de/competition/iv/
- Cho2017: https://www.nemar.org/dataset/nm000245
- HS-Stroke: https://github.com/KINLAWWW/HS-Stroke
- High Gamma: https://moabb.neurotechx.com/docs/generated/moabb.datasets.Schirrmeister2017.html
- MOABB: https://moabb.neurotechx.com/

## Core papers
- NeuRestore: https://repository.essex.ac.uk/42506/
- FBCSP: https://pmc.ncbi.nlm.nih.gov/articles/PMC3314883/
- Riemannian geometry: https://alexandre.barachant.org/papers/journals/IEEE-TBME-2012/
- Euclidean Alignment: https://pubmed.ncbi.nlm.nih.gov/31034407/
- RPA: https://pubmed.ncbi.nlm.nih.gov/30596565/
- EEGNet: https://pubmed.ncbi.nlm.nih.gov/29932424/
- FBCNet: https://pubmed.ncbi.nlm.nih.gov/33018625/
- ATCNet: https://ieeexplore.ieee.org/document/9852687/
- EEG Conformer: https://pubmed.ncbi.nlm.nih.gov/37015413/
- EEG-DG: https://pubmed.ncbi.nlm.nih.gov/39052465/
- Supervised Contrastive DG: https://pubmed.ncbi.nlm.nih.gov/39046861/
- ASFA: https://pubmed.ncbi.nlm.nih.gov/35439124/
- SSTDA: https://pubmed.ncbi.nlm.nih.gov/40031262/
- Cross-dataset multi-source DA: https://pubmed.ncbi.nlm.nih.gov/38772402/
- MSD-DDA: https://pubmed.ncbi.nlm.nih.gov/40966137/
- MIRepNet: https://doi.org/10.1016/j.knosys.2026.115966
- MIRepNet code: https://github.com/staraink/MIRepNet
- REVE: https://papers.neurips.cc/paper_files/paper/2025/hash/20a917f77773ac0fa8bea2bdd6606b66-Abstract-Conference.html
- EEGPT code: https://github.com/BINE022/EEGPT
- CBraMod code: https://github.com/wjq-learning/cbramod
- BIOT: https://proceedings.neurips.cc/paper_files/paper/2023/hash/f6b30f3e2dd9cb53bbf2024402d02295-Abstract.html

---

# 14. Final note for Codex

This pack is intentionally broader than the final Kaggle solution.

Use it to answer:
- What methods are genuinely robust across subjects/sessions?
- Which datasets resemble the target in **task**, **montage**, **sampling**, and **recording context**?
- Which external data may teach the wrong cue or wrong physiology?
- Which simple methods outperform fashionable deep models under honest validation?
- What could be transferred from movement execution to imagery/attempt?
- What is specific to this Kaggle experimental protocol that generic MI literature will miss?

The target is not to consume every resource mechanically.

The target is to extract **ideas worth testing on the actual Kaggle data**.
