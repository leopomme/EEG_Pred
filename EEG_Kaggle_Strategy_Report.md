# Strategic research report: Low Cost Motor Imagery Decoding for Rehab

Single Subject and Cross Subject competitions · Research completed 3 October 2026

**Recommendation:** organize the competition around the recording protocol, the information available at prediction time, and the distinction between cue responses, attempted movement, and decoder-driven feedback. Establish those before choosing an architecture. The most promising advantage is a better account of how these signals and their proportions change between training and test, supported by validation that reproduces those changes.

This report is strategic guidance for the subsequent experimental phase. No competition model was trained, no submission was optimized, and the competition EEG files were not inspected directly. Public competition pages, full rules, organizer discussion, notebook source, and primary research were investigated. The separate `PROMPT_FOR_CODEX.md` translates the findings into an experimental mission.

## The research map I chose

I initially asked five questions: What generates the labels? What physical events generate the EEG around those labels? What information will actually be available for each test prediction? Which changes distinguish training from test? Which apparently strong methods are ineligible or evaluated under a different information budget?

That led me to prioritize the experimental protocol and released-data structure over a broad search for high-scoring neural architectures. A small misunderstanding about a run, an event marker, or the available target labels can dominate an architectural improvement.

Your suggestions about validation, cue-related activity, the original paper, and competition rules were especially valuable. I retained alignment and foundation models as competing hypotheses. I deprioritized fixed class-balance enforcement, unconditional enthusiasm for Riemannian methods, automatic artifact removal, and model rankings detached from their evaluation protocols.

The directions that were missing or underdeveloped were **feedback as an endogenous consequence of another classifier**, **possible underrepresentation of feedback trials in the released training data**, **pre-cue activity as a paired measurement of nuisance state**, **reference-induced errors in channel ablations**, and **the limitations of validation when the only labeled feedback trials may occupy a different part of a run**.

My understanding changed in three stages. First, this is instructed rest versus motor attempt, despite the motor-imagery title. Second, run 3 is a different experimental condition, not merely another recording session. Third, public notebook sources suggest that the Cross Subject training release may also retain only ten labeled run-3 trials per person. If confirmed, the cross-subject problem combines participant shift with a substantial change in the mixture of experimental conditions. This last finding materially changes the priority of sampling, validation, and protocol-specific models.

An independent idea-generation pass then asked what remains after removing the suggested model families. The strongest answers were: use each trial's baseline to identify its nuisance state; model the transition into feedback explicitly; test whether an approximation of the original controller helps interpret late EEG; and distinguish representation failures from calibration failures before adding capacity.

## Evidence and its limits

The following labels are used throughout:

| Label | Meaning |
|---|---|
| **[KAGGLE FACT]** | Stated in the official competition material, public competition metadata, or an organizer's comment. Documentation can still be internally inconsistent. |
| **[PAPER FACT]** | Reported in NeuRestore. This does not automatically prove that every corresponding Kaggle file has the same properties. |
| **[OTHER LITERATURE]** | A finding or method from another primary research source. |
| **[PUBLIC NOTEBOOK]** | Present in a publicly shared notebook's code or narrative. These notebooks were read, not rerun, and their narratives are not authoritative dataset documentation. |
| **[INFERENCE]** | A conclusion or calculation from the evidence, with assumptions stated. |
| **[HYPOTHESIS]** | A plausible, untested explanation or method for these competitions. |

**[OBSERVED FROM DATA]: none in this Work phase.** Anonymous access to the actual competition data listing required authentication. I did not bypass that requirement. Consequently, exact file contents, signal quality, class counts, split chronology, and local decoding performance remain **NOT VERIFIED**. The later data audit is essential, not a formality.

The public Kaggle pages were read through the site's public page responses where ordinary page rendering was empty. Notebook source versions and the leaderboard snapshot were also retrieved. No private submissions, private leaderboard, or hidden labels were accessed.

## What the two competitions actually ask

Official sources: [Single Subject][K1], [Cross Subject][K2], their data descriptions [K3], [K4], evaluation pages [K5], [K6], and rules [K7], [K8].

| Property | Single Subject | Cross Subject |
|---|---|---|
| Prediction | `rest` versus `move`, where move denotes instructed motor attempt | Same |
| Essential generalization problem | Predict more trials from a known person's feedback run using that person's calibration and available labeled feedback data | Predict trials from entirely held-out people, across their recording runs |
| Official cohort description | Acquisition involved 20 healthy adults; the overview specifies 17 supplied participants | Same acquisition description; 17 training participants and 3 held-out participants |
| Test identities | The same supplied people; public loaders exclude S008, S013, S015; verify against the files | Officially S008, S013, S015 |
| Training information | Runs 001, 002, and the labeled part of 003 are available in principle | Other participants' labeled recordings; no labeled target-subject calibration is promised |
| Test construction | Official data page: 80% subset of each participant's run 003, originally recorded continuously | All sessions represented for the three withheld participants; exact trial-selection mechanism unspecified |
| Submission rows | 680 in public competition metadata | 360 in public competition metadata |
| Public notebook description | 17 test files × 40 trials; usually 50 + 50 + 10 labeled trials per person | 9 test files × 40 trials; a public notebook reports the same 1,795 labeled training-trial total as Single Subject |
| Metric | Classification accuracy | Classification accuracy |
| Submission schema | `ID,TARGET`, lowercase labels; participant order then epoch order | `ID,TARGET`; alphabetically sorted test files, then epoch order |
| Public leaderboard configuration | 60% | 59% |
| Final entries | Up to two; overview describes one ML and one DL entry | Same |

**[KAGGLE FACT] Single Subject explicitly requires training on each participant separately without using the other participants' data.** Direct pooled supervised training, competition-cohort pretraining, meta-learning from other supplied people, or distillation from a pooled competition model cannot be assumed eligible. Cross Subject explicitly asks for a model trained across the supplied training participants. This is an essential difference in allowable learning, not just in evaluation.

The public/private percentage is a configuration, not a verified row-by-row partition. Do not infer the public membership of a trial from it. The unusual 59% Cross Subject setting also makes an exact public-row count unsafe to infer by simple multiplication.

**[KAGGLE FACT]** Both competitions allow five submissions per day and teams of up to five. Public metadata gives an enabled date of 22 September 2026 and a final deadline of **19 February 2027 at 00:00 UTC**, or 01:00 in Paris. Separate entry and merger deadlines were **NOT VERIFIED** and should be rechecked before acting. Each competition advertises $500 first prize and $150 second prize, with paper co-authorship invitations for the top three. An organizer clarified that monetary prizes follow the overall leaderboard in each competition, rather than being duplicated for the ML and DL categories. [K7], [K8], [K9]

### Documentation discrepancies that matter

The Single Subject overview contains an older 20% holdout description, while the specific data page says 80% of run 003. One acquisition paragraph describes eight EEG channels but gives an inconsistent numerical column range. File-name examples also vary in hyphen/underscore formatting. These are reasons to build the data contract from headers, markers, sample submission, and actual counts—not from positional column slicing or filename assumptions.

**[PUBLIC NOTEBOOK]** The simple Single Subject baseline explicitly handles both `trial_start` and the misspelling `trail_start`, ignores trailing experiment markers, expects forty test cues per person, and reports variable cue lengths rather than exactly 1,250 samples. Its narrative gives preparation lengths of 659–756 samples and cue lengths of 1,242–1,279. Those ranges should be checked, not hard-coded. [N1]

Two notebooks report 1,795 training trials across 50 files, including a shortened S006 run and a missing S007 run. A second notebook supplies a detailed count table, but its narrative contains other unsupported interpretations. Treat the counts as specific audit targets, not independently verified observations. [N1], [N2], [N3]

**The important unresolved question is whether Cross Subject really has only ten labeled run-3 trials per training person.** If so, and if the reported counts are correct, feedback trials represent approximately 170/1,795 = 9.5% of training but 120/360 = 33.3% of test. That is a large protocol-mixture shift. It would justify comparing run-balanced sampling and shared models with run-specific components. It does not establish an optimal weight: uncritical weighting could amplify a small, chronologically unrepresentative sample.

## Reconstructing the experiment

**[KAGGLE FACT]** Recordings use an eight-channel g.tec Unicorn Hybrid system at 250 Hz. The named montage is **Fz, C3, Cz, C4, PO7, Pz, PO8, Oz**. The experimental run contains 50 trials, with 25 instructions of each class in randomized order. This describes acquisition, not necessarily every released file. [K3], [K4]

Participants attempt a pinch or squeeze with the thumb, index, and middle fingers of the dominant hand. Rest instructs relaxation. The labels encode the instruction, not an independent measurement that an attempted movement actually occurred. This distinction matters: instruction-related sensory activity can predict the scored label even when a participant's motor response is weak.

Use two explicit clocks:

| Event | Time since trial start | Time since cue onset, u | Modeling implication |
|---|---:|---:|---|
| Fixation/preparation | 0–3 s | −3–0 s | Candidate nuisance reference and negative control |
| Class cue and attempted movement/rest | 3–8 s | 0–5 s | Full instructed period |
| Early cue response | Begins at 3 s | Begins at 0 s | Test phase-locked sensory and attentional information |
| Before active feedback in run 3 | 3–5 s | 0–2 s | Most direct comparison with calibration runs |
| Feedback updates in run 3 | 5, 5.5, 6, 6.5, 7 s | 2, 2.5, 3, 3.5, 4 s | Late EEG can depend on the original decoder's decisions |
| Return of hand in the paper's online protocol | 7–8 s | 4–5 s | Separate late sensory/mechanical state |
| Inter-trial interval | After the trial | — | Approximately 2–3 s; not a new labeled trial |

The first two runs are calibration with a static exoskeleton. The third supplies visual and robotic feedback. Therefore “session shift” here should not casually imply separate days, headset replacement, or only electrode drift. Run order, fatigue, feedback, and distributional changes may coexist; their relative contributions are **NOT VERIFIED**.

### How securely does this identify NeuRestore?

**[INFERENCE: strong protocol-level attribution]** The match is unusually specific: authors/organizing context, device, exact montage, sampling rate, dominant-hand pinch task, three-run online design, five feedback updates, and robotic hand all agree with the online experiment in Chowdhury et al. (2026). [P1] This is much stronger than a title resemblance.

**Exact file-to-paper provenance is still NOT VERIFIED.** I did not find an authoritative manifest connecting Kaggle files and subject IDs to the paper's participant identifiers, nor an independently linked supplementary code/data release that establishes byte-level identity. Do not map S008 to a paper participant merely because both contain “08.”

**[PAPER FACT]** NeuRestore includes two disjoint groups of twenty participants: an offline cohort and a separate online cohort. The paper describes healthy, male, BCI-naive participants and added electrolyte for the hybrid electrodes. These are paper-cohort facts; the competition itself explicitly confirms healthy adults, but demographic details should not be imported uncritically without provenance. The paper's offline cohort is not twenty additional labeled Kaggle subjects available for free use.

The paper shows different class-specific visual/text and auditory instructions, including a hand-related move cue and a relaxing scene for rest. That supports investigating sensory differences; it does not establish their decoding strength in the released files. The exact screen content throughout every online rest/feedback phase remains less secure than the timing diagram.

The acquisition reference was **NOT VERIFIED** from the inspected text. The paper's later common-average rereferencing is a processing step, not proof of the acquisition reference.

### What the benchmark does—and does not—tell us

The complete paper and its protocol and result figures were inspected. These selected results are useful context, not target scores to reproduce mechanically:

| Paper comparison | Reported result | Correct interpretation |
|---|---|---|
| Offline within-person FBCNet with preprocessing | 73.7% at the 6 s feedback time point | A particular window and pipeline on the offline cohort |
| Offline CSP + SVM | 72.1% at that time point | Classical methods are credible competitors |
| Online CSP + SVM | 68.4% at that time point | Different cohort and online condition; not a paired causal estimate of feedback's effect |
| Offline cross-subject FBCNet | 66.6% with longest-consecutive-repetition aggregation, versus 62.67% mean across feedback windows | Evidence that within-trial aggregation matters, not a Kaggle performance ceiling |

The online controller was trained on a fixed three-second calibration window. Later predictions used overlapping three-second windows. A window ending at trial time 5 s starts at 2 s—one second before the cue. These windows are neither independent examples nor interchangeable experimental phases.

The paper's “longest consecutive repetition” operates on predictions **inside one trial**. It does not justify smoothing labels across successive randomly ordered trials.

Two reproduction traps deserve attention. The preprocessing description includes a 150 Hz notch despite a 250 Hz sampling rate; that cannot be applied literally below the 125 Hz Nyquist limit. It also uses Fz as an ocular proxy for ICA rejection, rather than a dedicated EOG recording. Removing activity correlated with Fz may also remove neural cue or error-related information. The paper's success with one cleaned deep pipeline does not establish that all deep learning requires ICA. Its nonsignificant comparisons do not prove equivalence. [P1]

## Rules that change the solution space

This is a reading of the competition's published requirements, not an organizer determination of every proposed pipeline. Specific exceptions matter. [K7], [K8]

| Status | Technique or action | Consequence |
|---|---|---|
| **Explicitly permitted, subject to conditions** | External data and models reasonably accessible to all participants, with suitable rights | Public external datasets must predate the competition start; check provenance and licenses |
| **Explicitly permitted** | DL input with a 50 Hz notch and broad 1–100 Hz filtering | This does not authorize any arbitrary narrow-band preprocessing recipe |
| **Explicitly permitted** | Time-axis-preserving expanded transforms, such as spectrograms and CWT, computed from raw data in the notebook | A time-frequency DL branch is a plausible compliant route |
| **Explicitly permitted** | A uniform normalization function whose parameters adapt at runtime to loaded data | No hard-coded, participant-specific normalization constants |
| **Explicitly prohibited** | Cross-referencing the other competition's test set; Single Subject overview more broadly prohibits using the other competition's data | Keep data, learned statistics, checkpoints, pseudo-labels, and predictions separate between competitions |
| **Explicitly prohibited in Single Subject** | Training a person's model using the other supplied participants' data | No pooled competition-cohort pretraining, learned cross-person prior, or cross-person distillation |
| **Explicitly prohibited** | Purpose-recording a new external dataset to improve the score | Do not solve calibration by collecting a bespoke new cohort |
| **Explicitly prohibited** | AutoML products such as the named Google/H2O examples, or similar tools | Do not deploy an automatic model-building service |
| **Explicitly prohibited** | Human labeling of validation/test records and multi-account submissions | No manual test-label construction or leaderboard probing through extra accounts |
| **Foundational code-license requirement** | Open-source submission code must use an OSI-approved license permitting commercial use unless a stated exception applies | Publicly readable or research-only code is not automatically eligible |
| **ML category restriction** | No neural-network-derived transformations; PyTorch/TensorFlow explicitly excluded | Neural embeddings followed by SVM are not a classical ML entry |
| **DL category restriction** | End-to-end EEG classification; classical LDA, PCA, ICA, FIR filters, SVMs excluded except stated exceptions | A paper's preprocessing cannot simply be attached to a network |
| **Organizer-confirmed** | The notebook producing the CSV must be submitted and run in Kaggle | Cluster success must be converted into a reproducible Kaggle execution |

The ML definition contains a narrow single-hidden-layer MLP exception. It is not a useful basis for creatively relabeling a hybrid neural pipeline as ML.

The foundational rules also prohibit private sharing of competition code/data outside the team, make the private leaderboard decisive, and break ties by earlier submission. The data license is CC BY-NC-SA 4.0; code and pretrained-model licensing are separate questions. A concrete trap is the public LatentAlignment implementation's CC BY-NC 4.0 license: do not simply copy it into a final submission under the default open-source code rule. Consider a lawful independent implementation of the published method or obtain clarification. The competition-specific pretrained-model exception does not automatically license arbitrary noncommercial source code.

**Conservative final-submission policy:** do not blend ML and DL predictions. Keep a pure candidate in each category. Within-category ensembling appears compatible with the categories but was not explicitly approved in the material inspected. Ask about it before committing a final ensemble.

### Important ambiguities to resolve with concrete pipeline descriptions

1. Is full spatial covariance whitening/EA an allowed DL normalization under the runtime-normalization exception? Scalar or per-channel normalization has a more direct reading; a spatial mixing transform is less clear.
2. May a prediction use statistics from all unlabeled trials of the same target participant/run? Does this permission also cover latent normalization, joint count inference, pseudo-labeling, or gradient adaptation? These are distinct operations, not one “TTA” question.
3. Given the explicit prohibition on using other supplied participants in Single Subject, what shared hyperparameter selection, if any, is allowed? How does permitted external pretraining interact with this requirement? Direct pooling of the supplied participants is not an ambiguous exception to assume.
4. Are same-category ensembles permitted? Must both selected final entries cover different categories?
5. Does the time-preserving-transform exception cover fixed narrow filter banks? Are required resampling/anti-aliasing operations for pretrained models acceptable?
6. Are auxiliary inertial/device streams admissible predictors, or only the eight EEG channels? Use EEG only until clarified.
7. Can an exact same-competition complementary trial count be used if run completeness is verified? Can predictions be jointly constrained by such a count?
8. What is the actual notebook runtime/hardware limit, and may externally trained checkpoints be attached? No numeric runtime limit was verified.
9. Where is the boundary between permitted researcher-designed parameter sweeps and prohibited AutoML? Ordinary controlled experiments are not the named turnkey products, but large automatic architecture-search systems should not be assumed acceptable.

These questions should not stop safe work. Investigate ambiguous methods using held-out labeled training data; exclude them from final submissions until clarified. Do not contact organizers automatically without the user's authorization to send a message.

The external-data cutoff explicitly concerns datasets. Do not silently convert it into a rule that all newly trained model weights must predate the competition. Nevertheless, pretrained-checkpoint provenance, accessibility, release dates, licenses, and contamination require an audit.

## What the public competition landscape reveals

**[KAGGLE FACT: snapshot, 3 October 2026, approximately 09:53 UTC]** The displayed leading public scores were **0.90 Single Subject** and **0.88 Cross Subject**. The next displayed scores were 0.88 and 0.87 respectively. The pages reported 22 teams/123 submissions for Single Subject and 16 teams/98 submissions for Cross Subject. These are rounded public displays, not verified private performance or evidence of any particular winning method.

Public notebook metadata reported the following best public scores:

| Public notebook | Competition | Displayed/metadata score | Strategic lesson from source inspection |
|---|---|---:|---|
| LCMID-S Machine learning baseline [N1] | Single | 0.66 | RBF SVM on time-domain epochs; careful marker handling; ordinary CV or a last-file validation option |
| Riemannian Tangent Space & EEGNet Rehab Decoding [N2] | Single | 0.75 | These families are already public; the source discards the first 0.5 s and uses shuffled within-person folds |
| Cross-Subject EEG Spatio-Temporal Rehab Decoding [N3] | Cross | 0.56 | Grouped subjects are useful, but reported CV is at overlapping-window level rather than the final trial metric |
| Organizer example notebook [N4] | Cross | 0.5308 | Helpful ordering example; not a validated optimal signal-processing recipe |

Scores belong to public notebook metadata and do not establish which variant produced them. In particular, the Single Subject notebook contains separate classical, neural, and mixed predictions. Its mixed prediction code is not proof that a mixed final submission is eligible.

There are concrete weaknesses to avoid copying:

- The organizer example constructs filters using `fs=512` for data documented at 250 Hz. A nominal 4–25 Hz filter therefore corresponds to approximately 1.95–12.21 Hz when applied to the recorded sampling grid. Its crop-level five-fold evaluation also fails to establish unseen-subject trial performance.
- One notebook calls a balanced experimental run evidence for an exactly balanced forty-trial test subset. That conclusion does not follow.
- The same notebook says S008/S013/S015 were absent from clinical collection. The official Cross Subject page instead identifies them as held-out participants.
- Equal averaging across sliding windows silently treats early cue, pre-feedback, and feedback periods as exchangeable evidence.

These observations identify openings in publicly visible approaches. They do **not** establish that the leading private teams make the same mistakes. The sizeable gap between public notebook scores and the leading public leaderboard means a standard baseline should not be expected to win by itself.

## Where the useful signal may be

### Cue, motor response, and feedback are separate sources

**[HYPOTHESIS]** Early visual/auditory responses, sustained sensorimotor desynchronization, diffuse attention/relaxation differences, and late feedback responses may all discriminate the instructions. Their importance can vary by person and run.

The four posterior channels are not automatically “the answer.” Nor should they be discarded because they are not over motor cortex. A posterior advantage could reflect visual evoked activity, alpha modulation, eye movements, or a referencing effect. A central advantage could also include sensory activity or volume-conducted artifacts. Electrode location alone does not identify a neural generator.

**[OTHER LITERATURE]** A particularly relevant study published on 28 September 2026 compared cue-inclusive versus cue-excluding windows in Dreyer2023 and Lee2019. Cross-subject DeepConvNet and REVE benefited from cue inclusion, with reported differences of roughly 5–13 percentage points across settings. [L1] This supports testing the hypothesis here, not transferring the effect size. The study changed the window's start and end together and used different preprocessing for CSP and neural models. Its title's contrast with “standard machine learning” should not be read as evidence that classical ERP decoders cannot exploit cue responses.

An earlier latent-alignment study also identified strong early, class-discriminative eye-movement activity in PhysioNet motor tasks. [L2] Thus high early decoding is not, by itself, proof of either pure visual cortical processing or motor intent.

For leaderboard purposes, legitimately recorded EEG correlates can be useful even when they would be confounds for a claim about autonomous motor control. This does not authorize extracting hidden labels from administrative fields, reverse engineering test labels, or using prohibited auxiliary data. Maintain separate conclusions about **predicting this competition's instruction labels** and **decoding uncued intent in a deployable BCI**.

### Feedback is not an independent cue

**[PAPER FACT]** In the online protocol, a rest trial can nevertheless produce hand closure if the original classifier predicts motor attempt. Feedback is therefore a consequence of that classifier's decisions, not a perfect copy of the true class. [P1]

The causal sequence is: instruction → early sensory and motor-related EEG → original decoder decision → visual/robotic feedback → later EEG. The instruction also affects whether that feedback is experienced as correct. Participant state influences several steps.

**[INFERENCE]** A late classifier may partly decode the previous controller's action. That can help when the controller is correct and propagate its errors when it is wrong. A feedback-only model can also fail when transferred from static calibration runs. An early/late model with interactions may be better than either independent expert or their average.

**[HYPOTHESIS]** A response to erroneous feedback could add information about whether the controller was wrong. Error-related potentials during robotic feedback have primary experimental precedent. [L3] Their presence, timing, and separability here are unverified. An error signal alone does not reveal the intended class unless the action being judged is also known or inferable. Five closely spaced updates further complicate simple event averaging.

One unusually specific experiment is to approximate the original CSP/SVM controller from each available person's calibration recordings, reconstruct its possible within-trial decision trajectory, and ask whether late EEG adds information conditional on that trajectory. This is most naturally a **classical ML research branch** in Single Subject. It must use only legal inputs, never the unknown true label to simulate feedback. The exact original controller cannot be assumed reconstructible. Its reconstructed scores are deterministic summaries of EEG, so any gain comes from useful representation and interactions, not newly created information. Do not feed a classical-controller feature pipeline into a supposedly pure DL final entry.

### Baseline as a paired nuisance measurement

**[HYPOTHESIS]** The fixation period can measure gain, background rhythms, electrode quality, and transient arousal immediately before a trial. Compare task activity with this baseline instead of requiring a model to learn invariance to every absolute amplitude scale.

Candidate representations include baseline-relative log power, task/baseline covariance changes, and an end-to-end model that receives baseline and task segments separately. The baseline should condition interpretation, not be relabeled as a supervised `rest` example. Instructed rest occurs after a different cue and is a different state.

There is a useful mathematical motive for a classical branch. Let B and C denote baseline and task covariance matrices. Generalized eigenvalues satisfying C v = λ B v are unchanged when the same invertible sensor mixing A acts on both: B′ = A B Aᵀ and C′ = A C Aᵀ. This follows by substituting v′ = A⁻ᵀv. Thus these eigenvalues can suppress a particular class of shared nuisance transforms.

This is not a theorem that they will classify well. Baseline and task may not share the same nuisance process; background activity is biologically variable; sorted eigenvalues discard anatomical orientation; regularization breaks exact invariance. Compare them with ordinary band power and tangent-space features, and preserve a branch retaining spatial information. Resting-state adaptation research supports examining reference activity, but does not validate this specific representation on NeuRestore. [L4]

### Reference, rank, and amplitude can defeat an otherwise good experiment

**[INFERENCE]** Common-average rereferencing across eight channels reduces the signal's spatial rank to at most seven. Covariance logarithms and inverse square roots require appropriate rank handling or regularization. Numerical stability is part of the model, not an implementation footnote.

A “central-channels-only” ablation performed **after** subtracting an average containing the posterior electrodes still contains posterior information through the reference. To assess regional dependence, distinguish dropping channels before rereferencing from dropping them afterward. Use a consistent reference comparison and avoid claiming anatomical localization from either alone.

Per-trial, per-channel variance normalization can remove precisely the amplitude differences that make band power useful. Conversely, uncorrected offsets and inconsistent units can overwhelm a pretrained model. Compare normalization choices deliberately: global scaling, per-channel centering, baseline scaling, trace normalization, and preservation of absolute power. Do not interpret a notebook's reported large offset as a verified unit conversion.

### Other high-value hypotheses

**Quality-conditioned model selection.** A flat, clipped, intermittently disconnected, or mechanically contaminated electrode should reduce reliance on the affected branch. Quality indicators derived without labels may be more trustworthy than confidence: a model can be confidently wrong on a bad channel. In Cross Subject, fit reliability rules on held-out source people/runs; in Single Subject, respect the separate-person training requirement. Do not invent per-test-person choices by visual inspection.

**Timing and phase alignment.** If cue timestamps or reaction latencies vary, small shared or label-blind adjustments may recover phase-locked signal. Do not estimate a separate alignment using each held-out label. Compare with a deliberately jittered cue control and matched-duration windows. Zero-phase filtering across cue boundaries can smear later information backward and invalidate an apparent pre-cue effect.

**Predictive source selection.** The most anatomically or covariance-similar source participant need not be the best source for classification. Select weighting rules using held-out transfer performance, and compare against equal participant weights. Protocol compatibility may matter more than participant similarity.

**Threshold adaptation versus representation adaptation.** Good within-target ranking with poor accuracy suggests a calibration/threshold problem. Poor ranking requires a different representation. Measure both before undertaking extensive TTA. A threshold is still selected without target labels in Cross Subject; use source-held-out evidence and any legally established priors.

**Aperiodic/background and global power.** Rest versus effort can differ in broad spectral level and alpha state, not only narrow motor bands. Test absolute power, relative power, and baseline-relative power separately. Expensive connectivity or aperiodic parameterization earns its place only if simpler summaries leave useful residual structure.

## Alignment and adaptation: useful, but conditional

EA estimates a reference covariance for a domain and whitens trials using its inverse square root. Riemannian recentering performs a related operation using a geometric reference. These are inexpensive and important controls. They can correct gain and covariance shifts, but matching marginal covariances does not guarantee matching class-conditional distributions. Nor does it remove all rotations or resolve a change in feedback policy. [L5], [L6]

A crucial choice is the domain used to estimate the reference: person, person-by-run, calibration versus feedback, a moving buffer, or pre-cue baselines. Pooling all runs can mix different class information and experimental conditions into the nuisance estimate. Small references require shrinkage and sensitivity analysis.

Do not conflate unsupervised recentering with every stage of Riemannian Procrustes Analysis. Class-matching rotations in supervised variants require labeled target examples. Such labels can exist in Single Subject and are absent in Cross Subject. Rotation estimated using hidden target labels would invalidate the comparison. [L7]

**Latent alignment deserves more attention than another large backbone.** It estimates subject-specific statistics inside the representation while retaining shared learned parameters. Its authors demonstrate subject-independent benefits and explicitly study vulnerability to class imbalance in the context set. [L2] Baseline-conditioned or shrinkage versions could reduce dependence on the unknown mixture of task classes. This is an experimental proposal, not a verified improvement.

**[OTHER LITERATURE]** An online adaptation study found that adaptive batch-normalization statistics could outperform input alignment; EA did not improve every cross-session condition. [L8] T-TIME combines incremental alignment, information maximization, and ensembles, and addresses the failure of a rigid balanced-batch assumption. Its experiments use conventional MI datasets and a defined sequential information budget, not these eight-channel feedback recordings. [L9]

A rational adaptation comparison is therefore:

1. Fixed source model.
2. Uniform runtime scaling/normalization.
3. Input alignment with alternative reference domains.
4. Adaptive latent statistics with shrinkage toward source statistics.
5. Only then, pseudo-label or entropy-based parameter updates.

This is a comparison hierarchy, not a demand to implement all five. A class collapse, sensitivity to target-trial order, or gains isolated to one held-out person should redirect the work. Independent seeds do not make ensemble errors conditionally independent; spectral ensemble-weight estimators still need a direct comparison with simple averaging.

## The class-balance question: a precise conditional opportunity

The premise “25 rest and 25 move per acquired run” is documented. The premise “every released test file has that balance” is false as a general deduction and unverified as a release property.

If a complete fifty-trial run is split into ten labeled and forty unlabeled trials, with no exclusions, then the remaining move count is

**K_test = 25 − K_labeled.**

It is not necessarily twenty. This could be useful in Single Subject, but requires proof that the files are exact complements of the same complete run, plus organizer clearance for constrained inference. Missing trials, incomplete runs, or post hoc selection invalidate the arithmetic.

If forty trials were sampled uniformly from fifty containing twenty-five moves, then K follows a hypergeometric distribution with mean 20 and variance 100/49. The probability of exactly twenty moves is approximately **27.48%**. Even starting from otherwise perfect predictions, forcing twenty would introduce an expected **2.75 percentage points** of errors under that sampling model. These are mathematical illustrations, not estimates of the actual split mechanism.

When an exact count is established, selecting the K highest move scores is the natural constrained assignment under common score assumptions. When only a sampling prior is justified, a soft count prior is more defensible than a hard quota. With no verified selection model, use no count constraint. In every case, evaluate on held-out training data with the actual selection pattern and distinguish ranking improvement from a prior imposed on the labels.

Do not infer repeated label sequences or smooth across random trials. Do not try to reproduce a hidden random seed or reconstruct organizer label-generation state. Do not use the other competition to complete a run or recover its labels.

## Literature: what is worth importing

The relevant literature spans ERP decoding, motor attempts/imagery, covariance estimation, participant transfer, feedback interpretation, and pretrained representations. The model name is less important than its information requirements.

| Family | Evidence and fit | Research priority and limitation |
|---|---|---|
| Time-domain ERP features + shrinkage LDA/ridge/logistic models | Small-data ERP covariance estimation has extensive primary evaluation [L10]; directly tests cue-phase information | **High** for ML. Avoid thousands of unconstrained time-point parameters with a hundred trials |
| Band power, regularized CSP/filter-bank CSP + linear classifiers or SVM | NeuRestore itself supplies evidence; interpretable control for oscillatory information | **High**. Do not assume only 8–30 Hz or central channels matter |
| Shrinkage covariance + tangent-space linear models; MDM | Strong general EEG benchmarking; inexpensive for eight channels [L11] | **High** comparator. Whole-epoch covariance loses temporal order; add phase-specific comparisons if warranted |
| ERP template covariance/xDAWN variants | Fits time-locked multichannel responses; useful bridge between ERP and covariance views | **Conditional high** after early response is established. Templates and filters must be learned inside folds |
| EEGNet, shallow temporal/spatial CNNs, EEGSimpleConv | Compact models with actual subject-transfer evaluations; EEGNet spans ERP, error, movement-potential, and SMR tasks [L12], [L13] | **High** for DL; relevant breadth and manageable variance |
| FBCNet | Relevant to NeuRestore and motor-band variance structure [L14] | **Medium–high**, but the original fixed-filter/ICA recipe is not automatically DL-compliant |
| ATCNet, EEG Conformer, modest temporal-convolution or attention variants | Plausible temporal modeling; established implementations [L15], [L16] | **Medium**, after matching preprocessing, windows, and validation |
| Latent alignment/AdaBN, EA/Riemannian recentering | Actual transfer/adaptation evidence, with clear failure modes [L2], [L5], [L6], [L7], [L8], [L9] | **High** for controlled comparisons, with rule-dependent final eligibility |
| Domain-adversarial, CORAL/MMD, episodic/meta-learning | Mechanisms address participant shift; can also remove label-relevant variation or overfit a small set of people | **Conditional**, after demonstrating which shift remains |
| Mamba/state-space architectures | Contemporary EEG examples exist [L17] | **Lower initially**. Linear sequence complexity is not obviously the bottleneck for eight channels and a few seconds |
| Connectivity, source localization, high-density graph models | Superficially relevant neuroscience | **Lower initially**. Eight sensors and limited independent trials constrain identification; require incremental evidence |

The large MOABB reproducibility study is useful evidence for robust classical baselines. Its main comparison is **within-session**, not a demonstration that its ranking transfers to unseen people in this competition. [L11]

The official ATCNet repository explicitly warns about the original training/testing methodology and supplies an improved training/validation/testing route. A published headline or a convenient implementation is not permission to select epochs on the eventual test fold. [L15]

### Foundation models: a targeted tournament, not a default solution

| Candidate | What the research actually supports | Fit and principal concern here |
|---|---|---|
| **REVE** | Flexible spatial positions; genuine held-out-subject PhysioNet evaluation; additional cue-onset study [L1], [L18] | Particularly interesting if phase-locked sensory information transfers. Verify eight-channel coordinates, sample rate, preprocessing, and checkpoint provenance |
| **MIRepNet** | MI-specific pretraining, channel unification and EA; original downstream headline results use **30% labeled target-session fine-tuning** [L19] | Relevant few-shot hypothesis, not evidence of zero-calibration Cross Subject success. The 8–30 Hz recipe may remove valuable cue information and raises DL-rule questions |
| **EEGPT** | Public pretrained representation and downstream adapters [L20] | Test an eligible end-to-end adaptation; channel mapping, temporal patches, and calibration regime matter |
| **CBraMod** | Public generalist model with flexible channel/time handling [L21] | Credible second generalist; clinical pretraining and head dimensions need adaptation |
| **LaBraM / BIOT** | Broad EEG representation learning; BIOT accommodates heterogeneous channels [L22], [L23] | Useful if other evidence supports transfer; no presumption of superiority on motor-attempt feedback |
| **Eight-channel HuBERT-style model, Ogg et al.** | Pretraining with Fz/Cz/C3/C4/P7/P8/Pz/Oz, close to this montage; subject-held-out BCI tests [L24] | Interesting low-channel evidence, but P7/P8 are not PO7/PO8, and a usable public checkpoint was **NOT VERIFIED** |
| **ST-EEGFormer** | Public model and multi-protocol benchmark; variable montage with pretrained weights [L25] | Optional additional comparison if available resources and provenance justify it; not a reason to begin with a 300M-parameter model |

Two broad benchmarks warn against expecting a frozen foundation embedding to solve the problem. Specialist models remain competitive, and full fine-tuning often materially changes the result; larger models do not have a uniform advantage. [L25], [L26] Compare pretrained initialization with a matched randomly initialized backbone, and compare frozen, partial, and fuller adaptation when justified. Otherwise architecture, optimization, and pretraining effects become inseparable.

The MIRepNet paper's five downstream tasks should not be read as five wholly independent subject cohorts: binary and four-class versions of BNCI2014001 reuse participants. More generally, check whether pretraining includes a downstream dataset before treating an evaluation as clean evidence of generalization.

Interpolation to a larger montage does not create missing information. It may help match an encoder's expected format, but can also blur the few posterior and central sensors that matter. Random zero-filled channels, wrong electrode identities, and unit mismatches can make an apparently negative foundation-model result meaningless.

## Validation must approximate the actual prediction problem

### Single Subject

The primary target is run 3. A random split of all labeled trials mostly evaluates calibration trials and can rank methods incorrectly. Training on run 1 and validating on run 2 is a useful stability check but also misses the feedback transition.

Use the labeled run-3 trials as a small target-condition evaluation resource. Cross-fit them: for each held-out subset, train the main model on that person's available calibration runs and adapt using only the remaining allowed run-3 labels. Vary the support size to determine whether supervised target calibration helps or destabilizes the model. Report **paired method differences across people** to understand variability, while keeping each person's model fitting and tuning isolated unless the organizer clarifies a permitted form of shared model selection. Avoid fitting a large independent hyperparameter search to each person's ten labels.

First determine whether those labels are the first ten trials, an interleaved selection, or another subset. The public baseline says file times increase, but chronology has not been checked on the recordings. If labels occupy the beginning and test the end, random support/query splits within the first ten cannot validate the full temporal extrapolation. Add forward/blocked stress tests and investigate drift using lawful unlabeled statistics. State that the late-feedback validation gap remains; do not disguise a proxy as a faithful replica.

Pooled-subject training on the supplied Single Subject cohort is explicitly excluded. Train each person's predictor only on that person's supplied data. Public external pretraining remains a separate rule question with its own conditions; do not treat a pooled competition checkpoint as external data.

### Cross Subject

Split by participant before learning features, making crops, selecting windows, or tuning normalization. Use outer held-out participants, with model selection and early stopping inside the remaining source participants. LOSO preserves more training people; repeated groups of roughly three held-out people better expose variability in a test set containing only three people. Both are informative, and neither needs an exhaustive combinatorial sweep.

Report scores separately for calibration runs and feedback runs. If training run 3 is truncated, a standard subject-held-out micro-average will underweight the test's feedback condition. Also report a target-protocol-weighted aggregate based on **verified test file counts**. Do not treat 7 overlapping windows as 7 independent trial labels or select a model using crop accuracy when the submission scores one decision per trial.

If only early feedback trials have labels, held-out source subjects still do not supply labeled late feedback data. This limitation survives perfect subject grouping. Investigate it explicitly and use multiple proxies; external eligible online datasets may test a mechanism, but do not replace direct validation of this cohort.

### Rules for both

- Fit supervised CSP, ERP templates, scalers, feature selection, tangent references, calibrators, and ensemble weights inside the appropriate training fold. Label-free use of a target batch is a different evaluation regime: reproduce it exactly and label it as such.
- Keep inductive inference, per-trial normalization, full-target-set normalization, sequential adaptation, and gradient TTA separate in the results. The available target context is part of the method.
- Keep all crops of a trial together. Respect temporal dependence and filter support near held-out boundaries. Never concatenate discontinuous files and filter across their join as if continuous.
- Use subject- or run-grouped uncertainty estimates and paired comparisons. Thousands of time samples or augmented crops do not increase the number of independent people.
- Test label permutations within legitimate groups and pre-cue negative controls. An apparent pre-cue predictor should trigger an audit of sequence constraints, filtering, carryover, and parsing before any claim of predictive brain state.
- Compare windows of matched duration and capacity when making causal interpretations. Use retrained region/time ablations alongside occlusion; a masked out-of-distribution input can mislead.
- Record trial-level out-of-fold probabilities, not only average accuracy. They permit error overlap, calibration, ensemble, and subject-specific failure analysis.
- Do not tune against the leaderboard. A single full-test trial changes accuracy by about 0.147 percentage points in Single Subject and 0.278 in Cross Subject; the public score is based on fewer trials and displayed coarsely. Domain uncertainty can exceed simple binomial uncertainty.

## Prioritized falsifiable hypotheses

Priority reflects expected information gain and plausible competitive value, not a predicted score gain. “Falsified” below means evidence against pursuing the proposed advantage, not proof of biological absence.

| ID / priority | Hypothesis and support | Observation that would support it | Observation that would weaken or falsify it |
|---|---|---|---|
| **H1 / first** | Parsing, phase coverage, or protocol proportions are materially misrepresented by standard loaders; official inconsistencies and notebook source motivate this | Corrected anchors/counts or run-weighted validation change model rankings | Independent loaders agree, timing is sound, and rankings are insensitive to verified protocol weighting |
| **H2 / first** | Early cue information contributes transferable accuracy; distinct cues and external cue studies support plausibility | Matched-window, held-out-person/run gains from early samples; reproducible temporal/region ablations | Early-only is weak; removing it does not hurt; gains vanish after timing/leakage controls |
| **H3 / first** | Feedback creates a distinct class-conditional signal requiring separate treatment | Run-by-time interactions; matched feedback training or conditional heads improve held-out trial accuracy | Pre-feedback and late effects transfer similarly, and protocol conditioning adds no reproducible value |
| **H4 / high** | Paired baseline reference removes nuisance without erasing task information | Baseline-relative power/covariance or baseline-conditioned DL improves hard held-out domains | Baseline is too noisy, references erase class separation, or gains fail under modest timing changes |
| **H5 / high** | Some failures are threshold/calibration failures | Strong within-target ranking but biased decisions; source-validated calibration improves accuracy | Ranking itself is poor, or recalibration gains require target labels unavailable at inference |
| **H6 / high** | Alignment's reference domain matters more than its name | Person-by-run, baseline, or shrinkage references beat global pooling consistently | Domain choices are immaterial or unaligned models are best |
| **H7 / high** | Quality-conditioned shrinkage/gating protects against unreliable sensors | Improvements concentrate on independently defined bad-channel cases and transfer across people | Quality indicators do not predict model failure or gating only fits a few unusual participants |
| **H8 / medium, high upside** | Late EEG can correct early/controller errors | Held-out improvement conditional on early scores; stable early/late interaction or controller-trajectory features | Late response merely repeats controller errors; interaction benefit disappears after matching capacity |
| **H9 / medium** | Latent normalization improves participant transfer beyond input whitening | Source-held-out gains survive context-size, imbalance, and order stress tests | Gains depend on artificial balanced batches or collapse with realistic target subsets |
| **H10 / conditional** | Exact/soft count information improves legal structured inference | Verified release mechanism, organizer approval, and gains on faithfully constructed held-out subsets | Incomplete complements, class-dependent exclusions, or harm under actual subset counts |
| **H11 / conditional** | Pretraining supplies transferable cue/motor features | Matched pretrained versus scratch comparison wins consistently, especially with scarce feedback labels | Benefit vanishes under matched training, correct units/montage, or subject/run holds |
| **H12 / later** | Gradient adaptation adds value beyond normalization | Gains recur across held-out subjects/seeds, with stable class proportions and limited order sensitivity | Entropy decreases without accuracy gain, errors reinforce themselves, or benefits are confined to one domain |

Three controls can rapidly redirect this agenda. If pre-feedback central band power already matches full-epoch performance, cue/feedback elaboration is lower priority. If low-frequency early EEG dominates and transfers reliably, ERP methods and a compact temporal neural model become the main branches. If all families fail on the same objectively corrupted recordings, improve quality handling before increasing capacity.

## Several plausible routes to a strong solution

**A phase-aware classical solution.** Shrinkage ERP features, regularized band-power/CSP features, and phase-specific covariance representations supply complementary evidence. Combine only when out-of-fold error structure demonstrates a benefit and category rules permit the final combination. Single Subject can emphasize conservative same-person run-3 calibration; Cross Subject emphasizes participant transfer and verified run weighting.

**A compact end-to-end solution.** A shallow or EEGNet-like temporal/spatial network preserves early responses and learns oscillatory summaries, with protocol-aware training and optional baseline conditioning. Treat trial phase explicitly rather than averaging indistinguishable crops. This has a clearer initial DL-rule path than copying ICA plus fixed narrow-band filtering.

**A participant-adaptive solution.** Begin with a good source model, then compare uniform runtime normalization, reference-domain alignment, and latent statistics. Add gradient adaptation only when evidence and rules justify it. The competitive mechanism is better handling of the actual target domain, not generic sophistication.

**A pretrained specialist or generalist solution.** A carefully audited REVE/CBraMod-type model or MIRepNet variant could add a representation unavailable from a small training release. It earns substantial compute after a matched comparison identifies a repeatable advantage. The winning representation may be a sensory-response representation rather than a motor-imagery one.

**A controller-aware solution.** For feedback runs, explicitly model pre-feedback evidence, potential device-state trajectory, and late response. This is the most unusual mechanistically grounded branch. It may fail because the controller cannot be reconstructed or the late response is too noisy; a modest controlled test has high information value before building a complex dynamical model.

There is no reliable basis yet to select one route as the winner or promise a score above the current public leaders.

## How I would spend substantial compute

Compute should buy stronger conclusions before it buys larger networks.

First, resolve the data contract and test an informative time × channel × frequency × run comparison with two or three complementary model families. Reuse predetermined folds and trial-level predictions. This answers which representation is worth optimizing.

Next, spend more on the branches whose effects survive participant/run holds: tuning regularization and windows, repeated seeds, reference-domain ablations, source weighting, and adaptation-context stress tests. Compare interactions selectively—baseline normalization may help a power model and hurt an ERP model; cue cropping may change the benefit of pretraining; class priors may change the behavior of latent normalization.

Then allocate a larger budget to a few survivors, including ensembles with genuinely different errors. Use a small screening budget for many plausible questions and a larger confirmation budget for important positive or negative results. Retain a confirmation split or locked evaluation procedure as model selection becomes extensive. Stop spending on a fashionable family when the evidence says it has no useful residual signal.

The experiment record should capture the question, competing explanation, information available at inference, category/legal status, expected deciding observation, result, and next decision. Failed hypotheses should change priorities. A leaderboard submission should test transfer of a defensible local conclusion, with its rationale recorded beforehand.

No cluster layout, software repository design, or implementation prescription is needed from this Work phase. The Codex agent should choose those after inspecting the actual environment and data.

## What I would investigate first if none of the proposed solutions had been suggested

**I would investigate the classifier–feedback loop and its representation in the released training set.** Specifically: are late EEG features generated under a different policy/condition from the majority of labeled examples, and does the test set put much more weight on that condition?

The first decisive experiment would compare pre-feedback versus full-trial prediction, separately by run, under validation weighted to the verified target protocol. I would then ask whether late EEG helps most when the early model is uncertain or wrong, and whether a simple early/late interaction beats an average of window predictions. In Single Subject, I would test a reconstruction of the old controller as an interpretable intermediate summary.

This could reveal that the main opportunity is neither a better MI feature extractor nor a generic transfer method. It might be correcting an experimentally induced change in how labels, actions, and sensory responses relate. If that hypothesis fails, the failure is valuable: it frees resources for early cue decoding, paired baseline representations, and conventional motor features.

The most consequential remaining unknowns are therefore not “which transformer?” They are the released run composition, the chronology of labeled versus test feedback trials, the amount of transferable early signal, the conditional value of late feedback, and the exact legal target-information budget.

## Primary-source reference guide

Competition pages and notebooks were inspected on 3 October 2026. Public notebook source versions: N1 353490236; N2 353478080; N3 353215421; N4 351887120. Notebook narratives are cited as notebook claims, never as independent experimental verification.

- **K1–K2:** [Single Subject overview][K1]; [Cross Subject overview][K2].
- **K3–K4:** [Single Subject data][K3]; [Cross Subject data][K4].
- **K5–K6:** [Single Subject evaluation][K5]; [Cross Subject evaluation][K6].
- **K7–K8:** [Single Subject rules][K7]; [Cross Subject rules][K8].
- **K9:** [Organizer discussion, 27 September 2026][K9]: prizes, Kaggle notebook execution, category naming.
- **N1:** [AmbrosM, LCMID-S Machine learning baseline][N1].
- **N2:** [Avik Das, Riemannian Tangent Space & EEGNet Rehab Decoding][N2].
- **N3:** [Avik Das, Cross-Subject EEG Spatio-Temporal Rehab Decoding][N3].
- **N4:** [Alexander Thomas, Motor Rehab Cross Subject Example Notebook][N4].
- **P1:** [Chowdhury et al., NeuRestore, IEEE Access 2026, DOI 10.1109/ACCESS.2026.3652957][P1]. [Full text][P1PDF]. Complete paper read; protocol and key result tables visually checked.
- **L1:** [Trocellier et al., Visual cues in MI-BCI induce biases…, Frontiers in Neuroergonomics, 28 September 2026][L1]. Directly relevant cue inclusion experiment; no assumed transfer of its effect sizes.
- **L2:** [Bakas et al., Latent alignment in deep learning models for EEG decoding, JNE 2025][L2]. [Public preprint read][L2PRE]; [author code][L2CODE]. Distinguish the preprint version from final publication.
- **L3:** [Online asynchronous decoding of error-related potentials during continuous control of a robot, Scientific Reports 2019][L3]. Mechanistic support for a hypothesis, not evidence of an ErrP in this dataset.
- **L4:** [An et al., Subject-Adaptive Transfer Learning Using Resting State EEG Signals…, MICCAI 2024][L4]. [Full preprint][L4PRE].
- **L5:** [He and Wu, Transfer Learning for BCIs: A Euclidean Space Data Alignment Approach, 2019/2020][L5].
- **L6:** [Wu, Revisiting Euclidean Alignment…, 2025][L6]. Includes correct placement and limitations.
- **L7:** [Rodrigues et al., Riemannian Procrustes Analysis, author repository and paper link][L7].
- **L8:** [Wimpff et al., Calibration-free online test-time adaptation…, full preprint][L8].
- **L9:** [Li et al., T-TIME, full preprint][L9]. Sequential information budget and marginal-prior assumptions matter.
- **L10:** [Sosulski et al., Improving Covariance Matrices Derived from Tiny Training Datasets…, Neuroinformatics 2021][L10]. [Author code][L10CODE].
- **L11:** [The largest EEG-based BCI reproducibility study for open science: the MOABB benchmark, 2024][L11]. Main evaluation is within-session.
- **L12:** [Lawhern et al., EEGNet, JNE 2018][L12].
- **L13:** [El Ouahidi et al., EEGSimpleConv, author implementation][L13]. Separately reports within-session, cross-subject, and fine-tuned cross-session results.
- **L14:** [Mane et al., FBCNet, 2021][L14].
- **L15:** [Altaheri et al., ATCNet, official repository][L15]. Includes methodological warning about the original evaluation implementation.
- **L16:** [Song et al., EEG Conformer, official repository][L16].
- **L17:** [EEGMamba, 2024 preprint][L17]. Representative of a lower-priority architecture family, not a recommended default.
- **L18:** [REVE, full 2025 preprint][L18]; [official project][L18CODE].
- **L19:** [MIRepNet, full 2025 preprint][L19]; [official repository][L19CODE]. Publication: Knowledge-Based Systems, DOI 10.1016/j.knosys.2026.115966. Detailed claims here refer to the inspected preprint.
- **L20:** [EEGPT, official NeurIPS 2024 paper][L20]; [official repository][L20CODE]. This is the universal-representation EEGPT, not the similarly named autoregressive model.
- **L21:** [CBraMod, official repository and paper][L21].
- **L22:** [LaBraM, ICLR 2024 paper][L22].
- **L23:** [BIOT, NeurIPS 2023 paper][L23].
- **L24:** [Ogg et al., EEG Foundation Models for BCI Learn Diverse Features of Electrophysiology, 2025][L24]. Full paper read; public checkpoint not established.
- **L25:** [Yang et al., Are EEG Foundation Models Worth It?, ICLR 2026][L25]; [author repository][L25CODE].
- **L26:** [Liu et al., EEG-FM-Compass, version 3, August 2026][L26]. Full PDF methods/results inspected; includes LOSO and within-person few-shot evaluation.

[K1]: https://www.kaggle.com/competitions/low-cost-motor-imagery-decoding-for-rehab
[K2]: https://www.kaggle.com/competitions/low-cost-motor-imagery-decoding-for-rehab-cross-subject
[K3]: https://www.kaggle.com/competitions/low-cost-motor-imagery-decoding-for-rehab/data
[K4]: https://www.kaggle.com/competitions/low-cost-motor-imagery-decoding-for-rehab-cross-subject/data
[K5]: https://www.kaggle.com/competitions/low-cost-motor-imagery-decoding-for-rehab/overview/evaluation
[K6]: https://www.kaggle.com/competitions/low-cost-motor-imagery-decoding-for-rehab-cross-subject/overview/evaluation
[K7]: https://www.kaggle.com/competitions/low-cost-motor-imagery-decoding-for-rehab/rules
[K8]: https://www.kaggle.com/competitions/low-cost-motor-imagery-decoding-for-rehab-cross-subject/rules
[K9]: https://www.kaggle.com/competitions/low-cost-motor-imagery-decoding-for-rehab/discussion/743366
[N1]: https://www.kaggle.com/code/ambrosm/lcmid-s-machine-learning-baseline
[N2]: https://www.kaggle.com/code/avikdas567/riemannian-tangent-space-eegnet-rehab-decoding
[N3]: https://www.kaggle.com/code/avikdas567/cross-subject-eeg-spatio-temporal-rehab-decoding
[N4]: https://www.kaggle.com/code/alexanderthomas2001/motor-rehab-cross-subject-example-notebook
[P1]: https://repository.essex.ac.uk/42506/
[P1PDF]: https://repository.essex.ac.uk/42506/1/Neurestore_A_New_Benchmark_for_Wearable_BCI-Based_Neuromotor_Training_in_Real-World.pdf
[L1]: https://www.frontiersin.org/journals/neuroergonomics/articles/10.3389/fnrgo.2026.1804143/full
[L2]: https://doi.org/10.1088/1741-2552/adb336
[L2PRE]: https://arxiv.org/abs/2311.17968
[L2CODE]: https://github.com/StylianosBakas/LatentAlignment
[L3]: https://www.nature.com/articles/s41598-019-54109-x
[L4]: https://papers.miccai.org/miccai-2024/740-Paper0192.html
[L4PRE]: https://arxiv.org/html/2405.19346v1
[L5]: https://pubmed.ncbi.nlm.nih.gov/31034407/
[L6]: https://arxiv.org/html/2502.09203v1
[L7]: https://github.com/plcrodrigues/RPA
[L8]: https://arxiv.org/html/2311.18520v2
[L9]: https://arxiv.org/html/2412.07228v1
[L10]: https://link.springer.com/article/10.1007/s12021-020-09501-8
[L10CODE]: https://github.com/jsosulski/time-decoupled-lda
[L11]: https://arxiv.org/html/2404.15319v1
[L12]: https://pubmed.ncbi.nlm.nih.gov/29932424/
[L13]: https://github.com/elouayas/EEGSimpleConv
[L14]: https://arxiv.org/abs/2104.01233
[L15]: https://github.com/Altaheri/EEG-ATCNet
[L16]: https://github.com/eeyhsong/EEG-Conformer
[L17]: https://arxiv.org/abs/2407.20254
[L18]: https://arxiv.org/html/2510.21585v1
[L18CODE]: https://brain-bzh.github.io/reve/
[L19]: https://arxiv.org/html/2507.20254v1
[L19CODE]: https://github.com/staraink/MIRepNet
[L20]: https://proceedings.neurips.cc/paper_files/paper/2024/hash/4540d267eeec4e5dbd9dae9448f0b739-Abstract-Conference.html
[L20CODE]: https://github.com/BINE022/EEGPT
[L21]: https://github.com/wjq-learning/CBraMod
[L22]: https://arxiv.org/abs/2405.18765
[L23]: https://arxiv.org/abs/2305.10351
[L24]: https://arxiv.org/abs/2506.01867
[L25]: https://openreview.net/forum?id=5Xwm8e6vbh
[L25CODE]: https://github.com/LiuyinYang1101/STEEGFormer
[L26]: https://arxiv.org/abs/2601.17883v3
