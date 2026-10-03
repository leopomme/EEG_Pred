# PROMPT FOR CODEX

You are taking over an autonomous EEG competition research and implementation project. Your objective is to develop the strongest legal Kaggle solution possible for **both** competitions below, ideally outperforming the field on the final private leaderboard:

1. **Low Cost Motor Imagery Decoding for Rehab — Single Subject**
   https://www.kaggle.com/competitions/low-cost-motor-imagery-decoding-for-rehab
2. **Low Cost Motor Imagery Decoding for Rehab — Cross Subject**
   https://www.kaggle.com/competitions/low-cost-motor-imagery-decoding-for-rehab-cross-subject

A Work research phase was completed on 3 October 2026. Its accompanying file, `EEG_Kaggle_Strategy_Report.md`, contains detailed reasoning, primary references, and the rule questions. Read it if available, but this prompt supplies enough context to start intelligently. Do not repeat the entire literature review. Verify current rules and resolve concrete factual gaps as needed.

**The Work report is strategic guidance, not ground truth.** If empirical evidence contradicts it, trust the evidence after checking the experiment. **You are expected to form new hypotheses based on the data and experimental results. The research strategy below is a starting point, not a ceiling.**

Unlike Work, your role is to inspect actual data, implement methods, run controlled experiments, revise the strategy, use substantial CPU/GPU compute intelligently, and eventually construct reproducible competition submissions. Do the work rather than stopping at a proposed plan. Choose your own implementation structure after inspecting the environment. Do not assume an unverified cluster configuration, data path, GPU allocation, or access credential.

## 1. Understand the prediction problem and allowed learning

Despite the titles, these are **instructed rest versus motor attempt**, not standard left/right motor-imagery tasks. `move` means an attempted dominant-hand pinch/squeeze; `rest` means relaxation. The subjects are healthy adults, with rehabilitation as the intended future application. Labels represent the instruction, not a separately measured confirmation of actual motor effort.

**Single Subject:** predict more trials from each known person's third, feedback-enabled run using that person's available calibration recordings and labeled part of run 3. The official overview explicitly says to train each participant separately **without using data from the other supplied participants**. Do not pool this competition's people, pretrain on its other people, learn a cross-person prior from them, or distill their pooled model into a person-specific model. Keep person-specific fitting and tuning separate unless an organizer explicitly clarifies a permitted shared-selection procedure. External pretrained models are a separate eligibility question under the external-data rules.

**Cross Subject:** train across the supplied training participants and predict entirely unseen participants. There are no promised labeled examples from these target people. Subject-independent validation is essential, but does not by itself address differences in run protocol or trial selection.

Both competitions distinguish **classical ML** and **end-to-end DL** entries. Maintain pure, independently reproducible candidates. Do not merge the two competitions' data, learned statistics, checkpoints, predictions, or pseudo-labels. General methodological understanding may transfer; competition-derived training information must not.

## 2. Facts established by Work, and what remains unverified

Official competition facts as of 3 October 2026:

- Acquisition involved 20 healthy adults. Single Subject supplies 17 people. Cross Subject uses 17 training people and holds out **S008, S013, S015**.
- Eight EEG channels: **Fz, C3, Cz, C4, PO7, Pz, PO8, Oz**; g.tec Unicorn Hybrid; nominal sampling rate **250 Hz**.
- The experimental design has three runs per person, nominally 50 trials per run, 25 `rest` and 25 `move`, in randomized order. This does **not** establish the counts in each delivered file.
- Runs 1 and 2 are calibration with a static exoskeleton. Run 3 includes active visual and robotic feedback.
- Trial timing: about 3 s fixation/preparation followed by 5 s cue/task, then an approximately 2–3 s inter-trial interval.
- Let **u = time since cue onset**. Preparation is approximately u ∈ [−3,0); task is [0,5]. Run-3 feedback updates occur at **u = 2, 2.5, 3, 3.5, 4 s**. The matching source paper describes return of the hand during u = 4–5 s.
- Train event markers include `rest` and `move`; test class cues are replaced by `cue_start`. Use markers for segmentation and training labels, not administrative fields as label surrogates.
- Single Subject data page says test is an **80% subset of run 003**, originally captured continuously. The remaining labeled run-3 data may be used for that person.
- Cross Subject evaluates sessions from the three unseen people. Determine the exact released trial selection from its own files.
- Accuracy is the metric. Submission columns are `ID,TARGET`, with lowercase labels. Single: participant order, then epoch order. Cross: sorted test files, then epoch order. Validate against each sample submission.
- Public metadata contains **680 Single Subject test rows** and **360 Cross Subject test rows**. Public leaderboard configuration is 60% and 59% respectively; do not infer individual public/private membership.
- Five submissions per day, at most two selected final submissions; overview describes one ML and one DL final entry. Public metadata deadline was **2027-02-19 00:00 UTC**. Recheck current timeline and requirements.
- An organizer confirmed that the notebook producing the CSV must be submitted and run in Kaggle. Actual numeric runtime limits and the permission to attach cluster-trained checkpoints were not established.

**Public notebook evidence—not direct data verification:**

- Sources corroborate 17 Single Subject test files × 40 trials and 9 Cross Subject test files × 40 trials.
- They report **1,795 labeled training trials across 50 files**, usually 50 + 50 + 10 trials per participant, with a shortened S006 run and a missing S007 run.
- Crucially, a Cross Subject notebook reports the same limited training total. Determine whether Cross Subject training also has only ten labeled run-3 trials per person. If so, feedback represents about 9.5% of labeled training versus one-third of test. Treat those percentages as conditional until verified.
- Public code handles both `trial_start` and `trail_start`, variable cue lengths, and trailing experiment markers. One source reports cue lengths of 1,242–1,279 samples. Do not hard-code those ranges or silently pad every irregular trial.
- Public notebook narratives contain mistakes. One incorrectly infers exact balance in a forty-trial subset; another assumption uses a 512 Hz filter design on 250 Hz data. Read source critically.

**Source-paper facts and limits:**

NeuRestore, Chowdhury et al., IEEE Access 2026, DOI **10.1109/ACCESS.2026.3652957**, closely matches the online protocol. It includes two disjoint twenty-person cohorts, offline and online. Do not count both as Kaggle training subjects. Exact Kaggle-file-to-paper-participant mapping is **NOT VERIFIED**. Do not infer that similarly numbered participants correspond.

The paper documents class-specific visual/text/auditory cues. In feedback runs, the device responds to the original classifier, including erroneous move predictions during rest. Late EEG can therefore reflect the controller's action and the participant's response to its correctness. It is not a perfect second label channel.

Work did **not** access the raw competition recordings or train models. File contents, chronology, per-file class counts, signal quality, actual references/units, and all proposed method gains remain unverified. Your data audit is the first empirical foundation.

## 3. Establish the data contract before large optimization

Inspect each competition independently through authorized access. Produce a concise factual audit that resolves the decisions below; do not turn this into cosmetic exploratory plotting.

1. What files, people, runs, labeled trials, and test cues actually exist? Identify incomplete runs, malformed markers, duplicate or overlapping records **within the same competition**, discontinuities, truncation, and unexpected ordering. Never match records between competitions.
2. Which columns are the eight named EEG channels? What are their units, offsets, sample spacing, acquisition reference if documented, clipping/flatline behavior, and auxiliary fields? Do not use inconsistent numerical column ranges from the description.
3. Where are trial start, cue onset, cue stop, and feedback phases relative to the actual sampling grid? Are marker timing and sample timestamps consistent? Do segmentation choices change labels or row order?
4. Which part of run 3 is labeled? Is the holdout chronological, interleaved, or another selection? Does Cross Subject training contain full or truncated feedback runs? Quantify the resulting train/test mixture by run, using no hidden labels.
5. Can a Single Subject labeled run-3 file and its test file be shown to be exact complements of a complete fifty-trial run? Do not impose any complementary count until this is established and the inference method is cleared by the organizer.
6. Which quality problems cluster by person, run, time, or channel? Could a preprocessing operation remove class information, rather than only nuisance?

Verify submission ordering and label mapping early. A silent loader error can overwhelm the performance difference between model families.

## 4. Highest-value scientific questions

Determine where useful class information occurs in **time, space, frequency, and experimental condition**, and whether it transfers under the relevant split.

Start with an informative comparison of early cue samples, pre-feedback samples, feedback samples, and the full trial. Suggested initial boundaries are u = 0–0.5/0.8 s, 0–2 s, 2–4 s, and 4–5 s; these are probes, not fixed final windows. Compare matched durations where needed to distinguish phase from more data.

Compare central, posterior, Fz, and full-montage information, with a reference-aware design. Dropping posterior channels **after** all-channel common-average rereferencing still leaves their contribution in the remaining channels. Common-average rereferencing also reduces rank to at most seven; regularize covariance operations appropriately.

Test low-frequency phase-locked responses, alpha/mu/beta power, broader spectral information, and task-to-baseline changes. Do not assume only 8–30 Hz matters. Do not label the fixation period as supervised rest. Determine whether early sensory responses, motor patterns, and late feedback supply complementary errors.

Measure trial-level accuracy, within-held-out-domain ranking, calibration, and error concentration by person/run/quality. Good ranking with poor accuracy suggests a decision-threshold problem; poor ranking suggests a representation problem. Do not use hidden target labels to fix either.

## 5. Prioritized hypotheses and falsification tests

These priorities may change immediately after the audit.

| Hypothesis | Why plausible / Work evidence | Supporting observation | Observation against it |
|---|---|---|---|
| **Protocol-mixture mismatch matters** | Public Cross Subject code suggests little labeled run-3 data but one-third feedback test trials | Run-aware validation changes rankings; sampling or protocol-conditioned models improve target-weighted held-out accuracy | File audit disproves mismatch or run treatment makes no reproducible difference |
| **Early cue EEG transfers** | Different sensory instructions; recent cross-subject cue-inclusion studies | Early samples add held-out accuracy under matched-duration and leakage controls | Early-only is weak; gains vanish with sound timing/reference controls |
| **Feedback requires conditional modeling** | Run 3 changes the physical experiment; device follows another decoder | Early/late interactions or protocol-specific components improve held-out trials | Late EEG adds no residual information or only repeats early errors |
| **Paired baseline suppresses nuisance** | Baseline and task share a person and recent recording state | Baseline-relative power/covariance or baseline-conditioned DL helps hard domains | Noisy baseline or normalization destroys useful separation |
| **Reference-domain choice dominates alignment choice** | All-run covariance mixes protocol and class information | Person-by-run, baseline, or shrinkage references beat pooled ones | Results are insensitive or unaligned models win |
| **Quality-based reliability helps** | Low-density wearable EEG is vulnerable to individual bad channels | Label-free quality measures predict failures; learned/fixed legal gating transfers | Gating helps only a few tuned participants or confidence is equally effective |
| **Latent normalization improves transfer** | Published subject-independent evidence beyond input EA | Gains survive realistic context size, class imbalance, and order changes | Gains require artificially balanced target contexts or collapse under subsets |
| **Controller-aware late decoding corrects mistakes** | Late response may depend on whether feedback matched the instruction | Late EEG improves predictions conditional on early/controller scores | It only decodes device action; no held-out interaction gain |
| **Count information can improve inference** | Original run design has 25/25 classes | Verified complementary structure, organizer clearance, and realistic held-out gains | Missing trials/unknown selection or hard balancing worsens predictions |
| **Pretraining supplies useful features** | Relevant MI and generalist models exist | Pretrained initialization beats a matched scratch model across domains/seeds | Gains disappear after matching preprocessing/training or correct input mapping |
| **Gradient TTA adds beyond statistics** | EEG TTA literature supports some settings | Repeatable gains over normalization with stable predictions and permitted information use | Entropy falls without accuracy gain, pseudo-label errors reinforce, or results depend on trial order |

For each experiment, state what decision either outcome would change. Do not keep a weak hypothesis alive merely because it was recommended here.

## 6. Validation principles

**Single Subject:** fit and tune each person's model only using that person's allowed data. Calibration run 1→2 checks are useful but insufficient because test is from run 3. Cross-fit the small labeled run-3 portion, training on runs 1/2 plus an allowed support subset of run 3 and scoring the untouched query subset. Vary support size to assess calibration value; shrink toward that person's calibration model, not a prior learned from other supplied people. Do not fit many per-person choices to ten target-condition labels.

If the labeled feedback trials are the beginning and test trials the end, acknowledge that resampling within the beginning does not validate late-run extrapolation. Add blocked/forward stress tests and relevant drift diagnostics. Report aggregate results across people, but do not use their data to tune another person's predictor without explicit clearance for that procedure.

**Cross Subject:** use outer participant-held-out evaluations, with inner source-participant splits for early stopping and tuning. LOSO and a limited set of held-out groups of approximately three people answer complementary questions. Never put crops from one person into both training and the unseen-person evaluation.

Report calibration-run and feedback-run performance separately, plus an aggregate weighted to verified target run counts. If source feedback labels cover only early trials, recognize the unresolved late-feedback validation gap even after correct subject grouping.

**Both:**

- Split before crop generation/augmentation. All windows of a trial stay together. Evaluate one prediction per trial, as Kaggle does.
- Fit supervised transforms, CSP, ERP templates, feature selection, covariance references, calibration, and ensemble weights within folds. For intentionally label-free target statistics, reproduce the allowed target context exactly and label the comparison as transductive where appropriate.
- Separate fixed inductive inference, per-trial normalization, full-target-batch statistics, sequential adaptation, and gradient updates. They have different information budgets.
- Do not filter across discontinuous file joins. Check temporal support near fold boundaries. Noncausal filtering can smear cue information into a purported baseline negative control.
- Keep trial-level out-of-fold probabilities and paired method comparisons. Estimate uncertainty across people/runs, not independent augmented windows.
- Include meaningful negative controls and permutation checks. Pre-cue decoding triggers an audit of sequence constraints, carryover, filtering, and parsing.
- Use held-out confirmation as the search grows. A repeated sweep over the same outer folds can itself overfit validation.

## 7. Method families worth investigating

Prioritize by the question each family answers, not by a fixed march from simple to complex.

**High initial value for ML:** regularized time-domain ERP features with shrinkage LDA/ridge/logistic classification; band power and regularized CSP/filter-bank CSP; shrinkage covariance with tangent-space linear models and MDM; phase-specific features; baseline-relative power/covariance. Consider xDAWN/template covariance after finding reproducible time-locked information. These methods can complement each other, but any final combination must satisfy category rules.

**High initial value for DL:** compact EEGNet-like, shallow temporal/spatial, or EEGSimpleConv-like models that retain cue timing; a baseline-conditioned or phase-aware model if the evidence supports it. Start from an eligible input pipeline. Compare oscillatory and phase-sensitive summaries instead of presuming average pooling is sufficient.

**High-value transfer comparisons:** runtime normalization, EA/Riemannian recentering with alternate reference domains, and adaptive/latent normalization with source-statistic shrinkage. Whole-set target methods and spatial whitening need the relevant rule clarification before final use.

**Conditional branches:** FBCNet, ATCNet, EEG Conformer, modest temporal attention/convolution; source weighting; domain-adversarial/CORAL/MMD or episodic approaches; pseudo-label/entropy adaptation. More layers do not establish better transfer. The original ATCNet repository warns about the old evaluation methodology; do not copy its old test-selection practice.

**Foundation-model tournament only after the signal audit:** REVE is interesting for flexible geometry and cue responses; CBraMod is another plausible generalist; MIRepNet is an MI-specific candidate whose original headline evaluations use 30% labeled target-session fine-tuning, not zero-calibration Cross Subject evaluation. EEGPT, LaBraM, BIOT, or ST-EEGFormer are optional competitors when there is a concrete rationale. Audit exact checkpoint, license, pretraining data, channels, units, sample rate, patch duration, normalization, and preprocessing legality. A larger interpolated montage does not create information. Compare against a matched randomly initialized backbone and appropriate fine-tuning regimes.

Deprioritize expensive source localization, high-density connectivity models, massive from-scratch transformers, Mamba merely for novelty, and indiscriminate synthetic augmentation unless the data indicate a specific advantage.

## 8. Unusual ideas with a dataset-specific mechanism

**A. Baseline-conditioned invariance.** For a classical branch, compare task and baseline covariance using generalized eigenvalues of C_task v = λ C_baseline v. These eigenvalues are invariant to the same invertible sensor mixing applied to both matrices. Test the assumptions, shrink noisy estimates, and compare against simpler ratios. Spatial orientation is lost, so retain a complementary spatial representation if useful. For DL, test an end-to-end baseline/task input rather than automatically importing handcrafted covariance features.

**B. Decode the transition into feedback.** Test whether late EEG helps conditional on pre-feedback evidence rather than treating all windows as interchangeable. Error-related responses are plausible but unverified; an error response alone does not reveal the true class without information about the action being evaluated.

**C. Reconstruct an approximate original controller.** In Single Subject, the person's calibration recordings may support an approximation of the paper's CSP/SVM controller and its within-trial decision trajectory. Ask whether that summary helps interpret late EEG. Use no true test labels or prohibited streams. This is a classical branch; do not feed classical-controller outputs into a pure DL final entry. Exact reproduction of the historical controller is not assumed possible.

**D. Prefer baseline context to class-mixture context when appropriate.** A target task batch's mean depends on its unknown class proportions; a pre-cue reference may be less entangled with those proportions. Test whether it stabilizes adaptation rather than assuming it is neutral.

**E. Make regional ablations reference-aware.** Apparent central decoding can contain posterior activity introduced by common-average referencing. This control can prevent choosing the wrong channels and model family.

**F. Learn which sources transfer, not only which sources look similar.** For Cross Subject, evaluate source weighting using held-out transfer performance and protocol compatibility. Keep an equal-weight control and protect against overfitting seventeen people.

These are hypotheses to earn or discard, not mandatory features.

## 9. Class-count inference requires evidence

Never infer exact 20/20 balance from forty test trials and a 25/25 acquisition design.

If ten labeled trials and forty test trials are proven exact complements of a complete fifty-trial run, the remaining move count would be **25 minus the number of labeled moves**. If forty were instead a random subset, the count would be hypergeometric, not fixed; under uniform sampling the probability of exactly twenty moves is only about 27.5%. Those are conditional mathematical statements, not established facts about this release.

Test hard counts, soft priors, or unconstrained inference only against realistic held-out selection patterns. Final count-constrained predictions require organizer clearance. Do not complete a run using another competition, reverse engineer hidden label-generation seeds, or use leaderboard responses to recover individual labels.

## 10. Legal boundaries

Re-read the current overview, data, specific rules, foundational rules, and organizer clarifications. The Work snapshot establishes:

- No cross-competition data use/cross-reference; no pooled supplied-person training in Single Subject.
- No human labeling/prediction of hidden validation or test records; no multiple accounts; no private competition code/data sharing outside the team.
- External data/models are conditionally permitted if reasonably and equally accessible. Public external datasets must predate competition start; purpose-recording a new dataset for the score is prohibited. Do not treat the other competition as external data.
- AutoML products or similar automatic model-building tools are prohibited. Use controlled researcher-designed experiments; clarify any large automated search scheme whose status is doubtful.
- ML entries cannot contain neural-derived transforms and explicitly cannot use PyTorch/TensorFlow. The narrow MLP exception does not authorize a hybrid neural pipeline.
- DL entries must be end-to-end. Classical LDA/PCA/ICA/FIR/SVM operations are excluded except the stated exceptions: 50 Hz notch plus broad 1–100 Hz filtering; runtime-computed transforms retaining a time axis; uniform runtime-adaptive normalization; specified similarity operations.
- Fixed narrow filter banks, spatial whitening as normalization, required pretrained-model resampling, full-target-set adaptation, gradient TTA, auxiliary device channels, same-category ensembles, and count inference need precise clarification where the rule text does not settle them. Keep safe alternatives active while these are unresolved.
- Do not blend ML and DL predictions in a final entry.
- Submission source-code licensing has an OSI/commercial-use requirement, subject to actual exceptions. Public availability alone is insufficient. In particular, the public LatentAlignment code is CC BY-NC 4.0: do not copy it into a final solution under the default rule. A lawful independent implementation of the published method is a separate possibility. Audit code and pretrained-weight licenses separately.
- The final notebook must run in Kaggle. Verify runtime, dependencies, input availability, checkpoint permissions, and reproducibility well before the deadline.

Research ambiguous methods on held-out labeled training data if useful, but **do not use them for final submissions until clarified**. Prepare concrete questions; do not send organizer messages without authorization. Do not halt unrelated safe work while awaiting a clarification.

## 11. Use substantial compute for evidence

Choose experiments by **expected information gain** and **expected competitive value**. A cheap experiment that falsifies the dominant explanation may be worth more than training ten new architectures.

Use CPUs for parsing, quality analysis, classical comparisons, covariance/reference ablations, and repeated grouped evaluation. Use GPUs for a small number of informative neural branches, then expand those that earn resources. Run independent jobs concurrently within available allocations and monitor failures rather than launching a large unattended search with an unreliable loader.

Spend compute on controlled contrasts, meaningful interactions, repeated seeds for important effects, robustness across people/runs, adaptation-context stress tests, and complementary ensembles. Match data, training budget, and validation when comparing models. Distinguish pretraining from architecture and window effects. Avoid exhaustive combinations with no scientific question.

Maintain a concise experiment record with: hypothesis; competing explanation; split and information budget; feature/model change; category and legal status; result and uncertainty; failure analysis; and next decision. Save enough state and out-of-fold predictions to resume and reproduce conclusions.

## 12. Operate as an autonomous research loop

Use:

**observe → hypothesize → design a discriminating experiment → run → evaluate → compare with existing evidence → update the hypothesis → choose the next highest-value experiment.**

Do not substitute “implement fifty models and keep the largest validation score.” Challenge this prompt, abandon weak directions, investigate unexpected observations, and propose better approaches. Explain major changes of direction with the evidence that caused them.

Initially prioritize data integrity, protocol composition, and the time/space/frequency map. If early sensory information dominates, pursue it through eligible methods. If clean sensorimotor features already capture nearly all useful signal, deprioritize cue/feedback machinery. If errors concentrate in corrupted recordings, solve quality handling before adding capacity. If normalization beats elaborate TTA, use the simpler robust method.

## 13. Leaderboard discipline and end goal

The 3 October Work snapshot showed leading public displays around 0.90 Single Subject and 0.88 Cross Subject. These are historical rounded public scores, not private results or a guarantee of an attainable score.

Build trustworthy local validation first. Use a small number of public submissions to test whether local conclusions transfer, with a recorded prediction of what each submission should teach. Do not reverse-engineer labels, chase tiny rounded movements, or let public feedback replace participant/run-held-out evaluation.

Eventually produce the strongest supported legal candidates for both competitions, with correct submission ordering, reproducible Kaggle notebooks, eligible dependencies and checkpoints, and a concise explanation of why the selected candidates should generalize. Preserve a safe fallback while investigating higher-risk methods. The objective is competitive performance, earned through reliable experimentation—not merely an elegant research benchmark.

Before expensive optimization, give a short evidence-based update: what the files actually contain, what changed relative to Work, which unknowns now matter most, and which experiments you chose first. Then continue executing and iterating autonomously.

Useful primary references if the full report is unavailable:

- NeuRestore: https://repository.essex.ac.uk/42506/
- Organizer clarification: https://www.kaggle.com/competitions/low-cost-motor-imagery-decoding-for-rehab/discussion/743366
- Cue-onset study: https://www.frontiersin.org/journals/neuroergonomics/articles/10.3389/fnrgo.2026.1804143/full
- Latent alignment: https://doi.org/10.1088/1741-2552/adb336
- Revisiting EA: https://arxiv.org/html/2502.09203v1
- Resting-state adaptation: https://arxiv.org/html/2405.19346v1
- Online adaptation: https://arxiv.org/html/2311.18520v2
- T-TIME: https://arxiv.org/html/2412.07228v1
- REVE: https://arxiv.org/html/2510.21585v1
- MIRepNet: https://arxiv.org/html/2507.20254v1
- EEG-FM-Compass: https://arxiv.org/abs/2601.17883v3

**Begin by inspecting the available data and environment, establishing the facts that would change the strategy, and running the most informative legal experiments.**
