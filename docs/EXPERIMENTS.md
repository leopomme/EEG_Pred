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

## Results

Pending the completed data audit and first allocated compute jobs.
