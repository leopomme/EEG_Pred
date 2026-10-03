# Delivered EEG data audit — 3 October 2026

Each competition was read and audited in its own process. Duplicate detection,
chronology checks and all statistics are confined to that competition. No
external research-pack data, hidden-label inference or inter-competition record
matching was used. The machine-readable evidence is in
`artifacts/<competition>/audit.json`, `metadata.csv` and `counts.csv`.

## Delivered trials and validation consequences

| Independently audited release | Train files / people / trials | Test files / people / trials | Training rest / move |
| --- | --- | --- | --- |
| Within subject | 50 / 17 / 1,795 | 17 / 17 / 680 | 900 / 895 |
| Cross subject | 50 / 17 / 1,795 | 9 / 3 / 360 | 900 / 895 |

In each release, training runs contain 825 run-1, 800 run-2 and 170 run-3
trials. S006 run 1 contains 25 complete trials (14 rest, 11 move) and an unused
final fixation-start marker. S007 run 2 is absent. Other supplied calibration
files have 50 trials, with 25 rest and 25 move; every training run-3 file has
10 trials. Run-3 labels total 86 rest and 84 move, with 2–8 moves per person.
Counts are observed, rather than assumed from the protocol.

Within-subject training/test people are S001–S007, S009–S012, S014 and
S016–S020. All test trials are run 3, 40 per person. For every person, the ten
labeled run-3 cues precede the forty test cues. The last training timestamp and
first test timestamp differ by one measured sample, with no overlap or internal
timestamp gaps. The test file begins at its first fixation marker. Thus the
released pieces reconstruct a continuous, chronologically ordered sequence of
50 observed trials: the labeled prefix followed by the test suffix. Random
support/query splits of the first ten do not validate extrapolation to the
last forty. Forward/blocked stress tests remain necessary.

This establishes the observed prefix/suffix structure. It does not independently
verify 25/25 hidden class counts or grant permission for count-constrained
inference. No complementary count or forced test balance is applied.

Cross-subject test people are S008, S013 and S015, absent from that release's
training people. Each has three files with 40 marked trials each: 120 test
trials per run. Feedback run 3 is therefore **9.47% of labeled training trials
and 33.33% of test trials**. Run-weighted participant-held-out evaluation is
necessary. The training run-3 files have no experiment-stop marker and cover
only ten trials; this release itself provides no later labeled run-3 tail.

Cross-subject test recordings retain **101.993–104.380 seconds of leading EEG
before the first fixation marker**. Their forty marked cues are consecutive,
about ten seconds apart, with no inter-cue gap above 15 seconds. The leading
signal has no trial/cue markers. This is consistent with a later forty-trial
block following an unmarked prefix; the original trial identities of the prefix
cannot be established from these files alone. The cache includes only explicit
marked trials. Prefix EEG is not treated as labeled calibration or supervised
rest.

## Columns, timing and segmentation

Every delivered file has exactly:
`time,Fz,C3,Cz,C4,PO7,Pz,PO8,Oz,Marker_val`.
The eight named EEG columns are the only predictor columns. There are no
auxiliary sensor fields. Numeric amplitude units and acquisition reference are
not established by these CSVs; no physical-unit conversion is applied.

Nominal sampling is 250 Hz. The numerical timestamp spacing ranges from
3.991293 to 3.997577 ms across files. Within a file the grid is almost exactly
affine (largest deviation from its median interval below 1.9 ns). This should
not be confused with a measurement of the hardware clock or used to invent
timestamp jitter. No resampling is applied. Metadata retains actual timestamps;
model-window names use the nominal 250-Hz grid.

Both `trial_start` and `trail_start` are recognized. A complete trial pairs its
own start, one `rest`/`move` (train) or `cue_start` (test), and the following
`cue_stop`. Unknown markers, split-incompatible cues, missing stops and invalid
boundaries fail explicitly. Extra stops outside trials are recorded: 43 in
within-subject files and 36 in cross-subject files. They do not create trials.
There are no unclosed labeled/test cues. No timestamp gaps, backwards steps,
nonfinite EEG, overlapping native trials, overlapping cached windows, exact
duplicate files or exact duplicate first-1,000-sample EEG segments were found
within either release. The duplicate check does not claim to detect arbitrary
near-duplicates or every possible partial overlap.

Fixation lasts 659–756 native samples (2.6312–3.0185 s); task cue lasts
1,242–1,279 samples (4.9589–5.1092 s). Assuming every task lasts exactly 1,250
rows would include post-cue samples in some trials. All trials support the
nominal `[-2,0)` baseline and `[0,4.8)` task windows. A nominal `[0,5)` task
extends beyond `cue_stop` in 41 within-subject training trials; cross-subject
has 41 such training trials and 3 test trials. All within-subject test trials
support that full window. Feedback update times are protocol information;
there are no separate feedback-update markers in the files to verify them.

## Quality observations and preprocessing implications

All EEG samples are finite. Across each independently audited release,
file/channel medians range from −714,859.7 to 308,650.39 in CSV units, and
file/channel standard deviations range from 19.93 to 4,571.20. These raw
statistics include large DC offsets and slow drift; they are not estimates of
physiological amplitude or pure noise. S019 run 1 has particularly large
whole-record variability (C4 standard deviation 4,571; C3 4,241); S003 run 1
also has broad variation. Cross-subject test S008 run 1 has C3 standard
deviation 3,191. Per-trial centering/normalization and explicit quality analyses
are therefore important controlled choices.

S017 run 2 contains a 20-sample constant segment in every channel. Other
recordings' longest constant segments are at most four samples. Recorded
extrema do not show substantial accumulation (maximum fraction at a file
minimum or maximum below 0.008%), but ADC limits are unknown, so this does not
establish an absence of saturation. No trial or channel is automatically
discarded based on these descriptive statistics.

`trial_quality.csv` and `quality_summary.json`, produced by
`scripts/summarize_audit.py`, record per-trial/channel standard deviation,
peak-to-peak range and largest step on the shared valid task window, exact
locations of long constant segments, and marker timing relative to file
boundaries. These are label-free measurements; any fitted reliability rule
must still be developed within the appropriate training folds.

## Reproducible cache contract

`artifacts/<competition>/epochs.npz` contains raw `float32` `X` of shape
`(N,8,2000)`, channel order `Fz,C3,Cz,C4,PO7,Pz,PO8,Oz`, nominal `fs=250`,
`cue_index=750`, and `cache_version=1`. The window is native cue-relative rows
`[-750,1250)`, conventionally labeled `[-3,5)` seconds. This is a finite window;
native cue lengths beyond its endpoint remain documented in metadata.

`y` is `int8`: rest=0, move=1, test=−1. `valid_mask` marks finite samples in
the cue's continuous recording segment. `phase_valid_mask` additionally
requires rows at or after fixation start and strictly before cue stop. Recorded
samples outside those phase bounds remain in raw `X`; model code must respect
both masks. Missing/discontinuous support would be NaN, never silent zeros or
edge padding. In the released data, all recording masks are true; phase masks
exclude 674 within-subject samples and 657 cross-subject samples.

Metadata rows align exactly to `epoch_index`; they include subject, numeric
run, split, filename, zero-based trial index, marker row/timestamps, native
phase lengths and window-validity flags. Training rows precede test rows.
Within each split, files are lexicographically sorted and trials follow marker
order. Test rows have contiguous zero-based `test_order`; train rows have −1.
No header/ID base is inferred from this internal index. Neither delivered ZIP
contains a sample submission, so exact Kaggle `ID` mapping still requires
checking the official sample before CSV submission.

The 11 loader tests cover marker errors/spelling, named channel order, phase
endpoints, out-of-file samples, timestamp gaps/nonfinite samples, native clock
preservation, and cache label/order round-tripping. The additional quality job
validates the actual cache alignment, channel contract and recording-mask
finiteness. Full audits ran through PBS: within-subject job `4263099.pbs-7`
(39 seconds, exit 0), cross-subject `4263100.pbs-7` (33 seconds, exit 0).
Rebuild with `jobs/audit_within_subject.pbs` and `jobs/audit_cross_subject.pbs`;
the quality supplement uses `jobs/audit_quality.pbs`. Direct logs are in `logs/`.

## First experimental decisions supported by this audit

Start with leakage-safe per-person and participant-held-out baselines using
`[-2,0)` baseline and up to `[0,4.8)` task support, then compare early versus
pre-feedback versus late windows without letting phase padding encode
duration. Report run-3 separately and weight cross-subject runs equally for
the verified target mixture. Check pre-cue decoding as a negative control
without filtering task signal into baseline. Use chronology-aware single-
subject stress tests and retain the unresolved early-to-late feedback gap in
cross-subject interpretation. Treat reference choice, per-trial normalization
and artifact sensitivity as measured comparisons before expanding models.
