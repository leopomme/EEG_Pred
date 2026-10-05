# ds003810: header, event and window audit

Audited **5 October 2026**, using Python 3.10.16, MNE 1.12.1, NumPy 1.26.4
and pandas 1.5.3 in `myenv`. The reusable script is
[audit_external_lowcost.py](../scripts/audit_external_lowcost.py).
It reads sidecars, EDF headers and annotations with `preload=False`; it never
loads EEG sample arrays, extracts epochs, trains a model, or modifies the
uploaded files.

**The imagery subset is ready for a controlled signal-quality/loader audit.**
Its 1,590 class cues have observed start/end markers and support a two-second
fixation baseline plus an early two-second task window. Full-corpus strict
marker validation correctly fails on one execution trial with a missing end
marker. Absolute unit conversion also requires explicit care.

## Input and reproducibility

Dataset root:
`extraernal_doc_eg_kagle/Datasets/ds003810_lowcost_mi_rest/v1.0.2/`.
This is **NEMAR on003810 v1.0.2**, derived from OpenNeuro ds003810 v2.0.2;
the two version numbers refer to different releases. NEMAR records 4 June 2026
and CC0 for this release, before the competition's 22 September 2026 start.
The local `dataset_description.json` also declares CC0.
[Primary dataset record](https://nemar.org/dataset/on003810)

The authoritative audit artifacts for this script version are in
`artifacts/external/ds003810_audit_v2/`:

- `audit.json`: aggregate contract checks and explicit errors/warnings.
- `recordings.csv`: 50 recordings with participant/run/class counts.
- `trials.csv`: all 1,690 cue rows, native boundaries and window validity.
- `headers.json`: exact EDF header fields and inherited recording metadata.
- `annotations_by_recording.csv`: annotation types/counts per recording.
- `counts_by_run_class.csv` and `input_manifest.json`: counts and provenance.

The first output at `artifacts/external/ds003810_audit/` is retained; v2 makes
the independent validity of pre-cue baseline and unknown task endpoints
explicit. The output directory must be fresh. For a later rerun:

```bash
/rds/general/user/lh5218/home/anaconda3/envs/myenv/bin/python \
  scripts/audit_external_lowcost.py \
  --output artifacts/external/ds003810_audit_next
```

The script's default output is `artifacts/external/ds003810_audit`. It returns
exit status **1** after writing the report when a strict metadata/marker issue
exists; that status is intentional for the known missing execution marker.
The manifest records the script hash, complete hashes for included text
sidecars, EDF byte sizes and **header-only hashes**. It explicitly leaves full
EDF hashes unset. Full file checksums and signal-quality checks precede any
future training use. No signals are included in Git.

## Classes, runs and counts

The shared event dictionary and the supplied demo notebook explicitly map:

| Native annotation | Event value | Meaning |
| --- | ---: | --- |
| `OVTK_GDF_Right` | 7 | Grasp task: **execution in RUN0**, imagery in RUN1–RUN4 |
| `OVTK_GDF_Tongue` | 9 | Rest/idle |

These OpenViBE token names do **not** describe a right-hand-versus-tongue task.
All ten people are marked right-hand dominant. Participant IDs are
02,03,04,05,06,07,08,09,10,12; the gaps are intentional, with no missing
participant relative to `participants.tsv`.

The file naming is zero-based: the **first recording** is RUN0 execution;
the **second through fifth recordings** are RUN1–RUN4 imagery. A loader using
one-based display names must preserve that distinction.

Counts below are **grasp/rest**, separately for each person/run:

| Participant | RUN0 execution | RUN1 imagery | RUN2 imagery | RUN3 imagery | RUN4 imagery | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sub-02 | 5/5 | 20/20 | 20/20 | **15/15** | 20/20 | 160 |
| sub-03 | 5/5 | 20/20 | 20/20 | 20/20 | 20/20 | 170 |
| sub-04 | 5/5 | 20/20 | 20/20 | 20/20 | 20/20 | 170 |
| sub-05 | 5/5 | 20/20 | 20/20 | 20/20 | 20/20 | 170 |
| sub-06 | 5/5 | 20/20 | 20/20 | 20/20 | 20/20 | 170 |
| sub-07 | 5/5 | 20/20 | 20/20 | 20/20 | 20/20 | 170 |
| sub-08 | 5/5 | 20/20 | 20/20 | 20/20 | 20/20 | 170 |
| sub-09 | 5/5 | 20/20 | 20/20 | 20/20 | 20/20 | 170 |
| sub-10 | 5/5 | 20/20 | 20/20 | 20/20 | 20/20 | 170 |
| sub-12 | 5/5 | 20/20 | 20/20 | 20/20 | 20/20 | 170 |
| **All** | **50/50** | **200/200** | **200/200** | **195/195** | **200/200** | **1,690** |

The imagery-only benchmark contains **40 files, 1,590 trials, 795 per class**.
RUN0 contributes 100 execution/rest trials. sub-02 RUN3 has 30 actual cues,
not 40; retain the observed count. Source class balance does not establish a
permitted rule for enforcing class counts in Kaggle predictions.

## Channels, scale and recording properties

All 50 EDFs agree with the channel tables: **15 EEG channels, 125 Hz**,
continuous EDF+C with an extra annotation signal. The EEG channel order is:

`Pz,Cz,T6,T4,F8,P4,C4,F4,Fz,T5,T3,F7,P3,C3,F3`.

Only **Fz,C3,Cz,C4,Pz** overlap our eight target EEG electrodes. PO7, PO8 and
Oz are absent. The sidecar reference is **left earlobe**; the ground and
manufacturer/model are unspecified. There are no EOG, EMG, ECG, miscellaneous
or trigger signal channels in this export. The README says EMG was acquired
for protocol control; its signal is not exported in these 15-channel EDFs.

The channel sidecars declare **µV**, but every EDF EEG physical-dimension field
is blank. MNE reports the original units as `n/a`, so it cannot infer ordinary
microvolt-to-volt scaling from this EDF header. A future loader must explicitly
document whether it adopts the sidecar's physical-unit interpretation. Do not
apply amplitude thresholds or noise augmentations in volts before that decision.
Scale-invariant trial normalization does not settle absolute physical units.

The README/root sidecar reports existing third-order Butterworth **0.5–45 Hz**
filtering. Native EDF prefilter fields are empty; MNE reports 0–62.5 Hz from
the header/default limits. Those values do not establish that unfiltered
broadband EEG is available. This dataset cannot teach frequencies removed by
its acquisition/export filtering.

Recordings last 125–450 s; imagery files last 339–450 s. Shared/per-file JSON
durations match `n_samples / 125` in all 50 files. No sample amplitudes,
clipping, flatness, channel noise or waveform artifacts were checked here.

## Native markers and timing

Native class-cue onset and rounded onset-sample indices exactly match all
1,690 TSV rows. For the 1,689 rows with observed stop markers, cue-to-stop
durations agree with TSV to numerical precision (maximum absolute error
below 6e-14 s). Event annotations have finer time precision than the 8 ms
sample interval; round cue onsets to the native grid when extracting later.

Every class cue has a preceding trial-start/cross marker. In imagery runs:

| Phase measurement | Minimum | Median | Maximum |
| --- | ---: | ---: | ---: |
| Trial-start/cross to class cue | 2.6223 s | 3.0032 s | 3.5900 s |
| Class cue to trial end | 3.8874 s | 4.0043 s | 4.3293 s |
| Last preparatory beep to cue | 0.9006 s | 1.0027 s | 1.9754 s |

Each file also has a separate prefix `BaselineStart`/`BaselineStop` segment:
approximately 10 s in execution RUN0 (9.9691–10.0198) and 20 s in imagery
runs (19.9193–20.0398). That prefix is not the trial-local fixation baseline
and is not automatically a labeled rest trial.

There are 1,689 `Feedback_Continuous` markers: 1,678 coincide with trial end;
the remaining 11 precede it by 1 ms. No such marker falls in the early two
seconds after a cue. These annotation names alone do not prove what visual
feedback was actually shown. `Correct`/`Incorrect` markers also occur; they
must not become predictors or alternative ground-truth labels.

The sole missing-end case is **sub-08 RUN0, zero-based trial 9**, move cue
at **111.9227 s**. The 126 s file has no subsequent `End_Of_Trial` marker,
although the curated TSV supplies **4.0000 s**. Therefore the README's claim
that every TSV duration comes from a next end marker does not hold for this
row. Preserve its endpoint as **unknown** rather than validating an invented
stop. This affects execution only; exclude RUN0 from the imagery benchmark.

## Compatible windows and next decision

| Cue-relative window | All-trial boundary check | Imagery-only boundary check |
| --- | --- | --- |
| Baseline [-2,0) | All 1,690 fit the observed start-to-cue phase | All 1,590 valid |
| Task [0,2) | 1,689 valid, one execution endpoint unknown | All 1,590 valid |
| Task [0,3) | 1,689 valid, one execution endpoint unknown | All 1,590 valid |
| Task [0,3.8) | 1,689 valid, one execution endpoint unknown | All 1,590 valid |
| Task [0,4) | 355 end-boundary violations, one unknown | **331 violations** |
| Task [0,4.8) | All 1,689 known trials cross the end, one unknown | **All 1,590 invalid** |

Every tested window fits within the overall recording. Checking only EOF
would therefore miss all phase-boundary violations. Keep the native markers
and explicit phase-validity masks in any future extraction.

A **five-channel, imagery-only, fixation-baseline versus early-cue comparison
is feasible** without endpoint reconstruction. At 125 Hz, 40 ms bins contain
five samples, so the same physical-time binning as our 250 Hz classical ERP
representation is possible. Use separately fitted external benchmark models;
hold out complete people/runs and report external performance as such. Do not
upsample to claim extra temporal information, invent posterior electrodes,
mix execution with imagery, or reuse the target's 4.8 s task endpoint.

Before external training, perform full file checksum verification and a small
waveform audit, confirm cue presentation semantics, declare units/reference
handling, and fix the transfer protocol. The dataset passes an initial release
date/license screen; final category-specific preprocessing and all other
competition requirements still apply. [Rules review](RULES_AND_SUBMISSION.md),
[research priorities](RESEARCH_PACK_REVIEW.md).

No external trial has entered any current competition model or submission.
