# ERP-template covariance experiment

Decision recorded before this branch's results: the first two seconds of EEG
support cross-person ERP classification, while a generic RBF classifier and
simple waveform/reference changes do not improve it convincingly. Test whether
an explicit comparison to class-average cue waveforms supplies useful nonlinear
spatial/temporal information.

Use exactly the trial-local 8-channel by50-bin representation from `erp_pre`:
0–2s,20Hz lowpass within that window, baseline-standardized,40ms averages.
Fit two class mean templates on source-fold labels. Concatenate both templates
with one query waveform, giving24channels. Compute covariance with10% isotropic
shrinkage, trace-normalize, and map to the log tangent representation around
the source-fold mean covariance. Standardization and logistic C=0.1 are fitted
on the same source fold. Neither templates nor reference depend on query EEG,
query labels, or other target trials. No neural library or external code/model
is used. No hyperparameter sweep is performed.

Cross development groups0–4 are evaluated; S019/S020 remain excluded from all
source and query sets. Save1575 OOF probabilities and compare run-weighted
accuracy, per-domain ranking and logloss with `erp_pre` and `erp_pre_centered`.
The latter comparison accounts for covariance discarding waveform DC means.
If the model does not improve robustly, deprioritize this representation instead
of expanding its hyperparameters. If it does, verify with a label permutation
control and participant-level paired comparisons before confirmation.

## First result and prespecified follow-up (4 October 2026)

The initial five development groups gave 73.43% target-run-weighted accuracy,
75.11% pooled accuracy, .525 log loss and .816 mean participant/run AUC.
The unchanged linear ERP comparator gave 69.98%, 71.87%, .577 and .772.
All 1,575 query trials and fold memberships match. The template/reference/scaler
use source people only; S019/S020 remain excluded. Tests also verify query
batch independence and immutable fitted templates/references at inference.

Before seeing further results, retain C=.1, shrinkage=.1, the same window and
normalization, and the .5 threshold. Two checks are now committed:

1. Permute source labels within participant/run, seed20261004, under the
   original five groups. This is a sanity check, not a significance test.
2. Regroup the same fifteen development people using NumPy's default_rng seed
   20261004 and five groups of three. Compare the exact fixed template model
   against the exact fixed linear ERP under these same alternate splits.
   This checks sensitivity to source/query composition without opening the
   reserved people or changing hyperparameters.

`scripts/run_erp_robustness.py` defaults to the second check. Its separate
`--confirmation` switch is not invoked by the development PBS script. If the
gain survives and the control remains near chance, freeze these two models
for one comparison on the two reserved people. Do not tune them on that result.

The alternate grouping completed: template73.94% versus linear69.41%, with a
paired conditional participant/run accuracy difference of+4.49points and a
descriptive95% bootstrap interval[+0.36,+9.42]points. Nine people improve, five
worsen, and one is unchanged. The permutation control gives53.16%.

The frozen confirmation comparison was then opened once, with its decision
record in CROSS_CONFIRMATION_FREEZE.md. The template did not improve the
primary metric:74.33% versus76.00%. Its run3 accuracy was55% versus65% on
twenty queries, despite better pooled accuracy and log loss. Keep linear ERP
as the default candidate. Preserve the template notebook for reproducibility,
not as a confirmed promotion. Do not modify settings using these results.

The standalone variant `notebooks/cross_subject_erp_template_ml.ipynb` now
passes complete raw-data execution (job4270273,39.65s). Its360 probabilities
exactly match an independent audited-cache refit and a saved-model reload.
The original linear notebook is unchanged. All raw-input hashes, metadata,
output hashes and source/target boundaries pass artifact validation. The
template remains an optional alternative, not a confirmation-supported upgrade.
