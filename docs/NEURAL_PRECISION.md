# Neural numerical reproducibility investigation

Status: updated 5 October 2026. This note records a read-only investigation plus a tiny
synthetic CPU calculation. It does not change the model, training configuration,
checkpoints, predictions, or notebook implementation. No confirmation labels
were inspected.

## Observed failure

The 17 saved Within Subject GPU models in
`results/within_subject/dl_pilot_full` were replayed on CPU in
`results/within_subject/dl_pilot_full_reloaded`. All 680 rows have corresponding
trial identities, but the strict replay check failed:

- Maximum absolute probability difference: **0.001675375835**.
- Mean absolute probability difference: **0.000083485744**.
- 679/680 hard labels match. Test ID 378, S011, changes from move
  (`p_move=0.5002217758301493`) to rest (`p_move=0.4998014271363707`).

The configured `1e-6` probability tolerance has not been relaxed. The absence of
`verification.json` in this CPU replay directory is expected: the verifier
writes success only after every check passes. Original pilot checkpoints lack
historical signal-cache and source hashes. A successful replay cannot recover
that missing historical provenance.

## Plausible mechanisms; cause not yet established

`TrialChannelStandardize.forward` receives float32 raw EEG and computes
`x - x.mean(-1)`, then the centered mean square, entirely in float32. The audit
finds file/channel offsets as large as approximately 715,000 CSV units, against
much smaller within-trial fluctuations. At that offset, float32 spacing is
0.0625. Different reduction orders can therefore change the estimated mean by
a meaningful fraction of the signal standard deviation. Subtracting a large
rounded mean can leave a small unintended channel offset after normalization.

CPU and GPU convolution implementations can also differ. TF32, when enabled
for eligible CUDA operations, is a separate possible contributor after
standardization. Disabling TF32 does not change the float32 input mean and
variance reductions. A GPU/CPU probability comparison alone cannot distinguish
these mechanisms.

The original checkpoint-only CPU replay did not explicitly restore the
deterministic backend settings established during training. The current runner
does so before inference, and records the actual TF32/determinism settings. This
addresses an avoidable configuration ambiguity; it does not promise equality
between devices or prove the cause of the earlier failure.

## Small synthetic evidence

In `myenv`, on CPU with one Torch thread, a fixed generator seed 20261003
produced a `(4, 8, 1200)` tensor of Gaussian samples with standard deviation 20.
Three constant offsets were added. The reference centers and standardizes the
same already-quantized input in float64. The alternative first subtracts the
first sample of each trial/channel, then uses the existing float32 standardizer.

| Added offset | Maximum native normalization error versus float64 | Maximum first-sample-shifted error versus float64 |
| ---: | ---: | ---: |
| 0 | 5.08e-7 | 5.49e-7 |
| 300,000 | 2.07e-3 | 4.04e-7 |
| -700,000 | 4.77e-3 | 4.02e-7 |

At -700,000 the native standardized output retained a maximum absolute channel
mean of 0.00474. This demonstrates numerical sensitivity at the audited scale;
it is **not** a real-data CPU/GPU diagnosis or an accuracy experiment.

## Completed GPU replay and remaining diagnostic

`jobs/neural_gpu_replay.pbs`, submitted as `4268135.pbs-7`, first replays the
historical checkpoints on an L40S with deterministic settings, then verifies
against the historical predictions, then runs a second replay with TF32
explicitly disabled. Both stages completed. Same-L40S probabilities and logits
match the historical outputs exactly for all680 rows. With TF32 disabled the
maximum probability difference is1.1451e-7 and no labels change. Thus TF32 does
not explain the approximately .001675 CPU discrepancy in this case. Complete
Within notebook retraining also exactly reproduces the original GPU predictions.
Cross-device identity remains unverified; large-DC centering and other backend
effects are possible contributors, not proven causes.

For the next diagnostic, use a copied S011 checkpoint and its 40 test trials,
without fitting, threshold tuning, or reading hidden labels:

1. Record CPU and GPU outputs of `model.standardize` on identical raw float32
   inputs; compare their maximum difference and residual channel means. Also
   compare each with float64 centering of those same input values.
2. Repeat the GPU calculation with TF32 disabled. The standardization output
   should be unchanged; compare final logits with TF32 enabled versus disabled.
3. Feed one identical, precomputed standardized float32 tensor to the remaining
   model on CPU and GPU, bypassing only the standardizer in temporary diagnostic
   model copies. This isolates downstream convolution/normalization differences
   from input centering differences. Keep weights, trial order, and batch size
   fixed, and record runtime flags and GPU model.
4. If needed, repeat a same-device replay before investigating CPU/GPU equality.
   A failure on the same GPU/settings is a different issue from portability.

## Minimal future numerical contrasts

After preserving the historical implementation and its results, test these as
explicit new configurations, with fresh training and validation:

- **First-sample shift:** compute `shifted = x - x[..., :1]`, then subtract
  `shifted.mean(-1)` and use the existing variance/epsilon calculation. The
  shift removes the large DC component before the reduction. In exact
  arithmetic this is the same per-trial, per-channel z-score. It neither uses
  other trials nor changes the temporal information budget.
- **Float64 centering:** cast the input to float64, subtract its float64 mean,
  then cast the centered values to float32 before the current variance and
  learned layers. A full float64 normalization followed by a cast is another
  useful reference, with a greater computation cost on some GPUs.

Neither method restores precision already lost when raw CSV values were cast
to float32 in the cache. Either changes numerical model behavior and should not
be silently substituted into a historical checkpoint replay. Fixing inference
alone would also mismatch the normalization used during training.

## Claims currently supported

The saved models are reloadable and produce finite, almost matching CPU
predictions; exact cross-device labels and the requested numerical tolerance
are **not verified**. Matching architecture/state shapes and deterministic
flags are necessary controls, not evidence of universal numerical equality.
Same-L40S reproducibility is verified exactly for all680 probabilities and
logits; full raw-data Within notebook retraining also reproduces them exactly.
The cross-device difference and historical provenance limits remain explicit.
