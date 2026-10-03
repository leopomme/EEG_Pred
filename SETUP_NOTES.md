# Preparation for the two EEG Kaggle challenges

Checked on 3 October 2026. This preparation covers extraction, the existing
`myenv` environment, and cluster GPU access. Modelling has not started;
`PROMPT_FOR_CODEX.md` and `EEG_Kaggle_Strategy_Report.md` are unchanged.

## Extracted data

| Challenge | Directory | Training CSVs | Test CSVs | Uncompressed bytes |
| --- | --- | ---: | ---: | ---: |
| Within subject | `data/within_subject/` | 50 | 17 | 626,354,491 |
| Cross subject | `data/cross_subject/` | 50 | 9 | 568,722,202 |

The original directory names inside each ZIP are preserved:

- `data/within_subject/Training set/`
- `data/within_subject/Single subject test set/`
- `data/cross_subject/Training set/`
- `data/cross_subject/Cross participant test set/`

All 126 extracted CSVs passed size and CRC checks against their ZIP entries.
The original ZIPs are retained. SHA-256 comparisons confirmed that the 50
training CSVs are byte-identical between challenges; both copies are retained
in their respective challenge directories. This comparison is an inventory
fact, not a decision about permitted competition data use.

All three inspected CSV headers were:
`time,Fz,C3,Cz,C4,PO7,Pz,PO8,Oz,Marker_val`.
Neither archive contains a sample submission or other non-EEG file.

## Existing environment

Conda prefix: `/rds/general/user/lh5218/home/anaconda3/envs/myenv`.
Activation was verified with:

```bash
source /rds/general/user/lh5218/home/anaconda3/etc/profile.d/conda.sh
conda activate myenv
```

Python is 3.10.16. Successful imports: NumPy 1.26.4, SciPy 1.15.3,
pandas 1.5.3, scikit-learn 1.2.1, PyTorch 2.3.0+cu121,
torchvision 0.18.0+cu121, MNE 1.12.1, h5py 3.9.0, joblib 1.4.2,
matplotlib 3.10.9, tqdm 4.68.4, and einops 0.6.1.

PyTorch was built for CUDA 12.1. This session is on `login-ai` and has no
allocated GPU: `torch.cuda.is_available()` returned false. GPU execution and
driver compatibility must be checked inside an allocated PBS job.

Not installed: pyriemann, braindecode, xgboost, lightgbm, catboost, torchaudio,
and the kaggle Python package. Whether these are needed depends on the later
implementation.

`pip check` reported three existing dependency conflicts:

- histolab 0.7.0 requires NumPy <=1.24.4; installed NumPy is 1.26.4.
- histolab 0.7.0 requires SciPy <1.10.1; installed SciPy is 1.15.3.
- TensorFlow 2.12.0 requires NumPy <1.24; installed NumPy is 1.26.4.

The checked PyTorch/scientific imports work, but the environment is not fully
dependency-consistent. TensorFlow was not validated. No packages were changed.

## Cluster GPUs and PBS

Current server: `pbs-7` (CX3); default routing queue: `cx`.
The live PBS node inventory was captured at 15:35 UTC / 17:35 Paris time.

| GPU type | VRAM per GPU (guide) | GPUs visible to this server | Assigned at snapshot | Node queue |
| --- | ---: | ---: | ---: | --- |
| L40S | 48 GB | 56 on 7 nodes | 56 | `v1_gpu72` |
| A100 | 40 GB | 4 on 2 nodes | 4 | `v1_gpu72` |
| Quadro RTX6000 (Turing) | 24 GB | 80 on 10 nodes | 51 | `v1_jupytergpu` |
| A40 | 48 GB | 8 on 2 nodes | 4 | `v1_jupytergpu` |

The unassigned A40 node was marked `state-unknown,down`. RTX6000 and A40
capacity is in the JupyterHub queue, not the ordinary batch GPU queue, according
to the live `Qlist` resources. The GPU guide has conflicting RTX6000 statements;
use the observed scheduler configuration for this session. Published cluster
totals include 88 RTX6000 GPUs, but this PBS server exposed 80 at the snapshot.
No batch GPU was unassigned at that time; availability changes with the queue.

L40S is the configured default for batch GPU jobs. A small initial GPU job can
use the following resource specification:

```bash
#!/bin/bash
#PBS -N eeg_gpu_check
#PBS -l select=1:ncpus=4:mem=24gb:ngpus=1:gpu_type=L40S
#PBS -l walltime=00:05:00
#PBS -j oe

set -e
cd "$PBS_O_WORKDIR"
source /rds/general/user/lh5218/home/anaconda3/etc/profile.d/conda.sh
conda activate myenv
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
nvidia-smi
python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); x = torch.ones(16, device="cuda"); print(x.sum().item())'
```

Submit a saved script with `qsub script.pbs`, without specifying an execution
queue. PBS routes to `v1_gpu72`; direct submission to that execution queue is
disabled (`from_route_only=True`). PBS sets `CUDA_VISIBLE_DEVICES` for the job.
Live queue limits include one node, up to 8 GPUs, and 72 hours. The guide lists
a per-user limit of 12 running GPUs across jobs. No job was submitted during
this preparation.

The separate HX1 facility offers A100 80 GB GPUs according to the guide.
Access to HX1 was not checked from this CX3 session.

Sources:

- [GPU jobs](https://icl-rcs-user-guide.readthedocs.io/en/latest/hpc/queues/gpu-jobs/)
- [Job sizing and routing](https://icl-rcs-user-guide.readthedocs.io/en/latest/hpc/queues/job-sizing-guidance/)
- [Cluster specifications](https://icl-rcs-user-guide.readthedocs.io/en/latest/hpc/cluster-specification/)
- [HX1](https://icl-rcs-user-guide.readthedocs.io/en/latest/hpc/hx1/)

## Recorded evidence

`setup/` contains extraction counts/checks, the training-file comparison,
package import results, Conda explicit package export, pip package inventory,
dependency-check output, and the live PBS node snapshot. These support the
later agent's environment inspection without changing the competition prompt.
