# HW6: CIFAR cat/dog classification on Delta

This is an experiment implementation and notebook scaffold. It contains no invented empirical results and is not a completed homework submission. The script and guide were prepared with generative AI; consult the syllabus for the required disclosure. You confirmed that SSH login now works. No GPU job has been submitted by this assistant.

## Your laptop and project

Verified on your laptop: **Apple Silicon (arm64), macOS 26.6.2, zsh**. macOS already provides `/usr/bin/ssh` and `/usr/bin/scp`; use Terminal for the commands below. Your course environment uses Python 3.9.6 at `/Users/wchae/Courseworks/26 Fall/DSO607/.venv/bin/python`.

| Setting | Your value |
| --- | --- |
| Local project | `/Users/wchae/Courseworks/26 Fall/DSO607/Solution/Chae_Woojin_DSO607_HW6` |
| Notebook | `Chae_Woojin_DSO607_HW6.ipynb` |
| NCSA username | `wchae` |
| Delta login | `login.delta.ncsa.illinois.edu` |
| GPU charge account | `bgxw-delta-gpu` (reported by your `accounts` output) |
| Delta storage project | `bgxw` (`/work/hdd/bgxw` appears in your directory listing; verify write access when creating the folder) |

Every Mac path containing `26 Fall` is quoted. Copy the commands without terminal prompt symbols. **Mac Terminal** means a local shell; **Delta Terminal** means the shell after SSH login. `module`, `accounts`, `sbatch`, and `squeue` belong in Delta Terminal. If unsure where you are, run `hostname`: your Mac and the Delta login node have different names. Use a second Terminal tab (Command-T) for local transfers while keeping the Delta session open.

## What runs where

Your Mac edits files and opens saved results. Delta's **login node** accepts SSH, file transfers, and Slurm commands. Slurm is the scheduler: `sbatch` queues a script and later runs it on a **compute node**. A GPU exists only inside a GPU allocation. The CNN uses PyTorch/CUDA on one NVIDIA GPU; scikit-learn's Lasso, KNN, and RF use the allocated CPUs. Asking for four GPUs would not accelerate this implementation.

These instructions target **NCSA Delta**, not DeltaAI (a separate system with different partitions). If your allocation is for DeltaAI, adapt the scheduler setup before using this script.

## How the pieces work together

```text
Your Mac --SSH--> Delta login node --sbatch--> Slurm queue
   |                                               |
   | scp                                           | starts when resources are free
   v                                               v
Shared Delta project files <---read/write--- GPU compute node
   |
   +--scp back to your Mac--> tables, plots, models, notebook
```

SSH gives you a remote shell; it does not reserve a GPU. `scp` copies files between computers; editing a Mac file does not automatically update its Delta copy. Slurm allocates compute resources. `sbatch` submits a script and returns a job ID immediately; the script starts later on a compute node. Inside the script, `srun` launches the Python processes in that allocation. This distinction is documented in [Delta's job guide](https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/running_jobs.html).

The login and compute nodes can both access your shared `/work/hdd/bgxw/wchae/Chae_Woojin_DSO607_HW6` directory. No extra copy to the GPU node is needed: the batch script reads the prepared arrays there and writes results there. Your Mac has a separate filesystem; use `scp` to retrieve those results. `cd` changes a terminal's working directory; it never copies files.

| Phase | Where it runs | What it accomplishes |
| --- | --- | --- |
| 1. Login and directory setup | Mac -> Delta login node | Establish access and a writable project directory |
| 2. Upload | Mac Terminal | Place your source files in Delta's shared project directory |
| 3. Environment check | Delta login node | Confirm the Python imports before requesting a GPU |
| 4. Synthetic smoke job | Prepare on login; train on compute node | Test the scheduler, GPU, and entire code path cheaply |
| 5. CIFAR pilot | Prepare on login; train on compute node | Check real-data fitting and estimate production runtime |
| 6. Full experiment | Prepare on login; train on compute node | Complete all 100 repetitions with frozen settings |
| 7. Final evaluation | Delta login node | Score saved test predictions after training is complete |
| 8. Download and reporting | Mac | Inspect outputs and complete your submission |

### What your job script requests

`run_delta.slurm` asks for one node, one task/process, eight CPU cores, one NVIDIA GPU, 24 GB of host RAM, and up to 12 hours. `gpuA100x4` names a partition whose nodes contain four A100 GPUs; `--gpus-per-node=1` requests only one of them. `--mem=24G` refers to CPU/system RAM, not the GPU's own memory. The wall-time request is an upper bound: Slurm stops a job that exceeds it. The smoke/pilot commands override that bound with shorter limits.

The script loads its own modules because a batch job needs a reproducible environment. The two `srun` commands first check GPU computation and then train. The methods run sequentially in one process: Lasso, KNN, and RF use CPUs; the CNN uses the GPU. GPU utilization may be low while classical methods run. This code does not distribute repetitions across GPUs.

`sbatch` acceptance means **queued**, not finished. `squeue` distinguishes pending/running jobs; `sacct` shows the final state and exit code. The job's standard output goes to `logs/hw6-cifar-JOBID.out`, and warnings/errors normally go to `.err`. A warning in `.err` alone does not establish failure; inspect the final job state and the expected completion messages. Create `logs/` before submission because Slurm opens these files before your script executes.

### What is saved and why

- `data/`: downloaded CIFAR source files on Delta, reused across runs.
- `runs/smoke`, `runs/pilot`, `runs/full`: separate prepared inputs, frozen settings, split indices, candidate logs, selected models, and saved test probabilities. Keeping these runs separate prevents synthetic and pilot outputs from being confused with the final experiment.
- `logs/`: scheduler job output, including setup/GPU diagnostics and exceptions.
- `delta_download/TIMESTAMP/` on your Mac: a snapshot of the retrieved final run and its code, for local inspection.

`prepare` creates and freezes the data and split indices. `train` fits models and saves test probabilities without computing test scores. `finalize` uses the saved probabilities to produce test errors and figures only after all 100 real-data repetitions finish. This separates model selection from final test evaluation, as required by HW6.

Copy command blocks as multiple lines. A trailing backslash means 'continue this same command on the next line'; add one only where shown. For example, `module reset` and `module load pytorch-conda/2.8` are separate commands with no trailing backslashes.

## Experiment choices and homework mapping

- Observation: one labeled CIFAR-10 image. Use only the original 50,000-image training pool; select cats (class 3 -> 0) and dogs (class 5 -> 1). Exclude all other classes and the original CIFAR test batch. Before duplicate removal this gives 10,000 images, 5,000 per class. This is a new binary experiment, not a standard CIFAR benchmark score.
- Remove byte-identical duplicate images globally before splitting: keep the first if labels agree; exclude the entire hash group if labels conflict. Save excluded source indices. Near duplicates and unobserved shared entities remain a limitation; CIFAR provides no subject/group IDs here. Inspect any available source provenance before asserting full independence.
- Fixed, nonlearned 2x2 averaging makes every image 16x16x3. All methods see this same information. No pretrained features, augmentation, imputation, feature selection, or external training data.
- Stratify a single 80/20 development/test split with `split_seed=6072026`. In each of R=100 repetitions, stratify development 80/20 using seed `160700+r`, shared by every method. Without duplicate exclusions: 6,400 train / 1,600 validation / 2,000 test; exact realized counts are in `split_counts.csv`.
- Fit channel means and standard deviations on the current training subset only. Apply the saved normalization to train/validation/test; flatten into 768 features for classical models, retain CHW layout for CNN. Training and validation outcomes alone determine models and settings.
- For each method choose minimum validation classification error; ties choose the earlier candidate in `config.json`. Class 1 iff probability >=0.5, including KNN vote ties. Never call an estimator's potentially different tie-breaking `predict` method. Retain the training-fitted selected model; do not refit on training plus validation.
- **Lasso:** L1 logistic regression, SAGA, unpenalized intercept, no class weights, `C` in {0.01, 0.1}, tolerance 0.001, max_iter 5,000. With n unweighted training samples, its objective is equivalent to average binary logistic loss + `||beta||_1/(C*n)`. Thus larger C means less regularization; C is not lambda. A convergence warning stops the run and is logged; resolve it using development data before final evaluation.
- **KNN:** Euclidean distance on normalized flattened pixels; brute-force search, uniform votes, k in {5,15}. Every k must be <= training size.
- **RF:** 100 trees, sqrt feature subsampling, min_samples_leaf=2, bootstrap=True, no class weights; candidate max_depth=10 or None. Initialization seed `260700+r`.
- **CNN:** input 3x16x16 -> Conv(3,16,3,padding=1), ReLU, 2x2 max pool -> Conv(16,32,3,padding=1), ReLU, 2x2 max pool -> flatten 512 -> dense 64, ReLU -> dense 1 logit. Spatial filters share weights, a natural inductive bias for image classification. All layers train from PyTorch's default random initialization, seed `260700+r`, reset for each candidate. BCEWithLogitsLoss, Adam learning rate 0.001, batch size 128, weight_decay in {0.0001,0.001}, no dropout. PyTorch Adam's standard coupled L2 decay applies to all parameters. Sigmoid converts logits to probabilities.
- **Early stopping:** at most EPOCHS=20 passes over training data, validation BCE after each pass; stop after 4 consecutive epochs with no strict decrease (min_delta=0). Restore the earliest lowest-validation-loss checkpoint. Choose between restored candidate models by validation classification error. Record both stopped and retained epochs. The plotted training loss is the online minibatch average during the epoch; validation loss uses end-of-epoch weights.
- Record split/model seeds, every candidate's settings, validation errors, warnings, fitting plus validation time, selected models, normalization, loss curves, probabilities, versions, and hardware. Test labels are saved separately during stratification; `train` never loads their values. `finalize` first requires every repetition to finish, then computes test scores. This is a workflow safeguard, not access-control encryption.
- The training-majority baseline predicts 1 when training prevalence >=0.5. Repeated errors are averaged, not predictions. `summary.csv` includes across-repetition sample SD (`ddof=1`). `paired_differences.csv` is method error minus Lasso error in the same repetition; negative favors that method.

## 1. Log in and identify your allocation (Mac Terminal)

**Purpose:** Establish an authenticated remote shell and identify where your allocation permits you to store files. SSH gets you onto a login node; it does not allocate a GPU. The directory creation gives the later upload a destination.

**Success check:** You have a Delta shell prompt, `accounts` lists `bgxw-delta-gpu`, and `mkdir` succeeds. This setup does not train a model.

You have resolved the login issue. Use your working NCSA credentials:

```bash
ssh -o PreferredAuthentications=keyboard-interactive,password \
  wchae@login.delta.ncsa.illinois.edu
```

Enter your NCSA password, then complete the NCSA Duo prompt (for a listed push option, enter `1` and approve it). Password typing is invisible. A `Permission denied, please try again` at the password prompt happens before Duo; verify the NCSA password through [NCSA account management](https://identity.ncsa.illinois.edu/). The [official Delta login guide](https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/login.html) documents the keyboard-interactive option above. Never save passwords or Duo codes in the project.

**On Delta**, run:

```bash
accounts
module spider pytorch-conda
```

Your `accounts` output identifies the GPU charge account as `bgxw-delta-gpu`, with 858 hours remaining when you copied it. The matching project code `bgxw` appears in `/work/hdd`; use `/work/hdd/bgxw/wchae` for this experiment. The balance can change as jobs run. The commands below already contain your account and path. Creating the directory below checks your write access; if it reports `Permission denied`, ask your project administrator to verify storage permissions.

```bash
ls /work/hdd
mkdir -p /work/hdd/bgxw/$USER/Chae_Woojin_DSO607_HW6
```

## 2. Transfer the files (a second Mac Terminal tab)

**Purpose:** Copy the seven source/configuration files from your Mac to the shared Delta project directory. Your Mac and Delta hold separate copies; transfer again when you intentionally change a source file before a new run.

**Success check:** The transfer completes without errors. In Delta Terminal, `ls /work/hdd/bgxw/wchae/Chae_Woojin_DSO607_HW6` shows the notebook, script, configuration, requirements, README, environment checker, and Slurm script.

First create the remote directory using Step 1. Then run this **on your Mac**; the destination is already filled in:

```bash
cd "/Users/wchae/Courseworks/26 Fall/DSO607/Solution/Chae_Woojin_DSO607_HW6"
scp -o PreferredAuthentications=keyboard-interactive,password \
  README.md experiment.py config.json check_environment.py run_delta.slurm \
  requirements-extra.txt Chae_Woojin_DSO607_HW6.ipynb \
  wchae@login.delta.ncsa.illinois.edu:/work/hdd/bgxw/wchae/Chae_Woojin_DSO607_HW6/
```

This transfers the seven project files without uploading local environments or downloaded results. To synchronize only this updated README later, run the following in Mac Terminal from the local project directory:

```bash
scp -o PreferredAuthentications=keyboard-interactive,password README.md \
  wchae@login.delta.ncsa.illinois.edu:/work/hdd/bgxw/wchae/Chae_Woojin_DSO607_HW6/README.md
```

 Complete NCSA password/Duo authentication again if prompted. If `No such file or directory` refers to a source file, confirm `pwd` shows your local homework folder. If it refers to the destination, create or correct the remote directory in Step 1.

## 3. Check the software (return to your Delta Terminal tab)

**Purpose:** Select Delta's installed Python/PyTorch stack and check that all required packages import. This isolates software problems from scheduling/GPU problems and avoids spending GPU allocation time on missing imports.

**Success check:** The checker prints versions for torch, torchvision, NumPy, SciPy, scikit-learn, pandas, Matplotlib, and joblib with no Python traceback. This confirms imports, not GPU execution.

```bash
cd /work/hdd/bgxw/$USER/Chae_Woojin_DSO607_HW6
module reset
module load pytorch-conda/2.8
export PYTHONNOUSERSITE=1
python3 check_environment.py
mkdir -p logs runs data
```

The documented module version is 2.8; confirm with `module spider` on your account. If the available version differs, update the module line in `run_delta.slurm` AND your setup commands before preparing the real run. An import check does not require a GPU; CUDA need not be available on the login node. Do not train on the shared login node.

### Your Step 3 output: what it means

Your reported imports succeeded with `torch 2.8.0+cu128`, `torchvision 0.23.0+cu128`, NumPy 2.2.6, SciPy 1.16.1, scikit-learn 1.7.2, pandas 2.3.2, Matplotlib 3.10.6, and joblib 1.5.2. The `+cu128` suffix identifies a CUDA 12.8 build; it does not itself prove that a GPU is accessible in the current shell. The checker in Step 3 imports libraries only. The Step 4 batch job runs the separate `--gpu` check and a real CUDA matrix calculation on an allocated GPU.

The `cray-mpich requires a network targeting module` warning occurred while the site's default Cray programming environment was being reset. Cray MPICH supplies MPI, used for communication among distributed processes; see [Delta's programming environment guide](https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/prog_env.html). Our Python code does not initialize MPI or distributed PyTorch. Given that all imports then succeeded, this warning does not invalidate Step 3. Proceed to the smoke job rather than changing the working module stack just to remove a warning. If the batch job aborts during module loading, or its GPU check fails, inspect the actual batch `.err` and `.out` files before proceeding to real data.

The module banner's `conda activate base` suggestion is optional for this session: your imports already used the desired PyTorch stack. You do not need to create the overlay `.venv` described below unless an import is missing. Loading a module adjusts environment variables and search paths for that terminal; it does not install packages on your Mac. `module reset` restores site defaults; `module load pytorch-conda/2.8` selects the site PyTorch environment. `PYTHONNOUSERSITE=1` prevents unrelated user-installed Python packages from shadowing it. `mkdir -p logs runs data` creates the directories used by later steps, and is safe to repeat.

If imports fail because supplementary packages are absent, make an overlay environment once, without changing the shared module:

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements-extra.txt
.venv/bin/python check_environment.py
```

If you created this overlay, run `source .venv/bin/activate` before subsequent manual Python commands in this Delta session; then `python3` uses the overlay. The batch script detects `.venv/bin/python` automatically. In a new login session, return to the remote project, load the same PyTorch module, and activate the overlay again. Do not copy a Mac virtual environment to Delta; Linux requires different binaries. No CUDA toolkit installation is needed with the site PyTorch module. Record actual loaded versions; the requirements file supplements the site stack rather than fully locking it.

## 4. Run a short GPU integration test (Delta Terminal)

**Purpose:** Exercise Slurm submission, CUDA computation, all four classifiers, and result saving using small synthetic data. Two repetitions and two CNN epochs keep this a setup test rather than a scientific experiment.

**Success check:** The batch job ends COMPLETED with exit code 0, prints the GPU matrix-test success and all-repetitions completion messages, and creates complete repetition folders. Do not interpret synthetic validation errors as CIFAR performance.

Create a tiny config with R=2 and EPOCHS=2, and synthetic images (no homework scores):

```bash
python3 - <<'PY'
import json
from pathlib import Path
c = json.loads(Path('config.json').read_text())
c.update(R=2, EPOCHS=2)
Path('smoke_config.json').write_text(json.dumps(c, indent=2))
PY
python3 experiment.py prepare --synthetic --config smoke_config.json --run runs/smoke
sbatch --account=bgxw-delta-gpu --time=00:15:00 run_delta.slurm runs/smoke
```

Slurm prints a job ID. Save it. Files under `logs/` appear when the job starts. Check progress:

```bash
squeue -u "$USER"
tail -f logs/hw6-cifar-JOBID.out
```

Replace JOBID with the printed number. Ctrl-C stops `tail`, not the job. `PD` means pending (queued); `R` means running. Read stderr too. To cancel:

```bash
scancel JOBID
```

For final status, including after the job leaves the queue:

```bash
sacct -j JOBID --format=JobID,JobName,State,Elapsed,ExitCode,MaxRSS
```

Success includes the GPU matrix test, all four methods for both repetitions, `All repetitions finished`, and COMPLETED / exit 0. Synthetic smoke performance has no substantive meaning. `finalize` is unnecessary for this test.

## 5. Prepare CIFAR and pilot runtime using development data only (Delta Terminal)

**Purpose:** Run two repetitions on the actual CIFAR data with the intended epoch budget. Use validation diagnostics and elapsed times to identify convergence problems and choose a realistic wall-time request before committing to the full run.

**Success check:** Both pilot repetitions finish, fitting diagnostics are acceptable, and you have a wall-time estimate. Leave test scores uncomputed; development-only decisions can still be made before the final configuration is frozen.

Download and split without holding an idle GPU:

```bash
python3 - <<'PY'
import json
from pathlib import Path
c = json.loads(Path('config.json').read_text())
c['R'] = 2
Path('pilot_config.json').write_text(json.dumps(c, indent=2))
PY
python3 experiment.py prepare --config pilot_config.json --data data --run runs/pilot
sbatch --account=bgxw-delta-gpu --time=02:00:00 run_delta.slurm runs/pilot
```

This keeps the full data and the production EPOCHS. Do NOT finalize the pilot or inspect test outcomes. Inspect validation errors, convergence warnings, stopped epochs, and timing only. Fit time may be dominated by CPU Lasso/KNN/RF even while you hold one GPU. A small CNN is a practical choice for 200 candidate fits; there is no promised runtime or accuracy. Estimate full wall time from the two `total_seconds` fields in `runs/pilot/repetitions/*/complete.json`: roughly 100 times their mean, plus margin. Early stopping makes this approximate. Adjust the sbatch wall time within the partition limit (currently 48h). If needed, resume completed repetitions in a subsequent job.

If the data URL fails on Delta, run preparation on another machine with the same dependencies and transfer the entire prepared run; do not manually redraw the test split. Confirm that only official CIFAR files are used. torchvision verifies its downloaded archive checksum. Preparing on Delta with the site environment is the preferred path.

Before the final run, make any development-informed choices to EPOCHS or candidate grids. Then freeze them for all 100 repetitions. If Lasso reaches its iteration limit, increase its budget or tolerance on development data, document the change, and create a fresh run. Code or config edits after prepare are intentionally rejected.

## 6. Run the real experiment (Delta Terminal)

**Purpose:** Run the homework's 100 shared development splits with the frozen candidate grids and epoch budget. The job retains selected training-fitted models and their fixed-test probabilities; it does not score the test outcomes.

**Success check:** All 100 repetition folders contain complete.json and the final message says all repetitions finished. A submitted job ID alone is not completion. Test scores are still uncomputed.

```bash
python3 experiment.py prepare --config config.json --data data --run runs/full
sbatch --account=bgxw-delta-gpu run_delta.slurm runs/full
```

`config.json` defaults to R=100 and EPOCHS=20. The batch script defaults to 12h; use `sbatch --time=24:00:00 ...` if pilot timing warrants it. Keep the dataset, source code, software versions, and run config unchanged during the experiment. Only run ONE training process per run directory.

If a wall-time limit interrupts a job, wait until it has ended, then submit the SAME command. Fully finished repetitions are skipped; an interrupted repetition is rerun from the start with the same seed. A partial repetition never counts as complete. Exceptions append to `failures.log` and terminate the job; no failed repetitions are silently omitted. Preserve that log and explain any resolutions. A new scientific configuration requires a new run, not deletion of inconvenient results.

## 7. Finalize once, then open results (Delta Terminal)

**Purpose:** Evaluate the selected models' saved test predictions after every repetition is finished. This produces the requested error summaries, paired comparisons, baseline, and figures without further model fitting.

**Success check:** FINALIZED.json, the 400-row repetition_results.csv, summary tables, and two plots exist. Do not use these test results to revise the experiment and claim a fresh evaluation on the same test set.

Only after all 100 repetitions have finished and settings are irrevocably settled:

```bash
python3 experiment.py finalize --run runs/full
```

This is a lightweight summary operation; it does not fit models. Real-data finalization requires R=100, so the R=2 pilot cannot accidentally reveal test scores. It writes a FINALIZED marker that blocks further training. Re-running finalize only regenerates summaries of the same frozen predictions.

Key outputs:

| File | Purpose |
| --- | --- |
| `indices.npz`, `split_counts.csv` | Source indices, local development split indices, counts, split seeds |
| `data_description.json`, `manifest.json`, `config.json` | Source, exclusions, integrity hashes, initial environment, frozen settings |
| `repetitions/001/candidates.json` | All candidates' validation errors, warnings, times, CNN epochs |
| `repetitions/001/*.joblib`, `DNN.pt`, `normalization.npz` | Selected training-fitted models and normalization |
| `repetitions/*/complete.json` | Actual training environment and total time per repetition |
| `repetition_results.csv` | Exactly 400 rows for R=100: four methods x 100 repetitions |
| `summary.csv`, `errors_with_baseline.csv` | Mean error, sample SD, majority baseline |
| `paired_differences.csv`, `paired_summary.csv` | Paired error differences relative to Lasso |
| `all_candidates.csv`, `timing_summary.csv`, `repetition_timings.csv` | Full tuning logs and computation summaries |
| `test_error_boxplot.png`, `DNN_loss_rep1.png` | Labeled figures for the report |

Every error is a proportion, not a percentage. The summary includes a fifth majority-baseline row; the requested four-method table is its first four rows. Model timings include candidate fitting and validation prediction; whole-repetition timings also include normalization, test prediction, and persistence.

## 8. Retrieve results and complete the notebook

**Purpose:** Bring the frozen experiment artifacts back to your Mac, display results in the notebook, and write the interpretation and LaTeX report. Downloading is a copy operation; it does not rerun training.

**Success check:** A timestamped local download contains runs/full, logs, matching code, and the notebook. The notebook displays saved results with RUN_EXPERIMENT=False and FINALIZE=False, ready for your interpretations.

Run the following in **Mac Terminal** after finalization. The remote paths are already filled in. Each download gets a timestamped folder under your homework project, preserving previous downloads:

```bash
cd "/Users/wchae/Courseworks/26 Fall/DSO607/Solution/Chae_Woojin_DSO607_HW6"
DOWNLOAD_DIR="$PWD/delta_download/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$DOWNLOAD_DIR/runs"
scp -o PreferredAuthentications=keyboard-interactive,password -r \
  wchae@login.delta.ncsa.illinois.edu:/work/hdd/bgxw/wchae/Chae_Woojin_DSO607_HW6/runs/full \
  "$DOWNLOAD_DIR/runs/"
scp -o PreferredAuthentications=keyboard-interactive,password -r \
  wchae@login.delta.ncsa.illinois.edu:/work/hdd/bgxw/wchae/Chae_Woojin_DSO607_HW6/logs \
  "$DOWNLOAD_DIR/"
scp -o PreferredAuthentications=keyboard-interactive,password \
  'wchae@login.delta.ncsa.illinois.edu:/work/hdd/bgxw/wchae/Chae_Woojin_DSO607_HW6/*.py' \
  'wchae@login.delta.ncsa.illinois.edu:/work/hdd/bgxw/wchae/Chae_Woojin_DSO607_HW6/*.json' \
  'wchae@login.delta.ncsa.illinois.edu:/work/hdd/bgxw/wchae/Chae_Woojin_DSO607_HW6/*.ipynb' \
  'wchae@login.delta.ncsa.illinois.edu:/work/hdd/bgxw/wchae/Chae_Woojin_DSO607_HW6/README.md' \
  'wchae@login.delta.ncsa.illinois.edu:/work/hdd/bgxw/wchae/Chae_Woojin_DSO607_HW6/requirements-extra.txt' \
  'wchae@login.delta.ncsa.illinois.edu:/work/hdd/bgxw/wchae/Chae_Woojin_DSO607_HW6/run_delta.slurm' \
  "$DOWNLOAD_DIR/"
open "$DOWNLOAD_DIR"
```

The quotes around remote wildcards prevent your Mac's zsh from expanding them locally. The transferred `experiment.py` must match the embedded notebook implementation and frozen source hash. This downloads results/models and logs without copying Delta's Linux virtual environment or the CIFAR download cache. Keep this Mac Terminal tab open if you want to reuse `$DOWNLOAD_DIR` in the next commands; shell variables belong to their current tab.

### Open the notebook locally on your Apple Silicon Mac (optional)

**Purpose:** Install a local notebook viewer/kernel and the imported scientific libraries so you can render saved tables and plots. This optional environment is separate from the Delta GPU environment. No local CUDA installation is required.

You can inspect CSVs and PNGs immediately in Finder. To run the notebook's display-only mode, create a separate local environment using your existing course Python:

```bash
cd "/Users/wchae/Courseworks/26 Fall/DSO607/Solution/Chae_Woojin_DSO607_HW6"
"/Users/wchae/Courseworks/26 Fall/DSO607/.venv/bin/python" -m venv .venv_mac
.venv_mac/bin/python -m pip install --upgrade pip
.venv_mac/bin/python -m pip install 'torch==2.8.0' 'torchvision==0.23.0' \
  -r requirements-extra.txt 'notebook>=7,<8'
.venv_mac/bin/python -m notebook --notebook-dir="$DOWNLOAD_DIR"
```

These commands install macOS packages into `.venv_mac`, leaving your course environment available for other homework. They do not install an NVIDIA CUDA toolkit on your Mac. Open `Chae_Woojin_DSO607_HW6.ipynb` from that downloaded folder, with `RUN_EXPERIMENT=False` and `FINALIZE=False`. Its relative `runs/full` path then finds the downloaded results. The notebook server stays in the foreground; stop it with Control-C when finished. Do not enable `DEVICE="cuda"` training locally: this project's CUDA path targets Delta's NVIDIA GPU. Local CPU training is possible with `DEVICE="cpu"`, but the full 100-repetition experiment is intended for Delta.

Back up run artifacts; shared work/project storage is not archival backup. Keep all prepared data, indices, configs, logs, predictions, and selected models. The notebook lives beside `experiment.py` and defaults to `runs/full`. It includes the full implementation as an embedded code cell, not just a shell call. Its default mode displays previously finalized results without retraining or opening sealed labels. It displays the tables (including all 400 rows), both figures, settings, epochs, and environments. Fill in the interpretation cells based on actual results.

For a full rebuild, use the notebook inside a Delta GPU notebook allocation, set `RUN_EXPERIMENT=True`, `FINALIZE=True`, and choose a fresh RUN_DIR. Top-level R and EPOCHS control its embedded config. A restart-and-run reuses a prepared frozen run and skips completed repetitions. A finalized run is display-only unless you choose a new directory. The notebook does not need a GPU to display exported results; it does need the imported scientific packages, including torch, installed locally. To use the Delta site notebook environment, follow the official Open OnDemand Jupyter guidance linked from the software page below; request a GPU allocation for rebuilding.

Your submission filenames are `Chae_Woojin_DSO607_HW6.pdf` and `Chae_Woojin_DSO607_HW6.ipynb`; produce the PDF using LaTeX. This bundle intentionally does not generate empirical interpretations or a submission PDF before results exist. Restart and run all cells, save visible outputs, and add the syllabus-compliant AI disclosure at the end of BOTH submitted files. The exact disclosure wording cannot be established from the homework alone.

## Interpretation prompts

Discuss average error AND split/fitting variability, training time, and interpretability. Lasso gives sparse pixel coefficients (correlations complicate interpretation); KNN retrieves examples but pixel distance can be poor semantically; RF importance is not causal; CNN uses spatial structure but is harder to explain. Balanced labels make the majority baseline close to 0.5; report actual frequencies and errors. Paired differences exploit the shared split, but reused test observations mean the repetitions are not independent test samples. Across-split SD quantifies variability for these fixed data. Do not use SD/sqrt(100) alone as a population-error confidence interval.

Two potential future improvements: training-only random flips/crops for the CNN, and PCA before KNN, fitted on each training subset only. Compare them using development-only resampling with fixed validation criteria; do not use current test errors to pick settings or claim a fresh unbiased test assessment. Once the final test has been examined, a confirmatory improved-method comparison needs a new untouched evaluation set.

## Common problems

Mac reminders: Control-C interrupts a foreground command; Command-T opens another local Terminal tab. Once `sbatch` accepts a job, the scheduler runs it independently of your laptop. You can disconnect with `exit` or close the SSH tab; check it later with `squeue`/`sacct`. Training through a direct interactive `srun` session has different session-lifetime behavior. Your Mac can sleep during a submitted batch job; reconnect afterward to monitor it.

| Symptom | Next step |
| --- | --- |
| `Permission denied` before Duo | Verify your NCSA username/password, then retry the SSH command in Step 1 |
| `zsh: command not found: sbatch` or `module` | You are probably in Mac Terminal; SSH into Delta first |
| `zsh: no matches found` during scp | Quote the remote path containing `*`, as in Step 8 |
| Invalid account/account-partition combination | Use `accounts` and select the correct GPU charge account; account access may need PI/admin help |
| Cray MPICH network warning during `module reset`, then imports succeed | Step 3 passed; validate the GPU with the Step 4 smoke job. See the Step 3 explanation above |
| `module load` fails | Inspect `module spider pytorch-conda`; keep setup and batch script versions identical |
| No module named sklearn/torchvision/etc. | Run the import check and use the overlay venv; don't mix unrelated conda environments |
| `CUDA unavailable` | Check allocation and node: login nodes have no GPU; CPU-only PyTorch also causes this |
| Missing logs | Create `logs/` before sbatch; pending jobs haven't started writing logs yet |
| TIMEOUT | Re-submit unchanged run; completed repetitions are retained |
| OUT_OF_MEMORY | Increase Slurm --mem for CPU memory; if CUDA OOM, adjust batch size on development data in a fresh run |
| Frozen input/code changed | Restore originals or prepare a new run; don't bypass the guard after viewing test results |

## Sources checked November 20, 2026

- [CIFAR source and technical report](https://www.cs.toronto.edu/~kriz/cifar.html)
- [Delta login and password/Duo prompts](https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/login.html)
- [Delta software and PyTorch modules](https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/software.html)
- [Delta Cray programming environment and MPI](https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/prog_env.html)
- [Delta running jobs and partition limits](https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/running_jobs.html)
- [Delta storage and project directories](https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/data_mgmt.html)
- [scikit-learn logistic regression objective](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
- [PyTorch binary cross-entropy with logits](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)

Assignment requirements were read from the user-provided homework6.pdf; they specify the experiment protocol and submission format, not authorization to submit work or connect to your account.

## Validation performed before delivery

Locally tested on CPU with two repetitions of synthetic images and two CNN epochs per candidate: all four methods, shared stratified split checks, model/checkpoint saving, finite probabilities, checkpoint restoration to the recorded lowest validation loss, resuming without rewriting completed models, refusal to finalize incomplete runs, refusal to train finalized runs, and agreement of saved means/SDs with direct NumPy calculations. Executed notebook code in default mode and against finalized synthetic results; checked shell syntax. Synthetic results remain in the scratch workspace and are not included as homework results.

The full CIFAR download/training, CUDA execution, and Slurm execution have NOT been completed here. The Delta smoke test and real-data development pilot are necessary setup checks. Fixed seeds and deterministic PyTorch operations improve repeatability, but do not promise bitwise equality across different hardware or package versions. Freeze the actual environment throughout a final run.
