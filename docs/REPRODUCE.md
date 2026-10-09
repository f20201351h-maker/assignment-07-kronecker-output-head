# Reproducing the results

All commands run from the repository root. Section 1 needs only this repository. Sections 3 and 4 re-run
analyses from the saved per-position losses and need the token streams of section 2. Section 5 retrains models
and needs a GPU. Nothing beyond section 1 is needed to check that the reported tables match the saved results.

## 0. Environment

```bash
python -m pip install -r requirements.txt
```

| Where | Python | Key packages |
|---|---|---|
| Modal training image (`scripts/modal_app.py`) | 3.11 | `torch==2.10.0`, `numpy==2.2.6` |
| Local CPU analysis and checks | 3.14 | torch 2.14 (CPU), numpy 2.4, matplotlib 3.11, pytest 9 |

The CPU work was done on an 8-thread laptop with 8 GB of RAM. On Windows, run Modal commands from Git Bash with
`export MSYS_NO_PATHCONV=1 PYTHONIOENCODING=utf-8 PYTHONUTF8=1`, otherwise Git Bash rewrites volume paths.

## 1. Checks that need only the repository (CPU, seconds)

```bash
python -m pytest -q tests
```

```bash
python scripts/build_summary.py --check
```

```bash
python scripts/check_readme_numbers.py README.md
```

* `pytest`: 75 tests covering the codec against the published `kronecker-embeddings` package, head
  equivalences, metrics, the Phase 2 dropout mask and generator input, and protocol 08's bias-only difference and
  optimiser grouping. Seven of them skip until `data/gpt2/tokenizer.json` exists (section 2 downloads it).
* `build_summary.py --check` regenerates every table between `<!-- table:NAME -->` markers in `README.md`,
  `docs/EXPERIMENTS.md` and `docs/EXPRESSIVITY.md` from the saved results and exits 1 if any differs. Without
  `--check` it rewrites those tables, `results/summary/` and figures 9 and 10.
* `check_readme_numbers.py` checks that every number in a document rounds to a value in some JSON file under
  `results/` or `experiments/**/summary.json`, or is listed with a reason in `scripts/number_allowlist.json`
  (configuration constants, values quoted from other work). It is a presence check, not a claim-level check: it
  rules out invented or stale digits, not misattributed ones. It also accepts `docs/EXPERIMENTS.md` and
  `docs/EXPRESSIVITY.md`.

## 2. Data

Token streams are not distributed (about 0.5 GB). Rebuild them deterministically:

```bash
python scripts/prepare_data.py download --train-tokens 250000000
```

```bash
python scripts/prepare_data.py build
```

`download` streams FineWeb-Edu `sample-10BT` in published order and fetches the GPT-2 `tokenizer.json` from
Hugging Face; `build` creates the working vocabulary, merges, held-out merges and unigram counts. Compare the
resulting files with the SHA-256 values in `data/tokens/manifest.json` and `data/raw/download_meta.json`;
`data/tokens/vocab.json` is included.

## 3. Analyses of the trained models (CPU; need the token streams)

Each run directory `experiments/<set>/runs/<run-id>/` holds `run_record.json` (exact configuration, environment,
data hashes), `summary.json`, `metrics.jsonl`, dev evaluation curves (`evalcurve/`), `nll_dev.npy` / `nll_test.npy`
(per-position NLL on every evaluation window) and `analysis_{dev,test}/` (minting, ablations, prior drift).
Checkpoints are not included.

| Set | Runs | Protocol | Arrays included |
|---|---|---|---|
| `modal-m3` | 14 Phase 1 runs (7 arms × 2 seeds) | 03, 03b, 05 | yes |
| `modal-p2` | 8 Phase 2 runs | 06, 06a | yes |
| `modal-p3` | 4 confirmation runs | 07 | yes |
| `modal-p4` | 4 prior-matched Dense runs | 08 | yes |
| `kq5-r3-canonical-x`, `kq5-r3-canonical-g` | the 14 Phase 1 configurations on Kaggle T4 | 03, 05 | summaries only |
| `kq5-r1-smoke`, `kq5-r1b-kasg`, `kq5-r2-pilot` | smoke tests and 21M-token pilot | 01, 02 | summaries only |

Phase 1 (verdicts H1 to H4, tables, figures 1 to 7):

```bash
python scripts/final_analysis.py --runs experiments/modal-m3/runs --prefix m3 --split test --out results/final
python scripts/collect_diagnostics.py experiments/modal-m3/runs test results/final/diagnostics_m3_test.json
python scripts/write_results_tables.py results/final/analysis_m3_test.json
python scripts/sensitivity_minting.py
python scripts/make_figures.py
```

The notebook `notebooks/01_analysis_from_results.ipynb` runs the same steps
(`python scripts/build_notebooks.py --execute` rebuilds it).

Phase 2 (Q1, H5 to H8, figure 8; notebook `notebooks/02_phase2_from_results.ipynb`):

```bash
python scripts/final_analysis.py --runs experiments/modal-m3/runs experiments/modal-p2/runs --prefix m3 p2 --split test --out results/phase2
python scripts/final_analysis.py --runs experiments/modal-m3/runs experiments/modal-p2/runs --prefix m3 p2 --split dev --out results/phase2
python scripts/phase2_analysis.py
python scripts/phase2_figures.py
python scripts/write_phase2_derived.py
python scripts/code_rank.py
```

Protocols 07 and 08:

```bash
python scripts/final_analysis.py --runs experiments/modal-m3/runs experiments/modal-p2/runs experiments/modal-p3/runs --prefix m3 p2 p3 --split test --out results/phase3
python scripts/confirm_analysis.py
python scripts/final_analysis.py --runs experiments/modal-m3/runs experiments/modal-p2/runs experiments/modal-p3/runs experiments/modal-p4/runs --prefix m3 p2 p3 p4 --split test --out results/phase4
python scripts/prior_analysis.py
python scripts/recompute_diagnostics.py
```

`recompute_diagnostics.py` is an independent check: it recomputes the protocol 08 arm means directly from the
twelve `nll_test.npy` arrays, asserts that each matches its `summary.json`, and writes the frequency-bucket
decomposition to `results/phase4/independent_diagnostics.json`.

## 4. Expressivity analysis

| Step | Command | Needs | Hardware |
|---|---|---|---|
| Audit of the saved fits: array checks, exact codec mapping, rank, same-length census, head accounting, non-EOS tables | `python scripts/expressivity/audit_evidence.py` | token streams | CPU, a few minutes |
| Four-position float64 convergence check | `python scripts/expressivity/check_fits.py` | checkpoint `p4-densep-s1` | CPU, about 2 minutes |
| Model-free targets and census | `python scripts/expressivity/floor_static.py` | token streams (train) | CPU, about 2 hours |
| 1,024-context teacher fits | `python scripts/expressivity/floor_contextual.py --windows 8` | checkpoint `p4-densep-s1` | CPU, about 2 hours |
| Learned row sets | `python scripts/expressivity/floor_rows.py` | checkpoints `p4-densep-s1`, `p2-kasu64-s1`, `m3-kasu0-s1`, `m3-dense-s1` | CPU, hours |
| 4,096-context matrix (2 teachers × 2 splits) | `modal run scripts/expressivity/modal_floor_matrix.py::stability` | Modal; checkpoints in the training volume | L4 GPUs, about $2 |

The scripts write to `results/expressivity/`, overwriting the included files; compare with `git diff` afterwards
(`check_fits.py` also records wall-clock seconds). Checkpoint-dependent scripts expect
`checkpoints/<run-id>/model.pt`. Checkpoints (150 to 240 MB each) are not distributed. Retraining a run with the
commands in section 5 gives a close but not bit-identical checkpoint (the T4 replication of Phase 1 differed from
the original runs by up to 0.0047 test BPB per run, one run by 0.0111). With access to the training volume, fetch
the original checkpoints instead:

```bash
modal volume get kq5-data /runs/p4-densep-s1/model.pt checkpoints/p4-densep-s1/model.pt
```

When this repository was prepared, `audit_evidence.py` and `recompute_diagnostics.py` were re-run from the
included files and reproduced every stored value.

## 5. Training (GPU)

Each canonical run trains 100,007,936 tokens in about 8 to 10 minutes on one NVIDIA RTX PRO 6000. The token
streams must first be uploaded to the Modal volume `kq5-data`. The historical runs read them from the volume path
recorded in each `run_record.json` as `data_dir` (`/data/C:/Program Files/Git/tokens`, produced by Git Bash path
conversion during upload); `scripts/modal_app.py` uses the same path, and the loader verifies every file's
SHA-256 against the manifest.

```bash
modal run --detach scripts/modal_app.py::canonical_detached --gpu RTX-PRO-6000
modal run --detach scripts/modal_app.py::phase2_detached --gpu RTX-PRO-6000
modal run --detach scripts/modal_app.py::confirm_detached --gpu RTX-PRO-6000
modal run --detach scripts/modal_app.py::prior_detached --gpu RTX-PRO-6000
```

Each entry point trains, analyses dev and test, and writes `/export/<run-id>.tar.gz` (without the checkpoint) to
the volume; download it with `modal volume get` and untar into `experiments/<set>/runs/`. Configurations are
generated by `scripts/make_jobs.py` and recorded in every `run_record.json`.

The pilot and the T4 replication ran as Kaggle script kernels. `kaggle/<kernel>/kernel.py` is the exact script
that ran (the source package is embedded; `stage_record.json` holds its SHA-256), and `configs/jobs/*.json`
lists the runs. To stage a kernel under your own account:

```bash
KAGGLE_USERNAME=<your-kaggle-user> python scripts/stage_kernel.py configs/jobs/r2_pilot.json
```

Kaggle account names in the included run records were replaced by `KAGGLE_USER`.

## 6. Compute used

Modal credit for the training runs: Phase 1 about $11 plus about $1.5 of checks; Phase 2 about $10.3 including
a GPU probe; protocol 07 about $2.1; protocol 08 between $2.2 and $2.8; the 4,096-context fit matrix about $2.3.
Kaggle T4 quota covered the pilot and the replication.
