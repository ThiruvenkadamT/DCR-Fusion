# DCR-Fusion: dual-test conformal reliability weighting for multi-sensor fault diagnosis

This repository contains the code for all experiments on reliability-weighted fusion of heterogeneous
sensors (vibration, motor current, temperature) for machine fault diagnosis when individual sensors degrade.

For every sensor group and every test window, DCR-Fusion computes two label-free tests:

* **Cross-sensor test** – a masked predictor estimates the group embedding from subsets of the other groups and the
  operating condition; the score is the smallest standardised residual over the admissible subsets (those that leave
  out `q` partners).
* **Typicality test** – the Mahalanobis distance of the group embedding to the nearest class mean.

Both scores are turned into split-conformal p-values on clean calibration windows and merged by the Bonferroni rule,
`p = min(1, 2 min(p_cs, p_ty))`. The reliability weight is `w = min(1, p / a)`, so healthy groups keep full weight and
the false-alarm rate per group stays below `a` (0.05). The weighted evidence of the groups is fused in a Dirichlet
model, with an uncertainty-based rejection rule and a certificate for the down-weighted groups.

## Repository layout

```
dcrfusion/
  config.py            all settings used in the experiments
  data/kaist.py        KAIST loader, 0.1 s windows, time-block split
  data/paderborn.py    Paderborn loader (real damage, 4 operating conditions), measurement or bearing split
  degradations.py      8 sensor degradation operators
  model.py             group encoders, evidence heads, masked predictors, evidential loss, TMC rule
  reliability.py       cross-sensor and typicality tests, conformal p-values, weights
  training.py          stage 1 (encoders and heads) and stage 2 (predictors), cross-sensor scores
  evaluation.py        one run: all methods on clean data, one and two degraded groups
run_experiments.py     runs all seeds for one dataset and writes the result tables
summarize_results.py   mean ± sd over runs and paired Wilcoxon tests
```

## Installation

Python 3.10 or newer is required.

```
pip install -r requirements.txt
```

A GPU is recommended. On one NVIDIA T4, one run takes about 21–23 min and five runs about 1.8 h per dataset.

## Data

* **KAIST** rotating machine dataset (vibration, temperature and motor current under 0, 2 and 4 Nm):
  Mendeley Data, https://doi.org/10.17632/ztmf3m7h5x.6. The vibration `.mat` files and the current/temperature
  `.tdms` files are read; the folder is searched recursively and `.zip` archives are extracted automatically.
* **Paderborn** bearing dataset (KAt-DataCenter, Paderborn University). The real-damage bearings K001–K005,
  KA04, KA15, KA16, KA22, KA30, KI04, KI14, KI16, KI18 and KI21 are used. Files named like
  `N15_M07_F10_KA04_1.mat` are searched recursively; `.rar` archives are extracted if `unrar`, `7z` or `bsdtar`
  is installed.

## Running the experiments

```
python run_experiments.py --dataset kaist     --data-root /path/to/kaist     --work-dir ./work
python run_experiments.py --dataset paderborn --data-root /path/to/paderborn --work-dir ./work
python run_experiments.py --dataset paderborn --data-root /path/to/paderborn --work-dir ./work --split bearing

python summarize_results.py --results ./work/results_kaist
python summarize_results.py --results ./work/results_paderborn
```

`--quick` runs a short smoke test. Settings can be changed with `--config '{"SEEDS": [0, 1]}'`.
The windows are cached in the work folder after the first run, and finished runs are skipped when a job is restarted.

## Protocol

| Item | KAIST | Paderborn |
|---|---|---|
| Classes | normal, inner race, outer race, misalignment, unbalance | healthy, outer race, inner race |
| Operating conditions | 0, 2, 4 Nm | N15_M07_F10, N09_M07_F10, N15_M01_F10, N15_M07_F04 |
| Sensor groups | vibration A (2 ch), vibration B (2 ch), current (1 ch), temperature (2 ch) | vibration, current 1, current 2, temperature |
| Split per run | 20 time blocks per recording, 14/3/3 | 20 measurements per bearing and condition, 14/3/3 |
| Runs | seeds 0–4 | seeds 0–4 |

Degradations are applied to the raw window of one group (all groups in turn) or two groups (all pairs):
noise (N), gain drift (G), dropout (Dr), failure (F), bias drift (B), clipping (Cl), stuck-at (St) and bandwidth
loss (Bw), at severities 0.2–1.0 (two groups: 0.6 and 1.0). N, G, Dr and F are also used to train the augmentation
baseline; B, Cl, St and Bw are never used in training. The reliability weights are calibrated on clean data only.

Compared methods: DCR-Fusion (q = 1, 2), the cross-sensor test alone with exponential weights (CS-Fusion, q = 1, 2),
P-MIF-style typicality p-value weights, equal late fusion, the TMC combination rule, EDL-uncertainty weights,
QMF-style energy weights, missing-modality training, augmentation training, the best single group and a drop-k oracle,
plus five ablations.

## Output

`run_experiments.py` writes one pickle per run to `results_<dataset>/runs/` and these tables:

| File | Content |
|---|---|
| `<dataset>_clean.csv` | macro-F1 and accuracy on clean test windows |
| `<dataset>_grid.csv` | one degraded group: run × group × type × severity × method |
| `<dataset>_grid_cond.csv` | the same per operating condition |
| `<dataset>_two.csv` | two degraded groups: run × pair × type × severity × method |
| `<dataset>_det.csv` | detection AUROC of each reliability score and mean weights |
| `<dataset>_far.csv` | false-alarm rate on clean windows |
| `<dataset>_cert.csv` | certificate and rejection statistics |
| `<dataset>_meta.json` | configuration, thresholds and confusion matrices |

## License

MIT License (see `LICENSE`).
