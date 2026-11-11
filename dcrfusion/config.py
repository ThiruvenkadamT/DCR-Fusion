"""Experiment configuration. All values used in the reported experiments are the defaults below."""
import copy
import json

DEFAULTS = dict(
    # signal and windowing
    FS=25600, WIN_SEC=0.1, TEMP_CONTEXT_SEC=5.0, TEMP_LEN=64,
    # KAIST: time-block split of every recording
    N_BLOCKS=20, BLOCK_SPLIT=(14, 3, 3), CURRENT_PHASES=1,
    # Paderborn: real-damage bearings, four operating conditions
    PU_CLASSES={"Healthy": ["K001", "K002", "K003", "K004", "K005"],
                "Outer race": ["KA04", "KA15", "KA16", "KA22", "KA30"],
                "Inner race": ["KI04", "KI14", "KI16", "KI18", "KI21"]},
    PU_CONDITIONS=["N15_M07_F10", "N09_M07_F10", "N15_M01_F10", "N15_M07_F04"],
    PU_MEAS_PER_COND=20, PU_SPLIT="measurement", PU_MEAS_SPLIT=(14, 3, 3), PU_FOLDS=[0, 1, 2, 3, 4],
    # repeated runs
    SEEDS=[0, 1, 2, 3, 4],
    # model
    D=64, E_MAX=20.0, SIGMA2_MIN=1e-3, PRED_HIDDEN=256,
    # reliability
    ALPHA=0.05, W_MIN=0.1,
    # training
    EPOCHS_1=30, EPOCHS_2=40, BATCH=128, BATCH_2=256, LR=1e-3, WD=1e-4, LAMBDA_G=1.0,
    MODDROP_P=0.25, TRAINED_BASELINES=True,
    # evaluation
    EVAL_MAX_WINDOWS=3000,
    SEVERITIES=[0.2, 0.4, 0.6, 0.8, 1.0],
    DEG_TYPES=["N", "G", "Dr", "F", "B", "Cl", "St", "Bw"],
    AUG_TYPES=["N", "G", "Dr", "F"],
    HELD_OUT_TYPES=["B", "Cl", "St", "Bw"],
    TWO_GROUP_SEVS=[0.6, 1.0],
)

QUICK = dict(EPOCHS_1=2, EPOCHS_2=2, EVAL_MAX_WINDOWS=150, SEVERITIES=[0.4, 1.0], TWO_GROUP_SEVS=[1.0],
             SEEDS=[0, 1], PU_FOLDS=[0, 1], DEG_TYPES=["N", "F", "B", "Bw"], PU_MEAS_PER_COND=3)


def make_config(dataset, data_root, work_dir, quick=False, overrides=None):
    cfg = copy.deepcopy(DEFAULTS)
    cfg.update(DATASET=dataset, DATA_ROOT=data_root, WORK=work_dir)
    if quick:
        cfg.update(QUICK)
    if overrides:
        cfg.update(json.loads(overrides) if isinstance(overrides, str) else overrides)
    return cfg
