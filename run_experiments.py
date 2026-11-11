"""Run all repeated experiments for one dataset.

Example:
    python run_experiments.py --dataset kaist --data-root /path/to/kaist --work-dir ./work
    python run_experiments.py --dataset paderborn --data-root /path/to/paderborn --work-dir ./work
Each finished run is saved, so an interrupted job resumes with the remaining runs.
"""
import argparse
import json
import os
import pickle
import time

import pandas as pd
import torch

from dcrfusion.config import make_config
from dcrfusion.data import kaist, paderborn
from dcrfusion.evaluation import evaluate_run

TABLES = ["clean", "far", "grid", "grid_cond", "det", "cert", "two"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["kaist", "paderborn"], required=True)
    ap.add_argument("--data-root", required=True, help="folder that contains the raw dataset files (searched recursively)")
    ap.add_argument("--work-dir", default="./work", help="folder for the window cache and the results")
    ap.add_argument("--split", choices=["measurement", "bearing"], default="measurement", help="Paderborn split (default: measurement)")
    ap.add_argument("--config", default=None, help="JSON string with configuration overrides")
    ap.add_argument("--quick", action="store_true", help="short smoke test with few epochs and cells")
    args = ap.parse_args()

    cfg = make_config(args.dataset, args.data_root, args.work_dir, args.quick, args.config)
    cfg["PU_SPLIT"] = args.split
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = args.dataset + ("_bybearing" if args.dataset == "paderborn" and args.split == "bearing" else "")
    out = os.path.join(args.work_dir, f"results_{tag}")
    os.makedirs(os.path.join(out, "runs"), exist_ok=True)
    print(f"Dataset: {args.dataset} | device: {device} | torch {torch.__version__}")

    loader = kaist if args.dataset == "kaist" else paderborn
    data = loader.load(cfg)
    runs = loader.make_runs(data, cfg)
    print(f"{len(runs)} runs:", [r["run"] for r in runs])
    for r in runs:
        print(f"  {r['run']}: training {len(r['tr'])}, calibration {len(r['cal'])}, test {len(r['test'])} windows")

    t_all = time.time()
    for r in runs:
        fn = os.path.join(out, "runs", f"{r['run']}.pkl")
        if os.path.exists(fn):
            print("already finished:", r["run"])
            continue
        print(f"\n=== run {r['run']} ===")
        res, meta = evaluate_run(data, cfg, r, device)
        with open(fn, "wb") as f:
            pickle.dump((res, meta), f)
        print(f"=== run {r['run']} finished; elapsed {(time.time() - t_all) / 60:.1f} min ===")

    tables, metas = {k: [] for k in TABLES}, []
    for r in runs:
        with open(os.path.join(out, "runs", f"{r['run']}.pkl"), "rb") as f:
            res, meta = pickle.load(f)
        for k in TABLES:
            tables[k].append(res[k])
        metas.append(meta)
    for k in TABLES:
        pd.concat(tables[k], ignore_index=True).to_csv(os.path.join(out, f"{tag}_{k}.csv"), index=False)
    json.dump(dict(cfg=cfg, groups=data.gnames, classes=data.classes, info=data.info, runs=metas),
              open(os.path.join(out, f"{tag}_meta.json"), "w"), indent=1, default=str)
    print("Results written to", out)


if __name__ == "__main__":
    main()
