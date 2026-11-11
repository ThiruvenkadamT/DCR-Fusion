"""KAIST rotating machine dataset (vibration .mat files and current/temperature .tdms files).

Groups: vibration A (2 channels), vibration B (2 channels), temperature (2 channels, 5 s history),
motor current (phase ai0). Each recording is cut into 0.1 s windows and divided into consecutive time
blocks; the first window of every block is dropped as a guard between blocks.
"""
import os
import re
import time

import numpy as np
import pandas as pd
from scipy.io import loadmat

from .common import Data, collect, to_fs, save_cache, load_cache

NAME_RE = re.compile(r"(\d+)\s*Nm[_\- ]*([A-Za-z]+)[_\- ]*([0-9A-Za-z.]*)", re.I)
CLASS_OF = {"normal": 0, "bpfi": 1, "bpfo": 2, "misalign": 3, "unbalance": 4}
CLASSES = ["Normal", "Inner race (BPFI)", "Outer race (BPFO)", "Misalignment", "Unbalance"]


def parse_name(path):
    m = NAME_RE.search(os.path.splitext(os.path.basename(path))[0])
    if not m:
        return None
    load, cond, sev = int(m.group(1)), m.group(2).lower(), m.group(3)
    for k, v in CLASS_OF.items():
        if cond.startswith(k[:5]):
            return load, v, re.sub(r"[^0-9]", "", sev or "").lstrip("0") or "0"
    return None


def read_mat_signal(path):
    S = loadmat(path)["Signal"][0, 0]
    v = np.asarray(S["y_values"][0, 0]["values"], dtype=np.float64)
    return v, 1.0 / float(np.asarray(S["x_values"][0, 0]["increment"]).ravel()[0])


def read_tdms(path, n_phases):
    """Current = NI 'Mod2/ai0' (the phase present in every recording); temperature = the two 'Mod1' channels."""
    from nptdms import TdmsFile
    cur, tem, allch = [], [], []
    with TdmsFile.open(path) as tf:
        for g in tf.groups():
            for ch in g.channels():
                inc = ch.properties.get("wf_increment", None)
                data = np.asarray(ch[:], dtype=np.float64)
                allch.append((ch.path, len(data)))
                if len(data) == 0:
                    continue
                item = (ch.name, data, 1.0 / float(inc) if inc else None)
                tag = (ch.path + ch.name).lower()
                (cur if "mod2" in tag else tem if "mod1" in tag else cur).append(item)
    pref = [c for c in cur if c[0].lower().endswith("ai0")]
    cur = (pref or sorted(cur, key=lambda t: t[0]))[:n_phases]
    tem = sorted(sorted(tem, key=lambda t: -len(t[1]))[:2], key=lambda t: t[0])
    if len(cur) < n_phases or len(tem) < 2:
        raise RuntimeError(f"{os.path.basename(path)}: unexpected channels {allch}")
    n = min(len(c[1]) for c in cur)
    nt = min(len(t[1]) for t in tem)
    return np.stack([c[1][:n] for c in cur], 1), cur[0][2], np.stack([t[1][:nt] for t in tem], 1), tem[0][2]


def load(cfg):
    FS = cfg["FS"]
    L = int(round(FS * cfg["WIN_SEC"]))
    groups = [dict(name="vibA", ch=2), dict(name="vibB", ch=2), dict(name="temp", ch=2), dict(name="current", ch=cfg["CURRENT_PHASES"])]
    gnames = [g["name"] for g in groups]
    cache = os.path.join(cfg["WORK"], f"cache_kaist_L{L}")
    cached = load_cache(cache, gnames)
    if cached is not None:
        print("Loading cached windows from", cache)
        X, META, info = cached
    else:
        extract_dir = os.path.join(cfg["WORK"], "extracted")
        vib_files = {parse_name(p): p for p in collect(cfg["DATA_ROOT"], (".mat",), extract_dir) if parse_name(p)}
        ct_files = {parse_name(p): p for p in collect(cfg["DATA_ROOT"], (".tdms",), extract_dir) if parse_name(p)}
        keys = sorted(set(vib_files) & set(ct_files))
        print(f"vibration files {len(vib_files)}, current/temperature files {len(ct_files)}, recordings with both: {len(keys)}")
        if not keys:
            raise RuntimeError("No KAIST recordings found under " + cfg["DATA_ROOT"])
        bufs = {g: [] for g in gnames}
        rows, info, t0 = [], {}, time.time()
        for ri, key in enumerate(keys):
            load_nm, cls, sev = key
            vib, fv = read_mat_signal(vib_files[key])
            cur, fc, tem, ft = read_tdms(ct_files[key], cfg["CURRENT_PHASES"])
            fc = fc or fv * len(cur) / len(vib)
            ft = ft or fv * len(tem) / len(vib)
            vib, cur = to_fs(vib, fv, FS), to_fs(cur, fc, FS)
            info.update(vibration_fs=fv, current_fs=fc, temperature_fs=ft)
            nwin = min(len(vib), len(cur)) // L
            tt = np.arange(len(tem)) / ft
            for i in range(nwin):
                b = i * cfg["N_BLOCKS"] // nwin
                if i > 0 and (i - 1) * cfg["N_BLOCKS"] // nwin != b:
                    continue   # guard: drop the first window of every block
                s0, s1 = i * L, (i + 1) * L
                bufs["vibA"].append(vib[s0:s1, 0:2].T.astype(np.float16))
                bufs["vibB"].append(vib[s0:s1, 2:4].T.astype(np.float16))
                bufs["current"].append(cur[s0:s1].T.astype(np.float16))
                tq = np.linspace(s1 / FS - cfg["TEMP_CONTEXT_SEC"], s1 / FS, cfg["TEMP_LEN"])
                bufs["temp"].append(np.stack([np.interp(tq, tt, tem[:, c]) for c in range(2)]).astype(np.float32))
                rows.append(dict(unit=ri, load=load_nm, cls=cls, sev=sev, win=i, block=b))
            print(f"  [{ri + 1}/{len(keys)}] {load_nm} Nm {CLASSES[cls]:<18} severity {sev:<6} windows {nwin}  ({time.time() - t0:.0f} s)")
        X = {g: np.stack(bufs[g]) for g in gnames}
        META = pd.DataFrame(rows)
        save_cache(cache, X, META, info)
    data = Data(X=X, Y=META["cls"].values.astype(np.int64),
                COND=(META["load"].values.astype(np.float32) / 4.0)[:, None],
                COND_LABEL=META["load"].astype(str).values + " Nm", META=META, groups=groups, classes=CLASSES, info=info)
    counts = data.counts("load")
    print("Windows per load and class:\n", counts)
    if (counts == 0).any().any():
        raise RuntimeError("Some load/class combinations have no windows.")
    return data


def make_runs(data, cfg):
    """One run per seed: a random assignment of each recording's time blocks to training / calibration / test."""
    runs = []
    meta = data.META
    for seed in cfg["SEEDS"]:
        split = np.empty(len(meta), dtype=object)
        for u in meta["unit"].unique():
            rng = np.random.RandomState(seed * 1000 + int(u))
            ntr, nca, nte = cfg["BLOCK_SPLIT"]
            bt = np.array(["tr"] * ntr + ["cal"] * nca + ["test"] * nte)[rng.permutation(cfg["N_BLOCKS"])]
            mm = (meta["unit"] == u).values
            split[mm] = bt[meta["block"].values[mm]]
        runs.append(dict(run=f"seed{seed}", seed=seed, tr=np.where(split == "tr")[0], cal=np.where(split == "cal")[0],
                         test=np.where(split == "test")[0]))
    return runs
