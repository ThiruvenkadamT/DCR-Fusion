"""Paderborn bearing dataset (real damage, three classes, four operating conditions).

Groups: vibration_1, phase_current_1, phase_current_2 (64 kHz resampled to 25.6 kHz) and
temp_2_bearing_module (5 s history). The condition vector holds speed, load torque and radial force.
"""
import os
import re
import shutil
import subprocess
import time
import zipfile
import zlib

import numpy as np
import pandas as pd
from scipy.io import loadmat

from .common import Data, to_fs, save_cache, load_cache

PU_RE = re.compile(r"(N\d{2}_M\d{2}_F\d{2})_(K[A-Z]?\d{2,3})_(\d+)\.mat$", re.I)
SIG = {"vib": "vibration_1", "cur1": "phase_current_1", "cur2": "phase_current_2", "temp": "temp_2_bearing_module"}
FS_RAW = 64000.0


def extract_archives(root, wanted, extract_dir):
    """The data are distributed as one .rar archive per bearing; extract .zip and (with unrar, 7z or bsdtar) .rar."""
    tools = [t for t in ["unrar", "7z", "bsdtar"] if shutil.which(t)]
    for dp, _, fs in os.walk(root):
        for f in fs:
            stem, ext = os.path.splitext(f)
            if stem.upper() not in wanted or ext.lower() not in (".zip", ".rar"):
                continue
            tgt = os.path.join(extract_dir, stem)
            if os.path.isdir(tgt):
                continue
            os.makedirs(tgt, exist_ok=True)
            p = os.path.join(dp, f)
            if ext.lower() == ".zip":
                with zipfile.ZipFile(p) as z:
                    z.extractall(tgt)
            elif tools:
                cmd = {"unrar": ["unrar", "x", "-o+", "-inul", p, tgt + "/"], "7z": ["7z", "x", "-y", f"-o{tgt}", p],
                       "bsdtar": ["bsdtar", "-xf", p, "-C", tgt]}[tools[0]]
                subprocess.run(cmd, check=True)
            else:
                raise RuntimeError("Found .rar archives but no unrar, 7z or bsdtar; please extract the archives first.")


def find_files(roots, wanted, conditions):
    out = {}
    for base in roots:
        if not os.path.isdir(base):
            continue
        for dp, _, fs in os.walk(base):
            for f in fs:
                m = PU_RE.search(f)
                if m and m.group(2).upper() in wanted and m.group(1).upper() in conditions:
                    out[(m.group(2).upper(), m.group(1).upper(), int(m.group(3)))] = os.path.join(dp, f)
    return out


def read_measurement(path):
    d = loadmat(path)
    key = [k for k in d if not k.startswith("__")][0]
    S = d[key][0, 0]
    Yf = S["Y"] if "Y" in (S.dtype.names or ()) else S[2]
    out = {}
    for el in np.ravel(Yf):
        names = el.dtype.names or ()
        nm = str(np.ravel(el["Name"] if "Name" in names else el[0])[0])
        out[nm] = np.asarray(el["Data"] if "Data" in names else el[2], dtype=np.float64).ravel()
    return out


def load(cfg):
    FS = cfg["FS"]
    L = int(round(FS * cfg["WIN_SEC"]))
    wanted = {b: c for c, bs in cfg["PU_CLASSES"].items() for b in bs}
    classes = list(cfg["PU_CLASSES"])
    groups = [dict(name="vib", ch=1), dict(name="cur1", ch=1), dict(name="cur2", ch=1), dict(name="temp", ch=1)]
    gnames = [g["name"] for g in groups]
    cache = os.path.join(cfg["WORK"], f"cache_paderborn_L{L}_m{cfg['PU_MEAS_PER_COND']}")
    cached = load_cache(cache, gnames)
    if cached is not None:
        print("Loading cached windows from", cache)
        X, META, info = cached
    else:
        extract_dir = os.path.join(cfg["WORK"], "extracted")
        files = find_files([cfg["DATA_ROOT"], extract_dir], wanted, cfg["PU_CONDITIONS"])
        if len(files) < 10:
            extract_archives(cfg["DATA_ROOT"], wanted, extract_dir)
            files = find_files([cfg["DATA_ROOT"], extract_dir], wanted, cfg["PU_CONDITIONS"])
        files = {k: v for k, v in files.items() if k[2] <= cfg["PU_MEAS_PER_COND"]}
        inv = pd.Series([k[0] for k in files]).value_counts().reindex(list(wanted), fill_value=0)
        print(f"Paderborn files found: {len(files)}")
        print(inv.to_frame("files").T.to_string())
        if (inv == 0).any():
            raise RuntimeError("Missing bearings: " + ", ".join(inv[inv == 0].index))
        bufs = {g: [] for g in gnames}
        rows, bad, t0 = [], [], time.time()
        for i, (key, path) in enumerate(sorted(files.items())):
            bearing, cond, meas = key
            try:
                sig = read_measurement(path)
            except Exception as e:
                bad.append((os.path.basename(path), str(e)[:80]))
                continue
            if any(SIG[g] not in sig for g in ["vib", "cur1", "cur2"]):
                bad.append((os.path.basename(path), "missing signals"))
                continue
            dur = len(sig[SIG["vib"]]) / FS_RAW
            hs = {g: to_fs(sig[SIG[g]][:, None], FS_RAW, FS)[:, 0] for g in ["vib", "cur1", "cur2"]}
            tem = sig.get(SIG["temp"], np.zeros(2))
            tt = np.linspace(0, dur, len(tem)) if len(tem) > 1 else np.array([0.0, dur])
            tem = tem if len(tem) > 1 else np.repeat(tem[:1] if len(tem) else [0.0], 2)
            nwin = min(len(v) for v in hs.values()) // L
            for w in range(nwin):
                s0, s1 = w * L, (w + 1) * L
                for g in ["vib", "cur1", "cur2"]:
                    bufs[g].append(hs[g][s0:s1][None].astype(np.float16))
                tq = np.linspace(s1 / FS - cfg["TEMP_CONTEXT_SEC"], s1 / FS, cfg["TEMP_LEN"])
                bufs["temp"].append(np.interp(tq, tt, tem)[None].astype(np.float32))
                rows.append(dict(unit=bearing, cls=classes.index(wanted[bearing]), cond=cond, meas=meas, win=w))
            if i % 50 == 0:
                print(f"  [{i + 1}/{len(files)}] {bearing} {cond} #{meas}: {nwin} windows ({time.time() - t0:.0f} s)")
        if bad:
            print("Skipped files:", bad)
        X = {g: np.stack(bufs[g]) for g in gnames}
        META = pd.DataFrame(rows)
        info = dict(raw_fs=FS_RAW, resampled_fs=FS, skipped=len(bad))
        save_cache(cache, X, META, info)
    cv = META["cond"].str.extract(r"N(\d{2})_M(\d{2})_F(\d{2})").astype(float).values
    cond = np.stack([(cv[:, 0] - 9) / 6, (cv[:, 1] - 1) / 6, (cv[:, 2] - 4) / 6], 1).astype(np.float32)
    data = Data(X=X, Y=META["cls"].values.astype(np.int64), COND=cond, COND_LABEL=META["cond"].values,
                META=META, groups=groups, classes=classes, info=info)
    print("Windows per operating condition and class:\n", data.counts("cond"))
    return data


def make_runs(data, cfg):
    """split 'measurement': per seed, the measurements of every bearing and condition are shuffled and split
    training / calibration / test (14/3/3 of 20). split 'bearing': fold f uses bearing f of each class for test
    and bearing f+1 for calibration."""
    runs = []
    meta = data.META
    u = meta["unit"].values
    if cfg["PU_SPLIT"] == "bearing":
        for f in cfg["PU_FOLDS"]:
            te, ca = set(), set()
            for bs in cfg["PU_CLASSES"].values():
                te.add(bs[f % len(bs)])
                ca.add(bs[(f + 1) % len(bs)])
            runs.append(dict(run=f"fold{f}", seed=f, test_units=sorted(te), cal_units=sorted(ca),
                             tr=np.where(~np.isin(u, list(te | ca)))[0], cal=np.where(np.isin(u, list(ca)))[0],
                             test=np.where(np.isin(u, list(te)))[0]))
        return runs
    ntr, nca, nte = cfg["PU_MEAS_SPLIT"]
    tot = ntr + nca + nte
    for seed in cfg["SEEDS"]:
        split = np.empty(len(meta), dtype=object)
        for (b, c), g in meta.groupby(["unit", "cond"]):
            ms = np.sort(g["meas"].unique())
            n = len(ms)
            rng = np.random.RandomState(seed * 1000 + zlib.crc32(f"{b}_{c}".encode()) % 997)
            k_te = max(1, int(round(n * nte / tot)))
            k_ca = max(1, int(round(n * nca / tot)))
            perm = ms[rng.permutation(n)]
            lab = {m: ("test" if i < k_te else "cal" if i < k_te + k_ca else "tr") for i, m in enumerate(perm)}
            split[g.index.values] = g["meas"].map(lab).values
        runs.append(dict(run=f"seed{seed}", seed=seed, tr=np.where(split == "tr")[0], cal=np.where(split == "cal")[0],
                         test=np.where(split == "test")[0]))
    return runs
