"""Evaluation of one run: clean data, one degraded group, two degraded groups, detection, certificate and rejection."""
import itertools
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

from .degradations import degrade, make_stats
from .model import tmc_combine
from .reliability import DualTest, TypicalityTest, calibrate_cs, cs_weights, p_weight
from .training import cs_scores, encode_idx, train_encoders, train_predictors

CS_VARIANTS = ["cs_q1", "cs_q2", "cs_nocond", "cs_mean"]
PER_COND_METHODS = ["DCR-Fusion (q=1)", "DCR-Fusion (q=2)", "CS-Fusion (q=1)", "CS-Fusion (q=2)", "Equal late fusion", "TMC rule",
                    "P-MIF reliability", "Missing-modality training", "Augmentation training", "Drop-k oracle"]


def fuse(E, W):
    return 1.0 + (W[..., None] * E).sum(1)


def margin(a):
    s = np.sort(a, 1)
    return s[:, -1] - s[:, -2]


def energy(H):
    mx = H.max(-1, keepdims=True)
    return -(mx[..., 0] + np.log(np.exp(H - mx).sum(-1)))


def certificate(alpha, E, w, e_max):
    """A flagged group (w < 1) cannot change the decision if margin > 2 delta / (A - delta)."""
    A = alpha.sum(1)
    mp = margin(alpha / A[:, None])
    worst = np.zeros(len(A))
    flagged = w < 1 - 1e-9
    for j in range(E.shape[1]):
        dev = w[:, j] * np.maximum(E[:, j], e_max - E[:, j]).sum(1)
        bound = np.where(A - dev > 0, 2 * dev / np.maximum(A - dev, 1e-9), np.inf)
        worst = np.where(flagged[:, j], np.maximum(worst, bound), worst)
    has = flagged.any(1)
    return has, has & (mp > worst)


@torch.no_grad()
def encode_group(net, m, raw, evid, K, device, bs=512):
    zs, hs = [], []
    for b in range(0, len(raw), bs):
        z = net.encode(m, torch.from_numpy(np.asarray(raw[b:b + bs], dtype=np.float32)).to(device))
        zs.append(z.cpu())
        hs.append(net.logits(m, z).cpu() if m in evid else torch.zeros(len(z), K))
    return torch.cat(zs), torch.cat(hs).numpy()


def evaluate_run(data, cfg, run, device):
    M, K, evid, gnames, Y = data.M, data.K, data.evid, data.gnames, data.Y
    EV, a_lvl, e_max = np.array(evid), cfg["ALPHA"], cfg["E_MAX"]

    def f1(y, p):
        return f1_score(y, p, average="macro", labels=list(range(K)))

    seed, tr, cal = run["seed"], run["tr"], run["cal"]
    te = np.sort(np.random.RandomState(seed + 99).choice(run["test"], min(cfg["EVAL_MAX_WINDOWS"], len(run["test"])), replace=False))
    y = Y[te]
    t0 = time.time()
    stats = make_stats(data, tr)
    nets = {"main": train_encoders(data, cfg, tr, stats, "main", seed, device)}
    if cfg["TRAINED_BASELINES"]:
        nets["moddrop"] = train_encoders(data, cfg, tr, stats, "moddrop", seed, device)
        nets["aug"] = train_encoders(data, cfg, tr, stats, "aug", seed, device)
    net = nets["main"]
    Z_tr, _, _ = encode_idx(net, data, tr, device)
    pred = train_predictors(data, cfg, Z_tr, data.COND[tr], True, seed, device)
    pred_nc = train_predictors(data, cfg, Z_tr, data.COND[tr], False, seed, device)
    Z_cal, E_cal, _ = encode_idx(net, data, cal, device)
    SC_cal = cs_scores(data, pred, pred_nc, Z_cal, data.COND[cal], device)
    rho, beta = calibrate_cs(SC_cal, a_lvl)
    typ = TypicalityTest(Z_tr, Y[tr], Z_cal, evid, K, cfg["D"])
    dual = DualTest(SC_cal, typ, evid)
    P_cal = dual.all(SC_cal, typ.dist(Z_cal))
    W_cal = {"CS-Fusion (q=1)": cs_weights(SC_cal["cs_q1"][0], "cs_q1", rho, beta)[:, EV],
             "CS-Fusion (q=2)": cs_weights(SC_cal["cs_q2"][0], "cs_q2", rho, beta)[:, EV],
             "DCR-Fusion (q=1)": p_weight(P_cal["dual_q1"], a_lvl), "DCR-Fusion (q=2)": p_weight(P_cal["dual_q2"], a_lvl)}
    u_th = {k: float(np.quantile(K / fuse(E_cal, w).sum(1), 0.95)) for k, w in W_cal.items()}
    best = int(np.argmax([accuracy_score(Y[cal], (1 + E_cal[:, j]).argmax(1)) for j in range(len(evid))]))
    print(f"    predictors and calibration done ({time.time() - t0:.0f} s); rho = {dict(zip(gnames, np.round(rho['cs_q1'], 2)))}")

    clean = {}
    for k, n_ in nets.items():
        zs, hs = zip(*[encode_group(n_, m, data.X[gnames[m]][te], evid, K, device) for m in range(M)])
        H = np.stack([hs[m] for m in evid], 1)
        clean[k] = dict(Z=torch.stack(zs, 1), H=H, E=e_max / (1 + np.exp(-H)))

    def outputs(corrupt):
        """corrupt: dict group -> degraded raw array. Returns predictions of every method and the reliability outputs."""
        O = {}
        for k in nets:
            Z, H = clean[k]["Z"].clone(), clean[k]["H"].copy()
            for m, raw in corrupt.items():
                z, h = encode_group(nets[k], m, raw, evid, K, device)
                Z[:, m] = z
                if m in evid:
                    H[:, evid.index(m)] = h
            O[k] = dict(Z=Z, H=H, E=e_max / (1 + np.exp(-H)))
        E, H, Z = O["main"]["E"], O["main"]["H"], O["main"]["Z"]
        SC = cs_scores(data, pred, pred_nc, Z, data.COND[te], device)
        W = {v: cs_weights(SC[v][0], v, rho, beta)[:, EV] for v in CS_VARIANTS}
        alpha = {"CS-Fusion (q=1)": fuse(E, W["cs_q1"]), "CS-Fusion (q=2)": fuse(E, W["cs_q2"]),
                 "CS-Fusion without condition input": fuse(E, W["cs_nocond"]), "CS-Fusion, mean over subsets": fuse(E, W["cs_mean"])}
        sm = np.exp(-SC["cs_q1"][0][:, EV])
        alpha["CS-Fusion, softmax weights"] = fuse(E, sm / sm.sum(1, keepdims=True) * len(EV))
        alpha["Equal late fusion"] = fuse(E, np.ones_like(W["cs_q1"]))
        alpha["EDL-uncertainty weights"] = fuse(E, 1 - K / (E.sum(-1) + K))
        qw = np.log1p(np.exp(-energy(H)))
        alpha["QMF-style energy weights"] = fuse(E, qw / (qw.sum(1, keepdims=True) + 1e-9) * len(EV))
        dist = typ.dist(Z)
        PV = dual.all(SC, dist)
        alpha["P-MIF reliability"] = fuse(E, PV["typ"])
        W["dual_q1"], W["dual_q2"] = p_weight(PV["dual_q1"], a_lvl), p_weight(PV["dual_q2"], a_lvl)
        alpha["DCR-Fusion (q=1)"], alpha["DCR-Fusion (q=2)"] = fuse(E, W["dual_q1"]), fuse(E, W["dual_q2"])
        alpha["DCR-Fusion without typicality test"] = fuse(E, p_weight(PV["cross_q1"], a_lvl))
        alpha["DCR-Fusion without cross-sensor test"] = fuse(E, p_weight(PV["typ"], a_lvl))
        Wo = np.ones_like(W["cs_q1"])
        for m in corrupt:
            if m in evid:
                Wo[:, evid.index(m)] = 0.0
        alpha["Drop-k oracle"] = fuse(E, Wo)
        if "moddrop" in O:
            alpha["Missing-modality training"] = fuse(O["moddrop"]["E"], np.ones_like(Wo))
            alpha["Augmentation training"] = fuse(O["aug"]["E"], np.ones_like(Wo))
        pr = {k: v.argmax(1) for k, v in alpha.items()}
        pr["TMC rule"] = tmc_combine(torch.from_numpy(E).float(), K)[0].numpy().argmax(1)
        pr["Best single group"] = (1 + E[:, best]).argmax(1)
        det = {v: SC[v][0] for v in CS_VARIANTS}
        for v in ["typicality", "qmf_energy", "edl_uncertainty", "dual_q1", "dual_q2"]:
            det[v] = np.full((len(te), M), np.nan)
        for j, m in enumerate(evid):
            det["typicality"][:, m] = dist[:, j]
            det["qmf_energy"][:, m] = energy(H)[:, j]
            det["edl_uncertainty"][:, m] = (K / (E.sum(-1) + K))[:, j]
            det["dual_q1"][:, m] = -np.log(PV["dual_q1"][:, j])
            det["dual_q2"][:, m] = -np.log(PV["dual_q2"][:, j])
        Wfull = {v: cs_weights(SC[v][0], v, rho, beta) for v in ["cs_q1", "cs_q2"]}
        for v in ["dual_q1", "dual_q2"]:
            Wfull[v] = np.full((len(te), M), np.nan)
            Wfull[v][:, EV] = W[v]
        return pr, alpha, W, Wfull, det, E

    def threshold(v, m):
        return rho[v][m] if v in rho else -np.log(a_lvl)

    R = dict(clean=[], far=[], grid=[], grid_cond=[], det=[], cert=[], two=[])
    base0 = dict(run=run["run"])
    pr0, a0, W0, Wf0, det0, E0 = outputs({})
    for k, p in pr0.items():
        R["clean"].append(dict(base0, method=k, macro_f1=f1(y, p), acc=accuracy_score(y, p)))
    for j, m in enumerate(evid):
        p = (1 + E0[:, j]).argmax(1)
        R["clean"].append(dict(base0, method=f"single:{gnames[m]}", macro_f1=f1(y, p), acc=accuracy_score(y, p)))
    for v in CS_VARIANTS:
        for m in range(M):
            R["far"].append(dict(base0, score=v, group=gnames[m], far=float((det0[v][:, m] > rho[v][m]).mean())))
    for v in ["dual_q1", "dual_q2"]:
        for m in evid:
            R["far"].append(dict(base0, score=v, group=gnames[m], far=float((det0[v][:, m] > -np.log(a_lvl)).mean())))
    cm = {k: confusion_matrix(y, pr0[k], labels=list(range(K))).tolist() for k in ["DCR-Fusion (q=1)", "CS-Fusion (q=1)", "Equal late fusion"]}
    cl = data.COND_LABEL[te]
    cert_methods = [("cs_q1", "CS-Fusion (q=1)"), ("cs_q2", "CS-Fusion (q=2)"), ("dual_q1", "DCR-Fusion (q=1)"), ("dual_q2", "DCR-Fusion (q=2)")]

    seed_c = seed * 100000
    for m in range(M):
        for tau in cfg["DEG_TYPES"]:
            for s in cfg["SEVERITIES"]:
                seed_c += 1
                rng = np.random.RandomState(seed_c)
                raw = degrade(np.asarray(data.X[gnames[m]][te], dtype=np.float32), tau, s, stats[gnames[m]][1], rng)
                pr, alpha, W, Wf, det, E = outputs({m: raw})
                b = dict(base0, group=gnames[m], tau=tau, held_out=tau in cfg["HELD_OUT_TYPES"], s=s)
                for k, p in pr.items():
                    R["grid"].append(dict(b, method=k, macro_f1=f1(y, p), acc=accuracy_score(y, p)))
                    if k in PER_COND_METHODS:
                        for cv in np.unique(cl):
                            mm = cl == cv
                            R["grid_cond"].append(dict(b, method=k, cond=cv, macro_f1=f1(y[mm], p[mm])))
                lab = np.r_[np.ones(len(y)), np.zeros(len(y))]
                for v, sc in det.items():
                    if np.isnan(sc[:, m]).all():
                        continue
                    row = dict(b, score=v, auroc=roc_auc_score(lab, np.r_[sc[:, m], det0[v][:, m]]))
                    if v in Wf:
                        row.update(weight_degraded=float(Wf[v][:, m].mean()), weight_healthy=float(np.nanmean(np.delete(Wf[v], m, 1))),
                                   flag_rate=float((sc[:, m] > threshold(v, m)).mean()))
                    R["det"].append(row)
                for v, key in cert_methods:
                    has, cert = certificate(alpha[key], E, W[v], e_max)
                    ok = pr[key] == y
                    keep = (K / alpha[key].sum(1) <= u_th[key]) & (W[v].max(1) >= cfg["W_MIN"])
                    R["cert"].append(dict(b, method=key, flagged=float(has.mean()), certified_of_flagged=float(cert.sum() / max(has.sum(), 1)),
                                          acc_certified=float(ok[cert].mean()) if cert.any() else np.nan,
                                          kept=float(keep.mean()), acc_kept=float(ok[keep].mean()) if keep.any() else np.nan))
        print(f"    one-group grid: {gnames[m]} done ({time.time() - t0:.0f} s)")
    for pair in itertools.combinations(range(M), 2):
        for tau in cfg["DEG_TYPES"]:
            for s in cfg["TWO_GROUP_SEVS"]:
                seed_c += 1
                rng = np.random.RandomState(seed_c)
                cor = {m: degrade(np.asarray(data.X[gnames[m]][te], dtype=np.float32), tau, s, stats[gnames[m]][1], rng) for m in pair}
                pr, *_ = outputs(cor)
                for k, p in pr.items():
                    R["two"].append(dict(base0, pair="+".join(gnames[i] for i in pair), tau=tau, s=s, held_out=tau in cfg["HELD_OUT_TYPES"],
                                         method=k, macro_f1=f1(y, p), acc=accuracy_score(y, p)))
    print(f"    two-group grid done ({time.time() - t0:.0f} s)")
    meta = dict(run=run["run"], rho={k: v.tolist() for k, v in rho.items()}, u_th=u_th, best_single=gnames[evid[best]], confusion=cm,
                n_train=len(tr), n_cal=len(cal), n_test_eval=len(te), test_units=[str(u) for u in run.get("test_units", [])])
    return {k: pd.DataFrame(v) for k, v in R.items()}, meta
