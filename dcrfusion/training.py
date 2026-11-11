"""Stage 1 (encoders and evidence heads), stage 2 (masked predictors) and cross-sensor scoring."""
import math
import random
import time

import numpy as np
import torch
import torch.nn as nn

from .degradations import degrade
from .model import FusionNet, Predictor, admissible_masks, edl_loss, partner_lists, residuals


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def train_encoders(data, cfg, tr, stats, kind, seed, device):
    """kind: 'main' (standard training), 'moddrop' (random evidence-group dropout) or 'aug' (degradation augmentation)."""
    set_seed(seed)
    M, K, evid, gnames, Y = data.M, data.K, data.evid, data.gnames, data.Y
    net = FusionNet(data.groups, K, cfg["D"], cfg["E_MAX"], stats).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["LR"], weight_decay=cfg["WD"])
    cnt = np.bincount(Y[tr], minlength=K).astype(np.float64)
    p = 1.0 / np.maximum(cnt[Y[tr]], 1)
    p /= p.sum()
    steps = int(math.ceil(len(tr) / cfg["BATCH"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["EPOCHS_1"] * steps)
    rng = np.random.RandomState(seed + 7)
    Yt = torch.from_numpy(Y)
    t0 = time.time()
    for ep in range(cfg["EPOCHS_1"]):
        net.train()
        lam = min(1.0, ep / 10.0)
        order = rng.choice(tr, size=len(tr), replace=True, p=p)
        for b in range(steps):
            idx = np.sort(order[b * cfg["BATCH"]:(b + 1) * cfg["BATCH"]])
            y = Yt[idx].to(device)
            raws = [np.asarray(data.X[g][idx], dtype=np.float32) for g in gnames]
            if kind == "aug":
                m = rng.randint(M)
                sel = rng.rand(len(idx)) < 0.5
                if sel.any():
                    raws[m][sel] = degrade(raws[m][sel], rng.choice(cfg["AUG_TYPES"]), rng.uniform(0.2, 1.0), stats[gnames[m]][1], rng)
            xs = [torch.from_numpy(r).to(device) for r in raws]
            ev = [net.evidence(m, net.encode(m, xs[m])) for m in range(M)]
            if kind == "moddrop":
                keep = (torch.rand(len(idx), len(evid), device=device) > cfg["MODDROP_P"]).float()
                keep[keep.sum(1) == 0, 0] = 1.0
                alpha = 1 + sum(keep[:, j:j + 1] * ev[m] for j, m in enumerate(evid))
            else:
                alpha = 1 + sum(ev[m] for m in evid)
            loss = edl_loss(alpha, y, lam, K) + cfg["LAMBDA_G"] * sum(edl_loss(1 + ev[m], y, lam, K) for m in range(M))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    print(f"    trained '{kind}' network: {cfg['EPOCHS_1']} epochs in {time.time() - t0:.0f} s")
    net.eval()
    for q in net.parameters():
        q.requires_grad_(False)
    return net


@torch.no_grad()
def encode_raw(net, raws_by_group, evid, device, bs=512):
    """raws_by_group: list (per group) of raw arrays (N, C, L). Returns Z (N, M, D), evidence E and logits H."""
    N, M = len(raws_by_group[0]), len(raws_by_group)
    Zs, Es, Hs = [], [], []
    for b in range(0, N, bs):
        Z = torch.stack([net.encode(m, torch.from_numpy(np.asarray(raws_by_group[m][b:b + bs], dtype=np.float32)).to(device))
                         for m in range(M)], 1)
        H = torch.stack([net.logits(m, Z[:, m]) for m in evid], 1)
        Zs.append(Z.cpu())
        Hs.append(H.cpu())
        Es.append((net.e_max * torch.sigmoid(H)).cpu())
    return torch.cat(Zs), torch.cat(Es).numpy(), torch.cat(Hs).numpy()


def encode_idx(net, data, idx, device):
    out = [], [], []
    for b in range(0, len(idx), 4096):
        ii = idx[b:b + 4096]
        z, e, h = encode_raw(net, [data.X[g][ii] for g in data.gnames], data.evid, device)
        out[0].append(z)
        out[1].append(e)
        out[2].append(h)
    return torch.cat(out[0]), np.concatenate(out[1]), np.concatenate(out[2])


def train_predictors(data, cfg, Z_tr, c_tr, use_cond, seed, device):
    """Stage 2: Gaussian negative log-likelihood with admissible (q = 1 or 2) or random partner masks."""
    set_seed(seed + 11)
    M = data.M
    others, adm = partner_lists(M), admissible_masks(M)
    preds = nn.ModuleList([Predictor(M, cfg["D"], data.COND.shape[1], cfg["SIGMA2_MIN"], use_cond, cfg["PRED_HIDDEN"])
                           for _ in range(M)]).to(device)
    op = torch.optim.AdamW(preds.parameters(), lr=cfg["LR"], weight_decay=cfg["WD"])
    Zd = Z_tr.to(device)
    cd = torch.from_numpy(c_tr).to(device)
    N, bs = len(Zd), cfg["BATCH_2"]
    for ep in range(cfg["EPOCHS_2"]):
        perm = torch.randperm(N, device=device)
        for b in range(0, N, bs):
            ii = perm[b:b + bs]
            Z, c, n = Zd[ii], cd[ii], len(ii)
            loss = 0.0
            for m in range(M):
                rnd = (torch.rand(n, M - 1, device=device) < 0.5).float()
                rnd[rnd.sum(1) == 0, 0] = 1.0
                q = 1 if torch.rand(1).item() < 0.5 else 2
                a = adm[q][m].to(device)[torch.randint(len(adm[q][m]), (n,), device=device)]
                mask = torch.where(torch.rand(n, 1, device=device) < 0.5, a, rnd)
                mu, var = preds[m](Z[:, others[m]], mask, c)
                loss = loss + (((Z[:, m] - mu) ** 2) / var + var.log()).mean()
            op.zero_grad()
            loss.backward()
            op.step()
    preds.eval()
    for q in preds.parameters():
        q.requires_grad_(False)
    return preds


@torch.no_grad()
def cs_scores(data, pred, pred_nc, Z, c, device, bs=2048):
    """Cross-sensor score variants -> (test score (N, M), calibration statistic (N, M)).
    cs_q1 / cs_q2: min over admissible subsets (calibration: max); cs_nocond: q = 1 without condition input;
    cs_mean: mean over subsets."""
    M = data.M
    others, adm = partner_lists(M), admissible_masks(M)
    out = {v: [np.zeros((len(Z), M)), np.zeros((len(Z), M))] for v in ["cs_q1", "cs_q2", "cs_nocond", "cs_mean"]}
    for b in range(0, len(Z), bs):
        Zd = Z[b:b + bs].to(device)
        cd = torch.from_numpy(c[b:b + bs]).to(device)
        sl = slice(b, b + len(Zd))
        for m in range(M):
            r1 = residuals(pred[m], m, Zd, adm[1][m], cd, others)
            r2 = residuals(pred[m], m, Zd, adm[2][m], cd, others)
            rn = residuals(pred_nc[m], m, Zd, adm[1][m], cd, others)
            out["cs_q1"][0][sl, m], out["cs_q1"][1][sl, m] = r1.min(1).values.cpu().numpy(), r1.max(1).values.cpu().numpy()
            out["cs_q2"][0][sl, m], out["cs_q2"][1][sl, m] = r2.min(1).values.cpu().numpy(), r2.max(1).values.cpu().numpy()
            out["cs_nocond"][0][sl, m], out["cs_nocond"][1][sl, m] = rn.min(1).values.cpu().numpy(), rn.max(1).values.cpu().numpy()
            out["cs_mean"][0][sl, m] = out["cs_mean"][1][sl, m] = r1.mean(1).cpu().numpy()
    return {k: (v[0], v[1]) for k, v in out.items()}
