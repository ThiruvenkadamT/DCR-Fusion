"""Reliability tests and weights.

cross-sensor test : r_m = min over admissible subsets of the standardised residual; calibrated on the max over subsets
typicality test   : eta_m = Mahalanobis distance of z_m to the nearest class mean (shared covariance)
dual-test p-value : p_m = min(1, 2 min(p_cs, p_ty));  weight w_m = min(1, p_m / a)
CS-Fusion weight  : w_m = exp(-beta_m max(0, r_m - rho_m)),  rho_m = conformal (1 - a) quantile, beta_m = max(1, 1/(4 rho_m))
"""
import math

import numpy as np


def conformal_quantile(s, a):
    n = len(s)
    k = int(math.ceil((n + 1) * (1 - a)))
    return float(np.sort(s)[min(k, n) - 1])


def conformal_p(cal_sorted, s):
    """Split-conformal p-value of test scores s against sorted clean calibration scores (larger = less typical)."""
    n = len(cal_sorted)
    return (1 + n - np.searchsorted(cal_sorted, s, side="left")) / (n + 1)


def calibrate_cs(SC, a):
    """Thresholds rho and slopes beta of the exponential cross-sensor weight, per score variant and group."""
    M = next(iter(SC.values()))[1].shape[1]
    rho = {v: np.array([conformal_quantile(SC[v][1][:, m], a) for m in range(M)]) for v in SC}
    return rho, {v: np.maximum(1.0, 1.0 / (4 * rho[v])) for v in rho}


def cs_weights(score, v, rho, beta):
    return np.exp(-beta[v][None] * np.maximum(score - rho[v][None], 0.0))


def p_weight(p, a):
    return np.minimum(1.0, p / a)


class TypicalityTest:
    """Distance of each evidence-group embedding to the nearest clean class mean, with conformal p-values."""

    def __init__(self, Z_tr, y_tr, Z_cal, evid, K, D):
        self.evid = evid
        self.mu, self.P = [], []
        for m in evid:
            z = Z_tr[:, m].numpy().astype(np.float64)
            mus = np.stack([z[y_tr == k].mean(0) if (y_tr == k).any() else z.mean(0) for k in range(K)])
            cov = np.cov((z - mus[y_tr]).T) + 1e-3 * np.eye(D)
            self.mu.append(mus)
            self.P.append(np.linalg.inv(cov))
        dc = self.dist(Z_cal)
        self.cal = [np.sort(dc[:, j]) for j in range(len(evid))]

    def dist(self, Z):
        out = np.zeros((len(Z), len(self.evid)))
        for j, m in enumerate(self.evid):
            z = Z[:, m].numpy().astype(np.float64)
            dd = z[:, None, :] - self.mu[j][None]
            out[:, j] = np.einsum("nkd,de,nke->nk", dd, self.P[j], dd).min(1)
        return out

    def pvalues(self, d):
        return np.stack([conformal_p(c, d[:, j]) for j, c in enumerate(self.cal)], 1)


class DualTest:
    """Cross-sensor and typicality p-values for every evidence group, merged by the Bonferroni rule."""

    def __init__(self, SC_cal, typ, evid):
        self.evid, self.typ = evid, typ
        self.cal = {v: [np.sort(SC_cal[v][1][:, m]) for m in evid] for v in ["cs_q1", "cs_q2"]}

    def p_cross(self, SC, v):
        return np.stack([conformal_p(self.cal[v][j], SC[v][0][:, m]) for j, m in enumerate(self.evid)], 1)

    def all(self, SC, dist):
        pt = self.typ.pvalues(dist)
        out = {"typ": pt}
        for q in ("q1", "q2"):
            pc = self.p_cross(SC, "cs_" + q)
            out["cross_" + q] = pc
            out["dual_" + q] = np.minimum(1.0, 2 * np.minimum(pc, pt))
        return out
