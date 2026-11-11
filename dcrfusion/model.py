"""Group encoders, bounded evidential heads, masked condition-conditioned predictors and fusion helpers."""
import itertools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, ci, co, stride):
        super().__init__()
        self.c1 = nn.Conv1d(ci, co, 7, stride, 3, bias=False)
        self.b1 = nn.BatchNorm1d(co)
        self.c2 = nn.Conv1d(co, co, 7, 1, 3, bias=False)
        self.b2 = nn.BatchNorm1d(co)
        self.sk = nn.Sequential(nn.Conv1d(ci, co, 1, stride, bias=False), nn.BatchNorm1d(co))

    def forward(self, x):
        return F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x))))) + self.sk(x))


class SignalEncoder(nn.Module):
    def __init__(self, C, D):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(C, 32, 64, 8, 28, bias=False), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2))
        self.body = nn.Sequential(ResBlock(32, 64, 2), ResBlock(64, 64, 2), ResBlock(64, 128, 2), ResBlock(128, 128, 2))
        self.proj = nn.Sequential(nn.Linear(128, D), nn.BatchNorm1d(D))

    def forward(self, x):
        return self.proj(self.body(self.stem(x)).mean(-1))


class TempEncoder(nn.Module):
    def __init__(self, C, D):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(C, 32, 7, 1, 3), nn.BatchNorm1d(32), nn.ReLU(),
                                 nn.Conv1d(32, 64, 5, 2, 2), nn.BatchNorm1d(64), nn.ReLU())
        self.proj = nn.Sequential(nn.Linear(64 + 2 * C, D), nn.BatchNorm1d(D))

    def forward(self, x):
        return self.proj(torch.cat([self.net(x).mean(-1), x.mean(-1), x[..., -1] - x[..., 0]], 1))


class FusionNet(nn.Module):
    """One encoder per sensor group and a spectrally normalised evidence head e = E_max * sigmoid(W z)."""

    def __init__(self, groups, K, D, e_max, stats):
        super().__init__()
        self.gnames = [g["name"] for g in groups]
        self.e_max = e_max
        self.enc = nn.ModuleList([TempEncoder(g["ch"], D) if g["name"] == "temp" else SignalEncoder(g["ch"], D) for g in groups])
        self.heads = nn.ModuleDict({str(m): nn.utils.parametrizations.spectral_norm(nn.Linear(D, K)) for m in range(len(groups))})
        for g in self.gnames:
            self.register_buffer("mu_" + g, torch.tensor(stats[g][0])[None, :, None])
            self.register_buffer("sd_" + g, torch.tensor(stats[g][1])[None, :, None])

    def encode(self, m, x_raw):
        g = self.gnames[m]
        return self.enc[m]((x_raw - getattr(self, "mu_" + g)) / getattr(self, "sd_" + g))

    def logits(self, m, z):
        return self.heads[str(m)](z)

    def evidence(self, m, z):
        return self.e_max * torch.sigmoid(self.logits(m, z))


def partner_lists(M):
    return [[j for j in range(M) if j != m] for m in range(M)]


def admissible_masks(M):
    """Masks over the partners of each group for the admissible subsets |S| = M - 1 - q, q in {1, 2}."""
    others = partner_lists(M)

    def masks(m, size):
        return torch.tensor([[1.0 if j in S else 0.0 for j in others[m]] for S in itertools.combinations(others[m], size)])
    return {q: [masks(m, max(1, M - 1 - q)) for m in range(M)] for q in (1, 2)}


class Predictor(nn.Module):
    """Masked heteroscedastic predictor g_m(z_S, 1_S, c) -> (mu, sigma^2) with FiLM conditioning on c."""

    def __init__(self, M, D, P, sigma2_min, use_cond=True, H=256):
        super().__init__()
        self.use_cond, self.H, self.D, self.sigma2_min = use_cond, H, D, sigma2_min
        din = (M - 1) * D + (M - 1)
        self.l1, self.l2, self.out = nn.Linear(din, H), nn.Linear(H, H), nn.Linear(H, 2 * D)
        self.phi = nn.Sequential(nn.Linear(P, 32), nn.SiLU(), nn.Linear(32, 4 * H))

    def forward(self, z_others, mask, c):
        v = torch.cat([(z_others * mask[..., None]).flatten(1), mask], 1)
        g1, b1, g2, b2 = self.phi(c if self.use_cond else torch.zeros_like(c)).split(self.H, 1)
        h = F.silu(self.l1(v)) * (1 + g1) + b1
        h = F.silu(self.l2(h)) * (1 + g2) + b2
        mu, raw = self.out(h).split(self.D, 1)
        return mu, self.sigma2_min + F.softplus(raw)


def residuals(pred, m, Z, masks, c, others):
    """Standardised residual r_{m|S} of group m for every subset mask; returns (B, n_subsets)."""
    B, nS = Z.shape[0], masks.shape[0]
    M, D = Z.shape[1], Z.shape[2]
    zo = Z[:, others[m]].unsqueeze(1).expand(B, nS, M - 1, D).reshape(B * nS, M - 1, D)
    mk = masks.to(Z.device).unsqueeze(0).expand(B, nS, M - 1).reshape(B * nS, M - 1)
    mu, var = pred(zo, mk, c.repeat_interleave(nS, 0))
    return ((Z[:, m].repeat_interleave(nS, 0) - mu) ** 2 / var).mean(1).view(B, nS)


def edl_loss(alpha, y, lam, K):
    """Evidential loss: expected cross-entropy under Dir(alpha) plus annealed KL of the misleading evidence."""
    S = alpha.sum(1, keepdim=True)
    yo = F.one_hot(y, K).float()
    nll = (yo * (torch.digamma(S) - torch.digamma(alpha))).sum(1)
    at = yo + (1 - yo) * alpha
    St = at.sum(1)
    kl = (torch.lgamma(St) - math.lgamma(K) - torch.lgamma(at).sum(1)
          + ((at - 1) * (torch.digamma(at) - torch.digamma(St)[:, None])).sum(1))
    return (nll + lam * kl).mean()


def tmc_combine(E, K):
    """Reduced Dempster combination of subjective opinions (trusted multi-view classification)."""
    b = E / (E.sum(-1, keepdim=True) + K)
    u = K / (E.sum(-1) + K)
    B1, U1 = b[:, 0], u[:, 0]
    for v in range(1, E.shape[1]):
        B2, U2 = b[:, v], u[:, v]
        C = B1.sum(1) * B2.sum(1) - (B1 * B2).sum(1)
        den = (1 - C).clamp_min(1e-6)
        B1 = (B1 * B2 + B1 * U2[:, None] + B2 * U1[:, None]) / den[:, None]
        U1 = U1 * U2 / den
    return B1, U1
