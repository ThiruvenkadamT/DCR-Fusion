"""Sensor degradation operators applied to raw windows of shape (B, C, L).

Severity s lies in (0, 1]. nu is the per-channel standard deviation of the clean training data.
N, G, Dr and F are also used for augmentation training; B, Cl, St and Bw are never used in training.
"""
import math

import numpy as np
from scipy.signal import butter, sosfiltfilt

NAMES = {"N": "Noise", "G": "Gain drift", "Dr": "Dropout", "F": "Failure",
         "B": "Bias drift", "Cl": "Clipping", "St": "Stuck-at", "Bw": "Bandwidth loss"}


def make_stats(data, idx):
    """Per-channel mean and standard deviation of the clean training windows."""
    st = {}
    sub = idx[:: max(1, len(idx) // 2000)]
    for g in data.gnames:
        xs = np.asarray(data.X[g][np.sort(sub)], dtype=np.float64)
        st[g] = (xs.mean(axis=(0, 2)).astype(np.float32), (xs.std(axis=(0, 2)) + 1e-8).astype(np.float32))
    return st


def degrade(x, tau, s, nu, rng):
    x = np.array(x, dtype=np.float32, copy=True)
    B, C, Ln = x.shape
    sig = nu[None, :, None]
    sign = rng.choice([-1.0, 1.0], size=(B, C, 1)).astype(np.float32)
    if tau == "N":       # additive noise, SNR = 20 - 25 s dB
        snr = 20.0 - 25.0 * s
        x = x + sig * 10 ** (-snr / 20.0) * rng.standard_normal(x.shape).astype(np.float32)
    elif tau == "G":     # gain drift
        x = (1.0 + sign * 0.9 * s) * x
    elif tau == "Dr":    # intermittent dropout of 32 equal segments
        nb = 32
        bl = int(math.ceil(Ln / nb))
        keep = (rng.random((B, C, nb)) >= s).astype(np.float32)
        x = x * np.repeat(keep, bl, axis=2)[:, :, :Ln]
    elif tau == "F":     # total failure: low-level noise only
        x = 0.01 * sig * rng.standard_normal(x.shape).astype(np.float32)
    elif tau == "B":     # bias drift growing over the window
        ramp = (0.5 + 0.5 * np.arange(Ln, dtype=np.float32) / Ln)[None, None, :]
        x = x + sign * 3.0 * s * sig * ramp
    elif tau == "Cl":    # clipping / saturation
        A = (1.0 - 0.95 * s) * np.abs(x).max(axis=2, keepdims=True)
        x = np.clip(x, -A, A)
    elif tau == "St":    # stuck-at: a segment held at its first value
        ln = int(math.ceil(s * Ln))
        for b in range(B):
            for c in range(C):
                t0 = rng.randint(0, max(1, Ln - ln + 1))
                x[b, c, t0:t0 + ln] = x[b, c, t0]
    elif tau == "Bw":    # bandwidth loss: zero-phase 4th-order Butterworth low-pass
        fc = max(1e-3, 1.0 - 0.95 * s)
        if fc < 0.999:
            sos = butter(4, fc, btype="low", output="sos")
            x = sosfiltfilt(sos, x, axis=2).astype(np.float32)
    else:
        raise ValueError(tau)
    return x
