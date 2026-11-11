"""Common data container and helpers."""
import os
import zipfile
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
import pandas as pd
from scipy.signal import resample_poly


@dataclass
class Data:
    X: dict                 # group name -> array (N, C, L)
    Y: np.ndarray           # class index per window
    COND: np.ndarray        # condition vector per window, scaled to [0, 1]
    COND_LABEL: np.ndarray  # readable operating condition per window
    META: pd.DataFrame      # unit, condition, block or measurement per window
    groups: list            # [dict(name=..., ch=...)]
    classes: list
    info: dict = field(default_factory=dict)

    @property
    def gnames(self):
        return [g["name"] for g in self.groups]

    @property
    def M(self):
        return len(self.groups)

    @property
    def K(self):
        return len(self.classes)

    @property
    def evid(self):
        """Groups with an evidence head (temperature is a reliability partner only)."""
        return [i for i, g in enumerate(self.gnames) if g != "temp"]

    def counts(self, by):
        c = self.META.groupby([by, "cls"]).size().unstack(fill_value=0).reindex(columns=range(self.K), fill_value=0)
        c.columns = self.classes
        return c


def to_fs(x, fs, target):
    """Resample along axis 0 from fs to target (polyphase)."""
    if fs is None or abs(fs - target) < 1e-6:
        return x
    fr = Fraction(target / fs).limit_denominator(2000)
    return resample_poly(x, fr.numerator, fr.denominator, axis=0)


def collect(root, exts, extract_dir):
    """All files under root with the given extensions; .zip archives are extracted once into extract_dir."""
    found = []
    if not os.path.isdir(root):
        return found
    for dp, _, fs in os.walk(root):
        for f in fs:
            p = os.path.join(dp, f)
            low = f.lower()
            if low.endswith(".zip"):
                tgt = os.path.join(extract_dir, os.path.splitext(f)[0])
                if not os.path.isdir(tgt):
                    with zipfile.ZipFile(p) as z:
                        z.extractall(tgt)
                found += collect(tgt, exts, extract_dir)
            elif low.endswith(exts):
                found.append(p)
    return sorted(set(found))


def save_cache(cache, X, META, info):
    import json
    os.makedirs(cache, exist_ok=True)
    for g, a in X.items():
        np.save(os.path.join(cache, g + ".npy"), a)
    META.to_csv(os.path.join(cache, "meta.csv"), index=False)
    json.dump(info, open(os.path.join(cache, "info.json"), "w"))


def load_cache(cache, gnames):
    import json
    if not os.path.exists(os.path.join(cache, "meta.csv")):
        return None
    X = {g: np.load(os.path.join(cache, g + ".npy"), mmap_mode="r") for g in gnames}
    return X, pd.read_csv(os.path.join(cache, "meta.csv")), json.load(open(os.path.join(cache, "info.json")))
