"""Summarise the result tables written by run_experiments.py: mean ± sd over runs and paired Wilcoxon tests.

Example:
    python summarize_results.py --results ./work/results_kaist
"""
import argparse
import glob
import os

import pandas as pd
from scipy.stats import wilcoxon

ORDER = ["DCR-Fusion (q=1)", "DCR-Fusion (q=2)", "CS-Fusion (q=1)", "CS-Fusion (q=2)", "P-MIF reliability", "Equal late fusion",
         "TMC rule", "EDL-uncertainty weights", "QMF-style energy weights", "Missing-modality training", "Augmentation training",
         "Best single group", "Drop-k oracle", "DCR-Fusion without typicality test", "DCR-Fusion without cross-sensor test",
         "CS-Fusion without condition input", "CS-Fusion, mean over subsets", "CS-Fusion, softmax weights"]
SCORES = ["dual_q1", "dual_q2", "cs_q1", "cs_q2", "cs_nocond", "cs_mean", "typicality", "edl_uncertainty", "qmf_energy"]


def ms(df, val="macro_f1", key="method", order=ORDER):
    per = df.groupby(["run", key])[val].mean().unstack()
    t = pd.DataFrame({"mean": per.mean(), "sd": per.std()})
    t["mean ± sd"] = t["mean"].map("{:.4f}".format) + " ± " + t["sd"].map("{:.4f}".format)
    return t.reindex([o for o in order if o in t.index] + [i for i in t.index if i not in order])[["mean ± sd"]]


def paired(df, ref, keys, val="macro_f1", col="method", others=None):
    piv = df.pivot_table(index=keys, columns=col, values=val)
    rows = []
    for o in others or [c for c in piv.columns if c != ref]:
        if o not in piv.columns or o == ref:
            continue
        d = (piv[ref] - piv[o]).dropna()
        p = wilcoxon(d).pvalue if (d != 0).any() else 1.0
        rows.append(dict(compared_with=o, mean_diff=round(d.mean(), 4), wins=int((d > 0).sum()), losses=int((d < 0).sum()), n=len(d), p_value=f"{p:.1e}"))
    return pd.DataFrame(rows).set_index("compared_with")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="results folder written by run_experiments.py")
    args = ap.parse_args()
    tag = os.path.basename(glob.glob(os.path.join(args.results, "*_clean.csv"))[0])[:-len("_clean.csv")]
    T = {k: pd.read_csv(os.path.join(args.results, f"{tag}_{k}.csv")) for k in ["clean", "far", "grid", "grid_cond", "det", "cert", "two"]}
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    G, TW = T["grid"], T["two"]
    det = T["det"][T["det"].group != "temp"]
    far = T["far"][T["far"].group != "temp"]
    worst = G.groupby(["run", "method"]).macro_f1.min().reset_index()

    print(f"== {tag}: runs {sorted(T['clean'].run.unique())} ==")
    one = pd.concat([ms(T["clean"]).rename(columns={"mean ± sd": "clean"}), ms(G).rename(columns={"mean ± sd": "one group"}),
                     ms(G[~G.held_out]).rename(columns={"mean ± sd": "augmentation types"}),
                     ms(G[G.held_out]).rename(columns={"mean ± sd": "held-out types"}), ms(worst).rename(columns={"mean ± sd": "worst cell"})], axis=1)
    print("\nMacro-F1: clean data and one degraded group\n", one.to_string())
    print("\nMacro-F1 by degradation type (one group)\n", G.pivot_table(index="method", columns="tau", values="macro_f1").reindex(ORDER).round(4).to_string())
    print("\nMacro-F1 by severity (one group)\n", G.pivot_table(index="method", columns="s", values="macro_f1").reindex(ORDER).round(4).to_string())
    print("\nMacro-F1 by operating condition (one group)\n", T["grid_cond"].pivot_table(index="method", columns="cond", values="macro_f1").round(4).to_string())
    two = pd.concat([ms(TW[TW.s == s]).rename(columns={"mean ± sd": f"s = {s}"}) for s in sorted(TW.s.unique())] +
                    [ms(TW).rename(columns={"mean ± sd": "all"})], axis=1)
    print("\nMacro-F1: two degraded groups\n", two.to_string())
    print("\nDetection AUROC of the degraded group (evidence groups)\n", ms(det, "auroc", "score", SCORES).to_string())
    print("\nDetection AUROC by degradation type\n", det.pivot_table(index="score", columns="tau", values="auroc").reindex(SCORES).round(3).to_string())
    print("\nFalse-alarm rate on clean data (evidence groups)\n", ms(far, "far", "score", SCORES).to_string())
    w = det[det.score.isin(["dual_q1", "dual_q2", "cs_q1", "cs_q2"])]
    print("\nMean weight of the degraded and healthy groups by severity\n",
          w.groupby(["score", "s"])[["weight_degraded", "weight_healthy", "flag_rate"]].mean().round(3).to_string())
    print("\nWeight of the degraded group by type and severity (DCR-Fusion, q=1)\n",
          w[w.score == "dual_q1"].pivot_table(index="tau", columns="s", values="weight_degraded").round(2).to_string())
    print("\nCertificate and rejection (one degraded group)\n",
          T["cert"].groupby("method")[["flagged", "certified_of_flagged", "acc_certified", "kept", "acc_kept"]].mean().round(3).to_string())
    print("\nWilcoxon: DCR-Fusion (q=1), one degraded group\n", paired(G, "DCR-Fusion (q=1)", ["run", "group", "tau", "s"], others=ORDER).to_string())
    print("\nWilcoxon: DCR-Fusion (q=2), two degraded groups\n", paired(TW, "DCR-Fusion (q=2)", ["run", "pair", "tau", "s"], others=ORDER).to_string())
    print("\nWilcoxon: detection AUROC of dual_q1\n", paired(det, "dual_q1", ["run", "group", "tau", "s"], "auroc", "score", SCORES).to_string())


if __name__ == "__main__":
    main()
