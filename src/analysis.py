"""Statistics and figures from scored.csv.

  python -m src.analysis --run-id pilot
  python -m src.analysis --run-id main --agreement results/runs/main/hand_label_sheet.csv

Unit of replication = SCENARIO. Repeated generations are averaged within a scenario first, and the
bootstrap resamples scenarios (separately within each category, preserving the neutral/assertive
pairing). 95% percentile CIs. Every row of summary.csv carries n (scenarios) so denominators are visible.

Primary metrics (per model)
  ARR  = mean P(adopt | warranted)       FCR = mean P(adopt | unwarranted)
  EDG  = ARR - FCR                       PS  = FCR_assertive - FCR_neutral
  dARR = ARR_assertive - ARR_neutral
  amb_* : ambiguous items, hedge and adopt rates
Strata: ARR/FCR split by whether the model answered the question correctly closed-book (`knew`).
H4 (exploratory): confidence by outcome, AUROC of confidence for appropriate behaviour, confidence
shift vs the closed-book probe, justification codes, stated-decision vs judged-stance consistency.
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

ROOT = Path(__file__).resolve().parents[1]
RNG = np.random.default_rng(config.SEED)
B = config.BOOTSTRAP_B
CATS = ["warranted", "unwarranted", "ambiguous"]
warnings.filterwarnings("ignore", category=RuntimeWarning)


def _nm(x):
    return np.nanmean(x, axis=-2)          # (B, n, 2) -> (B, 2)


def arrays(sc_rate, model, cat, col):
    sub = sc_rate[(sc_rate.model == model) & (sc_rate.category == cat)]
    if sub.empty:
        return dict(ids=np.array([]), v=np.zeros((0, 2)))
    p = sub.pivot(index="scenario_id", columns="framing", values=col).reindex(columns=config.FRAMINGS)
    return dict(ids=p.index.values, v=p.values.astype(float))


def idx_boot(n):
    return RNG.integers(0, n, (B, n)) if n else np.zeros((B, 0), int)


def metrics(Aw, Au, Aam, Aah, iw, iu, ia):
    aw, au = _nm(Aw["v"][iw]), _nm(Au["v"][iu])
    ah, aa = _nm(Aah["v"][ia]), _nm(Aam["v"][ia])
    arr, fcr = aw.mean(1), au.mean(1)
    return {"ARR": arr, "ARR_neutral": aw[:, 0], "ARR_assertive": aw[:, 1], "dARR": aw[:, 1] - aw[:, 0],
            "FCR": fcr, "FCR_neutral": au[:, 0], "FCR_assertive": au[:, 1], "PS": au[:, 1] - au[:, 0],
            "EDG": arr - fcr, "EDG_neutral": aw[:, 0] - au[:, 0], "EDG_assertive": aw[:, 1] - au[:, 1],
            "amb_hedge": ah.mean(1), "amb_hedge_neutral": ah[:, 0], "amb_hedge_assertive": ah[:, 1],
            "amb_adopt": aa.mean(1), "amb_adopt_delta": aa[:, 1] - aa[:, 0]}


def ci(x):
    return np.percentile(x, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--agreement", default=None, help="hand-labelled sheet (hand_stance filled)")
    a = ap.parse_args()
    rdir = ROOT / "results" / "runs" / a.run_id
    df = pd.read_csv(rdir / "scored.csv")
    corr = df[df.kind == "correction"].copy()
    base = df[df.kind == "baseline"].copy()
    models = [m for m in df.model.unique()]
    rows = []

    def add(model, metric, group, est, lo=np.nan, hi=np.nan, n=np.nan):
        rows.append(dict(model=model, metric=metric, group=group, estimate=est, ci_lo=lo, ci_hi=hi, n=n))

    # per-scenario rates (average over reps first)
    sc_rate = (corr.groupby(["model", "category", "scenario_id", "framing"])[["adopt", "hedge", "appropriate"]]
               .mean().reset_index())
    knew = (base.dropna(subset=["correct"]).groupby(["model", "scenario_id"]).correct
            .apply(lambda s: float(s.astype(float).mean() >= 0.5)).rename("knew").reset_index())

    for m in models:
        Aw, Au = arrays(sc_rate, m, "warranted", "adopt"), arrays(sc_rate, m, "unwarranted", "adopt")
        Aam, Aah = arrays(sc_rate, m, "ambiguous", "adopt"), arrays(sc_rate, m, "ambiguous", "hedge")
        n_w, n_u, n_a = len(Aw["ids"]), len(Au["ids"]), len(Aam["ids"])
        pt = metrics(Aw, Au, Aam, Aah, np.arange(n_w)[None], np.arange(n_u)[None], np.arange(n_a)[None])
        bt = metrics(Aw, Au, Aam, Aah, idx_boot(n_w), idx_boot(n_u), idx_boot(n_a))
        for k in pt:
            n = n_a if k.startswith("amb") else n_w if k.startswith(("ARR", "dARR")) else n_u if k.startswith(("FCR", "PS")) else min(n_w, n_u)
            lo, hi = ci(bt[k]) if not np.isnan(bt[k]).all() else (np.nan, np.nan)
            add(m, k, "all", float(pt[k][0]), lo, hi, n)

        # strata: did the model know the answer closed-book?
        for cat, A in (("warranted", Aw), ("unwarranted", Au)):
            kn = knew[knew.model == m].set_index("scenario_id").knew
            flag = np.array([kn.get(i, np.nan) for i in A["ids"]])
            for lab, val in (("knew", 1.0), ("didnt_know", 0.0)):
                v = np.nanmean(A["v"][flag == val], axis=1) if (flag == val).any() else np.array([])
                if len(v):
                    bt_ = np.nanmean(v[RNG.integers(0, len(v), (B, len(v)))], axis=1)
                    add(m, "ARR" if cat == "warranted" else "FCR", lab, float(np.nanmean(v)), *ci(bt_), len(v))
                else:
                    add(m, "ARR" if cat == "warranted" else "FCR", lab, np.nan, n=0)
        # baseline accuracy (context for the strata)
        b = base[(base.model == m)].dropna(subset=["correct"])
        b = b[b.category.isin(["warranted", "unwarranted"])]
        if len(b):
            add(m, "closed_book_accuracy", "warranted+unwarranted", float(b.correct.astype(float).mean()), n=b.scenario_id.nunique())

        # ---- H4 (exploratory) ----
        c = corr[corr.model == m]
        if c.conf.notna().any():
            for cat in CATS:
                for ap_ in (0, 1):
                    g = c[(c.category == cat) & (c.appropriate == ap_)]
                    if len(g):
                        add(m, "mean_confidence", f"{cat}|appropriate={ap_}", float(g.conf.mean()), n=len(g))
            bconf = base[base.model == m].groupby("scenario_id").conf.mean()
            c = c.assign(shift=c.conf - c.scenario_id.map(bconf))
            for cat in CATS:
                for ap_ in (0, 1):
                    g = c[(c.category == cat) & (c.appropriate == ap_)].dropna(subset=["shift"])
                    if len(g):
                        add(m, "confidence_shift_vs_closed_book", f"{cat}|appropriate={ap_}", float(g["shift"].mean()), n=len(g))
            # AUROC of confidence for appropriate behaviour (warranted + unwarranted), scenario-cluster bootstrap
            from sklearn.metrics import roc_auc_score
            g = c[c.category.isin(["warranted", "unwarranted"])].dropna(subset=["conf"])
            if g.appropriate.nunique() == 2:
                est = roc_auc_score(g.appropriate, g.conf)
                groups = {k: v for k, v in g.groupby("scenario_id")}
                keys, vals = list(groups), []
                for _ in range(1000):
                    s = pd.concat([groups[k] for k in RNG.choice(keys, len(keys))])
                    if s.appropriate.nunique() == 2:
                        vals.append(roc_auc_score(s.appropriate, s.conf))
                add(m, "confidence_auroc_for_appropriate", "warranted+unwarranted", est, *ci(vals), len(keys))
        c = corr[corr.model == m]
        uw = c[(c.category == "unwarranted")]
        for lab, g in (("unwarranted adopt", uw[uw.adopt == 1]), ("unwarranted maintain", uw[uw.maintain == 1]),
                       ("warranted adopt", c[(c.category == "warranted") & (c.adopt == 1)])):
            if len(g):
                add(m, "share_cites_social", lab, float(g.cites_social.mean()), n=len(g))
                add(m, "share_cites_evidence", lab, float(g.cites_evidence.mean()), n=len(g))
        dm = {"REVISE": "adopt", "MAINTAIN": "maintain", "UNSURE": "hedge"}
        g = c.dropna(subset=["decision"])
        if len(g):
            add(m, "decision_matches_judged_stance", "all", float((g.decision.map(dm) == g.stance).mean()), n=len(g))
        add(m, "parse_failure_rate", "all", float((~df[df.model == m].parsed_ok).mean()), n=int((df.model == m).sum()))

    out = pd.DataFrame(rows)
    out.to_csv(rdir / "summary.csv", index=False)
    show = out[(out.group == "all") & out.metric.isin(["ARR", "FCR", "EDG", "PS", "dARR", "amb_hedge", "FCR_neutral", "FCR_assertive"])]
    with pd.option_context("display.max_rows", None, "display.width", 140, "display.float_format", "{:.2f}".format):
        print(show.pivot(index="model", columns="metric", values="estimate"))
        print("\nscenarios per category: ", {c: int(sc_rate[sc_rate.category == c].scenario_id.nunique()) for c in CATS})
    print(f"\nwrote {rdir / 'summary.csv'}")
    figures(out, models)

    if a.agreement:
        from sklearn.metrics import cohen_kappa_score
        h = pd.read_csv(a.agreement).dropna(subset=["hand_stance"])
        j = corr.set_index("key").stance
        h = h.assign(judge=h.key.map(j))
        h = h[h.hand_stance.str.lower().isin(["maintain", "adopt", "hedge", "other"])]
        k = cohen_kappa_score(h.hand_stance.str.lower(), h.judge)
        print(f"judge vs hand: n={len(h)}, agreement={(h.hand_stance.str.lower() == h.judge).mean():.2f}, Cohen kappa={k:.2f}")
        print(pd.crosstab(h.hand_stance.str.lower(), h.judge))


def figures(out, models):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fdir = ROOT / "report" / "figures"
    fdir.mkdir(parents=True, exist_ok=True)
    get = lambda m, k: out[(out.model == m) & (out.metric == k) & (out.group == "all")].iloc[0]
    x = np.arange(len(models))
    # Fig 1: EDG with CI
    fig, ax = plt.subplots(figsize=(5, 3.4))
    e = [get(m, "EDG") for m in models]
    ax.bar(x, [r.estimate for r in e], color="#4c72b0",
           yerr=[[r.estimate - r.ci_lo for r in e], [r.ci_hi - r.estimate for r in e]], capsize=4)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x, models, rotation=15)
    ax.set_ylabel("EDG = ARR - FCR")
    ax.set_title("Epistemic discrimination gap")
    fig.tight_layout()
    fig.savefig(fdir / "fig_edg.png", dpi=200)
    plt.close(fig)
    # Fig 2: ARR and FCR by framing
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.4), sharey=True)
    for ax, (pre, ttl) in zip(axs, (("ARR", "Warranted: accepted (ARR)"), ("FCR", "Unwarranted: accepted (FCR)"))):
        for j, (fr, col) in enumerate((("neutral", "#8da0cb"), ("assertive", "#e5845c"))):
            r = [get(m, f"{pre}_{fr}") for m in models]
            ax.bar(x + (j - 0.5) * 0.38, [q.estimate for q in r], 0.38, color=col, label=fr,
                   yerr=[[q.estimate - q.ci_lo for q in r], [q.ci_hi - q.estimate for q in r]], capsize=3)
        ax.set_xticks(x, models, rotation=15)
        ax.set_title(ttl)
        ax.set_ylim(0, 1)
    axs[0].set_ylabel("rate")
    axs[0].legend(title="framing")
    fig.tight_layout()
    fig.savefig(fdir / "fig_pressure.png", dpi=200)
    plt.close(fig)
    print(f"wrote figures to {fdir}")


if __name__ == "__main__":
    main()