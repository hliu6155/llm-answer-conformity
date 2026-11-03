"""Analysis for the IN6208 answer-conformity experiment (prespecified analysis plan).

    python analyze.py --results path/to/results --materials path/to/materials --out path/to/analysis

Confirmatory sample: selected items (status 'selected') with valid responses in all 96 exposure cells.
Runs averaged within items; models and languages weighted equally; 10,000 item-level bootstrap
resamples; 95% and Bonferroni-adjusted 98.33% percentile intervals; worst-case bounds on all
selected items. Everything else is descriptive.
"""
import argparse
import gzip
import json
import os

import numpy as np
import pandas as pd

LANGS, CONDS, ROUNDS = ["en", "zh"], ["A", "B"], [1, 2, 3]
B = 10_000
SEED = 6208


def load_jsonl(path):
    """Read a JSON Lines log; falls back to a gzip copy (<path>.gz) when the plain file is absent."""
    if not os.path.exists(path) and os.path.exists(path + ".gz"):
        opener, path = gzip.open, path + ".gz"
    elif os.path.exists(path):
        opener = open
    else:
        return pd.DataFrame()
    rows = []
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.drop_duplicates("key", keep="last").reset_index(drop=True)


def pct(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(100 * float(x), 1)


def item_cube(exp, items, models, valid_col="valid"):
    """Array [item, model, lang, run, cond, round] of conform (1/0) or NaN when missing/invalid."""
    arr = np.full((len(items), len(models), 2, 2, 2, 3), np.nan)
    ii = {k: i for i, k in enumerate(items)}
    mi = {k: i for i, k in enumerate(models)}
    e = exp[exp["item_id"].isin(ii) & exp["model"].isin(mi)]
    for r in e.itertuples(index=False):
        if getattr(r, "ok") and getattr(r, valid_col):
            val = 1.0 if r.parsed == r.endorsed else 0.0
            arr[ii[r.item_id], mi[r.model], LANGS.index(r.language), r.run - 1,
                CONDS.index(r.condition), r.round - 1] = val
    return arr


def contrasts(cube):
    """cube: [item, model, lang, run, cond, round] with no NaN. Returns per-item H1, H2, H3 values."""
    runs = cube.mean(axis=3)                               # average runs within item
    delta = runs[:, :, :, 1, :] - runs[:, :, :, 0, :]      # B - A : [item, model, lang, round]
    h1 = delta[:, :, :, 0].mean(axis=(1, 2))
    h2 = (delta[:, :, :, 2] - delta[:, :, :, 0]).mean(axis=(1, 2))
    h3 = (delta[:, :, 1, 0] - delta[:, :, 0, 0]).mean(axis=1)
    return {"H1": h1, "H2": h2, "H3": h3}


def bootstrap(per_item, rng):
    n = len(per_item["H1"])
    idx = rng.integers(0, n, size=(B, n))
    out = {}
    for h, v in per_item.items():
        bs = v[idx].mean(axis=1)
        out[h] = {
            "estimate_pp": round(100 * float(v.mean()), 2),
            "ci95_pp": [round(100 * float(np.percentile(bs, 2.5)), 2), round(100 * float(np.percentile(bs, 97.5)), 2)],
            "ci9833_pp": [round(100 * float(np.percentile(bs, 100 / 120)), 2),
                          round(100 * float(np.percentile(bs, 100 - 100 / 120)), 2)],
        }
    return out


def worst_case(cube):
    """Bounds for each contrast when missing outcomes take the values that minimise/maximise it."""
    res = {}
    # sign of each (lang, cond, round) cell in each contrast
    signs = {"H1": {(l, c, 0): (1 if c == 1 else -1) for l in (0, 1) for c in (0, 1)},
             "H2": {(l, c, r): (1 if c == 1 else -1) * (1 if r == 2 else -1)
                    for l in (0, 1) for c in (0, 1) for r in (0, 2)},
             "H3": {(l, c, 0): (1 if c == 1 else -1) * (1 if l == 1 else -1) for l in (0, 1) for c in (0, 1)}}
    for h, sg in signs.items():
        lo, hi = cube.copy(), cube.copy()
        for (l, c, r), s in sg.items():
            sl = (slice(None), slice(None), l, slice(None), c, r)
            m = np.isnan(cube[sl])
            lo_sl, hi_sl = lo[sl], hi[sl]
            lo_sl[m] = 0.0 if s > 0 else 1.0
            hi_sl[m] = 1.0 if s > 0 else 0.0
            lo[sl], hi[sl] = lo_sl, hi_sl
        lo = np.nan_to_num(lo)
        hi = np.nan_to_num(hi)
        res[h] = [round(100 * float(contrasts(lo)[h].mean()), 2), round(100 * float(contrasts(hi)[h].mean()), 2)]
    return res


def cr_table(cube, models):
    rows = []
    runs = np.nanmean(cube, axis=3)  # [item, model, lang, cond, round]
    for mi, m in enumerate(models):
        for li, l in enumerate(LANGS):
            row = {"model": m, "language": "English" if l == "en" else "Chinese"}
            for ci, c in enumerate(CONDS):
                for r in ROUNDS:
                    row[f"{c}_R{r}"] = pct(np.nanmean(runs[:, mi, li, ci, r - 1]))
            rows.append(row)
    allrow = {"model": "All models (equal weights)", "language": "Both"}
    for ci, c in enumerate(CONDS):
        for r in ROUNDS:
            allrow[f"{c}_R{r}"] = pct(np.nanmean(np.nanmean(runs[:, :, :, ci, r - 1], axis=0)))
    rows.append(allrow)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--materials", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(SEED)

    items = pd.read_csv(os.path.join(a.materials, "items_main.csv"), dtype=str, keep_default_na=False)
    scr = load_jsonl(os.path.join(a.results, "screening.jsonl"))
    exp = load_jsonl(os.path.join(a.results, "exposure.jsonl"))
    sel = pd.read_csv(os.path.join(a.results, "selection.csv"), dtype=str, encoding="utf-8-sig")
    settings = json.load(open(os.path.join(a.results, "settings.json"), encoding="utf-8"))
    models = [m for m in ["Qwen3.8-Flash", "DeepSeek-V4.1-Flash", "GPT-6 Luna", "Claude Haiku 5.5"]
              if m in set(scr["model"])]
    S = {"models": models, "settings": {m: {k: v for k, v in settings.get(m, {}).items()} for m in models}}

    # ---- Screening ----
    scr_main = scr[scr["item_id"].isin(items["item_id"])]
    acc = (scr_main.assign(correct=scr_main["correct"].fillna(False).astype(bool))
           .groupby(["model", "language"]).agg(calls=("key", "size"), valid=("valid", "mean"),
                                                accuracy=("correct", "mean"),
                                                failed=("ok", lambda s: int((~s.astype(bool)).sum())))
           .reset_index())
    acc["valid"] = (100 * acc["valid"]).round(1)
    acc["accuracy"] = (100 * acc["accuracy"]).round(1)
    acc.to_csv(os.path.join(a.out, "screening_accuracy.csv"), index=False)
    S["screening"] = {"calls": int(len(scr_main)), "valid_pct": pct(scr_main["valid"].mean()),
                      "valid_strict_pct": pct(scr_main.get("valid_strict", scr_main["valid"]).mean()),
                      "failed": int((~scr_main["ok"].astype(bool)).sum())}
    elig = sel[sel["eligible"] == "1"]
    S["eligibility_by_subject"] = (sel.assign(eligible=sel["eligible"].astype(int))
                                   .groupby("subject")["eligible"].sum().to_dict())

    selected = sel.loc[sel["status"] == "selected", "item_id"].tolist()
    spares = sel.loc[sel["status"] == "spare", "item_id"].tolist()

    # ---- Exposure ----
    ex_sel = exp[exp["item_id"].isin(selected)]
    S["exposure"] = {"records_selected": int(len(ex_sel)), "valid_pct": pct(ex_sel["valid"].mean()),
                     "valid_strict_pct": pct(ex_sel.get("valid_strict", ex_sel["valid"]).mean()),
                     "failed_calls": int((~ex_sel["ok"].astype(bool)).sum()),
                     "expected_records": len(selected) * 96}
    inval = ex_sel[ex_sel["ok"].astype(bool) & ~ex_sel["valid"].astype(bool)]
    S["exposure"]["invalid_by_model_language"] = ({f"{m} | {l}": int(n) for (m, l), n in
                                                   inval.groupby(["model", "language"]).size().items()}
                                                  if len(inval) else {})
    S["exposure"]["invalid_examples"] = inval["response"].head(8).tolist()

    cube_all = item_cube(exp, selected, models)
    complete_mask = ~np.isnan(cube_all).reshape(len(selected), -1).any(axis=1)
    complete = [i for i, ok in zip(selected, complete_mask) if ok]
    cube = cube_all[complete_mask]
    S["sample_flow"] = {"screened": int(len(items)), "eligible": int(len(elig)), "selected": len(selected),
                        "spares_run": len(spares), "complete": len(complete)}

    # ---- Confirmatory tests ----
    if len(complete) >= 2:
        per = contrasts(cube)
        S["confirmatory"] = bootstrap(per, rng)
        S["confirmatory"]["n_items"] = len(complete)
        wc = worst_case(cube_all)
        for h in ("H1", "H2", "H3"):
            S["confirmatory"][h]["worst_case_pp"] = wc[h]
            lo, hi = S["confirmatory"][h]["ci9833_pp"]
            if h == "H1":
                S["confirmatory"][h]["decision"] = "supported" if lo > 0 else "not supported"
            else:
                S["confirmatory"][h]["decision"] = "supported" if (lo > 0 or hi < 0) else "not supported"
        S["feasibility_only"] = len(complete) < 30

        # language-specific Round-1 deltas for H3 reporting
        runs = cube.mean(axis=3)
        delta = runs[:, :, :, 1, :] - runs[:, :, :, 0, :]
        S["delta_round1_by_language_pp"] = {l: round(100 * float(delta[:, :, li, 0].mean()), 2)
                                            for li, l in enumerate(LANGS)}
        S["delta_by_round_pp"] = {f"R{r}": round(100 * float(delta[:, :, :, r - 1].mean()), 2) for r in ROUNDS}

        # per-model descriptive contrasts with bootstrap intervals
        pm = {}
        for mi, m in enumerate(models):
            v = delta[:, mi, :, 0].mean(axis=1)
            idx = rng.integers(0, len(v), size=(B, len(v)))
            bs = v[idx].mean(axis=1)
            pm[m] = {"H1_delta_pp": round(100 * float(v.mean()), 2),
                     "ci95_pp": [round(100 * float(np.percentile(bs, 2.5)), 2),
                                 round(100 * float(np.percentile(bs, 97.5)), 2)],
                     "delta_R3_minus_R1_pp": round(100 * float((delta[:, mi, :, 2] - delta[:, mi, :, 0]).mean()), 2)}
        S["per_model"] = pm

        # available-case sensitivity: each contrast only needs the cells it uses
        avail = {}
        for h, rounds_needed in (("H1", [0]), ("H2", [0, 2]), ("H3", [0])):
            sub = cube_all[..., rounds_needed]
            mask = ~np.isnan(sub).reshape(len(selected), -1).any(axis=1)
            if mask.sum() >= 2:
                c = np.nan_to_num(cube_all[mask])
                v = contrasts(c)[h]
                idx = np.random.default_rng(SEED + 2).integers(0, len(v), size=(B, len(v)))
                bs = v[idx].mean(axis=1)
                avail[h] = {"n_items": int(mask.sum()), "estimate_pp": round(100 * float(v.mean()), 2),
                            "ci9833_pp": [round(100 * float(np.percentile(bs, 100 / 120)), 2),
                                          round(100 * float(np.percentile(bs, 100 - 100 / 120)), 2)]}
        S["available_case_sensitivity"] = avail

        # strict-rule sensitivity
        cube_s_all = item_cube(exp, selected, models, valid_col="valid_strict") if "valid_strict" in exp else cube_all
        mask_s = ~np.isnan(cube_s_all).reshape(len(selected), -1).any(axis=1)
        if mask_s.sum() >= 2:
            S["strict_rule_sensitivity"] = bootstrap(contrasts(cube_s_all[mask_s]), np.random.default_rng(SEED + 1))
            S["strict_rule_sensitivity"]["n_items"] = int(mask_s.sum())
    else:
        S["confirmatory"] = None

    # ---- Descriptive tables ----
    t3 = cr_table(cube if len(complete) else cube_all, models)
    t3.to_csv(os.path.join(a.out, "table3_conformity_rates.csv"), index=False)
    by_round = []
    runs = np.nanmean((cube if len(complete) else cube_all), axis=3)
    for li, l in enumerate(LANGS):
        for ci, c in enumerate(CONDS):
            for r in ROUNDS:
                by_round.append({"language": l, "condition": c, "round": r,
                                 "cr_pct": pct(np.nanmean(np.nanmean(runs[:, :, li, ci, r - 1], axis=0)))})
    pd.DataFrame(by_round).to_csv(os.path.join(a.out, "figure2_data.csv"), index=False)
    S["round3_cr_by_model_pct"] = {m: pct(np.nanmean(runs[:, mi, :, :, 2])) for mi, m in enumerate(models)}

    # ---- Exploratory ----
    e = ex_sel[ex_sel["ok"].astype(bool) & ex_sel["valid"].astype(bool)].copy()
    e["other_error"] = (e["parsed"] != e["endorsed"]) & (e["parsed"] != e["answer"])
    S["other_error_pct"] = pct(e["other_error"].mean()) if len(e) else None
    S["other_error_by_condition_pct"] = {c: pct(g["other_error"].mean()) for c, g in e.groupby("condition")}
    br = e.pivot_table(index=["item_id", "model", "language", "run", "condition"], columns="round",
                       values="parsed", aggfunc="first")
    endo = e.drop_duplicates(["item_id"]).set_index("item_id")["endorsed"]
    ans = e.drop_duplicates(["item_id"]).set_index("item_id")["answer"]
    if {1, 3}.issubset(br.columns):
        br = br.dropna(subset=[1, 3]).reset_index()
        br["c1"] = br[1] == br["item_id"].map(endo)
        br["c3"] = br[3] == br["item_id"].map(endo)
        br["k3"] = br[3] == br["item_id"].map(ans)
        conf1 = br[br["c1"]]
        S["persistence"] = {"branches_conforming_at_R1": int(len(conf1)),
                            "still_conforming_at_R3_pct": pct(conf1["c3"].mean()) if len(conf1) else None,
                            "back_to_correct_at_R3_pct": pct(conf1["k3"].mean()) if len(conf1) else None,
                            "new_conformity_R1_to_R3_pct": pct((~br["c1"] & br["c3"]).mean())}
    S["reasoning_tokens_mean"] = {m: round(float(pd.to_numeric(g.get("reasoning_tokens"), errors="coerce").fillna(0).mean()), 2)
                                  for m, g in exp.groupby("model")}
    S["model_versions_returned"] = {m: sorted(set(map(str, g["model_returned"].dropna())))
                                    for m, g in pd.concat([scr, exp]).groupby("model")}
    times = pd.to_datetime(pd.concat([scr["time"], exp["time"]]), errors="coerce")
    S["collection_window_utc"] = [str(times.min()), str(times.max())]

    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(S, f, ensure_ascii=False, indent=2, default=str)
    make_figure(pd.DataFrame(by_round), os.path.join(a.out, "figure2_conformity_by_round.png"))
    print(json.dumps(S, ensure_ascii=False, indent=2, default=str))


def make_figure(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), sharey=True)
    colors = {"A": "#7A869A", "B": "#2F6DB5"}
    names = {"A": "A: one repeated source", "B": "B: three distinct sources"}
    ymax = max(10, float(df["cr_pct"].max() or 0) * 1.25)
    for ax, (lang, title) in zip(axes, (("en", "English prompts"), ("zh", "Chinese prompts"))):
        for c in ("A", "B"):
            d = df[(df.language == lang) & (df.condition == c)].sort_values("round")
            ax.plot(d["round"], d["cr_pct"], marker="o", color=colors[c], lw=2, label=names[c])
            for x, y in zip(d["round"], d["cr_pct"]):
                ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                            xytext=(0, 7) if c == "B" else (0, -13),
                            ha="center", fontsize=8, color=colors[c])
        ax.set_title(title, fontsize=10)
        ax.set_xticks([1, 2, 3])
        ax.set_xlabel("Round")
        ax.set_ylim(0, ymax)
        ax.grid(axis="y", alpha=0.3)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("Conformity rate (%)")
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)


if __name__ == "__main__":
    main()
