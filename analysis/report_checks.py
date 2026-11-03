"""Data-quality and descriptive counts quoted in the IN6208 2b report (Sections 4.1-4.3, 5.5; Appendix D).

    python report_checks.py path/to/results  > report_checks.json

Reads the raw logs written by run_experiment.py (pilot_*.jsonl, screening.jsonl, exposure.jsonl,
selection.csv). For every call key only the latest record counts, as in run_experiment.py.
"""
import collections
import csv
import gzip
import json
import os
import sys


def load(results, name):
    """Read results/<name>, or its gzip copy results/<name>.gz."""
    path = f"{results}/{name}"
    opener = open if os.path.exists(path) else gzip.open
    with opener(path if opener is open else path + ".gz", "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def latest(rows):
    out = {}
    for r in rows:
        out[r["key"]] = r
    return out


def main(results):
    out = {}
    raw = {n: load(results, f"{n}.jsonl") for n in ("pilot_screening", "pilot_exposure", "screening", "exposure")}

    # ---- collection windows, failures and retries (Section 4.3) ----
    for name, rows in raw.items():
        times = sorted(r["time"] for r in rows if r.get("time"))
        failed = [r for r in rows if not r.get("ok")]
        final = [r for r in latest(rows).values() if r.get("ok")]
        out[name] = {
            "window_utc": [times[0], times[-1]],
            "successful_calls": len(final),
            "calls_failed_after_retries": len(failed),
            "failed_by_model": dict(collections.Counter(r["model"] for r in failed)),
            "failure_messages": dict(collections.Counter(str(r.get("error"))[:60] for r in failed)),
            "calls_needing_in_call_retry": dict(collections.Counter(
                r["model"] for r in final if (r.get("attempts") or 1) > 1)),
            "rate_limit_waits": sum(r.get("waits") or 0 for r in final),
        }

    sel = list(csv.DictReader(open(f"{results}/selection.csv", encoding="utf-8-sig")))
    status = {s["item_id"]: s["status"] for s in sel}
    subject = {s["item_id"]: s["subject"] for s in sel}

    # ---- screening (Section 4.2, Table D1) ----
    scr = [r for r in latest(raw["screening"]).values() if r.get("ok")]
    acc = collections.defaultdict(lambda: [0, 0, 0])
    for r in scr:
        k = f"{r['model']} | {r['language']}"
        acc[k][0] += 1
        acc[k][1] += r.get("parsed") == r.get("answer")
        acc[k][2] += bool(r.get("valid_strict"))
    out["screening_accuracy_pct"] = {k: round(100 * v[1] / v[0], 2) for k, v in sorted(acc.items())}
    out["screening_strict_valid"] = sum(v[2] for v in acc.values())
    by_item = collections.defaultdict(list)
    for r in scr:
        by_item[r["item_id"]].append(r)
    inel = [i for i, s in status.items() if s == "not_eligible"]
    wrong = {i: [r for r in by_item[i] if r.get("parsed") != r.get("answer")] for i in inel}
    out["ineligible"] = {
        "items": len(inel),
        "missed_only_in_chinese": sum({r["language"] for r in w} == {"zh"} for w in wrong.values()),
        "missed_only_in_english": sum({r["language"] for r in w} == {"en"} for w in wrong.values()),
        "missed_by_one_model": sum(len({r["model"] for r in w}) == 1 for w in wrong.values()),
    }
    out["eligible_by_subject"] = dict(collections.Counter(subject[i] for i, s in status.items() if s != "not_eligible"))
    out["selected_by_subject"] = dict(collections.Counter(subject[i] for i, s in status.items() if s == "selected"))

    # ---- exposure (Sections 4.3 and 5.5, Table D4) ----
    ex = [r for r in latest(raw["exposure"]).values() if r.get("ok")]
    exs = [r for r in ex if status.get(r["item_id"]) == "selected"]
    out["exposure_selected_responses"] = len(exs)
    out["invalid_main_rule"] = [(r["item_id"], r["model"], r["language"], r["condition"], r["round"], r["response"][:200])
                                for r in exs if not r.get("valid")]
    strict = [r for r in exs if not r.get("valid_strict")]
    out["invalid_strict_rule_by_model_language"] = dict(collections.Counter(f"{r['model']} | {r['language']}" for r in strict))
    out["strict_invalid_responses"] = [(r["item_id"], r["model"], r["language"], r["run"], r["condition"], r["round"],
                                        r["response"][:300]) for r in strict]
    out["other_error_responses"] = sum(1 for r in exs if r.get("valid") and r["parsed"] not in (r["answer"], r["endorsed"]))

    branches = collections.defaultdict(dict)
    for r in exs:
        if r.get("valid"):
            branches[(r["item_id"], r["model"], r["language"], r["run"], r["condition"])][r["round"]] = r
    n = c1 = still = back = new = 0
    by_cond = collections.Counter()
    for key, d in branches.items():
        if 1 in d and 3 in d:
            n += 1
            r1 = d[1]["parsed"] == d[1]["endorsed"]
            r3 = d[3]["parsed"] == d[3]["endorsed"]
            if r1:
                c1 += 1
                by_cond[key[4]] += 1
                still += r3
                back += d[3]["parsed"] == d[3]["answer"]
            elif r3:
                new += 1
    out["persistence"] = {"branches": n, "conformed_at_round1": c1, "by_condition": dict(by_cond),
                          "still_conforming_round3": still, "back_to_correct_round3": back,
                          "first_conformed_after_round1": new}
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
