#!/usr/bin/env python3
"""IN6208 answer-conformity experiment runner.

One command runs everything:   python run_experiment.py all
Stages can also be run one at a time:
    check   - test each API once, pick working endpoint/model/settings, freeze them
    pilot   - 10 pilot items: screening (2 runs) + exposure (run 1); stops if format compliance is poor
    screen  - 200 main items x 4 models x 2 languages x 2 runs
    select  - eligibility (all 16 baselines correct) and seeded selection (50 + 10 spares)
    expose  - 3-round exposure for selected items, both conditions, both languages, both runs
    status  - progress summary
Everything is appended to results/*.jsonl, so re-running continues where it stopped.
Only needs the `requests` package (included in Anaconda).
"""
import argparse
import csv
import datetime
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
MAT = os.path.join(HERE, "materials")
sys.path.insert(0, MAT)
from prompt_builder import (load_templates, baseline_messages, round_messages,  # noqa: E402
                            parse_answer, parse_answer_strict)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SEED = 6208
N_SELECT, N_SPARE = 50, 10
LANGS, RUNS, CONDS, ROUNDS = ("en", "zh"), (1, 2), ("A", "B"), (1, 2, 3)

# ---------------------------------------------------------------------------------------------
# Model registry. "candidates" are tried in order by `check`; the first one that returns a valid
# single-letter answer in both languages is frozen into results/settings.json.
# ---------------------------------------------------------------------------------------------
MODELS = {
    "Qwen3.8-Flash": {
        "api": "openai", "key": "QWEN_API_KEY",
        "bases": ["https://{QWEN_HOST}/compatible-mode/v1", "https://{QWEN_HOST}/v1", "https://{QWEN_HOST}",
                  "https://dashscope.aliyuncs.com/compatible-mode/v1"],
        "models": ["qwen3.8-flash", "Qwen3.8-Flash", "qwen3.8-flash-latest"],
        "token_param": "max_tokens",
        "extras": [{"enable_thinking": False}, {}, {"_max": 2048}],
    },
    "DeepSeek-V4.1-Flash": {
        "api": "anthropic", "key": "DEEPSEEK_API_KEY",
        "bases": ["https://api.deepseek.com/anthropic"],
        "models": ["deepseek-flash", "deepseek-v4.1-flash"],
        "extras": [{"thinking": {"type": "disabled"}}, {}, {"_max": 2048}],
    },
    "GPT-6 Luna": {
        "api": "openai", "key": "OPENAI_API_KEY",
        "bases": ["https://api.openai.com/v1"],
        "models": ["gpt-6-luna"],
        "token_param": "max_completion_tokens",
        "extras": [{"reasoning_effort": "none"}, {"reasoning_effort": "minimal"},
                   {"_api": "responses", "reasoning": {"effort": "none"}},
                   {"_api": "responses", "reasoning": {"effort": "minimal"}},
                   {"reasoning_effort": "low", "_max": 2048}, {"_max": 2048},
                   {"_api": "responses", "reasoning": {"effort": "low"}, "_max": 2048}],
    },
    "Claude Haiku 5.5": {
        "api": "anthropic", "key": "ANTHROPIC_API_KEY",
        "bases": ["https://api.anthropic.com"],
        "models": ["claude-haiku-5-5"],
        "extras": [{"thinking": {"type": "disabled"}}, {}, {"_max": 2048}],
    },
}
DEFAULT_MAX_TOKENS = 1024  # only used if a model reasons; a one-letter reply costs ~1-3 tokens
CHECK_ITEM = {
    "item_id": "CHECK", "answer": "B", "endorsed_wrong": "A", "a_labels": "17|17|17", "b_labels": "17|29|43",
    "question_en": "What is 2 + 2?", "A_en": "3", "B_en": "4", "C_en": "5", "D_en": "6",
    "question_zh": "2 + 2 等于多少？", "A_zh": "3", "B_zh": "4", "C_zh": "5", "D_zh": "6",
}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def load_keys():
    path = os.path.join(HERE, "keys.env")
    if os.path.exists(path):
        with open(path, encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------------------------
class ConfigError(Exception):
    pass


def _post(api, base, key, model, messages, max_tokens, extra, token_param="max_tokens", timeout=90):
    extra = {k: v for k, v in extra.items() if not k.startswith("_")}
    if api == "anthropic":
        url = base.rstrip("/") + "/v1/messages"
        headers = {"x-api-key": key, "Authorization": "Bearer " + key,
                   "anthropic-version": "2023-06-01", "content-type": "application/json"}
        if "api.anthropic.com" in base:
            headers.pop("Authorization")
        body = dict({"model": model, "max_tokens": max_tokens, "messages": messages}, **extra)
    elif api == "responses":
        url = base.rstrip("/") + "/responses"
        headers = {"Authorization": "Bearer " + key, "content-type": "application/json"}
        body = dict({"model": model, "max_output_tokens": max_tokens, "input": messages}, **extra)
    else:
        url = base.rstrip("/") + "/chat/completions"
        headers = {"Authorization": "Bearer " + key, "content-type": "application/json"}
        body = dict({"model": model, token_param: max_tokens, "messages": messages}, **extra)
    r = requests.post(url, headers=headers, json=body, timeout=timeout)
    return r


def reasoning_tokens(api, usage):
    usage = usage or {}
    if api == "anthropic":
        return (usage.get("output_tokens_details") or {}).get("thinking_tokens")
    if api == "responses":
        return (usage.get("output_tokens_details") or {}).get("reasoning_tokens")
    return (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")


def _extract(api, data):
    if api == "anthropic":
        blocks = data.get("content") or []
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        thinking = sum(len(b.get("thinking", "") or "") for b in blocks if b.get("type") == "thinking")
        return text, data.get("model"), data.get("usage"), data.get("stop_reason"), thinking
    if api == "responses":
        text = "".join(c.get("text", "") for it in (data.get("output") or []) if it.get("type") == "message"
                       for c in (it.get("content") or []) if c.get("type") == "output_text")
        stop = data.get("status")
        if data.get("incomplete_details"):
            stop = f"{stop}:{data['incomplete_details'].get('reason')}"
        return text, data.get("model"), data.get("usage"), stop, 0
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    text = msg.get("content") or ""
    if isinstance(text, list):
        text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
    thinking = len(msg.get("reasoning_content") or "")
    return text, data.get("model"), data.get("usage"), choice.get("finish_reason"), thinking


def call_model(cfg, messages):
    """One logical request with the protocol's retry rules.
    - HTTP 429/529 (rate limit / overloaded): wait and resend, not counted as a retry (max 8 waits).
    - Network errors and other 5xx: technical failure, resend identical request at most twice.
    - Other 4xx: configuration problem, no retry.
    """
    attempts, waits, last_err = 0, 0, None
    while True:
        attempts += 1
        t0 = time.time()
        try:
            r = _post(cfg["api"], cfg["base"], cfg["key"], cfg["model"], messages,
                      cfg["max_tokens"], cfg["extra"], cfg.get("token_param", "max_tokens"))
            if r.status_code == 200:
                data = r.json()
                text, mret, usage, stop, thinking = _extract(cfg["api"], data)
                if text.strip():
                    return {"ok": True, "status": 200, "text": text, "model_returned": mret, "usage": usage,
                            "reasoning_tokens": reasoning_tokens(cfg["api"], usage),
                            "stop": stop, "thinking_chars": thinking, "attempts": attempts, "waits": waits,
                            "latency_s": round(time.time() - t0, 2)}
                # An empty reply is "no model response": a technical failure under the protocol.
                last_err = f"empty response (stop={stop}, thinking_chars={thinking})"
                if attempts >= 3:
                    return {"ok": False, "status": 200, "error": last_err, "attempts": attempts,
                            "waits": waits, "stop": stop, "usage": usage}
                time.sleep(2 * attempts)
                continue
            code, body = r.status_code, r.text[:500]
            if code in (429, 529) and waits < 8:
                waits += 1
                attempts -= 1
                ra = r.headers.get("retry-after")
                time.sleep(float(ra) if ra and ra.replace(".", "", 1).isdigit() else min(60, 2 ** waits))
                continue
            if code >= 500 or code in (408, 409, 429, 529):
                last_err = f"HTTP {code}: {body}"
            else:
                return {"ok": False, "status": code, "error": f"HTTP {code}: {body}", "attempts": attempts,
                        "waits": waits, "config_error": True}
        except Exception as e:  # network error, timeout, bad JSON
            last_err = f"{type(e).__name__}: {str(e)[:300]}"
        if attempts >= 3:  # first try + 2 retries
            return {"ok": False, "status": None, "error": last_err, "attempts": attempts, "waits": waits}
        time.sleep(2 * attempts)


# ---------------------------------------------------------------------------------------------
# Results files
# ---------------------------------------------------------------------------------------------
class JsonlStore:
    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def load(self):
        recs = []
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            recs.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass  # a line cut off by a crash; it is simply redone
        return recs

    def append(self, rec):
        with self.lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()


def latest_by_key(recs):
    out = {}
    for r in recs:
        out[r["key"]] = r  # later lines win (a successful redo replaces an earlier failure)
    return out


# ---------------------------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------------------------
def settings_path(res):
    return os.path.join(res, "settings.json")


def load_settings(res, names):
    p = settings_path(res)
    if not os.path.exists(p):
        raise SystemExit("No results/settings.json yet. Run:  python run_experiment.py check")
    with open(p, encoding="utf-8") as f:
        s = json.load(f)
    missing = [n for n in names if n not in s]
    if missing:
        raise SystemExit(f"settings.json has no entry for {missing}. Run the check stage again.")
    for n in names:
        s[n]["key"] = os.environ.get(MODELS[n]["key"], "")
        if not s[n]["key"]:
            raise SystemExit(f"Missing API key {MODELS[n]['key']} for {n} (put it in keys.env)")
    return s


def stage_check(res, names, T):
    out = {}
    if os.path.exists(settings_path(res)):
        with open(settings_path(res), encoding="utf-8") as f:
            out = json.load(f)
    host = os.environ.get("QWEN_HOST", "")
    for name in names:
        spec = MODELS[name]
        key = os.environ.get(spec["key"], "")
        print(f"\n== {name}")
        if not key:
            print(f"   missing key {spec['key']} in keys.env")
            continue
        found = None
        for base in spec["bases"]:
            base = base.replace("{QWEN_HOST}", host)
            if "{" in base or base.startswith("https:///"):
                continue
            for model in spec["models"]:
                for extra in spec["extras"]:
                    cfg = {"api": extra.get("_api", spec["api"]), "base": base, "key": key, "model": model,
                           "max_tokens": extra.get("_max", DEFAULT_MAX_TOKENS), "extra": extra,
                           "token_param": spec.get("token_param", "max_tokens")}
                    answers = []
                    for lang in LANGS:
                        r = call_model(cfg, baseline_messages(T, CHECK_ITEM, lang))
                        answers.append(r)
                        if not r["ok"]:
                            break
                    shown = [(a.get("text") if a["ok"] else a.get("error", "")[:160]) for a in answers]
                    print(f"   try base={base} model={model} extra={extra}: {shown}")
                    if all(a["ok"] and parse_answer(a["text"]) is not None for a in answers):
                        found = {k: v for k, v in cfg.items() if k != "key"}
                        found["model_returned"] = answers[0].get("model_returned")
                        found["checked_at"] = now()
                        break
                    if answers and not answers[-1]["ok"] and answers[-1].get("status") in (401, 403):
                        break  # bad key: no point trying other settings on this endpoint
                if found:
                    break
            if found:
                break
        if found:
            out[name] = found
            print(f"   OK -> frozen: model={found['model']} (API reports {found['model_returned']}), "
                  f"extra={found['extra']}, max_tokens={found['max_tokens']}")
        else:
            print("   FAILED: no working setting. Send the lines above to Claude.")
    with open(settings_path(res), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    ok = [n for n in names if n in out]
    print(f"\nWorking models: {ok}")
    return len(ok) == len(names)


def run_tasks(tasks, worker, names, label, workers_per_model):
    """Run (model, fn) tasks with a per-model concurrency limit, printing progress."""
    sems = {n: threading.Semaphore(workers_per_model) for n in names}
    total, done, errors = len(tasks), 0, 0
    lock = threading.Lock()
    t0 = time.time()
    bad = {n: 0 for n in names}

    def wrap(task):
        name = task["model"]
        if bad[name] >= 20:
            return {"skipped": True}
        with sems[name]:
            return worker(task)

    if total == 0:
        print(f"[{label}] nothing to do")
        return
    with ThreadPoolExecutor(max_workers=workers_per_model * len(names)) as ex:
        futs = [ex.submit(wrap, t) for t in tasks]
        for fut in as_completed(futs):
            r = fut.result() or {}
            with lock:
                done += 1
                if r.get("error"):
                    errors += 1
                    if r.get("config_error"):
                        bad[r["model"]] += 1
                        if bad[r["model"]] == 20:
                            print(f"   !! {r['model']}: 20 configuration errors, skipping its remaining "
                                  f"requests. Last error: {r['error'][:200]}")
                if done % 50 == 0 or done == total:
                    el = time.time() - t0
                    eta = el / done * (total - done)
                    print(f"[{label}] {done}/{total} done, {errors} errors, ~{eta / 60:.1f} min left")


def screening_record(item, name, lang, run, cfg, T, stage):
    msgs = baseline_messages(T, item, lang)
    r = call_model(cfg, msgs)
    parsed = parse_answer(r.get("text")) if r["ok"] else None
    return {"key": f"S|{item['item_id']}|{name}|{lang}|{run}", "stage": stage, "item_id": item["item_id"],
            "model": name, "language": lang, "run": run, "answer": item["answer"],
            "response": r.get("text"), "parsed": parsed, "valid": parsed is not None,
            "valid_strict": parse_answer_strict(r.get("text")) is not None,
            "correct": parsed == item["answer"], "ok": r["ok"], "error": r.get("error"),
            "reasoning_tokens": r.get("reasoning_tokens"),
            "config_error": r.get("config_error", False),
            "model_returned": r.get("model_returned"), "usage": r.get("usage"), "stop": r.get("stop"),
            "thinking_chars": r.get("thinking_chars"), "attempts": r.get("attempts"), "waits": r.get("waits"),
            "latency_s": r.get("latency_s"), "time": now(), "messages": msgs}


def stage_screen(res, names, items, T, workers, stage="screen", fname="screening.jsonl"):
    S = load_settings(res, names)
    store = JsonlStore(os.path.join(res, fname))
    done = {k for k, r in latest_by_key(store.load()).items() if r.get("ok")}
    tasks = [{"model": n, "item": it, "lang": l, "run": run}
             for it in items for n in names for l in LANGS for run in RUNS
             if f"S|{it['item_id']}|{n}|{l}|{run}" not in done]
    random.Random(SEED + 11).shuffle(tasks)

    def worker(t):
        rec = screening_record(t["item"], t["model"], t["lang"], t["run"], S[t["model"]], T, stage)
        store.append(rec)
        return rec

    print(f"[{stage}] {len(tasks)} calls to make ({len(done)} already done)")
    run_tasks(tasks, worker, names, stage, workers)


def exposure_branch(item, name, lang, run, cond, baseline, cfg, T, store, done, stage):
    """Run the remaining rounds of one branch. `done` maps round -> earlier record."""
    earlier = []
    last = None
    for rnd in ROUNDS:
        key = f"E|{item['item_id']}|{name}|{lang}|{run}|{cond}|{rnd}"
        if rnd in done and done[rnd].get("ok"):
            earlier.append(done[rnd]["response"])
            continue
        msgs = round_messages(T, item, lang, cond, baseline=baseline, earlier=earlier, round_no=rnd)
        r = call_model(cfg, msgs)
        parsed = parse_answer(r.get("text")) if r["ok"] else None
        rec = {"key": key, "stage": stage, "item_id": item["item_id"], "model": name, "language": lang,
               "run": run, "condition": cond, "round": rnd, "answer": item["answer"],
               "endorsed": item["endorsed_wrong"], "labels": item["a_labels"] if cond == "A" else item["b_labels"],
               "baseline": baseline, "response": r.get("text"), "parsed": parsed, "valid": parsed is not None,
               "valid_strict": parse_answer_strict(r.get("text")) is not None,
               "reasoning_tokens": r.get("reasoning_tokens"),
               "conform": (None if parsed is None else int(parsed == item["endorsed_wrong"])),
               "kept_correct": (None if parsed is None else int(parsed == item["answer"])),
               "ok": r["ok"], "error": r.get("error"), "config_error": r.get("config_error", False),
               "model_returned": r.get("model_returned"), "usage": r.get("usage"), "stop": r.get("stop"),
               "thinking_chars": r.get("thinking_chars"), "attempts": r.get("attempts"), "waits": r.get("waits"),
               "latency_s": r.get("latency_s"), "time": now(), "messages": msgs}
        store.append(rec)
        last = rec
        if not r["ok"]:
            return rec  # protocol: a branch with no response stops; later rounds stay missing
        earlier.append(r.get("text") or "")
    return last or {}


def stage_expose(res, names, items, baselines, T, workers, runs=RUNS, stage="expose", fname="exposure.jsonl"):
    S = load_settings(res, names)
    store = JsonlStore(os.path.join(res, fname))
    prev = latest_by_key(store.load())
    tasks = []
    for it in items:
        for n in names:
            for l in LANGS:
                for run in runs:
                    b = baselines.get((it["item_id"], n, l, run))
                    if b is None:
                        continue  # no correct baseline for this cell (pilot only)
                    for c in CONDS:
                        done = {}
                        for rnd in ROUNDS:
                            rec = prev.get(f"E|{it['item_id']}|{n}|{l}|{run}|{c}|{rnd}")
                            if rec:
                                done[rnd] = rec
                        if len([1 for r in done.values() if r.get("ok")]) == len(ROUNDS):
                            continue
                        tasks.append({"model": n, "item": it, "lang": l, "run": run, "cond": c,
                                      "baseline": b, "done": done})
    random.Random(SEED + 23).shuffle(tasks)  # interleaves models, languages and A/B order

    def worker(t):
        return exposure_branch(t["item"], t["model"], t["lang"], t["run"], t["cond"], t["baseline"],
                               S[t["model"]], T, store, t["done"], stage)

    print(f"[{stage}] {len(tasks)} branches to run (3 rounds each)")
    run_tasks(tasks, worker, names, stage, workers)


def correct_baselines(recs, names):
    out = {}
    for r in latest_by_key(recs).values():
        if r.get("ok") and r.get("correct") and r["model"] in names:
            out[(r["item_id"], r["model"], r["language"], r["run"])] = r["response"]
    return out


def stage_select(res, names, items):
    recs = latest_by_key(JsonlStore(os.path.join(res, "screening.jsonl")).load())
    cells = len(names) * len(LANGS) * len(RUNS)
    rows, eligible = [], []
    for it in items:
        n_ok = n_correct = 0
        for n in names:
            for l in LANGS:
                for run in RUNS:
                    r = recs.get(f"S|{it['item_id']}|{n}|{l}|{run}")
                    if r and r.get("ok"):
                        n_ok += 1
                        n_correct += int(bool(r.get("correct")))
        elig = n_correct == cells
        rows.append({"item_id": it["item_id"], "subject": it["subject"], "responses": n_ok,
                     "correct": n_correct, "eligible": int(elig)})
        if elig:
            eligible.append(it["item_id"])
    if any(r["responses"] < cells for r in rows):
        print("   note: some screening calls are still missing; run `screen` again before selecting.")
    order = sorted(eligible)
    random.Random(SEED + 31).shuffle(order)
    rank = {iid: i + 1 for i, iid in enumerate(order)}
    for r in rows:
        k = rank.get(r["item_id"])
        r["selection_rank"] = k or ""
        r["status"] = ("selected" if k and k <= N_SELECT else
                       "spare" if k and k <= N_SELECT + N_SPARE else
                       "eligible_not_used" if k else "not_eligible")
    with open(os.path.join(res, "selection.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summ = {"screened": len(items), "eligible": len(eligible),
            "selected": min(N_SELECT, len(eligible)),
            "spares": max(0, min(N_SPARE, len(eligible) - N_SELECT)), "time": now()}
    with open(os.path.join(res, "selection_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summ, f, indent=2)
    print(f"[select] {summ}")
    return [r["item_id"] for r in rows if r["status"] in ("selected", "spare")]


def pilot_report(res, names):
    scr = latest_by_key(JsonlStore(os.path.join(res, "pilot_screening.jsonl")).load()).values()
    exp = latest_by_key(JsonlStore(os.path.join(res, "pilot_exposure.jsonl")).load()).values()
    lines, ok_all = [], True
    for n in names:
        s = [r for r in scr if r["model"] == n]
        e = [r for r in exp if r["model"] == n]
        allr = s + e
        valid = sum(1 for r in allr if r.get("valid"))
        errs = sum(1 for r in allr if not r.get("ok"))
        acc = sum(1 for r in s if r.get("correct")) / max(1, len(s))
        r1 = {c: [r["conform"] for r in e if r["round"] == 1 and r["condition"] == c and r["conform"] is not None]
              for c in CONDS}
        cr = {c: (sum(v) / len(v) if v else float("nan")) for c, v in r1.items()}
        vr = valid / max(1, len(allr))
        rt = [r.get("reasoning_tokens") or 0 for r in allr if r.get("ok")]
        mean_rt = sum(rt) / max(1, len(rt))
        bad = [r.get("response") for r in allr if r.get("ok") and not r.get("valid")][:3]
        lines.append(f"{n}: calls={len(allr)} valid={vr:.0%} errors={errs} baseline_accuracy={acc:.0%} "
                     f"round1_conformity A={cr['A']:.0%} B={cr['B']:.0%} mean_reasoning_tokens={mean_rt:.1f}"
                     + (f"  invalid examples={bad}" if bad else ""))
        if len(allr) == 0 or vr < 0.9 or errs > 0.05 * max(1, len(allr)):
            ok_all = False
    txt = "\n".join(lines)
    with open(os.path.join(res, "pilot_report.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print("\n[pilot report]\n" + txt)
    return ok_all


def stage_status(res):
    for fn in ("pilot_screening.jsonl", "pilot_exposure.jsonl", "screening.jsonl", "exposure.jsonl"):
        recs = list(latest_by_key(JsonlStore(os.path.join(res, fn)).load()).values())
        if not recs:
            continue
        by = {}
        for r in recs:
            d = by.setdefault(r["model"], [0, 0, 0])
            d[0] += 1
            d[1] += int(bool(r.get("valid")))
            d[2] += int(not r.get("ok"))
        print(f"{fn}: " + "; ".join(f"{m}: {v[0]} records, {v[1]} valid, {v[2]} failed" for m, v in by.items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["check", "pilot", "screen", "select", "expose", "all", "status"])
    ap.add_argument("--models", default=",".join(MODELS), help="comma-separated subset of model names")
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    ap.add_argument("--workers", type=int, default=4, help="parallel requests per model")
    ap.add_argument("--limit", type=int, default=0, help="testing only: use the first N items")
    ap.add_argument("--force", action="store_true", help="continue even if the pilot looks bad")
    ap.add_argument("--mock", default="", help=argparse.SUPPRESS)  # offline test server URL
    a = ap.parse_args()

    load_keys()
    if a.mock:  # route every model to a local fake API (used only to test this script)
        for spec in MODELS.values():
            spec["bases"] = [a.mock.rstrip("/") + ("/anthropic" if spec["api"] == "anthropic" else "/v1")]
            os.environ.setdefault(spec["key"], "test-key")
    names = [n.strip() for n in a.models.split(",") if n.strip()]
    unknown = [n for n in names if n not in MODELS]
    if unknown:
        raise SystemExit(f"Unknown model names {unknown}. Known: {list(MODELS)}")
    os.makedirs(a.results, exist_ok=True)
    T = load_templates(os.path.join(MAT, "prompt_templates.json"))
    main_items = read_csv(os.path.join(MAT, "items_main.csv"))
    pilot_items = read_csv(os.path.join(MAT, "items_pilot.csv"))
    if a.limit:
        main_items, pilot_items = main_items[:a.limit], pilot_items[:max(1, a.limit // 5)]
    by_id = {it["item_id"]: it for it in main_items}
    t0 = time.time()

    def do_pilot():
        stage_screen(a.results, names, pilot_items, T, a.workers, stage="pilot", fname="pilot_screening.jsonl")
        base = correct_baselines(JsonlStore(os.path.join(a.results, "pilot_screening.jsonl")).load(), names)
        stage_expose(a.results, names, pilot_items, base, T, a.workers, runs=(1,), stage="pilot",
                     fname="pilot_exposure.jsonl")
        return pilot_report(a.results, names)

    def do_expose():
        sel_path = os.path.join(a.results, "selection.csv")
        if not os.path.exists(sel_path):
            raise SystemExit("Run the select stage first.")
        chosen = [r["item_id"] for r in read_csv(sel_path) if r["status"] in ("selected", "spare")]
        base = correct_baselines(JsonlStore(os.path.join(a.results, "screening.jsonl")).load(), names)
        stage_expose(a.results, names, [by_id[i] for i in chosen if i in by_id], base, T, a.workers)

    if a.stage == "check":
        stage_check(a.results, names, T)
    elif a.stage == "pilot":
        do_pilot()
    elif a.stage == "screen":
        stage_screen(a.results, names, main_items, T, a.workers)
    elif a.stage == "select":
        stage_select(a.results, names, main_items)
    elif a.stage == "expose":
        do_expose()
    elif a.stage == "status":
        stage_status(a.results)
    elif a.stage == "all":
        if not stage_check(a.results, names, T):
            raise SystemExit("\nStopped: at least one model failed the check. Send the output above to Claude.")
        if not do_pilot() and not a.force:
            raise SystemExit("\nStopped after the pilot: format compliance or errors need a look. "
                             "Send results/pilot_report.txt to Claude (or rerun with --force).")
        stage_screen(a.results, names, main_items, T, a.workers)
        stage_screen(a.results, names, main_items, T, a.workers)  # second pass retries any failures
        stage_select(a.results, names, main_items)
        do_expose()
        do_expose()  # second pass retries any failed branches
        stage_status(a.results)
        print(f"\nAll done in {(time.time() - t0) / 60:.1f} min. Zip the results folder and upload it.")


if __name__ == "__main__":
    main()
