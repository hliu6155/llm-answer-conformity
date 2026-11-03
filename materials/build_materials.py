"""Build the item lists, review workbook and prompt examples for the IN6208
answer-conformity experiment from the English MMLU test set and its Chinese
MMMLU translation (OpenAI simple-evals files mmlu.csv and mmlu_ZH-CN.csv).

    python build_materials.py --en mmlu.csv --zh mmlu_ZH-CN.csv --out materials

Everything is reproducible from --seed. Re-running with the same inputs and
seed gives byte-identical item lists.
"""
import argparse
import itertools
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from prompt_builder import load_templates, baseline_messages, round_messages  # noqa: E402

DEFAULT_SUBJECTS = ["astronomy", "high_school_biology", "high_school_geography",
                    "prehistory", "nutrition"]
CODES = {"astronomy": "AST", "high_school_biology": "BIO", "high_school_geography": "GEO",
         "prehistory": "PRE", "nutrition": "NUT"}
MODELS = ["Qwen3.8-Flash", "DeepSeek-V4.1-Flash", "GPT-6 Luna", "Claude Haiku 5.5"]
LETTERS = ["A", "B", "C", "D"]
FIELDS = ["Question", "A", "B", "C", "D"]


def read_mmlu(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)
    first = df.columns[0]
    df = df.rename(columns={first: "idx"})
    need = {"idx", "Question", "A", "B", "C", "D", "Answer", "Subject"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"{path}: missing columns {missing}")
    df["idx"] = df["idx"].astype(int)
    for c in FIELDS + ["Answer", "Subject"]:
        df[c] = df[c].astype(str).str.strip()
    return df


def align(en, zh):
    m = en.merge(zh, on=["Subject", "idx"], suffixes=("_en", "_zh"), how="outer", indicator=True)
    report = {
        "rows_en": int(len(en)), "rows_zh": int(len(zh)),
        "only_en": int((m["_merge"] == "left_only").sum()),
        "only_zh": int((m["_merge"] == "right_only").sum()),
    }
    both = m[m["_merge"] == "both"].copy()
    report["matched"] = int(len(both))
    report["answer_key_agreement"] = float((both["Answer_en"] == both["Answer_zh"]).mean())
    return both.drop(columns="_merge"), report


def usable(row):
    """Prespecified technical exclusions (before any model sees an item)."""
    if row["Answer_en"] not in LETTERS or row["Answer_en"] != row["Answer_zh"]:
        return False
    for lang in ("en", "zh"):
        vals = [row[f"{f}_{lang}"] for f in FIELDS]
        if any(v == "" for v in vals):
            return False
        opts = vals[1:]
        if len(set(opts)) < 4:          # duplicated options make letters ambiguous
            return False
    return True


def balanced_perms(labels, n, rng):
    perms = list(itertools.permutations(labels))
    pool = (perms * (n // len(perms) + 1))[:n]
    order = rng.permutation(len(pool))
    return [pool[i] for i in order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--en", required=True)
    ap.add_argument("--zh", required=True)
    ap.add_argument("--out", default="materials")
    ap.add_argument("--seed", type=int, default=6208)
    ap.add_argument("--subjects", default=",".join(DEFAULT_SUBJECTS))
    ap.add_argument("--n-main", type=int, default=40)
    ap.add_argument("--n-pilot", type=int, default=2)
    ap.add_argument("--n-reserve", type=int, default=5)
    args = ap.parse_args()

    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    os.makedirs(args.out, exist_ok=True)
    T = load_templates(os.path.join(HERE, "prompt_templates.json"))
    rng = np.random.default_rng(args.seed)

    en, zh = read_mmlu(args.en), read_mmlu(args.zh)
    both, report = align(en, zh)
    # Every chosen subject must line up item by item (identical answer keys in both files).
    # Mismatches elsewhere (MMMLU's high_school_world_history block is shifted) do not matter here.
    agree = (both["Answer_en"] == both["Answer_zh"]).groupby(both["Subject"]).mean()
    report["subjects_with_key_mismatches"] = {s: round(float(a), 4) for s, a in agree.items() if a < 1}
    for subj in subjects:
        if subj not in agree.index:
            raise SystemExit(f"{subj}: not found in both files")
        if agree[subj] < 1:
            raise SystemExit(f"{subj}: English and Chinese answer keys disagree on "
                             f"{(1 - agree[subj]):.1%} of items; pick another subject")

    rows = []
    sizes = {}
    for subj in subjects:
        sub = both[both["Subject"] == subj].sort_values("idx")
        ok = sub[sub.apply(usable, axis=1)]
        need = args.n_pilot + args.n_main + args.n_reserve
        sizes[subj] = {"test_items": int(len(sub)), "usable": int(len(ok))}
        if len(ok) < need:
            raise SystemExit(f"{subj}: only {len(ok)} usable items, need {need}")
        perm = rng.permutation(len(ok))
        picked = ok.iloc[perm[:need]]
        code = CODES.get(subj, subj[:3].upper())
        for k, (_, r) in enumerate(picked.iterrows()):
            if k < args.n_pilot:
                role, item_id = "pilot", f"{code}-P{k + 1}"
            elif k < args.n_pilot + args.n_main:
                role, item_id = "main", f"{code}-{k - args.n_pilot + 1:02d}"
            else:
                role, item_id = "reserve", f"{code}-R{k - args.n_pilot - args.n_main + 1}"
            wrong = [L for L in LETTERS if L != r["Answer_en"]]
            rows.append({
                "item_id": item_id, "set": role, "subject": subj, "subject_index": int(r["idx"]),
                "answer": r["Answer_en"], "endorsed_wrong": str(rng.choice(wrong)),
                "question_en": r["Question_en"], "A_en": r["A_en"], "B_en": r["B_en"],
                "C_en": r["C_en"], "D_en": r["D_en"],
                "question_zh": r["Question_zh"], "A_zh": r["A_zh"], "B_zh": r["B_zh"],
                "C_zh": r["C_zh"], "D_zh": r["D_zh"],
            })

    items = pd.DataFrame(rows)
    # Counterbalancing: B gets one of the six label orders (balanced); A repeats B's first label,
    # so Statement 1 is identical across conditions and only distinctness differs.
    for role, labels in (("main", T["labels"]["main"]), ("reserve", T["labels"]["main"]),
                         ("pilot", T["labels"]["pilot"])):
        mask = items["set"].eq(role)
        perms = balanced_perms(labels, int(mask.sum()), rng)
        items.loc[mask, "b_labels"] = ["|".join(map(str, p)) for p in perms]
        items.loc[mask, "a_labels"] = ["|".join([str(p[0])] * 3) for p in perms]

    cols = ["item_id", "set", "subject", "subject_index", "answer", "endorsed_wrong",
            "a_labels", "b_labels", "question_en", "A_en", "B_en", "C_en", "D_en",
            "question_zh", "A_zh", "B_zh", "C_zh", "D_zh"]
    items = items[cols]
    for role, fname in (("main", "items_main.csv"), ("pilot", "items_pilot.csv"),
                        ("reserve", "items_reserve.csv")):
        items[items["set"] == role].to_csv(os.path.join(args.out, fname), index=False,
                                           encoding="utf-8-sig")

    # Screening schedule: every main item x model x language x run, in a random interleaved order.
    trials = [{"item_id": i, "model": m, "language": l, "run": r}
              for i in items.loc[items["set"] == "main", "item_id"]
              for m in MODELS for l in ("en", "zh") for r in (1, 2)]
    trials = pd.DataFrame(trials).iloc[rng.permutation(len(trials))].reset_index(drop=True)
    trials.insert(0, "order", range(1, len(trials) + 1))
    trials.to_csv(os.path.join(args.out, "screening_schedule.csv"), index=False, encoding="utf-8-sig")

    write_review_workbook(items, os.path.join(args.out, "bilingual_review.xlsx"))
    write_examples(T, items, os.path.join(args.out, "example_prompts.md"))

    main_items = items[items["set"] == "main"]
    summary = {
        "seed": args.seed, "subjects": subjects, "alignment": report, "subject_sizes": sizes,
        "counts": items["set"].value_counts().to_dict(),
        "main_answer_letters": main_items["answer"].value_counts().sort_index().to_dict(),
        "main_endorsed_letters": main_items["endorsed_wrong"].value_counts().sort_index().to_dict(),
        "main_b_label_orders": main_items["b_labels"].value_counts().sort_index().to_dict(),
        "screening_calls": int(len(trials)),
    }
    with open(os.path.join(args.out, "build_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    for fn in ("prompt_templates.json", "prompt_builder.py", "build_materials.py", "README.md"):
        shutil.copy(os.path.join(HERE, fn), os.path.join(args.out, fn))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def write_review_workbook(items, path):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation

    import re

    wb = Workbook()
    ws = wb.active
    ws.title = "Review"
    review = items.copy()
    # Spread each subject's items across the three reviewers.
    review["reviewer"] = (review.groupby("subject").cumcount() % 3 + 1).map(lambda k: f"Reviewer {k}")
    country = re.compile(r"\b(United States|U\.S\.|USA|Americans?|America|UK|U\.K\.|United Kingdom|"
                         r"Britain|British|England|Congress|Supreme Court|FDA|USDA|NHS|federal)\b")
    timely = re.compile(r"\b(1[89]\d\d|20\d\d)\b|\b(currently|current|today|recent|recently|latest|nowadays)\b",
                        re.I)
    crossref = re.compile(r"\b(all|none|both) (of )?(the )?(above|these)\b|\bboth [A-D] and [A-D]\b", re.I)

    def flags(r):
        text = " ".join(r[c] for c in ("question_en", "A_en", "B_en", "C_en", "D_en"))
        out = []
        if country.search(text):
            out.append("country-specific?")
        if timely.search(text):
            out.append("time-specific?")
        if crossref.search(text):
            out.append("option refers to other options")
        if len(r["question_en"]) > 600:
            out.append("long item")
        return "; ".join(out)

    review["auto_flags"] = review.apply(flags, axis=1)
    checks = ["Same meaning EN/ZH? (Y/N)", "Answer key correct? (Y/N)",
              "Free of culture/time-specific content? (Y/N)", "Formatting OK? (Y/N)",
              "Decision (keep/replace)", "Notes"]
    head = ["item_id", "set", "reviewer", "subject", "answer", "endorsed_wrong", "auto_flags",
            "question_en", "A_en", "B_en", "C_en", "D_en",
            "question_zh", "A_zh", "B_zh", "C_zh", "D_zh"] + checks
    ws.append(head)
    for _, r in review.iterrows():
        ws.append([r[c] for c in head[:17]] + [""] * len(checks))
    bold = Font(bold=True)
    fill = PatternFill("solid", fgColor="DDEBF7")
    for c in ws[1]:
        c.font = bold
        c.fill = fill
        c.alignment = Alignment(wrap_text=True, vertical="top")
    widths = {"A": 10, "B": 9, "C": 12, "D": 20, "E": 8, "F": 9, "G": 18, "H": 48, "I": 18,
              "J": 18, "K": 18, "L": 18, "M": 40, "N": 16, "O": 16, "P": 16, "Q": 16,
              "R": 14, "S": 14, "T": 16, "U": 12, "V": 14, "W": 30}
    for col, w in widths.items():
        ws.column_dimensions[col].width = w
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
    yn = DataValidation(type="list", formula1='"Y,N"', allow_blank=True)
    kr = DataValidation(type="list", formula1='"keep,replace"', allow_blank=True)
    ws.add_data_validation(yn)
    ws.add_data_validation(kr)
    last = ws.max_row
    yn.add(f"R2:U{last}")
    kr.add(f"V2:V{last}")
    ws.freeze_panes = "B2"
    ws.auto_filter.ref = ws.dimensions

    g = wb.create_sheet("Guide")
    guide = [
        "How to review (each reviewer checks the rows marked with their name)",
        "1. Same meaning EN/ZH: the Chinese question and all four options say the same thing as the English.",
        "2. Answer key correct: the 'answer' letter is really the correct option (check the English).",
        "3. Free of culture/time-specific content: no content tied to one country's laws/policies or to a date that may have changed.",
        "4. Formatting OK: no broken symbols, missing text, or options that refer to a figure.",
        "The auto_flags column is a keyword hint only (e.g. a country name or a year); it never decides keep/replace.",
        "If any check is N, set Decision = replace and write why in Notes.",
        "Replacement rule: a replaced main item is swapped for the next unused reserve item of the same subject (R1, then R2, ...).",
        "Pilot items (set = pilot) are reviewed the same way but are never used in the main study.",
    ]
    for line in guide:
        g.append([line])
    g["A1"].font = bold
    g.column_dimensions["A"].width = 120
    wb.save(path)


def write_examples(T, items, path):
    it = items[items["set"] == "main"].iloc[0].to_dict()
    pil = items[items["set"] == "pilot"].iloc[0].to_dict()
    ans = it["answer"]
    out = [f"# Example prompts (item {it['item_id']}, correct = {ans}, endorsed wrong = {it['endorsed_wrong']})", ""]

    def block(title, msgs):
        out.append(f"## {title}")
        out.append("")
        for m in msgs:
            out.append(f"**{m['role']}:**")
            out.append("")
            out.append("```")
            out.append(m["content"])
            out.append("```")
            out.append("")

    block("Screening call, English", baseline_messages(T, it, "en"))
    block("Screening call, Chinese", baseline_messages(T, it, "zh"))
    block(f"Round 1, Condition A (labels {it['a_labels']}), English",
          round_messages(T, it, "en", "A", baseline=ans, earlier=[], round_no=1))
    block(f"Round 1, Condition B (labels {it['b_labels']}), English",
          round_messages(T, it, "en", "B", baseline=ans, earlier=[], round_no=1))
    block(f"Round 1, Condition B, Chinese",
          round_messages(T, it, "zh", "B", baseline=ans, earlier=[], round_no=1))
    block("Round 3, Condition A, English (earlier answers kept in history; shown here as "
          f"{it['endorsed_wrong']} then {ans} for illustration)",
          round_messages(T, it, "en", "A", baseline=ans, earlier=[it["endorsed_wrong"], ans], round_no=3))
    block(f"Pilot item {pil['item_id']}, Round 1, Condition B (pilot labels {pil['b_labels']}), English",
          round_messages(T, pil, "en", "B", baseline=pil["answer"], earlier=[], round_no=1))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))


if __name__ == "__main__":
    main()
