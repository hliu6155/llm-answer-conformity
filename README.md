# Answer Conformity in LLMs: Source Attribution, Repeated Rounds and Prompt Language

Data and code for a group research report in IN6208 Research Methods for Information Professionals (Nanyang Technological University, 2026).

## The study in brief

Four models (Qwen3.8-Flash, DeepSeek-V4.1-Flash, GPT-6 Luna and Claude Haiku 5.5) first answered an MMLU multiple-choice question correctly. They then saw the same wrong answer endorsed three times per round, attributed either to one repeated source (Condition A: `Source 17` × 3) or to three distinct sources (Condition B: `Source 17, 29, 43`), over three rounds, with English and Chinese prompts. Two runs were collected per model, language, condition and round.

Main results on the 49 items with complete data (item-level bootstrap, 10,000 resamples):

| Test | Estimate (percentage points) | 98.33% CI | Result |
| --- | --- | --- | --- |
| H1: B − A at Round 1 | +5.36 | [2.42, 8.55] | Supported |
| H2: change in B − A from Round 1 to Round 3 | −0.13 | [−2.30, 2.04] | Not supported |
| H3: Chinese − English difference in B − A at Round 1 | −1.53 | [−7.65, 4.59] | Not supported |

## Repository layout

| Path | Contents |
| --- | --- |
| `materials/items_main.csv` | 200 screening items (40 from each of astronomy, high school biology, high school geography, prehistory and nutrition), English and Chinese text, answer key, endorsed wrong option and source labels |
| `materials/items_pilot.csv`, `materials/items_reserve.csv` | 10 pilot items (labels 58, 61, 76) and 25 reserve items |
| `materials/prompt_templates.json`, `materials/prompt_builder.py` | Prompt templates (no system prompt) and the two response-validity rules |
| `materials/build_materials.py`, `materials/build_summary.json` | Rebuilds the item lists from the MMLU and MMMLU files (seed 6208) and records the alignment checks |
| `materials/example_prompts.md` | One item rendered in full for both languages and conditions |
| `run_experiment.py` | Data collection: model check, pilot, screening, selection and exposure, with automatic resumption |
| `keys.env.example` | Empty template for `keys.env`; API keys are never committed |
| `results/` | Raw logs of every call as gzip-compressed JSON Lines (`pilot_screening`, `pilot_exposure`, `screening`, `exposure`; `.jsonl.gz`), plus `selection.csv`, `settings.json` and `pilot_report.txt` |
| `analysis/analyze.py` | Confirmatory tests, worst-case bounds, sensitivity analysis and descriptive tables |
| `analysis/report_checks.py` | Collection, data-quality and persistence counts quoted in the report |
| `analysis/outputs/` | The outputs of both scripts on the data in `results/` |

## Reproducing the analysis

Python 3.11 or later:

```
pip install pandas numpy matplotlib
python analysis/analyze.py --results results --materials materials --out analysis/outputs
python analysis/report_checks.py results > analysis/outputs/report_checks.json
```

## Re-running data collection

Copy `keys.env.example` to `keys.env` (next to `run_experiment.py`) and add your own API keys; `QWEN_HOST` is the host name of your Alibaba Cloud Model Studio endpoint. Then run `python run_experiment.py all` (about 10,000 calls). `keys.env` is excluded by `.gitignore`, so it stays on your computer. Model aliases such as `deepseek-flash` may point to newer versions later; `results/settings.json` records the settings used and the model identifiers returned on 3 November 2026.

## Fields in the JSONL logs

Both analysis scripts read the `.jsonl.gz` files directly; `gunzip -k results/exposure.jsonl.gz` gives a plain copy. Each line is one model call. `key` identifies the call (stage, item, model, language, run, condition, round); `messages` is the exact conversation sent; `response` is the raw reply; `parsed` is the extracted letter; `valid` and `valid_strict` apply the main and strict validity rules; `answer`, `endorsed` and `labels` describe the item and condition; `conform` is 1 when the reply selects the endorsed wrong option; `model_returned`, `usage`, `attempts`, `latency_s` and `time` record the API's returned model identifier, token usage, attempt count, latency and UTC timestamp. When a call key appears more than once, the last record is the one used.

## Sources

Item texts come from the MMLU test set (Hendrycks et al., 2021, *Measuring Massive Multitask Language Understanding*, ICLR) and its professional Chinese translation in MMMLU (OpenAI, 2024); they remain under their original licences. The repository contains no API keys or account credentials.
