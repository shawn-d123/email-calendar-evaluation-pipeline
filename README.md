<div align="center">

# 📧 Email Calendar Evaluation Pipeline

![Python](https://img.shields.io/badge/Python-3.14-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-Qwen_3_8B-black?style=for-the-badge&logo=ollama&logoColor=white)
![Licence](https://img.shields.io/badge/Licence-MIT-green?style=for-the-badge)

**Rules vs. LLM — who extracts calendar events from messy emails better?**

A Python evaluation pipeline that benchmarks a rule-based baseline against a local LLM<br>on structured extraction from raw email-style messages.

[Getting Started](#-quickstart) · [Results](#-results-at-a-glance) · [Dataset](#-dataset) · [Methods](#-methods)

</div>

---

## 🧠 The Problem

Assistant-style products need to pull actionable structure out of messy text — calendar events, reminders, deadlines, action items. That gets hard fast when messages contain multiple dates, relative time expressions, cancellations, ambiguous phrasing, and inconsistent formatting.

This project builds a repeatable evaluation pipeline around that problem rather than just running a single model and eyeballing the output.

```
📨 Raw Email  ──▶  🔧 Rule-Based + LLM Extractor  ──▶  📊 Evaluate & Score  ──▶  📈 Compare & Visualise
```

---

## 🔍 What It Extracts

| Field | Description |
|:------|:------------|
| `calendar_event_required` | Should an event be created? |
| `event_category` | Type of event |
| `event_date` | When it happens |
| `start_time` / `end_time` | Time window |
| `action_required` | Does the reader need to do something? |
| `action_type` | What kind of action |
| `action_deadline` | By when |
| `summary` | Short description |

Two extraction methods are benchmarked side-by-side against a **40-row labelled benchmark** (20 synthetic + 20 Enron-derived).

---

## 📊 Results at a Glance

| Metric | Rule-Based Baseline | Qwen 3 8B | Edge |
|:-------|:-------------------:|:---------:|:----:|
| **Avg latency** | `95 ms` | `24,516 ms` | 🟦 Baseline |
| **Calendar event F1** | `0.917` | `0.902` | 🟦 Baseline |
| **Action required F1** | `0.760` | `0.964` | 🟧 Qwen |
| **Event category macro F1** | `0.776` | `0.733` | 🟦 Baseline |
| **Action type macro F1** | `0.629` | `0.819` | 🟧 Qwen |
| **Event date accuracy** | `0.700` | `0.875` | 🟧 Qwen |
| **Action deadline accuracy** | `0.800` | `0.800` | ⬜ Tie |

> [!TIP]
> **Bottom line:** Neither system wins outright. The strongest practical outcome is a **hybrid approach** — let rules handle the easy stuff fast, route the harder cases to an LLM.

<details>
<summary>📈 <strong>View charts</strong></summary>
<br>

### Metric Comparison
![Metric Comparison](outputs/charts/metric_comparison.png)

### Failure Count by Field
![Failure Comparison](outputs/charts/failure_comparison.png)

### Latency
![Latency Comparison](outputs/charts/latency_comparison.png)

</details>

---

## 📁 Dataset

### Benchmark composition

| Source | Rows | Examples |
|:-------|:----:|:---------|
| **Synthetic** | 20 | Trip reminders, parent meetings, club updates, payment deadlines, cancellations |
| **Enron-derived** | 20 | Real corporate email language, messier formatting, less predictable structure |

### Enron data pipeline

```
1,000 raw emails  →  -32 dupes  →  778 clean  →  120 candidates  →  20 labelled
```

> Raw Enron maildir is not committed (too large). The repo includes cleaned artefacts, labelled data, and all outputs.

---

## ⚙️ Methods

### 🟦 Rule-based baseline

Keyword matching, regex, date parsing, and priority rules. Fast, interpretable, easy to debug. Falls over when wording is indirect or when multiple temporal cues compete.

### 🟧 Qwen 3 8B (Ollama)

Local LLM with structured JSON output, fixed schema, and deterministic prompting (`temperature=0`). Better at reading between the lines on action intent and date/time extraction. The trade-off? **~257× slower** (95 ms vs 24.5 seconds per message).

---

## 📏 Evaluation Approach

| Type | Fields | Metrics |
|:-----|:-------|:--------|
| **Classification** | `calendar_event_required`, `action_required`, `event_category`, `action_type` | Precision, recall, F1, macro F1 |
| **Extraction** | `event_date`, `start_time`, `end_time`, `action_deadline` | Exact match accuracy |
| **Operational** | — | Average latency (ms) |
| **Error analysis** | All fields | Field-level failure counts |

---

## 🗂️ Extraction Schema

<details>
<summary>Event categories</summary>

`none` · `meeting_admin` · `club_activity` · `trip` · `payment_deadline` · `cancellation_change` · `reminder_other`
</details>

<details>
<summary>Action types</summary>

`none` · `attend` · `pay` · `reply_confirm` · `bring_item` · `submit_form`
</details>

---

## 🚀 Quickstart

### Requirements

- **Python 3.14**
- [**Ollama**](https://ollama.com/) with Qwen 3 8B pulled locally (LLM evaluation only)

### Install

```bash
pip install -r requirements.txt
```

### Run the pipeline

```bash
# 1. Validate and split the dataset
python src/build_dataset.py

# 2. Run the rule-based baseline
python src/baseline_extractor.py

# 3. Evaluate baseline
python src/evaluate_predictions.py
python src/analyse_failures.py

# 4. Run the LLM extractor (requires Ollama + Qwen 3 8B)
python src/llm_extractor.py

# 5. Evaluate LLM output
#    Update file paths in evaluate_predictions.py and analyse_failures.py
#    to point to the Qwen output, then:
python src/evaluate_predictions.py
python src/analyse_failures.py

# 6. Generate comparison charts
python src/generate_visualisations.py
```

> [!NOTE]
> Steps 3 and 5 require you to update the input file paths in the evaluation scripts depending on which extractor output you're evaluating. This is documented in the script comments.

<details>
<summary>🔄 <strong>Rebuild the Enron data stages</strong></summary>

```bash
python src/extract_enron_messages.py
python src/clean_real_world_data.py
python src/select_enron_eval_candidates.py
python src/build_enron_label_template.py
python src/append_enron_labels.py
```

</details>

---

## 🗃️ Repo Structure

```
email-calendar-evaluation-pipeline/
│
├── 📂 data/
│   ├── raw/                          # Enron maildir (local only, not committed)
│   ├── intermediate/
│   │   ├── enron_messages_raw.csv
│   │   ├── enron_messages_clean.csv
│   │   └── enron_eval_candidates.csv
│   └── processed/
│       ├── eval_dataset.csv
│       ├── dev_dataset.csv
│       ├── test_dataset.csv
│       ├── enron_label_template.csv
│       └── enron_label_template_labeled.csv
│
├── 📂 docs/
│   └── label_guide.md
│
├── 📂 outputs/
│   ├── baseline_predictions.csv
│   ├── qwen_predictions.csv
│   ├── summary_metrics.csv
│   ├── field_metrics.csv
│   ├── failure_summary.csv
│   ├── qwen_summary_metrics.csv
│   ├── qwen_field_metrics.csv
│   ├── qwen_failure_summary.csv
│   └── charts/
│       ├── metric_comparison.png
│       ├── failure_comparison.png
│       └── latency_comparison.png
│
├── 📂 src/
│   ├── build_dataset.py
│   ├── baseline_extractor.py
│   ├── llm_extractor.py
│   ├── evaluate_predictions.py
│   ├── analyse_failures.py
│   ├── extract_enron_messages.py
│   ├── clean_real_world_data.py
│   ├── select_enron_eval_candidates.py
│   ├── build_enron_label_template.py
│   ├── append_enron_labels.py
│   ├── generate_visualisations.py
│   └── schemas.py
│
├── requirements.txt
└── README.md
```

---

## 🛠️ Implementation Notes

A few things that mattered more than expected in practice:

- Handling both clean benchmark timestamps and messy Enron-style timestamps required separate parsing paths
- Enron filenames with trailing dots caused issues on Windows
- Schema consistency across baseline and LLM outputs needed explicit enforcement
- Separating raw → intermediate → processed data stages kept things debuggable

---

## ⚠️ Known Limitations

| Limitation | Detail |
|:-----------|:-------|
| Single-message only | No email threading support |
| One event + one action | Per message |
| No rich media | No attachment, image, or PDF processing |
| No location extraction | Not in schema |
| No recurring events | Single occurrence only |
| Small benchmark | 40 rows — directional, not production-grade |
| One LLM tested | Only Qwen 3 8B in the final comparison |

---

## 🔮 Possible Extensions

- Expand the labelled benchmark with more Enron rows
- Benchmark a second local model (Mistral, Llama, etc.)
- Build a hybrid router that sends easy cases to rules and hard cases to the LLM
- Improve action deadline handling (weakest field for both methods)
- Add confusion matrices and per-category breakdowns
- Introduce softer scoring for the summary field

---

## 🧰 Built With

| Tool | Role |
|:-----|:-----|
| **Python 3.14** | Core pipeline |
| **Qwen 3 8B** | Local LLM via [Ollama](https://ollama.com/) |
| **pandas** | Data wrangling and evaluation |
| **matplotlib** | Visualisations |
| **Enron Email Corpus** | Real-world test data |

---

<div align="center">

**Built by [Shawn D'Souza](https://github.com/shawn-d123)**

Licensed under [MIT](LICENCE)

</div>
