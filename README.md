# GPT-4o Web Break Classification: Reasoning Failure Under Quantitative Hint Injection

---

## Overview

This repository documents a natural experiment in LLM classification failure from MSc thesis work at Bertelsmann SE & Co. KGaA. The task: classify industrial web break events in a high-speed printing press as either **machine_problem** or **paper_problem**, using GPT-4o on 300-frame multivariate time-series data represented as compressed text sequences.

The central finding is a paradox:

> **Injecting correct, XGBoost-learned quantitative decision rules into the prompt increased GPT-4o's overall accuracy by 7 percentage points, while collapsing its recall on the minority class (paper breaks) by 61%.**

This paradox improved surface accuracy, degraded reasoning quality is the behavioral signature that needs to be studied.

---

## The Experiment

### Dataset
- **14,073 labelled web break events** from an active Bertelsmann printing facility (2021–2026)
- Events represented as compressed frame sequences + operational metadata (speed, grammage, paper length)
- Class distribution: 84% `machine_problem`, 16% `paper_problem`
- All GPT-4o experiments use a shared held-out test set of **100 events** sampled with `SEED=42`

### Conditions

| Version | Condition | Prompt Design | Paper Recall | Paper F1 | Overall Acc |
|---------|------------------------|---------------|:------------:|:--------:|:-----------:|
| **v3** | Clean hint | 10 few-shot examples (5 per class), semantic descriptions only | **0.46** | **0.22** | 0.64 |
| **v5** | Misleadingly hinted | 24 few-shot examples + exact XGBoost decision rules with threshold values | **0.18** | **0.12** | 0.71 |

The XGBoost baseline achieves **mean test AUC 0.86** on the full dataset under stratified 3-fold CV, establishing the ceiling for what the signals can support.

### The Failure Mode

In v3 (clean), GPT-4o relies on **visual sequence shortcuts**: it associates "long no_defect run followed by sudden tear" with paper breaks, even though XGBoost's #1 feature (`cam1_tear_ratio`) treats a long clean run as a *machine* signal (XGBoost feature importance rank 12 for `cam1_longest_no_defect_run` — machine lean). The model's written reasoning cites this pattern confidently and is wrong ~30% of the time on machine events.

In v5 (misleadingly hinted), the system prompt provides the actual XGBoost threshold values as explicit rules:

```
cam1 tear_ratio < 0.019  → machine
grammage > 60 g/m²       → paper
pap_len > 25000 m        → paper
...
```

The model **cites these numbers** in almost every reasoning trace. Accuracy on the majority class improves because the thresholds happen to be correct and the majority class is large. But paper recall collapses from 0.46 to 0.18. Why? The model **cannot reliably execute threshold comparisons**:

1. `cam1_tear_ratio` (XGBoost rank #1) requires computing a ratio from the frame sequence — an operation GPT-4o cannot perform. It is told the threshold but cannot measure the operand.
2. Grammage and paper length are visible in the metadata and easy to cite. The model over-weights these convenient scalar signals (XGBoost ranks 20 and 5) while ignoring the harder-to-compute ratio features.
3. XGBoost's ranks 2–3 (`grade_encoded`, `printer_encoded`) are completely absent from the event text — the model never had access to them in either condition.

The result is a model that writes structured, threshold-citing reasoning while making systematically incorrect predictions on the minority class — **correct-sounding reasoning, wrong answers, for wrong reasons**.

---

## Interpretability Research Extensions

This work raises questions about LLM reasoning that behavioral metrics alone cannot answer:

### 1. Labeled behavioral dataset

`results/labeled_response_dataset.csv` contains each response annotated with:
- `event_id`, `true_label`, `predicted_label`, `prompt_condition`, `reasoning`
- `failure_type`: `correct` | `shortcut_visual_pattern` | `majority_class_bias` | `threshold_misexecution` | `correct_threshold_use`

This dataset captures not just prediction correctness but the *type* of reasoning failure, enabling mechanistic analysis beyond accuracy metrics.

### 2. A matched clean / rule-injected condition pair

v3 and v5 are a controlled pair: the same 100 events, same model, same output format, but different internal reasoning pathways. For many events the final label is the same across conditions — meaning behavioral accuracy alone cannot detect the reasoning failure.

### 3. Candidate failure signatures for activation analysis

The `shortcut_visual_pattern` events in v3 and `threshold_misexecution` events in v5 are candidate inputs for mechanistic analysis. Specifically:

| Analysis Target | Hypothesis |
|----------------|------------|
| Activations at the class-label commit token in v3 shortcuts | SAE features for "sequence pattern completion" should dominate over "quantitative comparison" features |
| Activations at threshold-citation tokens in v5 | If the model is executing thresholds, inequality-evaluation features should activate; if pattern-matching, label-association features should dominate |
| Attention to few-shot paper examples in v5 vs v3 | v5 paper recall collapse may correlate with reduced attention weight on the few-shot paper examples (drowned out by the rule text) |

### 4. Open questions

- Do internal representations distinguish a model that *knows* a threshold from one that *mimics* citing it?
- Is the paper-recall collapse in v5 driven by a change in the decision boundary in representation space, or by a retrieval failure (rule text suppresses paper example retrieval)?
- Can a linear probe trained on residual stream activations predict `failure_type` above the behavioral baseline?

---

## Repository Structure

```
web-break-llm-reasoning/
│
├── experiments/           # Experimental notebooks (source)
│   ├── v3_baseline_fewshot.ipynb     → LLMClassifier_Paper_Web_Breaks_v1.ipynb
│   └── v5_xgboost_rules.ipynb        → LLMClassifier_Paper_Web_Breaks_v2.ipynb
│
├── results/
│   ├── v3_results.csv                # 100-event test set, v3 predictions + reasoning
│   └── labeled_response_dataset.csv  # v3 (n=100) + v5 sample (n=10), with failure_type
│
├── prompts/
│   ├── v3_system_prompt.txt          # Clean few-shot prompt template + annotation
│   └── v5_system_prompt.txt          # XGBoost-rule-injected prompt template + annotation
│
├── analysis/
│   └── behavioral_analysis.ipynb     # Confusion matrices, reasoning analysis, CSV generation
│
├── docs/
│   └── thesis_context.md             # Full thesis context and methodology
│
└── README.md
```

**Data note**: The raw event data (14,073 JSON files) is proprietary to Bertelsmann SE & Co. KGaA and is not included. The results CSVs, prompts, and notebook outputs contain no raw sensor data.

---

## Running the Analysis

```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

# Run the analysis notebook from the analysis/ directory
cd analysis
jupyter notebook behavioral_analysis.ipynb
```

The notebook produces four figures and regenerates `results/labeled_response_dataset.csv`.  
To regenerate v5 predictions, run `experiments/v5_xgboost_rules.ipynb` with a valid `OPENAI_API_KEY`.

---

## Key Results

### Confusion matrices

| | v3 (Clean) | | v5 (Misleadingly Hinted) | |
|---|:---:|:---:|:---:|:---:|
| | pred: machine | pred: paper | pred: machine | pred: paper |
| **true: machine** (n=89) | 59 (66%) | 30 (34%) | 69 (78%) | 20 (22%) |
| **true: paper** (n=11) | 6 (55%) | 5 (45%) | 9 (82%) | 2 (18%) |

### Reasoning signal analysis (v3)

GPT-4o's reasoning cites these signals most frequently in v3 false positives (machine misclassified as paper):
- "no_defect" + "tear" co-occurrence (the visual shortcut pattern)
- "sudden" / "abrupt" tear onset
- "stable" pre-break period

These surface signals have **low-to-moderate XGBoost importance** (`cam1_longest_no_defect_run` is rank 12 and points to *machine*, not paper). The model's cited evidence is systematically pointing in the wrong direction for the minority class.

### Feature visibility gap

Of XGBoost's top 20 features:
- Only **4 are directly visible** in the event text: `pap_len`, `cam1_rollenwechsel_count`, `speed`, `grammage`
- The top feature (`cam1_tear_ratio`) requires ratio computation from the sequence — not observable directly
- Ranks 2–3 (`grade_encoded`, `printer_encoded`) are not in the event text at all

This means GPT-4o is working with a structurally incomplete view of the feature space it cannot, in principle, replicate XGBoost's decision boundary even with perfect threshold execution.

---

## Thesis Context

**Title**: Automated Web Break Cause Classification Using Time-Series and LLM Approaches 
**Graded**: 1,1
**Degree**: MSc, Bertelsmann SE & Co. KGaA  
**Dataset**: 14,073 labelled break events, 9 model families evaluated  
**XGBoost baseline**: Mean test AUC 0.86 (3-fold stratified CV)  
**GPT-4o v3**: paper recall 0.46, F1 0.22 (100-event test, 5-shot per class)  
**GPT-4o v5**: paper recall 0.18, F1 0.12 (100-event test, 12-shot per class + XGBoost rules)  
**Key conclusion**: Providing exact decision rules worsened performance, isolating the failure as architectural — the model cannot execute quantitative threshold logic regardless of information provided.

---


*Data: proprietary (Bertelsmann SE & Co. KGaA). Code and results: research use.*
