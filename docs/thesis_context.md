# Thesis Context and Methodology

## Thesis Summary

**Task**: Automated classification of web break root cause (machine_problem vs paper_problem)  
**Dataset**: 14,073 labelled break events from a Bertelsmann offset printing facility, 2021–2026  
**Data format**: Each event = 300-frame multivariate time-series (camera defect detections) + operational metadata  
**Class balance**: ~84% machine_problem, ~16% paper_problem

## Model Families Evaluated

Nine model families were evaluated under stratified 3-fold cross-validation:

| Model | Architecture | AUC (mean test) |
|-------|-------------|-----------------|
| XGBoost (v3 features) | Gradient boosted trees, 75 engineered features | **0.86** |
| CNN | 1D ConvNet on raw sequence | ~0.82 |
| ConvTimeNet | Deep conv time-series | ~0.81 |
| LSTM-FCN | Hybrid LSTM + FCN | ~0.80 |
| TapNet | Time-series attentive prototype | ~0.79 |
| TST | Time-series transformer | ~0.78 |
| InceptionTime | Inception module for time series | ~0.79 |
| GPT-4o (v3) | Few-shot text classification | paper recall 0.46 |
| GPT-4o (v5) | Few-shot + XGBoost rules | paper recall 0.18 |

XGBoost with engineered features is the best-performing model overall.

## XGBoost Feature Engineering

Features were engineered in three rounds:

**Original features**: Camera-level statistics — tear_ratio, defect_ratio, rollenwechsel_count, rollenwechsel_in_last_50/100, first/last defect position, longest no_defect run, dominant label entropy, Kantenfehler count/ratio, defect burst acceleration, defect agreement between cameras, etc.

**v2 features**: Sequence-derived signals — first_problem_pos_any, k_present_early, k_asymmetry, cam problem_span, defect bursts in first half, k_escalation, max_pre_tear_density

**v3 features**: Operational metadata — grade_encoded, printer_encoded (machine ID), web_width_mm, supplier_encoded, detector, month

The top XGBoost features (by gain importance):
1. `cam1_tear_ratio` — fraction of cam1 frames classified as tear
2. `grade_encoded` — paper grade (not visible in event text)
3. `printer_encoded` — which printing unit (not in event text)
4. `first_problem_pos_any` — relative position of first defect/tear in sequence
5. `pap_len` — paper roll length in metres

## LLM Experiment Design

### Text representation
Each event is serialised as:
```
Speed: X m/s | Grammage: X g/m2 | Paper length: X m
Camera1: no_defect x257 -> defect x1 -> no_defect x3 -> tear x212
Camera2: no_defect x1139 -> defect x5 -> tear x213
```

Frame sequences are run-length encoded: consecutive frames with the same dominant label are compressed into `label xN`.

### v3 Prompt (baseline few-shot)
- 10 few-shot examples: 5 machine_problem, 5 paper_problem
- Selected from training split with `SEED=42`
- Prompt asks model to cite "concrete signals and how they align with the examples"
- No quantitative guidance provided
- Batch size: 5 events per API call

### v5 Prompt (XGBoost rule injection)
- 24 few-shot examples: 12 machine_problem, 12 paper_problem
- Full set of XGBoost-derived decision rules with threshold values
- Explicit 5-step classification procedure (read metadata → apply rules → count signals → set confidence → write reasoning)
- Prompt length: ~56,000 characters vs ~25,700 for v3

### Test set
Both experiments classify the same 100 events sampled from the held-out test set using `rng.choice(len(test_texts), size=100, replace=False)` with `SEED=42`. This ensures the comparison is directly controlled.

## Key Findings

### Finding 1: Performance paradox
Adding quantitatively correct rules degraded minority-class performance (paper recall: 0.46 → 0.18) while improving overall accuracy (0.64 → 0.71). This pattern is consistent with a model that can leverage rules as confirmation bias for the majority class but cannot use them to rescue minority-class decisions.

### Finding 2: Feature visibility gap
GPT-4o cannot access the features that actually matter most:
- #1 feature (cam1_tear_ratio) requires ratio computation from raw sequence data
- #2 and #3 features (grade_encoded, printer_encoded) are absent from the event text entirely
- Only 4 of the top 20 XGBoost features are directly readable from the text representation

### Finding 3: Shortcut reasoning pattern
In v3, the dominant error pattern on machine events is: model sees `no_defect x[large number] -> tear x[large number]` and cites "sudden onset of tearing after stable run → paper_problem". This visual pattern is a shortcut — XGBoost treats long clean runs as machine signals (cam1_longest_no_defect_run is rank 12, slight machine lean). The model's written reasoning is coherent but its causal logic is inverted.

### Finding 4: Architectural failure isolation
The v5 experiment was designed to test whether the model *could* execute threshold logic if given the rules explicitly. It cannot — at least not reliably for the minority class. This isolates the failure as architectural (an inability to execute quantitative comparisons reliably) rather than informational (lack of the right decision boundaries). The model knows the rules in the sense that it can recite them, but it does not know them in the sense of consistently applying them.

## Connections to LLM Interpretability

This work motivates several interpretability questions:

**Representation of numerals**: When GPT-4o reads "Grammage: 57.0 g/m2" in a prompt containing the rule "grammage > 60 g/m² → paper", does the representation of "57.0" activate comparison-related features? Or does it merely activate "grammage" as a semantic concept co-occurring with "paper" in the rule text?

**Attention to examples vs rules**: Does the v5 prompt's large rule section suppress attention to the few-shot paper examples? Tracking attention weights across the full prompt could explain why paper recall collapses even with more paper examples in v5 (12 vs 5).

**Feature circuits for shortcut reasoning**: In v3, the shortcut pattern ("long clean run → paper") might correspond to identifiable attention heads that attend to run-length counts and co-activate with the "paper" output token. Identifying and ablating these circuits would test whether they are causally responsible for the shortcut errors.

These are the empirical questions that connect this thesis work to the SOAR I-1 research agenda.
