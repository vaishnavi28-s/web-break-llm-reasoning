# Experiments

The source notebooks are stored at the thesis root and referenced here.

| File here | Source notebook | Condition | Description |
|-----------|----------------|-----------|-------------|
| v3_baseline_fewshot.ipynb | LLMClassifier_Paper_Web_Breaks_v1.ipynb | Clean / baseline few-shot | GPT-4o with 10 few-shot examples (5 per class), semantic prompt only |
| v5_xgboost_rules.ipynb | LLMClassifier_Paper_Web_Breaks_v2.ipynb | Misleadingly hinted | GPT-4o with 24 examples + explicit XGBoost decision thresholds |

To use: copy the source notebooks from the thesis root into this folder, or symlink them.

## Running Notes

Both experiments require:
- `OPENAI_API_KEY` in your `.env` file
- Access to the event data ZIPs in `data/events/` (proprietary, not distributed)
- Python environment: `pip install -r requirements.txt`

Cost per 100-event run:
- v3: ~$0.61 USD
- v5: ~$0.89 USD (longer prompt due to rule injection)

## Reproducing Results

Both notebooks use `SEED=42` for the random test set sample. To reproduce exactly:
1. Load the same 14,073 events from `data/events/`  
2. Apply the 80/20 stratified split (`random_state=42`)
3. Sample 100 events from the test split with `np.random.default_rng(42).choice(...)`

The resulting 100 event IDs are stable across runs and match `results/v3_results.csv`.
