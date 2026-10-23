# ACLAgent

## Random forest baseline

`src/random_forest_loo_baseline.py` evaluates separate random forest regressors for postoperative Tegner and ACL-RSI scores. Each patient is held out once, and the remaining patients form the training set. For a cohort of 115 patients, each fold therefore uses 114 training patients. Predictions are summarized for the full cohort and for each center group specified in the cohort file.

### Environment

Use Python 3.10 or later with NumPy and pandas. The script includes its own regression-tree and random forest implementation and does not require scikit-learn or a language-model API.

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-random-forest.txt
python src/random_forest_loo_baseline.py --help
```

### Required inputs

The clinical input files are supplied separately. By default, the script reads:

```text
data/
  raw_processed/<center>/<patient>/Admission note.csv
  interim/<center>/<patient>/acl_rsi_predict_report.json
  interim/<center>/<patient>/tegner_predict_report.json
  reports/patient_referral_centers.json
```

The admission CSV provides `Sex`, `Age`, `Chief complaint`, `History of present illness`, `Past medical history`, and `Personal history`. The script uses the first patient row in each admission CSV. UTF-8 with or without a byte-order mark and GB18030 encodings are supported. Patient directory names must be unique within the cohort.

Source filenames, column headers, and clinical narrative values are expected in English. The random forest uses the English narrative field labels listed above when constructing text features.

The outcome JSON files must contain numeric `ground_truth_ACL-RSI` and `ground_truth_Tegner` fields, respectively. These fields supply the supervised training labels and the held-out evaluation outcomes. They are not predictor features. ACL-RSI values must be between 0 and 100, and Tegner values between 0 and 10.

The cohort JSON maps each evaluation group to its patient identifiers. For example, the following illustrates the schema, not a study dataset:

```json
{
  "Center-A": ["A001", "A002"],
  "Center-B": ["B001"]
}
```

### Fixed default configuration

| Component | Configuration |
| --- | --- |
| Forest | 200 trees, each trained on a bootstrap sample of the training fold |
| Tree limits | Maximum depth 6, minimum split size 4, minimum leaf size 2 |
| Split criterion | Minimum within-node sum of squared errors |
| Features considered per split | Integer part of the square root of the encoded feature count |
| Age | Training-fold median imputation with a missingness indicator. If every training age is missing, the replacement is 0. |
| Sex | One-hot encoding based on training-fold categories, plus missing and unseen-category indicators |
| Narrative text | The four narrative fields and their field labels, converted to lowercase and normalized for whitespace, without truncation |
| Text encoding | Character 2–4-grams, with whitespace removed during n-gram extraction. CRC32 maps each n-gram to one of 256 bins. Counts are divided by the number of n-grams in that record. No cohort vocabulary is fitted. |
| Hyperparameters | Identical settings across folds and outcomes. No inner hyperparameter search is performed. |
| Random seeds | Base seed 42 for ACL-RSI and 1042 for Tegner. The zero-based held-out-patient index is added for each fold. |
| Patient order | Sorted center/patient directory paths |
| Prediction | Mean of the tree predictions, without score rounding |
| Classification thresholds | Tegner ≥6 and ACL-RSI ≥65, applied to both observed scores and predictions |
| Uncertainty | 2,000 patient-level bootstrap draws. Classification draws are stratified by observed threshold status. Percentile intervals use the finite metric estimates. |

The bootstrap intervals describe uncertainty conditional on the saved out-of-fold predictions. Bootstrap resampling does not refit the forest. All configuration values can be inspected using `--help` and are recorded in the output JSON files.

### Run

With the default input layout:

```bash
python src/random_forest_loo_baseline.py
```

With separately stored inputs:

```bash
python src/random_forest_loo_baseline.py \
  --raw-processed-root /path/to/raw_processed \
  --interim-root /path/to/interim \
  --referral-centers-json /path/to/patient_referral_centers.json \
  --output-dir /path/to/random_forest_results \
  --n-estimators 200 \
  --max-depth 6 \
  --min-samples-split 4 \
  --min-samples-leaf 2 \
  --max-features sqrt \
  --text-hash-dim 256 \
  --text-ngram-min 2 \
  --text-ngram-max 4 \
  --bootstrap-repeats 2000 \
  --seed 42
```

### Outputs

Results are written to `data/baselines_ml/` by default:

| File | Contents |
| --- | --- |
| `rf_feature_matrix.csv` | Source predictor fields, narrative text, patient/group identifiers and outcome labels |
| `rf_acl_rsi_loo_predictions.csv` | ACL-RSI observations, held-out predictions, residuals, absolute errors and training counts |
| `rf_tegner_loo_predictions.csv` | Tegner observations, held-out predictions, residuals, absolute errors and training counts |
| `rf_acl_rsi_loo_metrics.json` | ACL-RSI model/encoding settings, cohort and center metrics, confusion matrices and confidence intervals |
| `rf_tegner_loo_metrics.json` | Corresponding Tegner settings and metrics |
| `rf_loo_run_summary.json` | Input/output locations, cohort sizes, common settings and task summaries |

Residuals are observed scores minus predicted scores. The reported regression metrics are MAE, RMSE, RMSE normalized by the observed score range, Pearson correlation and Spearman correlation. Classification metrics are AUC, recall, precision and F1 score. The metrics JSON files also report the numbers of finite bootstrap estimates used for each interval.

## ACLRPRAgent tools and terminology

The agent modules cover clinical data curation, retrieval, Tegner prediction and
attribution, ACL-RSI prediction and attribution, and clinical decision reasoning.
English prompt text and structured output formats are centralized in
`src/prompt_templates.json`, following Supplementary Table S10. Retrieval actions
and task goals follow Table S9 in `src/retrieval_tasks.json`. The prompt builder
inserts runtime values without translating patient evidence or changing the field
names returned by the retrieval model.

| Manuscript term | Public code term or output key |
| --- | --- |
| Record selection | `selected_records` |
| Field selection | `selected_fields` with `record` and `field` |
| Tegner score prediction | `predicted_Tegner` |
| ACL-RSI score prediction | `predicted_ACL-RSI` |
| Rehabilitation plan | `rehabilitation_plan` |
| RTA pathway recommendations | `RTA_pathway_recommendations` |
| Attribution insights | `reasoning` and `key_factors` |

The retrieval adapter maps the appendix output keys to the existing internal
`selected_documents` and `document` fields. This preserves source-record lookup.
The original Tegner prompt does not specify one follow-up timepoint; the ACL-RSI
prompt specifies 6–12 postoperative months. Both predictions were evaluated
against 12-month follow-up outcomes. The English templates preserve this distinction.

Clinical input supplied to the agent must be restricted to information available
by discharge. Recorded postoperative outcome scores are evaluation labels, not
prompt evidence. The separate follow-up CSV used when saving evaluation reports
is `data/raw/<center>/follow_up_scores.csv`, with `Patient name`, `ACL-RSI`,
`Preoperative Tegner` and `Postoperative Tegner` columns.

The random forest baseline requires only the dependencies documented above. The
agent modules additionally use the packages in `requirements-agent.txt` and
provider API credentials configured locally in `src/llm_api_config.py`. Provider
calls incur external requests; the supplied credentials are placeholders.

For the agent workflow, run from `src` because its existing relative data paths
are resolved from the working directory:

```bash
python -m pip install -r requirements-agent.txt
cd src
python main.py
```

Set the center and synthetic example patient identifiers in the entry point to
your local study configuration before running. No clinical data are distributed.
Generated prescriptions require clinician review before use. The English prompt
edition has not been used to regenerate the manuscript's LLM predictions;
translation does not establish numerical equivalence for stochastic model outputs.
