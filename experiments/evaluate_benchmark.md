# Benchmark Evaluation Script

Evaluates SeededSFS benchmark runs across multiple models (Logistic Regression, Decision Tree, Random Forest, SVM). Compares accuracy, fit time, and feature count.

## Usage

```bash
python experiments/evaluate_benchmark.py <dataset> <variant> [-c N]
```

### Arguments

| Argument    | Description                                                |
|-------------|------------------------------------------------------------|
| `dataset`   | Dataset name (e.g. `Brain`, `Breast3classes`, `colon1`, `Leukemia_3c1`) |
| `variant`   | `raw` or `union` — which feature set variant to evaluate   |
| `-c, --cv`  | Cross-validation folds (default: `4`, min: `2`)            |

### Example

```bash
python experiments/evaluate_benchmark.py colon1 raw --cv 5
```

## Pipeline

### Phase 1 — Baseline Evaluation
Runs cross-validation on the full feature set (`raw_path`) and logs accuracy for each model class (log, dt, rf, svm).

### Phase 2 — SFS Model Collection
For each model:
1. **Metrics** — reads latest `*benchmark_{model}_1seeds*/metrics/metrics.csv` containing fit time, features selected, and SFS internal score.
2. **Cross-validation** — loads the corresponding SFS-selected feature CSV (`*{model}_1seeds*{variant}.csv`) and runs independent CV via `ModelEvaluator.evaluate_custom_file()`.

### Phase 3 — Accuracy Report
Aggregates all CV fold results into a grouped accuracy bar chart + CSV report.

### Phase 4 — Time & Feature Trade-off
Generates a side-by-side bar plot:
- **Left**: SFS fit time (seconds) per internal model
- **Right**: Number of features selected

Saves plot to `evaluation_dir/plots/benchmark_time_features_{variant}_{dataset}.png`

## Outputs

All outputs under `evaluation_dir` (typically `results/<dataset>/evaluation/`):

| Artifact                                  | Description                              |
|-------------------------------------------|------------------------------------------|
| `reports/benchmark_accuracy_*.csv`        | Per-fold accuracy for baseline + SFS     |
| `plots/benchmark_accuracy_*.png`          | Accuracy comparison chart                |
| `plots/benchmark_time_features_*.png`     | Time & feature count trade-off plot      |
| `reports/benchmark_metrics_summary_*.csv` | Summary table (Model, Features, Score, Time) |

Wrapper metrics are read from:
```
results/<dataset>/wrapper/<variant>/seededsfsselector/*benchmark_{model}_1seeds*/metrics/metrics.csv
```

## Dependencies

- `src.config.ProjectPath` — path resolution
- `src.modeling.evaluation.ModelEvaluator` — CV evaluation, reports, plots
- `pandas`, `matplotlib`, `seaborn` — data handling and visualization
