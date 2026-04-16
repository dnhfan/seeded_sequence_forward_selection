# Leukemia_4c1 Results and Evaluation

[Back to index](results.md)

## 1) EDA (Exploratory Data Analysis)

- Notebook entry point(s):
- `notebook/Leukemia_4c1/01_eda.ipynb`

[Insert Chart: EDA Summary]
![Leukemia_4c1 EDA](../results/Leukemia_4c1/eda/plot/countplot.png)

## 2) Data Preprocessing

- Notebook entry point(s):
- `notebook/Leukemia_4c1/02_preprocess.ipynb`
- Output location convention: `data/processed/Leukemia_4c1/01_clean/`

## 3) Filter Selection

- Notebook entry point(s):
- `notebook/Leukemia_4c1/03_filter_selection.ipynb`
- Report artifact: `results/Leukemia_4c1/filter/reports/evaluation_Leukemia_4c1.txt`

[Insert Chart: Filter Selection Comparison]
![Leukemia_4c1 Filter Selection](../results/Leukemia_4c1/filter/plots/evaluation_Leukemia_4c1.png)

## 4) Modeling (Filter-stage comparison)

- Notebook entry point(s):
- `notebook/Leukemia_4c1/04_modeling.ipynb`
- Modeling outputs are tracked under `results/Leukemia_4c1/filter/` when available.

## 5) Ensemble Filter (Voting + union feature set)

- Notebook entry point(s):
- `notebook/Leukemia_4c1/05_esemble_filter.ipynb`
- Seed pool file: `data/processed/Leukemia_4c1/03_ensemble/top*_features_voting*.csv`
- Seed pool size: 10
- Top voting features: `1881(5)`, `4195(4)`, `759(4)`, `6040(4)`, `4049(4)`

[Insert Chart: Ensemble Voting / Union Features]
![Leukemia_4c1 Ensemble Voting](../results/Leukemia_4c1/ensemble/plots/top50_features_voting.png)

## 6) Wrapper: Sklearn SFS (Raw vs Union execution)

- Script entry point(s):
- `notebook/Leukemia_4c1/06_sklearn_sfs-raw.py`
- `notebook/Leukemia_4c1/06_sklearn_sfs-union.py`

| Variant | Sklearn Selected | Sklearn Global Best | Sklearn Fit Time (ms) |
|---|---:|---:|---:|
| Raw | 3 | 0.9722 | 370,856 |
| Union | 3 | 0.9722 | 10,241 |

## 7) Wrapper: Seeded SFS (Raw vs Union execution)

- Script entry point(s):
- `notebook/Leukemia_4c1/07_sfs-raw.py`
- `notebook/Leukemia_4c1/07_sfs-union.py`

| Variant | Seeded Selected | Seeded Global Best | Seeded Fit Time (ms) |
|---|---:|---:|---:|
| Raw | 6 | 0.9861 | 62,965 |
| Union | 5 | 0.9722 | 3,200 |

## 8) Accuracy Evaluation (Comparing Raw vs Union)

- Notebook entry point(s):
- `notebook/Leukemia_4c1/8_accuracu_evaluate.ipynb`
- `notebook/Leukemia_4c1/8_accuracu_evaluate_union.ipynb`

[Insert Chart: Accuracy Comparison Raw vs Union]
![Leukemia_4c1 Accuracy Evaluation](../results/Leukemia_4c1/evaluation/plots/wrapper_sfs_comparison_sk_raw_seeded_raw_Leukemia_4c1.png)
![Leukemia_4c1 Accuracy Evaluation](../results/Leukemia_4c1/evaluation/plots/wrapper_sfs_comparison_sk_union_seeded_union_Leukemia_4c1.png)

- **Observation:** Class distribution is explicitly visualized before feature engineering steps.
- **Explanation:** Early imbalance detection helps interpret downstream metric variance and model bias.
- **Takeaway:** Keep class-aware validation decisions consistent across later stages.

- Raw best configuration: `sklearn + Tree`, mean accuracy **0.9722**, std 0.0321
- Union best configuration: `sklearn + Tree`, mean accuracy **0.9722**, std 0.0321

## 9) Time Evaluation (Comparing fit times for Raw vs Union)

- Notebook entry point(s):
- `notebook/Leukemia_4c1/9_time_evaluate.ipynb`
- `notebook/Leukemia_4c1/9_time_evaluate_union.ipynb`

[Insert Chart: Time Comparison Raw vs Union]
![Leukemia_4c1 Time Evaluation](../results/Leukemia_4c1/evaluation/plots/time_comparison_raw_seeded_vs_raw_sklearn.png)
![Leukemia_4c1 Time Evaluation](../results/Leukemia_4c1/evaluation/plots/time_comparison_union_seeded_vs_union_sklearn.png)

- **Observation:** Union runs are generally faster than raw runs across wrapper methods.
- **Explanation:** Union reduces candidate-space size, reducing total model-fit operations.
- **Takeaway:** Use union for rapid iteration; use raw when chasing peak wrapper score.
