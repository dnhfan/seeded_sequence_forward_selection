# colon1 Results and Evaluation

[Back to index](../results.md)

## 1) EDA (Exploratory Data Analysis)

- Notebook entry point(s):
- `notebook/colon1/01_eda.ipynb`

[Insert Chart: EDA Summary]
![colon1 EDA](../../results/colon1/eda/plot/countplot.png)

## 2) Data Preprocessing

- Notebook entry point(s):
- `notebook/colon1/02_preprocess.ipynb`
- Output location convention: `data/processed/colon1/01_clean/`

## 3) Filter Selection

- Notebook entry point(s):
- `notebook/colon1/03_filter_selection.ipynb`
- Report artifact: `results/colon1/filter/reports/evaluation_colon1.txt`

[Insert Chart: Filter Selection Comparison]
![colon1 Filter Selection](../../results/colon1/filter/plots/evaluation_colon1.png)

## 4) Modeling (Filter-stage comparison)

- Notebook entry point(s):
- `notebook/colon1/04_modeling.ipynb`
- Modeling outputs are tracked under `results/colon1/filter/` when available.

## 5) Ensemble Filter (Voting + union feature set)

- Notebook entry point(s):
- `notebook/colon1/05_esemble_filter.ipynb`
- Seed pool file: `data/processed/colon1/03_ensemble/top50_features_voting.csv`
- Seed pool size: 10
- Top voting features: `T95018(5)`, `M63391(5)`, `M76378(4)`, `T60155(4)`, `M22382(4)`

[Insert Chart: Ensemble Voting / Union Features]
![colon1 Ensemble Voting](../../results/colon1/ensemble/plots/top50_features_voting.png)

## 6) Wrapper: Sklearn SFS (Raw vs Union execution)

- Script entry point(s):
- `notebook/colon1/06_sklearn_sfs-raw.py`
- `notebook/colon1/06_sklearn_sfs-union.py`

| Variant | Sklearn Selected | Sklearn Global Best | Sklearn Fit Time (ms) |
|---|---:|---:|---:|
| Raw | 4 | 0.9513 | 193,300 |
| Union | 3 | 0.9032 | 13,822 |

## 7) Wrapper: Seeded SFS (Raw vs Union execution)

- Script entry point(s):
- `notebook/colon1/07_sfs-raw.py`
- `notebook/colon1/07_sfs-union.py`

| Variant | Seeded Selected | Seeded Global Best | Seeded Fit Time (ms) |
|---|---:|---:|---:|
| Raw | 8 | 0.9346 | 188,278 |
| Union | 5 | 0.9359 | 7,700 |

## 8) Accuracy Evaluation (Comparing Raw vs Union)

- Notebook entry point(s):
- `notebook/colon1/8_accuracu_evaluate.ipynb`
- `notebook/colon1/8_accuracu_evaluate_union.ipynb`

[Insert Chart: Accuracy Comparison Raw vs Union]
![colon1 Accuracy Evaluation](../../results/colon1/evaluation/plots/wrapper_sfs_comparison_sk_raw_seeded_raw_colon1.png)
![colon1 Accuracy Evaluation](../../results/colon1/evaluation/plots/wrapper_sfs_comparison_sk_union_seeded_union_colon1.png)

- **Observation:** Score improves strongly in early iterations and then plateaus near 0.9346.
- **Explanation:** Early selected candidates add strong signal; later candidates contribute marginal gain.
- **Takeaway:** The model is effectively optimized with a low feature count.

- Raw best configuration: `seeded + LogReg`, mean accuracy **0.9179**, std 0.1021
- Union best configuration: `seeded + LogReg`, mean accuracy 0.8872, std 0.0436
- Final selected features (winning setup, raw seeded):
  `T95018, T63508, X14958, T57780, H09263, T49423, H16991, T83673`

## 9) Time Evaluation (Comparing fit times for Raw vs Union)

- Notebook entry point(s):
- `notebook/colon1/9_time_evaluate.ipynb`
- `notebook/colon1/9_time_evaluate_union.ipynb`

[Insert Chart: Time Comparison Raw vs Union]
![colon1 Time Evaluation](../../results/colon1/evaluation/plots/time_comparison_raw_seeded_vs_raw_sklearn.png)
![colon1 Time Evaluation](../../results/colon1/evaluation/plots/time_comparison_union_seeded_vs_union_sklearn.png)

- **Observation:** Union runs are generally faster than raw runs across wrapper methods.
- **Explanation:** Union reduces candidate-space size, reducing total model-fit operations.
- **Takeaway:** Use union for rapid iteration; use raw when chasing peak wrapper score.
