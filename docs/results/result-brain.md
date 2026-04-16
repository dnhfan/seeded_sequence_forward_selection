# Brain Results and Evaluation

[Back to index](../results.md)

## 1) EDA (Exploratory Data Analysis)

- Notebook entry point(s):
- `notebook/Brain/01_eda.ipynb`

[Insert Chart: EDA Summary]
![Brain EDA](../../results/Brain/eda/plot/countplot.png)

**Caption:**
- Purpose: Check whether the dataset is imbalanced.
- How to read: The x-axis (V1) shows class labels (0 and 1), and the y-axis (count) shows the number of samples in each class.

## 2) Data Preprocessing

- Notebook entry point(s):
- Not explicitly available in current notebook folder.
- Output location convention: `data/processed/Brain/01_clean/`

## 3) Filter Selection

- Notebook entry point(s):
- `notebook/Brain/02_Filter_selection.ipynb`
- Report artifact: `results/Brain/filter/reports/filter_compare_50features_Brain.txt`

[Insert Chart: Filter Selection Comparison]
![Brain Filter Selection](../../results/Brain/filter/plots/filter_compare_50features_Brain.png)

**Caption:**
- Purpose: Compare filter-method performance to select the best feature set for the next stage.
- How to read: The x-axis lists filter methods, and the y-axis shows evaluation scores; higher bars/scores indicate better methods.

## 4) Modeling (Filter-stage comparison)

- Notebook entry point(s):
- `notebook/Brain/03_Modeling.ipynb`
- Modeling outputs are tracked under `results/Brain/filter/` when available.

## 5) Ensemble Filter (Voting + union feature set)

- Notebook entry point(s):
- `notebook/Brain/04_Ensemble_fitler_selection.ipynb`
- `notebook/Brain/05_Union.ipynb`
- Seed pool file: `data/processed/Brain/03_ensemble/top50_features_voting.csv`
- Seed pool size: 10
- Top voting features: `V1893(5)`, `V523(4)`, `V541(4)`, `V1050(3)`, `V2332(3)`

[Insert Chart: Ensemble Voting / Union Features]
![Brain Ensemble Voting](../../results/Brain/ensemble/plot/top50_features_voting.png)

**Caption:**
- Purpose: Show agreement among filter methods when voting for features.
- How to read: The x-axis lists feature names, and the y-axis shows vote counts; features with higher votes are prioritized.

## 6) Wrapper: Sklearn SFS (Raw vs Union execution)

- Script entry point(s):
- `notebook/Brain/06_sklearn_sfs-raw.py`
- `notebook/Brain/06_sklearn_sfs-union.py`

| Variant | Sklearn Selected | Sklearn Global Best | Sklearn Fit Time (ms) |
|---|---:|---:|---:|
| Raw | 6 | 0.9295 | 732,895 |
| Union | 4 | 0.8795 | 13,169 |

## 7) Wrapper: Seeded SFS (Raw vs Union execution)

- Script entry point(s):
- `notebook/Brain/07_sfs-raw.py`
- `notebook/Brain/07_sfs-union.py`

| Variant | Seeded Selected | Seeded Global Best | Seeded Fit Time (ms) |
|---|---:|---:|---:|
| Raw | 11 | 0.9295 | 112,591 |
| Union | 6 | 0.9 | 3,555 |

## 8) Accuracy Evaluation (Comparing Raw vs Union)

- Notebook entry point(s):
- `notebook/Brain/08_accuracy_evaluate.ipynb`
- `notebook/Brain/08_accuracy_evaluate_union.ipynb`

[Insert Chart: Accuracy Comparison Raw vs Union]
![Brain Accuracy Evaluation](../../results/Brain/evaluation/plots/evaluation_Brain.png)

**Caption:**
- Purpose: Compare accuracy across wrapper configurations (Sklearn SFS and Seeded SFS) for each data variant.
- How to read: The x-axis shows configurations/methods, and the y-axis shows accuracy; higher values indicate better performance.
![Brain Accuracy Evaluation](../../results/Brain/evaluation/plots/wrapper_sfs_comparison_union_Brain.png)

**Caption:**
- Purpose: Compare accuracy across wrapper configurations (Sklearn SFS and Seeded SFS) for each data variant.
- How to read: The x-axis shows configurations/methods, and the y-axis shows accuracy; higher values indicate better performance.

- **Observation:** Score trajectory shows stepwise improvements with intermittent regressions.
- **Explanation:** Feature interactions are non-monotonic; global-best tracking preserves optimal subset.
- **Takeaway:** Retaining global-best rollback is important for robust final subset selection.

- Raw best configuration: see evaluation report.
- Union best configuration: see evaluation report.
- Final selected features (raw seeded): 11 features

## 9) Time Evaluation (Comparing fit times for Raw vs Union)

- Notebook entry point(s):
- `notebook/Brain/9_time_evaluate.ipynb`
- `notebook/Brain/9_time_evaluate_union.ipynb`

[Insert Chart: Time Comparison Raw vs Union]
![Brain Time Evaluation](../../results/Brain/evaluation/plots/time_comparison_seeded3_vs_sklearn_brain.png)

**Caption:**
- Purpose: Compare training-time cost across wrapper methods on the same dataset.
- How to read: The x-axis shows methods/configurations, and the y-axis shows total fit time (ms); lower bars mean faster runtime.
![Brain Time Evaluation](../../results/Brain/evaluation/plots/time_comparison_union_seeded_vs_union_sklearn.png)

**Caption:**
- Purpose: Compare training-time cost across wrapper methods on the same dataset.
- How to read: The x-axis shows methods/configurations, and the y-axis shows total fit time (ms); lower bars mean faster runtime.

- **Observation:** Union runs are generally faster than raw runs across wrapper methods.
- **Explanation:** Union reduces candidate-space size, reducing total model-fit operations.
- **Takeaway:** Use union for rapid iteration; use raw when chasing peak wrapper score.
