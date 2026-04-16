# NCI Results and Evaluation

[Back to index](../results.md)

## 1) EDA (Exploratory Data Analysis)

- Notebook entry point(s):
- `notebook/NCI/01_eda.ipynb`

[Insert Chart: EDA Summary]
![NCI EDA](../../results/NCI/eda/plot/countplot.png)

**Caption:**
- Purpose: Check whether the dataset is imbalanced.
- How to read: The x-axis (V1) shows class labels (0 and 1), and the y-axis (count) shows the number of samples in each class.

## 2) Data Preprocessing

- Notebook entry point(s):
- Not explicitly available in current notebook folder.
- Output location convention: `data/processed/NCI/01_clean/`

## 3) Filter Selection

- Notebook entry point(s):
- `notebook/NCI/02_filter_selection.ipynb`
- Report artifact: `results/NCI/filter/reports/evaluation_NCI.txt`

[Insert Chart: Filter Selection Comparison]
![NCI Filter Selection](../../results/NCI/filter/plots/evaluation_NCI.png)

**Caption:**
- Purpose: Compare filter-method performance to select the best feature set for the next stage.
- How to read: The x-axis lists filter methods, and the y-axis shows evaluation scores; higher bars/scores indicate better methods.

## 4) Modeling (Filter-stage comparison)

- Notebook entry point(s):
- `notebook/NCI/03_modeling.ipynb`
- Modeling outputs are tracked under `results/NCI/filter/` when available.

## 5) Ensemble Filter (Voting + union feature set)

- Notebook entry point(s):
- `notebook/NCI/04_ensemble.ipynb`
- Seed pool file: `data/processed/NCI/03_ensemble/top50_features_voting.csv`
- Seed pool size: 10
- Top voting features: `V155(4)`, `V4968(4)`, `V4516(4)`, `V4618(4)`, `V4581(4)`

[Insert Chart: Ensemble Voting / Union Features]
![NCI Ensemble Voting](../../results/NCI/ensemble/plots/top50_features_voting.png)

**Caption:**
- Purpose: Show agreement among filter methods when voting for features.
- How to read: The x-axis lists feature names, and the y-axis shows vote counts; features with higher votes are prioritized.

## 6) Wrapper: Sklearn SFS (Raw vs Union execution)

- Script entry point(s):
- `notebook/NCI/06_sklearn_sfs-raw.py`
- `notebook/NCI/06_sklearn_sfs-union.py`

| Variant | Sklearn Selected | Sklearn Global Best | Sklearn Fit Time (ms) |
|---|---:|---:|---:|
| Raw | 9 | 0.8551 | 825,324 |
| Union | 8 | 0.8192 | 29,053 |

## 7) Wrapper: Seeded SFS (Raw vs Union execution)

- Script entry point(s):
- `notebook/NCI/07_sfs-raw.py`
- `notebook/NCI/07_sfs-union.py`

| Variant | Seeded Selected | Seeded Global Best | Seeded Fit Time (ms) |
|---|---:|---:|---:|
| Raw | 7 | 0.8833 | 223,827 |
| Union | 6 | 0.8359 | 7,373 |

## 8) Accuracy Evaluation (Comparing Raw vs Union)

- Notebook entry point(s):
- `notebook/NCI/8_accuracu_evaluate.ipynb`
- `notebook/NCI/8_accuracu_evaluate_union.ipynb`

[Insert Chart: Accuracy Comparison Raw vs Union]
![NCI Accuracy Evaluation](../../results/NCI/evaluation/plots/wrapper_sfs_comparison_sk_raw_seeded_raw_NCI.png)

**Caption:**
- Purpose: Compare accuracy across wrapper configurations (Sklearn SFS and Seeded SFS) for each data variant.
- How to read: The x-axis shows configurations/methods, and the y-axis shows accuracy; higher values indicate better performance.
![NCI Accuracy Evaluation](../../results/NCI/evaluation/plots/wrapper_sfs_comparison_sk_union_seeded_union_NCI.png)

**Caption:**
- Purpose: Compare accuracy across wrapper configurations (Sklearn SFS and Seeded SFS) for each data variant.
- How to read: The x-axis shows configurations/methods, and the y-axis shows accuracy; higher values indicate better performance.

- **Observation:** Raw variant outperforms union in both wrapper and evaluation scores.
- **Explanation:** Additional raw features appear to preserve useful discriminatory signal.
- **Takeaway:** Keep raw variant as default for NCI benchmark runs.

- Raw best configuration: `seeded + LogReg`, mean accuracy **0.7872**, std 0.0937
- Union best configuration: `sklearn + LogReg`, mean accuracy 0.7385, std 0.1053

## 9) Time Evaluation (Comparing fit times for Raw vs Union)

- Notebook entry point(s):
- `notebook/NCI/9_time_evaluate.ipynb`
- `notebook/NCI/9_time_evaluate_union.ipynb`

[Insert Chart: Time Comparison Raw vs Union]
![NCI Time Evaluation](../../results/NCI/evaluation/plots/time_comparison_raw_seeded_vs_raw_sklearn.png)

**Caption:**
- Purpose: Compare training-time cost across wrapper methods on the same dataset.
- How to read: The x-axis shows methods/configurations, and the y-axis shows total fit time (ms); lower bars mean faster runtime.
![NCI Time Evaluation](../../results/NCI/evaluation/plots/time_comparison_union_seeded_vs_union_sklearn.png)

**Caption:**
- Purpose: Compare training-time cost across wrapper methods on the same dataset.
- How to read: The x-axis shows methods/configurations, and the y-axis shows total fit time (ms); lower bars mean faster runtime.

- **Observation:** Union runs are generally faster than raw runs across wrapper methods.
- **Explanation:** Union reduces candidate-space size, reducing total model-fit operations.
- **Takeaway:** Use union for rapid iteration; use raw when chasing peak wrapper score.
