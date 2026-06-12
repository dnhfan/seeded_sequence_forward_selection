# DLBCL Results and Evaluation

[Back to index](./README.md) ## 1) EDA (Exploratory Data Analysis)

- Notebook entry point(s):
- `notebook/DLBCL/01_eda.ipynb`
- Shape: (77, 5470)

[Insert Chart: EDA Summary]
![DLBCL EDA](../../results/DLBCL/eda/plot/countplot.png)

**Caption:**

- Purpose: Check whether the dataset is imbalanced.
- How to read: The x-axis (V1) shows class labels (0 and 1), and the y-axis (count) shows the number of samples in each class.

## 2) Data Preprocessing

- Notebook entry point(s):
- `notebook/DLBCL/02_preprocess.ipynb`
- Output location convention: `data/processed/DLBCL/01_clean/`

## 3) Filter Selection

- Notebook entry point(s):
- `notebook/DLBCL/03_filter_selection.ipynb`
- Results data: `data/processed/DLBCL/02_filter`

## 4) Modeling (Filter-stage comparison)

- Notebook entry point(s):
- `notebook/DLBCL/04_modeling.ipynb`
- Report artifact: `results/DLBCL/filter/reports/evaluation_DLBCL.txt`

[Insert Chart: Filter Selection Comparison]
![DLBCL Filter Selection](../../results/DLBCL/filter/plots/evaluation_DLBCL.png)

**Caption:**

- Purpose: Compare filter-method performance to select the best feature set for the next stage.
- How to read: The x-axis lists filter methods, and the y-axis shows evaluation scores; higher bars/scores indicate better methods.

## 5) Ensemble Filter (Voting + union feature set)

- Notebook entry point(s):
- `notebook/DLBCL/05_esemble_filter.ipynb`
- Seed pool file: `data/processed/DLBCL/03_ensemble/top50_features_voting.csv`
- Seed pool size: 10
- Top voting features: `V3128(5)`, `V4553(5)`, `V1056(4)`, `V1601(4)`, `V3468(4)`

[Insert Chart: Ensemble Voting / Union Features]
![DLBCL Ensemble Voting](../../results/DLBCL/ensemble/plots/top50_features_voting.png)

**Caption:**

- Purpose: Show agreement among filter methods when voting for features.
- How to read: The x-axis lists feature names, and the y-axis shows vote counts; features with higher votes are prioritized.

## 6) Wrapper: Sklearn SFS (Raw vs Union execution)

- Script entry point(s):
- `notebook/DLBCL/06_sklearn_sfs-raw.py`
- `notebook/DLBCL/06_sklearn_sfs-union.py`

| Variant | Sklearn Selected | Sklearn Global Best | Sklearn Fit Time (s) |
| ------- | ---------------: | ------------------: | -------------------: |
| Raw     |                2 |                   1 |              229.964 |
| Union   |                3 |                   1 |               13.010 |

## 7) Wrapper: Seeded SFS (Raw vs Union execution)

- Script entry point(s):
- `notebook/DLBCL/07_sfs-raw.py`
- `notebook/DLBCL/07_sfs-union.py`

| Variant | Seeded Selected | Seeded Global Best | Seeded Fit Time (s) |
| ------- | --------------: | -----------------: | ------------------: |
| Raw     |               5 |           1.000000 |             114.106 |
| Union   |               8 |           1.000000 |               6.418 |

## 8) Accuracy Evaluation (Comparing Raw vs Union)

- Notebook entry point(s):
- `notebook/DLBCL/8_accuracu_evaluate.ipynb`
- `notebook/DLBCL/8_accuracu_evaluate_union.ipynb`

[Insert Chart: Accuracy Comparison Raw vs Union]
![DLBCL Accuracy Evaluation](../../results/DLBCL/evaluation/plots/wrapper_sfs_comparison_sk_raw_seeded_raw_DLBCL.png)

**Caption:**

- Purpose: Compare accuracy across wrapper configurations (Sklearn SFS and Seeded SFS) for each data variant.
- How to read:
  - The x-axis shows configurations/methods, and the y-axis shows accuracy; higher values indicate better performance.
  - Vertical black lines (error bars) show Standard Deviation across cross-validation folds. Shorter bars indicate more stable model performance.

![DLBCL Accuracy Evaluation](../../results/DLBCL/evaluation/plots/wrapper_sfs_comparison_sk_union_seeded_union_DLBCL.png)

**Caption:**

- Purpose: Compare accuracy across wrapper configurations (Sklearn SFS and Seeded SFS) for each data variant.
- How to read:
  - The x-axis shows configurations/methods, and the y-axis shows accuracy; higher values indicate better performance.
  - Vertical black lines (error bars) show Standard Deviation across cross-validation folds. Shorter bars indicate more stable model performance.

- **Observation:** Seeded SFS achieves perfect accuracy (1.0000) in both raw and union variants.
- **Explanation:** The dataset's predictive signal is highly concentrated and consistently captured by seeded SFS.
- **Takeaway:** Use seeded as the primary configuration; either variant reaches ceiling performance.

- Raw best configuration: `seeded + LogReg`, mean accuracy **1.0000**, std 0.0000
- Union best configuration: `seeded + LogReg`, mean accuracy **1.0000**, std 0.0000

## 9) Time Evaluation (Comparing fit times for Raw vs Union)

- Notebook entry point(s):
- `notebook/DLBCL/9_time_evaluate.ipynb`
- `notebook/DLBCL/9_time_evaluate_union.ipynb`

[Insert Chart: Time Comparison Raw vs Union]
![DLBCL Time Evaluation](../../results/DLBCL/evaluation/plots/time_comparison_raw_seeded_vs_raw_sklearn.png)

**Caption:**

- Purpose: Compare training-time cost across wrapper methods on the same dataset.
- How to read: The x-axis shows methods/configurations, and the y-axis shows total fit time (ms); lower bars mean faster runtime.

![DLBCL Time Evaluation](../../results/DLBCL/evaluation/plots/time_comparison_union_seeded_vs_union_sklearn.png)

**Caption:**

- Purpose: Compare training-time cost across wrapper methods on the same dataset.
- How to read: The x-axis shows methods/configurations, and the y-axis shows total fit time (ms); lower bars mean faster runtime.

- **Observation:** Union runs are generally faster than raw runs across wrapper methods.
- **Explanation:** Union reduces candidate-space size, reducing total model-fit operations.
- **Takeaway:** Use union for rapid iteration; use raw when chasing peak wrapper score.

## 10) Final Evaluation (All Methods Comparison)

- Notebook entry point(s):
- `notebook/DLBCL/10_final_evaluate.ipynb`
- Report artifact: `results/DLBCL/evaluation/reports/final_evaluation_all_methods_dlbcl_DLBCL.txt`

[Insert Chart: Final Evaluation - All Methods]
![DLBCL Final Evaluation](../../results/DLBCL/evaluation/plots/final_evaluation_all_methods_dlbcl_DLBCL.png)

**Caption:**

- Purpose: Compare all feature selection methods (Filter, Ensemble, Sklearn SFS, Seeded SFS) with both LogReg and Tree models.
- How to read:
  - The x-axis lists all method/model combinations (e.g., "Sklearn_SFS_Raw + LogReg").
  - The y-axis shows cross-validation accuracy; higher bars indicate better performance.
  - Vertical error bars show Standard Deviation across folds; shorter bars indicate more stable models.

| Rank | Method + Model              | CV Folds | Mean Accuracy |    Std | Median |    Min |    Max |
| ---- | --------------------------- | -------: | ------------: | -----: | -----: | -----: | -----: |
| 1    | Seeded_SFS_Union + LogReg   |        5 |        1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| 1    | Seeded_SFS_Raw + LogReg     |        5 |        1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| 2    | MUTUAL_INFORMATION + LogReg |        5 |        0.9875 | 0.0280 | 1.0000 | 0.9375 | 1.0000 |
| 3    | None + LogReg               |        5 |        0.9733 | 0.0596 | 1.0000 | 0.8667 | 1.0000 |
| 4    | Sklearn_SFS_Raw + Tree      |        5 |        0.9608 | 0.0358 | 0.9375 | 0.9333 | 1.0000 |
| 5    | Sklearn_SFS_Union + LogReg  |        5 |        0.9483 | 0.0290 | 0.9375 | 0.9333 | 1.0000 |

**Key Observations:**

- Best configuration: Seeded_SFS_Union + LogReg and Seeded_SFS_Raw + LogReg both achieve 1.0000 accuracy (σ=0.0000)
- Seeded union is the most cost-effective (6.418s fit time vs 229.964s for sklearn raw)
