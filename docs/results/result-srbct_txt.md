# SRBCT_txt Results and Evaluation

[Back to index](../results.md)

## 1) EDA (Exploratory Data Analysis)

- Notebook entry point(s):
- `notebook/SRBCT_txt/01_eda.ipynb`

[Insert Chart: EDA Summary]
![SRBCT_txt EDA](../../results/SRBCT_txt/eda/plot/countplot.png)

**Caption:**
- Purpose: Check whether the dataset is imbalanced.
- How to read: The x-axis (V1) shows class labels (0 and 1), and the y-axis (count) shows the number of samples in each class.

## 2) Data Preprocessing

- Notebook entry point(s):
- `notebook/SRBCT_txt/02_preprocess.ipynb`
- Output location convention: `data/processed/SRBCT_txt/01_clean/`

## 3) Filter Selection

- Notebook entry point(s):
- `notebook/SRBCT_txt/03_filter_selection.ipynb`
- Report artifact: `results/SRBCT_txt/filter/reports/evaluation_SRBCT_txt.txt`

[Insert Chart: Filter Selection Comparison]
![SRBCT_txt Filter Selection](../../results/SRBCT_txt/filter/plots/evaluation_SRBCT_txt.png)

**Caption:**
- Purpose: Compare filter-method performance to select the best feature set for the next stage.
- How to read: The x-axis lists filter methods, and the y-axis shows evaluation scores; higher bars/scores indicate better methods.

## 4) Modeling (Filter-stage comparison)

- Notebook entry point(s):
- `notebook/SRBCT_txt/04_modeling.ipynb`
- Modeling outputs are tracked under `results/SRBCT_txt/filter/` when available.

## 5) Ensemble Filter (Voting + union feature set)

- Notebook entry point(s):
- `notebook/SRBCT_txt/05_ensemble.ipynb`
- Seed pool file: `data/processed/SRBCT_txt/03_ensemble/top50_features_voting.csv`
- Seed pool size: 10
- Top voting features: `V510(5)`, `V546(5)`, `V247(5)`, `V1956(5)`, `V1955(5)`

[Insert Chart: Ensemble Voting / Union Features]
![SRBCT_txt Ensemble Voting](../../results/SRBCT_txt/ensemble/plots/top50_features_voting.png)

**Caption:**
- Purpose: Show agreement among filter methods when voting for features.
- How to read: The x-axis lists feature names, and the y-axis shows vote counts; features with higher votes are prioritized.

## 6) Wrapper: Sklearn SFS (Raw vs Union execution)

- Script entry point(s):
- `notebook/SRBCT_txt/06_sklearn_sfs-raw.py`
- `notebook/SRBCT_txt/06_sklearn_sfs-union.py`

| Variant | Sklearn Selected | Sklearn Global Best | Sklearn Fit Time (ms) |
|---|---:|---:|---:|
| Raw | 4 | 1 | 151,507 |
| Union | 4 | 1 | 11,372 |

## 7) Wrapper: Seeded SFS (Raw vs Union execution)

- Script entry point(s):
- `notebook/SRBCT_txt/07_sfs-raw.py`
- `notebook/SRBCT_txt/07_sfs-union.py`

| Variant | Seeded Selected | Seeded Global Best | Seeded Fit Time (ms) |
|---|---:|---:|---:|
| Raw | 4 | 1 | 38,541 |
| Union | 4 | 1 | 4,077 |

## 8) Accuracy Evaluation (Comparing Raw vs Union)

- Notebook entry point(s):
- `notebook/SRBCT_txt/8_accuracu_evaluate.ipynb`
- `notebook/SRBCT_txt/8_accuracu_evaluate_union.ipynb`

[Insert Chart: Accuracy Comparison Raw vs Union]
![SRBCT_txt Accuracy Evaluation](../../results/SRBCT_txt/evaluation/plots/wrapper_sfs_comparison_sk_raw_seeded_raw_SRBCT_txt.png)

**Caption:**
- Purpose: Compare accuracy across wrapper configurations (Sklearn SFS and Seeded SFS) for each data variant.
- How to read:
  - The x-axis shows configurations/methods, and the y-axis shows accuracy; higher values indicate better performance.
  - Vertical black lines (error bars) show Standard Deviation across cross-validation folds. Shorter bars indicate more stable model performance.
![SRBCT_txt Accuracy Evaluation](../../results/SRBCT_txt/evaluation/plots/wrapper_sfs_comparison_sk_union_seeded_union_SRBCT_txt.png)

**Caption:**
- Purpose: Compare accuracy across wrapper configurations (Sklearn SFS and Seeded SFS) for each data variant.
- How to read:
  - The x-axis shows configurations/methods, and the y-axis shows accuracy; higher values indicate better performance.
  - Vertical black lines (error bars) show Standard Deviation across cross-validation folds. Shorter bars indicate more stable model performance.

- **Observation:** Accuracy remains constant while union runtime decreases substantially.
- **Explanation:** Union filtering does not remove core predictive features for this dataset.
- **Takeaway:** Use union seeded for faster cycles without loss in quality.

- Raw best configuration: `seeded + LogReg`, mean accuracy **0.9846**, std 0.0344
- Union best configuration: `seeded + LogReg`, mean accuracy **0.9846**, std 0.0344

## 9) Time Evaluation (Comparing fit times for Raw vs Union)

- Notebook entry point(s):
- `notebook/SRBCT_txt/9_time_evaluate.ipynb`
- `notebook/SRBCT_txt/9_time_evaluate_union.ipynb`

[Insert Chart: Time Comparison Raw vs Union]
![SRBCT_txt Time Evaluation](../../results/SRBCT_txt/evaluation/plots/time_comparison_raw_seeded_vs_raw_sklearn.png)

**Caption:**
- Purpose: Compare training-time cost across wrapper methods on the same dataset.
- How to read: The x-axis shows methods/configurations, and the y-axis shows total fit time (ms); lower bars mean faster runtime.
![SRBCT_txt Time Evaluation](../../results/SRBCT_txt/evaluation/plots/time_comparison_union_seeded_vs_union_sklearn.png)

**Caption:**
- Purpose: Compare training-time cost across wrapper methods on the same dataset.
- How to read: The x-axis shows methods/configurations, and the y-axis shows total fit time (ms); lower bars mean faster runtime.

- **Observation:** Union runs are generally faster than raw runs across wrapper methods.
- **Explanation:** Union reduces candidate-space size, reducing total model-fit operations.
- **Takeaway:** Use union for rapid iteration; use raw when chasing peak wrapper score.
