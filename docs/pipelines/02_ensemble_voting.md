# Ensemble Voting Pipeline
**1. Overview / Purpose:**  
This pipeline aggregates selected features from multiple filter methods and applies a vote-count mechanism to identify robust "seed" features. The purpose is to create a consensus ranking that initializes Seeded SFS with strong starting points.

**2. Inputs:**  
- Filtered CSV files produced in stage `02_filter`, one per method:
  - `data/processed/<dataset>/02_filter/<dataset>_<method>_<n_features>features.csv`
- Parameters:
  - `data_name`
  - `valid_methods` (typically 5 methods)
  - `n_features` (used in file naming and output naming)
  - `data_dir` (source folder containing filtered CSVs)
- Core implementation:
  - `src/filter/ensemble.py` (`EnsembleFeatureSelector`)

**3. Step-by-Step Execution Logic:**  
1. **Initialize ensemble selector**  
   `EnsembleFeatureSelector(data_name, valid_methods, n_features, data_dir)` sets paths using `ProjectPath` and ensures output directories exist:
   - report dir (`results/.../ensemble/reports`)
   - plot dir (`results/.../ensemble/plots`)
   - csv dir (`data/processed/.../03_ensemble`)

2. **Collect selected columns from each filter output**  
   In `run_voting()`, for each method:
   - read only CSV header (`nrows=0`) for efficiency,
   - ignore first column (target),
   - collect selected feature names from remaining columns.

3. **Count votes across methods**  
   Features are pooled from all methods and counted using `Counter`.  
   A feature receives higher votes if it appears in more methods, which indicates cross-method robustness.

4. **Build ranked voting table**  
   Create DataFrame with columns:
   - `Feature`
   - `Votes`  
   Then sort descending by `Votes`.

5. **Generate artifacts (`generate_report_and_plot`)**  
   - Keep top-N rows for visualization/reporting (`top_n_plot`).
   - Save bar chart of votes.
   - Save human-readable text report.
   - Save CSV containing ranked top voted features.

6. **Use voting output as Seeded SFS initialization source**  
   Seeded SFS later reads `Feature` column from this ranking file and takes top `n_seeds` entries as starting features (`load_seed_from_csv`).

**4. Outputs / Artifacts:**  
- Machine-readable seed ranking CSV:
  - `data/processed/<dataset>/03_ensemble/top<n_features>_features_voting.csv`
- Human-readable artifacts:
  - `results/<dataset>/ensemble/plots/top<n_features>_features_voting.png`
  - `results/<dataset>/ensemble/reports/top<n_features>_features_voting.txt`
- In-memory ranked DataFrame (`df_counts`) used for downstream logic.
