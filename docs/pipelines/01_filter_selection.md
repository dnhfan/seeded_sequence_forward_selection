# Filter Selection Pipeline
_Read in [Vietnamese](01_filter_selection.vi.md)._ 

**1. Overview / Purpose:**  
This pipeline performs the first-stage feature reduction using multiple filter-based methods. Its purpose is to quickly remove weak or redundant features before expensive wrapper methods run, while preserving diverse candidate signals for later ensemble voting and SFS.

**2. Inputs:**  
- Raw dataset CSV with structure: first column = target label (`y`), remaining columns = features (`X`)  
  - Canonical path pattern: `data/raw/<dataset>.csv` (via `ProjectPath.raw_path`)  
- Configuration parameters:
  - `data_name` (dataset id)
  - `n_features` (top-k features to keep per method, commonly 50)
  - `method` in:
    - `variance`
    - `correlation`
    - `chi_squared`
    - `mutual_information`
    - `anova_f_test`
  - `random_state` (used by MI)
- Core implementation:
  - `src/filter/filter_selection.py`
  - `src/filter/filter_algorithms.py`
  - `src/utils/Timer.py` (`TimerContext` — used to measure per-method wall-clock time)

**3. Step-by-Step Execution Logic:**  
1. **Initialize the selector (`FeatureFilter`)**  
   The pipeline creates `FeatureFilter(method, n_features, random_state)`.  
   It validates the method name and prepares internal state (`selected_features_`, `feature_scores_`, `metrics_`).

2. **Split input data into `X` and `y`**  
   The target stays separate because several filters are supervised (need labels), while variance is unsupervised.

3. **Run method-specific scoring logic**  
   Depending on `method`, the selector dispatches to one algorithm:
   - **Variance (`calc_variance`)**: compute per-feature variance and rank descending.
   - **Correlation (`calc_correlation`)**:  
     a) build absolute feature-feature correlation matrix,  
     b) drop highly correlated columns (`|corr| > 0.95`),  
     c) rank remaining features by ANOVA F-score against target.  
     This prevents selecting many near-duplicate genes/features.
   - **Chi-squared (`calc_chi_squared`)**:  
     a) detect negative values,  
     b) if negatives exist, apply `MinMaxScaler` to make values non-negative,  
     c) compute chi2 scores with `SelectKBest`.  
     This step ensures chi2 assumptions are respected.
   - **Mutual Information (`calc_mutual_info`)**: estimate non-linear dependency between each feature and target using `mutual_info_classif`.
   - **ANOVA F-test (`calc_anova`)**: compute F-statistics with `SelectKBest(f_classif)`.

   Each method dispatch is wrapped in a `TimerContext` that records elapsed time in milliseconds. After each method completes, a metrics entry `{algorithm, elapsed_ms, n_features_selected}` is stored in `self.metrics_`.

4. **Select top-k features**  
   For each method, features are sorted by score and top `n_features` are retained.  
   If requested `n_features` exceeds available features, it is clipped safely.

5. **Transform dataset**  
   `transform()` returns `X` restricted to selected columns.  
   `fit_transform()` executes fit + transform in one call for convenience.

6. **Persist filtered CSV**  
   `save_filtered_data()` reconstructs `[target + selected features]` and writes to:  
   `data/processed/<dataset>/02_filter/<dataset>_<method>_<n_features>features.csv`

7. **Persist timing metrics (optional)**  
   `save_metrics(data_name, output_dir, append=True)` exports `metrics_` to:  
   `results/<dataset>/filter/metrics/metrics.csv`  
   Each row contains `algorithm`, `elapsed_ms`, and `n_features_selected` for that run.  
   - If `append=True` (default) and the file already exists, new rows are concatenated onto the existing CSV rather than overwriting it. This allows accumulating metrics across multiple `FeatureFilter` runs (e.g., running all 5 methods sequentially and collecting results in one file).  
   - If `append=False`, the file is overwritten each time.

**4. Outputs / Artifacts:**  
- One filtered CSV per method in `02_filter/`:
  - Example: `data/processed/colon1/02_filter/colon1_anova_f_test_50features.csv`
- In-memory artifacts after fit:
  - `selected_features_` (ordered selected feature names)
  - `feature_scores_` (feature -> score mapping)
  - `metrics_` (per-algorithm timing dict: `{algorithm, elapsed_ms, n_features_selected}`)
  - optional scaler object for chi2 path
- Optional timing artifact:
  - `results/<dataset>/filter/metrics/metrics.csv` (written by `save_metrics()`)
- These per-method filtered files become inputs for:
  - Ensemble voting (`03_ensemble`)
  - Union feature generation
  - Baseline filter-method evaluation
