# Wrapper Pipeline - Sklearn Sequential Feature Selector (SFS)
**1. Overview / Purpose:**  
This pipeline runs the standard scikit-learn Sequential Feature Selector as the reference wrapper baseline. It measures how well standard forward SFS performs (accuracy + fit time) on both Raw and Union feature pools, then saves standardized experiment artifacts for fair comparison against custom Seeded SFS.

**2. Inputs:**  
- Input dataset DataFrame format: first column target, remaining columns features.
- Data variants:
  - **Raw**: loaded from `data/raw/<dataset>.csv`
  - **Union**: generated via `create_union_features(...)` from filter outputs
- Runner scripts (example dataset):
  - `notebook/<dataset>/06_sklearn_sfs-raw.py`
  - `notebook/<dataset>/06_sklearn_sfs-union.py`
- Core classes/files:
  - `src/wrapper/sklearn_sfs.py` (`SklearnSFSSelector`)
  - `src/wrapper/base.py` (`BaseWrapperSelector`)
  - `src/wrapper/models.py` (`get_model`)
  - `src/utils/experiment_paths.py` (run folder contract)

**3. Step-by-Step Execution Logic:**  
1. **Initialize wrapper selector**  
   Notebook creates `SklearnSFSSelector(...)` with:
   - `data_name`, `n_features`
   - `dataset_variant` (`raw` or `union`)
   - `voting_csv_name` (kept for interface consistency)
   - timer config (`using_timer=True`, `unit="ms"`)

2. **Prepare dataset**  
   `BaseWrapperSelector.run_sfs()` splits DataFrame:
   - `y_in = df.iloc[:, 0]`
   - `X_in = df.iloc[:, 1:]`  
   This enforces a single data contract for all wrappers.

3. **Build SFS runtime config**  
   Parameters include:
   - `max_features` (`"auto"` or integer)
   - `cv`
   - `model` (mapped via `get_model`)
   - `scoring` (commonly accuracy)
   - `direction` (forward by default)

4. **Create and fit sklearn SFS**  
   In `_execute_core()`:
   - create deterministic `StratifiedKFold(shuffle=True, random_state=42)`,
   - instantiate `SequentialFeatureSelector` with given model/scoring/cv,
   - fit selector with timer (`TimerContext`) to capture total fit duration.

5. **Extract selected features and build final dataset**  
   After fit:
   - get support mask from selector,
   - create `X_selected_df`,
   - concatenate target and selected features into final output DataFrame.

6. **Recompute global best score on selected subset**  
   Since sklearn SFS does not expose full iteration history, pipeline explicitly runs `cross_val_score` on final selected set to compute `global_best_score` (mean CV score).

7. **Write history text (summary style)**  
   A text report is generated with:
   - configuration
   - total fit time
   - final best score
   - selected features  
   It explicitly states that per-iteration logs are unavailable in sklearn SFS.

8. **Persist artifacts via base wrapper contract**  
   `BaseWrapperSelector._save_sfs_output()` writes:
   - processed wrapper CSV in `data/processed/.../04_wrapper/...`
   - run artifacts in `results/<dataset>/wrapper/<variant>/sklearnsfsselector/run_<timestamp>/...`
   - metrics including `total_fit_time_ms`

**4. Outputs / Artifacts:**  
- Selected wrapper dataset:
  - `data/processed/<dataset>/04_wrapper/<variant>/sklearnsfsselector/<dataset>_sklearnsfsselector_*.csv`
- Run artifact structure:
  - `results/<dataset>/wrapper/<variant>/sklearnsfsselector/run_YYYYMMDD_HHMMSS/`
    - `history/history.txt`
    - `features/selected_features.csv`
    - `metrics/metrics.json`
    - `metrics/metrics.csv` (contains `total_fit_time_ms`)
- Key recorded metrics:
  - `n_features_selected`
  - `global_best_score`
  - `total_fit_time_ms`
  - dataset + variant metadata
