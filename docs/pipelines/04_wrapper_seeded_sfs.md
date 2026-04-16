# Wrapper Pipeline - Custom Seeded Forward Selection (Seeded SFS)

_Read in [Vietnamese](04_wrapper_seeded_sfs.vi.md)._

**1. Overview / Purpose:**  
This pipeline executes a custom Seeded Forward Selection algorithm that starts from ensemble-voted seed features and expands feature subsets greedily using cross-validated performance. It is designed to improve search efficiency and stability compared to blind forward selection.

You can read how this algorithm work more specific in [SeededForwardSelection](../seeded_sfs_core.md)
**2. Inputs:**

- Input dataset DataFrame: first column target, remaining columns features.
- Seed source CSV (from ensemble voting):
  - `data/processed/<dataset>/03_ensemble/top<n_features>_features_voting.csv`
  - required columns: `Feature`, `Votes`
- Main runtime parameters:
  - `n_seeds` (number of starting features from voting file)
  - `max_features`
  - `patience` (early stop when no global improvement)
  - `model`, `scoring`, `cv`, `random_state`
  - `n_jobs` for candidate parallel evaluation
- Runner scripts:
  - `notebook/<dataset>/07_sfs-raw.py`
  - `notebook/<dataset>/07_sfs-union.py`
- Core classes/files:
  - `src/wrapper/seeded.py` (`SeededSFSSelector`)
  - `src/wrapper/forward_selection.py` (`SeededForwardSelection`)
  - `src/wrapper/base.py` (orchestration + artifact saving)
  - `src/utils/utils.py` (`load_seed_from_csv`, `validate_features`)

**3. Step-by-Step Execution Logic:**

1. **Initialize wrapper and resolve seed file**  
   `SeededSFSSelector` builds seed CSV path from `ProjectPath.ensemble_dir / voting_csv_name`.

2. **Split input data and pass config**  
   `run_sfs()` in base wrapper splits `[y, X]`, builds parameter dictionary, and forwards all seeded-specific args (`n_seeds`, `patience`, etc.) into core selector construction.

3. **Initialize SeededForwardSelection internals**  
   During `fit()`:
   - resolve model instance (`get_model` if string alias),
   - map feature names to indices,
   - load top-`n_seeds` from voting CSV (or provided list),
   - validate all seeds exist in current dataset columns.

4. **Freeze deterministic CV splits once**  
   `_build_cv()` creates `StratifiedKFold` (or `KFold`), then precomputes/freeze splits in `_cv_splits_`.  
   This keeps candidate comparisons fair because every candidate is scored on identical folds.

5. **Compute baseline score from seed set**  
   The initial seed subset is evaluated first.  
   Baseline initializes:
   - `current_score`
   - `global_best_score`
   - `global_best_indices`

6. **Iterative forward expansion**  
   For each iteration:
   - generate candidates = all not-yet-selected features,
   - evaluate each candidate subset (`selected + candidate`) using CV mean score,
   - run candidate scoring in parallel with `joblib.Parallel(n_jobs=self.n_jobs)`,
   - pick best candidate and append to selected set,
   - record step improvement and iteration timing,
   - update global best + patience counter.

7. **Stopping criteria**  
   The loop stops if:
   - no candidates remain, or
   - `max_features` reached, or
   - `patience` exhausted (no new global peak for N iterations).

8. **Global-best rollback behavior**  
   Final selected features are taken from `global_best_indices`, not necessarily from the last iteration.  
   This prevents keeping late-added features that degraded the score.

9. **Generate report and return SFSResult**  
   The selector builds a detailed text report including iteration logs and timings, then returns:
   - final selected dataset
   - selected feature list
   - total fit time
   - global best score

10. **Persist artifacts (same wrapper contract)**  
    Base wrapper saves processed CSV and experiment artifacts under standardized run folders.

**4. Outputs / Artifacts:**

- Selected wrapper dataset:
  - `data/processed/<dataset>/04_wrapper/<variant>/seededsfsselector/<dataset>_seededsfsselector_*.csv`
- Run artifacts:
  - `results/<dataset>/wrapper/<variant>/seededsfsselector/run_YYYYMMDD_HHMMSS/`
    - `history/history.txt` (full iteration log)
    - `features/selected_features.csv`
    - `metrics/metrics.json`
    - `metrics/metrics.csv` (includes `total_fit_time_ms`)
- Key behavior-specific outputs:
  - iteration-level score trajectory
  - per-iteration elapsed time
  - global-best subset retained for final output
