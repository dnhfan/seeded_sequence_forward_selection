# Evaluation Metrics Pipeline (Accuracy + Fit Time, Raw vs Union)
**1. Overview / Purpose:**  
This pipeline evaluates and compares feature-selection outcomes across methods, models, and data variants (Raw vs Union). It reports predictive performance (accuracy statistics) and training efficiency (fit time), enabling fair comparison between Seeded SFS and Sklearn SFS.

**2. Inputs:**  
- Evaluation engine:
  - `src/modeling/evaluation.py` (`ModelEvaluator`)
- Accuracy evaluation inputs:
  - Any CSV where first column is target and remaining columns are features
  - Common sources:
    - filtered outputs (`02_filter`)
    - wrapper outputs (`04_wrapper`)
    - custom files passed via `evaluate_custom_file(...)`
- Time evaluation inputs:
  - `metrics/metrics.csv` from wrapper run folders, each containing `total_fit_time_ms`
- Typical compared dimensions:
  - Method labels (e.g., `Sklearn_SFS_Raw`, `Seeded_SFS_Union`)
  - Models: Logistic Regression (`LogReg`) and Decision Tree (`Tree`)
  - Variants: Raw vs Union

**3. Step-by-Step Execution Logic:**  
1. **Initialize evaluator**  
   `ModelEvaluator(data_name, n_features, max_iter, use_scaler, custom_base_dir)` prepares output dirs:
   - reports dir
   - plots dir  
   If `custom_base_dir` is provided, artifacts are routed there (used by wrapper comparison notebooks).

2. **Load data uniformly**  
   `_load_data(file_path)` enforces a standard schema:
   - `y = first column`
   - `X = remaining columns`

3. **Run cross-validated model training (`_train_and_evaluate`)**  
   For each method label:
   - build deterministic `StratifiedKFold(shuffle=True, random_state=42)`,
   - define models:
     - Logistic Regression (optionally wrapped in `StandardScaler` pipeline)
     - Decision Tree (`max_depth=5`)
   - run `cross_validate(..., scoring="accuracy")`,
   - store fold-level accuracy for each model and fold.

4. **Evaluate either grouped sources or explicit files**  
   - `evaluate_filtered_features(data_dir)` loops filter methods automatically.
   - `evaluate_baseline(raw_path)` evaluates full raw feature baseline.
   - `evaluate_custom_file(file_path, method_label)` supports wrapper outputs and arbitrary experiments.

5. **Generate accuracy report + chart (`generate_report_and_plot`)**  
   - Build fold-level DataFrame from all stored results.
   - Create bar chart (`Method` x `Model` with accuracy labels).
   - Compute grouped summary statistics:
     - mean, std, min, max, median, fold count
   - Derive ranking and stability:
     - `cv_stability = 1 - std_accuracy`
     - dense rank by mean accuracy descending
   - Save:
     - plot PNG
     - text report with metadata + executive summary + ranked summary + fold-level audit table.

6. **Compare fit time between wrappers (`plot_fit_time_comparison`)**  
   - Read two metrics CSV files (seeded vs sklearn).
   - Validate required column: `total_fit_time_ms`.
   - Build comparison table (ms + converted seconds).
   - Plot 2-bar chart and annotate execution times.
   - Save chart to evaluation plot directory (or custom `save_dir`).

7. **Raw vs Union comparison protocol**  
   In wrapper comparison notebooks, method labels encode variant identity and charts/reports include variant-specific prefixes (for example `..._raw` or `..._union`) to avoid overwriting and to preserve side-by-side traceability.

**4. Outputs / Artifacts:**  
- Accuracy artifacts:
  - `.../plots/<experiment_prefix>_<dataset>.png`
  - `.../reports/<experiment_prefix>_<dataset>.txt`
  - Report includes:
    - best method/model configuration
    - per-method/model summary statistics
    - fold-level results for auditability
- Time artifacts:
  - `.../plots/time_comparison_*.png`
  - returned comparison table containing:
    - `algorithm`
    - `total_fit_time_ms`
    - `total_fit_time_sec`
- Upstream dependency for time:
  - Wrapper run-level `metrics/metrics.csv` from:
    - `results/<dataset>/wrapper/raw/.../metrics/metrics.csv`
    - `results/<dataset>/wrapper/union/.../metrics/metrics.csv`
