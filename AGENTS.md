# AGENTS.md

## Repo Reality Check
- This is a plain Python repo (no `pyproject.toml`, no `Makefile`, no CI workflow, no pre-commit config).
- Main runnable workflow is script/notebook-driven, especially under `notebook/<dataset>/`.
- Existing guidance files besides this one are effectively absent; trust code paths in `src/` and scripts in `notebook/`.

## Environment + Dependencies
- Install from `requirements.txt`, then manually install missing ML deps used by source code: `scikit-learn`, `imbalanced-learn`, `seaborn`, `xgboost`.
- Quick sanity check command before running wrappers/evaluation:
  - `python -c "import sklearn, imblearn, seaborn, xgboost"`

## Canonical Pipeline Layout (Do Not Invent New Paths)
- Path conventions are centralized in `src/config.py` (`ProjectPath`). Use it instead of hardcoded paths.
- Processed data is under `data/processed/<dataset>/` by stage:
  - `01_clean`, `02_filter`, `03_ensemble`, `04_wrapper`
- Human-readable experiment outputs are under `results/<dataset>/...`.

## Wrapper Output Contract
- Wrapper runs are structured by `src/utils/experiment_paths.py` + `src/wrapper/base.py`:
  - `results/<dataset>/wrapper/<variant>/<algorithm>/run_YYYYMMDD_HHMMSS[_tag]/`
  - Subdirs: `history/`, `features/`, `metrics/`, `artifacts/`
- Metrics consumed by time-evaluate notebooks are always `metrics/metrics.csv` with `total_fit_time_ms`.

## How Wrapper Scripts Actually Run
- Dataset scripts live in `notebook/<dataset>/` as Python scripts:
  - `06_sklearn_sfs-raw.py`, `07_sfs-raw.py`
  - `06_sklearn_sfs-union.py`, `07_sfs-union.py`
- They rely on `sys.path` injection to import `src.*`; run from repo root to avoid path surprises:
  - `python notebook/colon1/06_sklearn_sfs-raw.py`
- Union scripts call `create_union_features(...)` from `src/utils/utils.py` and regenerate `data/processed/<dataset>/03_ensemble/<dataset>_Union_<n>features.csv`.

## Variant Rules (Raw vs Union)
- Keep raw and union artifacts separated by `dataset_variant` (`raw` / `union`) in wrapper/evaluation scripts.
- For comparison notebooks, always tag report/plot names with variant to avoid overwrite.
  - Current pattern in accuracy notebooks:
    - `experiment_prefix=f"wrapper_sfs_comparison_sk_{sk_data_variant}_seeded_{data_variant}"`
    - `chart_title=f"Sklearn sfs({sk_data_variant}) vs Seeded sfs({data_variant}) Performance"`

## Evaluation Entry Points
- Accuracy evaluation logic is in `src/modeling/evaluation.py` (`ModelEvaluator`).
- Time comparison uses `ModelEvaluator.plot_fit_time_comparison(...)`; pass explicit metrics CSV paths.
- `generate_report_and_plot(...)` writes to `<custom_base_dir>/reports` and `<custom_base_dir>/plots` if `custom_base_dir` is provided.

## High-Value Verification Commands
- Syntax-check key module after edits:
  - `python -m py_compile src/modeling/evaluation.py`
- Validate wrapper outputs exist for one dataset/variant before notebook edits:
  - `python - <<'PY'\nfrom pathlib import Path\np=Path('results/colon1/wrapper/union')\nprint(p.exists(), list(p.glob('*/*/metrics/metrics.csv'))[:3])\nPY`
- When bulk-editing notebooks, verify expected pattern via ripgrep (or equivalent):
  - check `data_variant = "union"` in union notebooks
  - check metrics paths include `/wrapper/union/`

## Known Gotchas
- `requirements.txt` does not include all imports used by `src/`.
- Dataset notebook naming is inconsistent (`8_accuracu...` typo, Brain uses `08_accuracy...`, Lung_cancer uses `7_...`/`8_...`). Keep local naming conventions when adding sibling files.
- Some notebooks are generated/edited JSON directly; preserve notebook metadata/ids where possible and avoid unnecessary churn.
