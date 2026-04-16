# Pipelines Documentation Index

This folder documents the end-to-end feature selection and evaluation workflow used in this project.

## Pipeline Order

1. Filter Selection
   - English: [01_filter_selection.md](01_filter_selection.md)
   - Tiếng Việt: [01_filter_selection.vi.md](01_filter_selection.vi.md)

2. Ensemble Voting
   - English: [02_ensemble_voting.md](02_ensemble_voting.md)
   - Tiếng Việt: [02_ensemble_voting.vi.md](02_ensemble_voting.vi.md)

3. Wrapper - Sklearn SFS
   - English: [03_wrapper_sklearn_sfs.md](03_wrapper_sklearn_sfs.md)
   - Tiếng Việt: [03_wrapper_sklearn_sfs.vi.md](03_wrapper_sklearn_sfs.vi.md)

4. Wrapper - Seeded SFS
   - English: [04_wrapper_seeded_sfs.md](04_wrapper_seeded_sfs.md)
   - Tiếng Việt: [04_wrapper_seeded_sfs.vi.md](04_wrapper_seeded_sfs.vi.md)

5. Evaluation Metrics (Accuracy + Time)
   - English: [05_evaluation_metrics.md](05_evaluation_metrics.md)
   - Tiếng Việt: [05_evaluation_metrics.vi.md](05_evaluation_metrics.vi.md)

## Notes

- Data flow is standardized via `ProjectPath` (`src/config.py`).
- Wrapper run artifacts follow `results/<dataset>/wrapper/<variant>/<algorithm>/run_YYYYMMDD_HHMMSS[_tag]/`.
- Time comparison reads `metrics/metrics.csv` and uses `total_fit_time_ms`.
