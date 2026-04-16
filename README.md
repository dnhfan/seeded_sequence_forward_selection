# Feature Selection Pipeline (Filter + Wrapper)

_Read this in [Vietnamese](README.vi.md)_

This repository provides an end-to-end, script-driven feature selection pipeline for multiple cancer datasets, including:

- Filter methods (variance, correlation, chi-squared, mutual information, ANOVA)
- Wrapper methods (Sklearn SFS, Seeded SFS)
- Two dataset variants: Raw and Union

**Note:** _Seeded SFS_ is a hybrid approach that combines Filter + Wrapper.

## Table of Contents

- [End-to-End Pipeline Stages](#end-to-end-pipeline-stages)
- [Folder Structure](#folder-structure)
- [Setup](#setup)
- [How to Run (End-to-End)](#how-to-run-end-to-end)
  - [1) Run notebook stages (EDA -> Ensemble)](#1-run-notebook-stages-eda---ensemble)
  - [2) Run wrapper scripts (raw + union)](#2-run-wrapper-scripts-raw--union)
  - [3) Run evaluation notebooks](#3-run-evaluation-notebooks)
- [Variant Rules (Raw vs Union)](#variant-rules-raw-vs-union)
- [Key Source Entry Points](#key-source-entry-points)
- [Known Gotchas](#known-gotchas)
- [Documentation Index](#documentation-index)

---

## End-to-End Pipeline Stages

![graph](./docs/graph.png)

Each dataset follows these stages:

1. EDA
2. Preprocess (can vary by dataset)
3. Filter Selection
4. Modeling (filter-stage comparison)
5. Ensemble Filter (voting + union feature set)
6. Wrapper: Sklearn SFS (raw/union)
7. Wrapper: Seeded SFS (raw/union)
8. Accuracy Evaluation (raw/union)
9. Time Evaluation (raw/union)

Canonical processed-data layout:

- `data/processed/<dataset>/01_clean`
- `data/processed/<dataset>/02_filter`
- `data/processed/<dataset>/03_ensemble`
- `data/processed/<dataset>/04_wrapper`

## Folder Structure

Top-level layout (trimmed to important paths):

```text
.
├── data/
│   ├── raw/
│   └── processed/
│       └── <dataset>/
│           ├── 01_clean/
│           ├── 02_filter/
│           ├── 03_ensemble/
│           └── 04_wrapper/
├── docs/
├── notebook/
│   └── <dataset>/
│       ├── 01_eda.ipynb
│       ├── ...
│       ├── 06_sklearn_sfs-raw.py
│       ├── 06_sklearn_sfs-union.py
│       ├── 07_sfs-raw.py
│       └── 07_sfs-union.py
├── results/
│   └── <dataset>/
│       ├── filter/
│       ├── wrapper/
│       │   └── <variant>/<algorithm>/run_YYYYMMDD_HHMMSS[_tag]/
│       │       ├── history/
│       │       ├── features/
│       │       ├── metrics/
│       │       └── artifacts/
│       └── evaluation/
├── src/
│   ├── config.py
│   ├── utils/
│   ├── wrapper/
│   ├── modeling/
│   └── filter/
├── README.md
├── README.vi.md
├── requirements.txt
└── run_sfs.sh
```

---

## Setup

Run from repository root:

```bash
pip install -r requirements.txt
```

Quick dependency check:

```bash
python -c "import sklearn, imblearn, seaborn, xgboost"
```

`requirements.txt` is intended to cover end-to-end runtime dependencies.

---

## How to Run (End-to-End)

## 1) Run notebook stages (EDA -> Ensemble)

For each dataset in `notebook/<dataset>/`, run notebooks in stage order.

Example dataset folder:

- `notebook/colon1/`

Typical files include:

- `01_eda.ipynb`
- `02_preprocess.ipynb`
- `03_filter_selection.ipynb`
- `04_modeling.ipynb`
- `05_esemble_filter.ipynb`

Naming is not fully consistent across datasets (for example `8_accuracu...`, `08_accuracy...`, and `7_.../8_...` in `Lung_cancer`).

## 2) Run wrapper scripts (raw + union)

Run from repo root (example: `colon1`):

```bash
python notebook/colon1/06_sklearn_sfs-raw.py
python notebook/colon1/07_sfs-raw.py
python notebook/colon1/06_sklearn_sfs-union.py
python notebook/colon1/07_sfs-union.py
```

You can also use helper scripts:

```bash
bash run_sfs.sh --list
bash run_sfs.sh colon1 raw sklearn
bash run_sfs.sh colon1 raw seeded
bash run_sfs.sh colon1 union sklearn
bash run_sfs.sh colon1 union seeded
```

Batch run sklearn wrapper scripts for all datasets:

```bash
bash run_all_sklearn_sfs.sh
```

## 3) Run evaluation notebooks

Per dataset, run:

- Accuracy comparison notebooks (`8_...evaluate...ipynb` or `08_...`)
- Time comparison notebooks (`9_time_evaluate...ipynb` or `8_time...` in `Lung_cancer`)

Both raw and union variants are available.

---

## Variant Rules (Raw vs Union)

- Keep raw and union artifacts strictly separated via `dataset_variant` (`raw`, `union`).
- Include variant tags in output names to avoid overwrite collisions.
- Accuracy notebook naming pattern:

```python
experiment_prefix=f"wrapper_sfs_comparison_sk_{sk_data_variant}_seeded_{data_variant}"
chart_title=f"Sklearn sfs({sk_data_variant}) vs Seeded sfs({data_variant}) Performance"
```

---

## Key Source Entry Points

- Path conventions and directory structure: `src/config.py` (`ProjectPath`)
- Wrapper run path contract: `src/utils/experiment_paths.py`, `src/wrapper/base.py`
- Evaluation logic: `src/modeling/evaluation.py`
- Union feature generation: `src/utils/utils.py` (`create_union_features`)

---

## Known Gotchas

- Notebook naming is inconsistent across datasets; preserve local naming conventions.
- Many notebooks are edited directly as JSON; avoid unnecessary metadata/id churn.
- Wrapper scripts rely on `sys.path` injection and should be run from repo root.

---

## Documentation Index

You can find out how each **pipeline** work at [Pipeline Documentation](./docs/pipelines/README.md)
You can look at the **results** in [Results Report](./docs/results/README.md)
