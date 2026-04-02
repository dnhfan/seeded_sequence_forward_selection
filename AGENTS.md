# AGENTS.md - Coding Agent Guidelines

This document provides essential information for AI coding agents working in this repository.

## Project Overview

**Type:** Python Data Science / Machine Learning Project  
**Focus:** Feature selection combining wrapper and filter methods for cancer genomics datasets  
**Python Version:** 3.12.12  
**Environment:** Uses `.venv` with `uv` package manager

**Directory Structure:**

```
wrapper-w-filter/
├── data/
│   ├── raw/          # 16 cancer genomics CSV datasets (target in first column)
│   └── processed/      # For processed/intermediate data
├── notebook/         # Jupyter notebooks for exploratory analysis
│   └── 01_eda.ipynb
├── result/           # Output directory for results (models, reports, plots)
├── src/              # Source code modules
│   ├── preprocessing.py      # Data loading and SMOTE balancing
│   └── filter_selection.py   # Feature selection (skeleton only)
└── requirements.txt
```

## Build, Test, and Lint Commands

### Environment Setup

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies (requirements.txt is incomplete!)
pip install -r requirements.txt

# Install missing critical dependencies
pip install scikit-learn imbalanced-learn seaborn scipy

# Verify installation
pip list | grep -E "scikit-learn|imbalanced-learn|seaborn"
```

### Running Code

```bash
# Run preprocessing module (manual test)
python src/preprocessing.py

# Run specific module
python src/<module_name>.py

# Run Jupyter notebooks
jupyter notebook notebook/01_eda.ipynb
```

### Testing

**Current state:** No formal test framework configured.

```bash
# Manual testing via __main__ blocks
python src/preprocessing.py

# Future: Add pytest
# pip install pytest
# pytest tests/test_preprocessing.py          # Run specific test file
# pytest tests/test_preprocessing.py::test_load  # Run single test
# pytest -v                                   # Verbose output
# pytest -k "test_smote"                      # Run tests matching pattern
```

### Linting and Formatting

**Current state:** No linting/formatting tools configured.

```bash
# Recommended setup:
# pip install ruff black mypy

# Ruff (fast linter + formatter)
# ruff check .                # Lint all files
# ruff check src/             # Lint specific directory
# ruff check --fix .          # Auto-fix issues
# ruff format .               # Format code

# Black (code formatter)
# black src/                  # Format source code
# black --check src/          # Check without modifying

# MyPy (type checker)
# mypy src/                   # Type check source code
# mypy --strict src/          # Strict type checking
```

## Code Style Guidelines

### Import Organization

Order imports in three groups with blank lines between:

1. Standard library imports
2. Third-party imports (numpy, pandas, sklearn, etc.)
3. Local application imports

```python
import os
import sys

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from src.preprocessing import load_and_preprocess
```

### Naming Conventions

- **Functions/Variables:** `snake_case` (e.g., `load_and_preprocess`, `x_resampled`)
- **Classes:** `PascalCase` (e.g., `FeatureFilter`, `WrapperSelector`)
- **Constants:** `UPPER_SNAKE_CASE` (e.g., `RANDOM_STATE`, `DEFAULT_TARGET_INDEX`)
- **Private methods:** `_leading_underscore` (e.g., `_validate_input`)

### Type Hints

- Use type hints for function signatures when clear
- Use `# type: ignore` to suppress mypy warnings when needed (e.g., with SMOTE)
- For complex types, import from `typing` module

```python
from typing import Tuple
import pandas as pd

def load_and_preprocess(filepath: str, target_index: int = 0) -> Tuple[pd.DataFrame, pd.Series]:
    """Load and preprocess dataset."""
    # ... code using imblearn that may need type: ignore
    x_resampled, y_resampled = smote.fit_resample(x, y)  # type: ignore
    return x_final, y_resampled
```

### Docstrings

Use triple-quoted docstrings with informal parameter descriptions:

```python
def load_and_preprocess(filepath, target_index=0):
    """
    Pipeline for clean and balances the dataset:
        - filepath: data filepath
        - target_index: index of target column (default: 0)
    """
```

For more complex functions, consider NumPy-style docstrings with Parameters/Returns sections.

### Formatting

- **Indentation:** 4 spaces (no tabs)
- **Line length:** Aim for 88 characters (Black default), max 100
- **Blank lines:** 2 blank lines before top-level function definitions
- **Quotes:** Prefer double quotes for strings

### Error Handling

- Use explicit exception handling for I/O operations:

  ```python
  try:
      df = pd.read_csv(filepath)
  except FileNotFoundError:
      print(f"Error: File not found: {filepath}")
      raise
  except pd.errors.EmptyDataError:
      print(f"Error: File is empty: {filepath}")
      raise
  ```

- Let pandas/sklearn raise their own exceptions for invalid data
- Add informative error messages for user-facing functions

### Print Statements and Logging

- Use f-strings for formatted output
- Include emoji/icons for visual clarity (terminal-friendly): `󰄲`, `󰓫`
- Print progress information for long-running operations:

```python
print(f"󰄲 Preprocessed file: {filepath}")
print(f"󰓫 X.Shape before preprocess: {x.shape}")
print(f"󰓫 X.Shape after preprocess: {x_final.shape}")
```

### Module Testing Pattern

Include manual testing block at the end of each module:

```python
if __name__ == "__main__":
    # Test with sample dataset
    x_train, y_train = load_and_preprocess("data/raw/Lung_cancer.csv")
    print(f"Loaded {len(x_train)} samples with {len(x_train.columns)} features")
```

## Data Workflow

### Input Data

- Location: `data/raw/*.csv`
- Format: CSV with target variable in first column (default: `target_index=0`)
- 16 cancer genomics datasets available
- High dimensionality (thousands of gene expression features)

### Processing Pipeline

1. Load raw CSV data
2. Split features (X) and target (y)
3. Apply SMOTE for class balancing (random_state=42)
4. Return balanced DataFrames

### Output Structure

Store results in `result/` directory:

- Models: `*.pkl` files
- Reports: `*.txt` files with metrics
- Visualizations: `*.png` files

## Important Notes for Agents

1. **Missing Dependencies:** `requirements.txt` is incomplete. Always verify critical packages are installed:
   - `scikit-learn` (not in requirements.txt)
   - `imbalanced-learn` (not in requirements.txt)
   - `seaborn` (not in requirements.txt)

2. **Random State:** Use `random_state=42` for all random operations (SMOTE, train/test split, etc.)

3. **No Git:** This is not currently a git repository

4. **Testing Infrastructure:** No formal testing setup exists. When adding tests:
   - Create `tests/` directory
   - Use pytest framework
   - Name test files: `test_<module>.py`
   - Use fixtures for sample data

5. **Code Organization:** Keep preprocessing, feature selection, and evaluation in separate modules

6. **Performance:** Datasets are large. Consider memory usage and use incremental processing when possible

## Common Tasks

### Adding a New Feature Selection Method

1. Create function/class in appropriate module (`src/filter_selection.py` or new module)
2. Follow naming conventions and docstring format
3. Add manual test in `if __name__ == "__main__"` block
4. Test with small dataset first (e.g., `colon1.csv` is smallest)

### Creating Output Files

```python
import os

# Ensure output directory exists
os.makedirs("result", exist_ok=True)

# Save results
output_path = "result/model_name.pkl"
# ... save logic
```

### Working with Datasets

```python
# Load and preprocess
from src.preprocessing import load_and_preprocess

x, y = load_and_preprocess("data/raw/Lung_cancer.csv", target_index=0)

# Verify data shape
print(f"Features: {x.shape[1]}, Samples: {x.shape[0]}")
```
