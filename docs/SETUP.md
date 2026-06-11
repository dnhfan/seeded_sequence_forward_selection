# Project Setup Guide

Step-by-step guide to set up this project from scratch so `import src.*` works from anywhere (scripts, notebooks, REPL).

## Prerequisites

- Python ≥ 3.8
- `git`
- `venv` (built-in) or `virtualenv`

## 1. Clone the repository

```bash
git clone <repo-url> <project-dir>
cd <project-dir>
```

## 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows: `.venv\Scripts\activate`

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

The project also needs these ML libraries (not all in `requirements.txt`):

```bash
pip install scikit-learn imbalanced-learn seaborn xgboost
```

Quick sanity check:

```bash
python -c "import sklearn, imblearn, seaborn, xgboost"
```

## 4. Understand `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "Seeded_Forward_Selection"
version = "0.1.0"

[tool.setuptools.packages.find]
include = ["src", "src.*"]
```

Key points:

- `[build-system]` — declares setuptools as the build backend.
- `[project]` — package metadata (name, version).
- `[tool.setuptools.packages.find]` — tells setuptools to discover `src` (a package thanks to `src/__init__.py`) and all its subpackages (`src.config`, `src.wrapper`, `src.utils`, `src.modeling`, `src.preprocess`).

> **Why not `where = ["src"]`?** That would look *inside* `src/` for packages, installing them as `wrapper`, `utils` (no `src.` prefix). Our notebooks use `import src.*` — so we discover from the root and include `src` itself.

## 5. Why `src/__init__.py`

An empty file at `src/__init__.py` makes Python treat `src/` as a regular package. Without it, `import src` won't resolve even if the package is installed.

## 6. Install the package in editable mode

```bash
pip install -e .
```

This installs `Seeded_Forward_Selection` in editable (development) mode. Changes to source files take effect immediately without reinstalling.

If you see `error: externally-managed-environment`, either:

- Use a virtual environment (recommended — see step 2).
- Or pass `--break-system-packages` (not recommended for production).

## 7. Verify the installation

From **any** directory:

```bash
cd /tmp && python -c "
import src
import src.config
import src.wrapper
import src.utils
import src.modeling
import src.preprocess
print('src:', src.__file__)
print('src.config:', src.config.__file__)
"
```

Expected output (paths will match your system):

```
src: /home/user/project/src/__init__.py
src.config: /home/user/project/src/config.py
```

## 8. Using in Jupyter notebooks

Notebooks opened from the activated venv automatically see the installed package — no `sys.path` hacks needed. Run notebooks as usual:

```bash
jupyter notebook
# or
jupyter lab
```

The `import src.config` and `from src.wrapper import ...` lines in `notebook/*/` scripts will resolve correctly from any working directory.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'src'` | `src/__init__.py` missing, or package not installed | Create `src/__init__.py`, run `pip install -e .` |
| Package installed but `import src.wrapper` fails | `where = ["src"]` used in config | Change to `include = ["src", "src.*"]` (no `where`) |
| `import wrapper` works but `import src.wrapper` doesn't | Same cause as above | Same fix as above |
| `pip install -e .` fails | Externally-managed environment | Activate venv first, or use `--break-system-packages` |
| Changes not reflected | Package not reinstalled after `__init__.py` changes | `pip install -e .` again |
| Notebook can't import `src` | Notebook kernel is not the venv's Python | Select correct kernel (the one from `.venv/`) |
