#!/usr/bin/env bash
set -euo pipefail

scripts=(
  "notebook/adenocarcinoma/06_sklearn_sfs-raw.py"
  "notebook/adenocarcinoma/06_sklearn_sfs-union.py"
  "notebook/Brain/06_sklearn_sfs-raw.py"
  "notebook/Brain/06_sklearn_sfs-union.py"
  "notebook/Breast2classes/06_sklearn_sfs-raw.py"
  "notebook/Breast2classes/06_sklearn_sfs-union.py"
  "notebook/Breast3classes/06_sklearn_sfs-raw.py"
  "notebook/Breast3classes/06_sklearn_sfs-union.py"
  "notebook/CNS1/06_sklearn_sfs-raw.py"
  "notebook/CNS1/06_sklearn_sfs-union.py"
  "notebook/colon1/06_sklearn_sfs-raw.py"
  "notebook/colon1/06_sklearn_sfs-union.py"
  "notebook/DLBCL/06_sklearn_sfs-raw.py"
  "notebook/DLBCL/06_sklearn_sfs-union.py"
  "notebook/Leukemia_3c1/06_sklearn_sfs-raw.py"
  "notebook/Leukemia_3c1/06_sklearn_sfs-union.py"
  "notebook/Leukemia_4c1/06_sklearn_sfs-raw.py"
  "notebook/Leukemia_4c1/06_sklearn_sfs-union.py"
  "notebook/Lung_cancer/06_sklearn_sfs-raw.py"
  "notebook/Lung_cancer/06_sklearn_sfs-union.py"
  "notebook/Lymphoma/06_sklearn_sfs-raw.py"
  "notebook/Lymphoma/06_sklearn_sfs-union.py"
  "notebook/NCI/06_sklearn_sfs-raw.py"
  "notebook/NCI/06_sklearn_sfs-union.py"
  "notebook/Prostate/06_sklearn_sfs-raw.py"
  "notebook/Prostate/06_sklearn_sfs-union.py"
  "notebook/SRBCT_txt/06_sklearn_sfs-raw.py"
  "notebook/SRBCT_txt/06_sklearn_sfs-union.py"
  "notebook/Tumors9/06_sklearn_sfs-raw.py"
  "notebook/Tumors9/06_sklearn_sfs-union.py"
)

for script in "${scripts[@]}"; do
  echo "\n=== Running ${script} ==="
  python "${script}"
done
