#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash run_sfs.sh --list
  bash run_sfs.sh <dataset_name> <dataset_variant> <algorithm>

Arguments:
  dataset_name     Folder name under notebook/ (e.g. Leukemia_3c1)
  dataset_variant  raw | union
  algorithm        sklearn | seeded

Examples:
  bash run_sfs.sh --list
  bash run_sfs.sh Leukemia_3c1 raw sklearn
  bash run_sfs.sh Lung_cancer union seeded
EOF
}

if [[ $# -eq 1 && ( "$1" == "--list" || "$1" == "-l" ) ]]; then
  if [[ ! -d "notebook" ]]; then
    printf "Error: notebook directory not found.\n" >&2
    exit 1
  fi

  printf "Available datasets:\n"
  for d in notebook/*/; do
    [[ -d "$d" ]] || continue
    name="${d#notebook/}"
    name="${name%/}"
    [[ "$name" == .* ]] && continue
    printf -- "- %s\n" "$name"
  done
  exit 0
fi

if [[ $# -ne 3 ]]; then
  usage
  exit 1
fi

dataset_name="$1"
dataset_variant="${2,,}"
algorithm="${3,,}"

if [[ "$dataset_variant" != "raw" && "$dataset_variant" != "union" ]]; then
  printf "Error: dataset_variant must be 'raw' or 'union'.\n" >&2
  usage
  exit 1
fi

case "$algorithm" in
  sklearn|sklearn_sfs|sk)
    algorithm="sklearn"
    ;;
  seeded|seeded_sfs|sfs|custom)
    algorithm="seeded"
    ;;
  *)
    printf "Error: algorithm must be 'sklearn' or 'seeded'.\n" >&2
    usage
    exit 1
    ;;
esac

dataset_dir="notebook/${dataset_name}"
if [[ ! -d "$dataset_dir" ]]; then
  printf "Error: dataset folder not found: %s\n" "$dataset_dir" >&2
  exit 1
fi

if [[ "$algorithm" == "sklearn" ]]; then
  script_path="${dataset_dir}/06_sklearn_sfs-${dataset_variant}.py"
else
  script_path="${dataset_dir}/07_sfs-${dataset_variant}.py"
fi

if [[ ! -f "$script_path" ]]; then
  printf "Error: script not found: %s\n" "$script_path" >&2
  printf "Hint: expected standardized naming (06_sklearn_*, 07_sfs_*).\n" >&2
  exit 1
fi

printf "Running: python %s\n" "$script_path"
python "$script_path"
