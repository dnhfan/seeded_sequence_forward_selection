"""Shared helpers for benchmark scripts."""

from pathlib import Path

_ALGO_SHORT: dict[str, str] = {
    "seededsfsselector": "seeded_sfs",
    "sklearnsfsselector": "sk_sfs",
}

_CONFIG_TOKENS = {"max", "seeds", "pat", "cv"}


def _parse_filename_tokens(filename: str, data_name: str) -> dict:
    """Parse SFS filename into structured tokens."""
    stem = Path(filename).stem
    stem = stem.replace(f"{data_name}_", "", 1)
    stem = stem.rsplit("_union", 1)[0].rsplit("_raw", 1)[0]
    parts = stem.split("_")

    algo = _ALGO_SHORT.get(parts[0], parts[0]) if parts else ""
    model = parts[1] if len(parts) > 1 else ""
    config = {t for t in parts[2:] if any(t.endswith(k) for k in _CONFIG_TOKENS)}

    return {"algorithm": algo, "model": model, "config": config}


def build_smart_labels(files: list[Path], data_name: str) -> list[str]:
    """Build labels that only include tokens differentiating the files.

    Algorithm names are shortened (seededsfsselector -> seeded_sfs, etc.).
    Config tokens shared by ALL files are dropped; only differing ones are kept.

    Examples:
        1 file (..._20max_1seeds_5pat_5cv_):
            -> 'seeded_sfs_log'

        2 files (..._20max_1seeds_5pat_5cv_ + ..._20max_1seeds_5pat_3cv_):
            -> ['seeded_sfs_log_5cv', 'seeded_sfs_log_3cv']

        2 files (..._20max_1seeds_5pat_5cv_ + ..._10max_1seeds_3pat_5cv_):
            -> ['seeded_sfs_log_20max_5pat', 'seeded_sfs_log_10max_3pat']
    """
    if not files:
        return []

    parsed = [_parse_filename_tokens(f.name, data_name) for f in files]

    # Find config tokens common to ALL files
    common_config = parsed[0]["config"]
    for p in parsed[1:]:
        common_config = common_config & p["config"]

    # Build labels: algorithm + model + differing config tokens
    labels = []
    for p in parsed:
        diff_config = sorted(p["config"] - common_config)
        label_parts = [p["algorithm"], p["model"]] + diff_config
        labels.append("_".join(label_parts))

    return labels


def label_from_filename(filename: str, data_name: str) -> str:
    """Extract a human-readable method label from an SFS result filename.

    Example:
        colon1_seededsfsselector_log_accuracy_20max_1seeds_5pat_5cv_union.csv
        -> 'seededsfsselector_log_20max_1seeds_5pat'

        colon1_sklearnsfsselector_logisticregression_accuracy_automax_5cv_union.csv
        -> 'sklearnsfsselector_logisticregression_automax'
    """
    stem = Path(filename).stem  # drop .csv
    stem = stem.replace(f"{data_name}_", "", 1)  # drop dataset prefix
    stem = stem.rsplit("_union", 1)[0].rsplit("_raw", 1)[0]  # drop variant suffix
    parts = stem.split("_")

    if len(parts) < 2:
        return stem

    # Always include algorithm + model
    label_parts = [parts[0], parts[1]]

    # Include config tokens: *max, *seeds, *pat
    for token in parts[2:]:
        if token.endswith("max") or token.endswith("seeds") or token.endswith("pat"):
            label_parts.append(token)

    return "_".join(label_parts)


def discover_sfs_files(
    wrapper_dir: Path,
    variant: str,
    algorithms: list[str] | None = None,
    models: list[str] | None = None,
) -> list[Path]:
    """Auto-discover *_union.csv (or *_raw.csv) under 04_wrapper/<variant>/.

    Args:
        wrapper_dir: Root wrapper directory (e.g. data/processed/<dataset>/04_wrapper).
        variant: Data variant ("raw" or "union").
        algorithms: If provided, keep only files whose name contains any of these algorithm tokens.
        models: If provided, keep only files whose name contains any of these model tokens.
    """
    variant_dir = wrapper_dir / variant
    if not variant_dir.exists():
        return []

    files = sorted(variant_dir.rglob(f"*_{variant}.csv"))

    if algorithms:
        files = [
            f for f in files
            if any(algo.lower() in f.name.lower() for algo in algorithms)
        ]

    if models:
        files = [
            f for f in files
            if any(model.lower() in f.name.lower() for model in models)
        ]

    return files
