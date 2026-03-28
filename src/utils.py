import os
from typing import List

import pandas as pd


def load_seed_from_csv(csv_path: str, n_seeds: int = 1) -> list:
    """
    Read csv voting
    """

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f" Error: not found csv in {csv_path}")

    df = pd.read_csv(csv_path)

    if "Feature" not in df.columns or "Votes" not in df.columns:
        raise ValueError(f" Error: csv file is not votings file")

    seed_features = df["Feature"].head(n_seeds).tolist()

    print(f"󰄬 Taken {n_seeds} seed from file: {os.path.basename(csv_path)}")

    return seed_features


def validate_features(features: list, X_columns: list) -> bool:
    """
    Validate if seed features exists in dataset
    """

    invalid_features = [f for f in features if f not in X_columns]

    if invalid_features:
        raise ValueError(f" Error: seed is not found in dataset")

    print(f"󰄬 Seed exists in dataset")
    return True


def create_union_features(
    data_name: str,
    valid_method: List[str],
    n_features: int,
    filter_dir: str,
    raw_path: str,
    ensemble_dir: str,
) -> pd.DataFrame:
    """
    [Private] Aggregates all features from the filter files into a unique Union set.
    Extracts the corresponding data from the raw dataset to construct a new DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing the target variable and the union of all filtered features.
    """

    all_features = set()

    # 1. Colllect featrues from all specified filter medthods
    for m in valid_method:
        path = f"{filter_dir}/{data_name}_{m}_{n_features}features.csv"
        try:
            # read header
            cols = pd.read_csv(path, nrows=0).columns.to_list()
            all_features.update(cols[1:])
        except FileNotFoundError:
            print(f" Error: file not found {path}")

    union_features = sorted(all_features)

    print(
        f" Union: Collected {len(union_features)} unique features from {len(valid_method)} methods."
    )

    # 2. Extract data from the raw dataset
    raw = pd.read_csv(raw_path)
    target_col = raw.columns[0]

    # select the target column and the union features
    y = raw[target_col]
    X_union = raw[union_features]

    # Concatenate target and features
    df_union = pd.concat([y, X_union], axis=1)

    # 3. Save the Union dataset to the ensemble directory
    os.makedirs(ensemble_dir, exist_ok=True)
    save_path = f"{ensemble_dir}/{data_name}_Union_{n_features}features.csv"
    df_union.to_csv(save_path, index=False)
    print(f"󰈙 Saved Union data to: {save_path} | Shape: {df_union.shape}")

    return df_union
