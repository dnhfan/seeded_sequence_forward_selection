import os

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
