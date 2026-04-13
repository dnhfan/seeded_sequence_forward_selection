import os
import sys

import pandas as pd

# Tính toán lùi về 2 cấp thư mục: sfs.py -> Tumors9 -> notebook -> wrapper-w-filter (Gốc)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.config import ProjectPath
from src.wrapper import SeededSFSSelector


def main():
    print("󰜎 Running Wrapper Features Slection..")

    # 1. setup conf
    data_name = "Prostate"
    n_features = 50

    path = ProjectPath(data_name=data_name, n_features=n_features)

    voting_csv_name = f"top{n_features}_features_voting.csv"

    # 2. Init WrapperSelector
    wrapper = SeededSFSSelector(
        data_name=data_name,
        n_features=n_features,
        voting_csv_name=voting_csv_name,
        using_timer=True,
        unit="ms",
        dataset_variant="raw",
    )

    df = pd.read_csv(path.raw_path)

    df_final = wrapper.run_sfs(
        df=df,
        max_features=20,
        patience=3,
        n_seeds=1,
        model="log",
        scoring="accuracy",
        cv=5,
    )

    # View data
    print("\n󰔂  Preview head of FINAL DATASET:")
    print(df_final.head())


if __name__ == "__main__":
    main()
