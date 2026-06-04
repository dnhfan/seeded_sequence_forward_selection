import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
import pandas as pd

from src.config import ProjectPath
from src.wrapper import SeededSFSSelector


def main():
    data_name = "colon1"

    models = ["log", "dt", "rf", "svm"]

    # [TÙY CHỈNH CHO TỪNG DATASET TẠI ĐÂY]
    n_features = 50
    cv_value = 5
    max_features_sfs = 20
    patience_sfs = 5
    data_variant = "raw"

    path = ProjectPath(data_name, n_features)
    voting_csv_name = f"top{n_features}_features_voting.csv"

    # Load data
    try:
        if data_variant == "union":
            df = pd.read_csv(path.ensemble_file("union"))
        else:
            df = pd.read_csv(path.raw_path)
    except FileNotFoundError as e:
        print(f" Error: Not FileNotFoundError {e}")
        return

    print(f" Setup benchmark for dataset: {data_name} (Variant: {data_variant})")
    for model in models:
        print("\n Running SFS benchmark for model:", model.upper())

        wrapper = SeededSFSSelector(
            data_name=data_name,
            n_features=n_features,
            voting_csv_name=voting_csv_name,
            dataset_variant=data_variant,
            run_tag=f"benchmark_{model}_1seeds",
            using_timer=True,
            unit="ms",
        )

        wrapper.run_sfs(
            df=df,
            max_features=max_features_sfs,
            n_seeds=1,
            patience=patience_sfs,
            cv=cv_value,
            scoring="accuracy",
            model=model,
        )


if __name__ == "__main__":
    main()
