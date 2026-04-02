import os
import sys

# Tính toán lùi về 2 cấp thư mục: sfs.py -> Tumors9 -> notebook -> wrapper-w-filter (Gốc)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.config import ProjectPath
from src.utils import create_union_features
from src.wrapper.wrapper_selector import WrapperSelector


def main():
    print("󰜎 Running Wrapper Features Slection using Union data set..")

    # 1. setup conf
    data_name = "SRBCT_txt"
    n_features = 50

    valid_methods = [
        "variance",
        "correlation",
        "chi_squared",
        "mutual_information",
        "anova_f_test",
    ]

    path = ProjectPath(data_name=data_name, n_features=n_features)

    voting_csv_name = f"top50_features_voting_2026-04-03.csv"

    # 2. Init WrapperSelector
    wrapper = WrapperSelector(
        data_name=data_name,
        valid_method=valid_methods,
        n_features=n_features,
        voting_csv_name=voting_csv_name,
        dataset_variant="union",
    )

    df = create_union_features(
        data_name=data_name,
        valid_method=valid_methods,
        n_features=n_features,
        filter_dir=str(path.filter_dir()),
        raw_path=str(path.raw_path),
        ensemble_dir=str(path.ensemble_dir()),
    )

    df_final = wrapper.run_sfs(
        df=df,
        file_suffix="Union",
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
