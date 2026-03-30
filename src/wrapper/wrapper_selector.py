import os
from datetime import datetime
from typing import List

import pandas as pd

from src.config import ProjectPath
from src.wrapper import SeededForwardSelection


class WrapperSelector:
    """
    A class to perform Wrapper-based Feature Selection (SFS).
    It first creates a Union of features from various filter methods,
    then runs Seeded Forward Selection to find the optimal subset.

    Attributes:
        data_name (str): The name of the dataset (e.g., 'NCI', 'Lymphoma').
        valid_methods (List[str]): List of filtering methods used previously.
        n_features (int): The number of top features selected from each method.
        voting_csv_name (str): The filename of the voting results (used as seed).
    """

    def __init__(
        self,
        data_name: str,
        valid_method: List[str],
        n_features: int,
        voting_csv_name: str,
    ) -> None:

        self.data_name = data_name
        self.valid_method = valid_method
        self.n_features = n_features
        self.voting_csv_name = voting_csv_name

        # Dir path
        # self.filter_dir = f"data/processed/{self.data_name}/filter{n_features}"
        # self.ensemble_dir = f"data/processed/{self.data_name}/ensemble{n_features}"
        # self.wrapper_dir = f"data/processed/{self.data_name}/wrapper{n_features}"
        # self.raw_path = f"data/raw/{self.data_name}.csv"

        self.path = ProjectPath(data_name=data_name, n_features=n_features)

        self.timestamp = datetime.now().strftime("%Y-%m-%d")

        os.makedirs(self.path.wrapper_dir, exist_ok=True)

        self.report_dir = f"results/{self.data_name}/{self.timestamp}/report"

    # def _create_union_features(self) -> pd.DataFrame:
    #     """
    #     [Private] Aggregates all features from the filter files into a unique Union set.
    #     Extracts the corresponding data from the raw dataset to construct a new DataFrame.
    #
    #     Returns:
    #         pd.DataFrame: A DataFrame containing the target variable and the union of all filtered features.
    #     """
    #
    #     return create_union_features(
    #         self.data_name,
    #         self.valid_method,
    #         self.n_features,
    #         self.filter_dir,
    #         self.raw_path,
    #         self.ensemble_dir,
    #     )

    def _execute_sfs_core(
        self, X_in: pd.DataFrame, y_in: pd.Series, sfs_params: dict
    ) -> tuple[pd.DataFrame, SeededForwardSelection]:
        """
        [Private] Execute the core SFS aglorithsm

        Returns:
            df_final: DataFrame which restore the results of sfs
            selector: Instance of SFS
        """
        # 2. Config SFS
        voting_csv_path = f"{self.path.ensemble_dir}/{self.voting_csv_name}"

        selector = SeededForwardSelection(seed_source=voting_csv_path, **sfs_params)

        # 3. Fit the model
        selector.fit(X_in, y_in)

        # 4. Transformdata and package the final dataset
        X_selected = selector.transform(X_in)
        X_selected_df = pd.DataFrame(
            X_selected, columns=selector.get_feature_names_out()
        )

        df_final = pd.concat([y_in.reset_index(drop=True), X_selected_df], axis=1)

        return df_final, selector

    def _save_sfs_output(
        self,
        df_final: pd.DataFrame,
        selector: SeededForwardSelection,
        file_suffix: str,
        n_seeds: int,
        patience: int,
        max_features: int,
    ) -> None:
        # 5. Save the final dataset to the wrapper directory
        save_path = f"{self.path.wrapper_dir}/{self.data_name}_SFS_{n_seeds}seed_{file_suffix}.csv"
        df_final.to_csv(save_path, index=False)

        print(f" Saved Final data to: {save_path}")

        # 6. Save the SFS Execution history
        history_path = f"{self.report_dir}/sfs_history_{n_seeds}seed_{patience}patience_{max_features}max_{file_suffix}.csv"
        selector.save_history(history_path)

    def run_sfs(
        self,
        df: pd.DataFrame,
        file_suffix: str = "custom",
        max_features: int = 20,
        patience: int = 3,
        n_seeds: int = 1,
        model: str = "logistic",
        scoring: str = "accuracy",
        cv: int = 5,
        verbose: int = 2,
    ) -> pd.DataFrame:
        """
        Executes Seeded Forward Selection (SFS) on the dataset.

        Args:
            max_features (int): The maximum number of features the SFS is allowed to select. Defaults to 20.
            patience (int): The number of iterations to wait for an improvement before early stopping. Defaults to 3.

        Returns:
            pd.DataFrame: The final DataFrame containing only the target and the SFS-selected features.
        """
        print("\n Starting Seeded Forward Selection (SFS)...")

        # 1. Prepare the data

        X_in: pd.DataFrame = df.iloc[:, 1:]
        y_in: pd.Series = df.iloc[:, 0]

        # 2. executes sfs
        sfs_params = {
            "max_features": max_features,
            "patience": patience,
            "n_seeds": n_seeds,
            "model": model,
            "scoring": scoring,
            "cv": cv,
            "verbose": verbose,
        }

        df_final, selector = self._execute_sfs_core(X_in, y_in, sfs_params)

        print(f"\n SFS completed! Final dataset shape: {df_final.shape}")
        print(" Selected Features: ", selector.get_feature_names_out())

        # 3. Save the result
        self._save_sfs_output(
            df_final,
            selector,
            file_suffix,
            n_seeds,
            patience,
            max_features,
        )

        return df_final
