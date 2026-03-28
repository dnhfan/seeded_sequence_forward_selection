import os
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ProjectPath:
    """
    Initialize all the path of the Project
    """

    data_name: str
    n_features: int

    @property
    def raw_path(self) -> str:
        return f"data/raw/{self.data_name}.csv"

    @property
    def filter_dir(self) -> str:
        return f"data/processed/{self.data_name}/filter{self.n_features}"

    @property
    def ensemble_dir(self) -> str:
        return f"data/processed/{self.data_name}/ensemble{self.n_features}"

    @property
    def wrapper_dir(self) -> str:
        return f"data/processed/{self.data_name}/wrapper{self.n_features}"

    @property
    def report_dir(self) -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d")
        return f"results/{self.data_name}/{timestamp}/report"

    def create_all_dirs(self) -> None:
        directories = [
            self.filter_dir,
            self.ensemble_dir,
            self.wrapper_dir,
            self.report_dir,
        ]
        for d in directories:
            os.makedirs(d, exist_ok=True)
        print(" setuped all the dirs")
