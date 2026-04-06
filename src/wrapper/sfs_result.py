from dataclasses import dataclass
from typing import List

import pandas


@dataclass()
class SFSResult:
    df_final: pandas.DataFrame
    selected_features: List[str]
    total_fit_time_ms: float
    global_best_score: float
    history_text: str = ""
