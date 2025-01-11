# pyre-strict

from dataclasses import dataclass
from typing import List
import pandas as pd


@dataclass
class ParetoData:
    """
    Data structure for holding Pareto optimization data.

    Attributes:
        decomp_spend_dist: Decomposition spending distribution
        result_hyp_param: Model hyperparameters and results
        x_decomp_agg: Aggregated decomposition data
        pareto_fronts: List of Pareto front indices
    """
    decomp_spend_dist: pd.DataFrame
    result_hyp_param: pd.DataFrame
    x_decomp_agg: pd.DataFrame
    pareto_fronts: List[int]
