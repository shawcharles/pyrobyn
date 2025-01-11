from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd
import numpy as np


@dataclass
class ParetoResult:
    """
    Holds the results of Pareto optimization for marketing mix models.

    Attributes:
        pareto_solutions (List[str]): List of solution IDs that are Pareto-optimal.
        pareto_fronts (int): Number of Pareto fronts considered in the optimization.
        result_hyp_param (pd.DataFrame): Hyperparameters of Pareto-optimal solutions.
        x_decomp_agg (pd.DataFrame): Aggregated decomposition results for Pareto-optimal solutions.
        result_calibration (Optional[pd.DataFrame]): Calibration results, if calibration was performed.
        media_vec_collect (pd.DataFrame): Collected media vectors for all Pareto-optimal solutions.
        x_decomp_vec_collect (pd.DataFrame): Collected decomposition vectors for all Pareto-optimal solutions.
        plot_data_collect (Dict[str, pd.DataFrame]): Data for various plots, keyed by plot type.
        df_caov_pct_all (pd.DataFrame): Carryover percentage data for all channels and Pareto-optimal solutions.
        x_decomp_agg_refresh (Optional[pd.DataFrame]): Refresh aggregated decomposition results for Pareto-optimal solutions.
        x_decomp_agg_calib (Optional[pd.DataFrame]): Calibration aggregated decomposition results for Pareto-optimal solutions.
        pareto_front (Optional[pd.DataFrame]): Pareto front dataframe.
        hyperparameters (Optional[pd.DataFrame]): Hyperparameters dataframe.
        media_transforms (Optional[pd.DataFrame]): Media transforms dataframe.
        all_decomp (Optional[pd.DataFrame]): All decomposition dataframe.
    """

    pareto_solutions: List[str]
    pareto_fronts: int
    result_hyp_param: pd.DataFrame
    x_decomp_agg: pd.DataFrame
    result_calibration: Optional[pd.DataFrame]
    media_vec_collect: pd.DataFrame
    x_decomp_vec_collect: pd.DataFrame
    plot_data_collect: Dict[str, pd.DataFrame]
    df_caov_pct_all: pd.DataFrame
    x_decomp_agg_refresh: Optional[pd.DataFrame] = None
    x_decomp_agg_calib: Optional[pd.DataFrame] = None
    pareto_front: Optional[pd.DataFrame] = None
    hyperparameters: Optional[pd.DataFrame] = None
    media_transforms: Optional[pd.DataFrame] = None
    all_decomp: Optional[pd.DataFrame] = None

    def get_hyperparameters_df(self) -> Optional[pd.DataFrame]:
        """Get hyperparameters dataframe for export."""
        if self.hyperparameters is not None:
            return self.hyperparameters.copy()
        return None

    def get_aggregated_decomposition(self) -> Optional[pd.DataFrame]:
        """Get aggregated decomposition dataframe for export."""
        if self.x_decomp_agg is not None:
            return self.x_decomp_agg.copy()
        return None

    def get_media_transform_matrix(self) -> Optional[pd.DataFrame]:
        """Get media transformation matrix dataframe for export."""
        if self.media_transforms is not None:
            return self.media_transforms.copy()
        return None

    def get_all_decomposition_matrix(self) -> Optional[pd.DataFrame]:
        """Get all decomposition matrix dataframe for export."""
        if self.all_decomp is not None:
            return self.all_decomp.copy()
        return None
