"""
Manager class for exporting Robyn results.
"""

import os
from pathlib import Path
from typing import Dict, Optional
import pandas as pd

from robyn.modeling.entities import ParetoResult
from robyn.data.entities.mmmdata import MMMData


class ExportManager:
    """Manages the export of Robyn results to various formats."""

    def __init__(self, working_dir: str):
        """
        Initialize ExportManager.

        Args:
            working_dir: Directory to export files to
        """
        self.working_dir = working_dir
        os.makedirs(working_dir, exist_ok=True)

    def export_pareto_results(self, pareto_result: ParetoResult) -> Dict[str, str]:
        """
        Export Pareto optimization results to CSV files.

        Args:
            pareto_result: ParetoResult object containing results to export

        Returns:
            Dictionary mapping export type to file path
        """
        export_paths = {}

        # Export hyperparameters
        if pareto_result.hyperparameters is not None:
            path = os.path.join(self.working_dir, "pareto_hyperparameters.csv")
            pareto_result.hyperparameters.to_csv(path, index=False)
            export_paths["hyperparameters"] = path

        # Export aggregated decomposition
        if pareto_result.x_decomp_agg is not None:
            path = os.path.join(self.working_dir, "pareto_aggregated.csv")
            pareto_result.x_decomp_agg.to_csv(path, index=False)
            export_paths["aggregated"] = path

        # Export media transforms
        if pareto_result.media_transforms is not None:
            path = os.path.join(self.working_dir, "pareto_media_transform_matrix.csv")
            pareto_result.media_transforms.to_csv(path, index=False)
            export_paths["media_transforms"] = path

        # Export all decomposition
        if pareto_result.all_decomp is not None:
            path = os.path.join(self.working_dir, "pareto_alldecomp_matrix.csv")
            pareto_result.all_decomp.to_csv(path, index=False)
            export_paths["all_decomp"] = path

        return export_paths

    def get_export_paths(self) -> Dict[str, str]:
        """
        Get paths to exported files.

        Returns:
            Dictionary mapping export type to file path
        """
        return {
            "hyperparameters": os.path.join(self.working_dir, "pareto_hyperparameters.csv"),
            "aggregated": os.path.join(self.working_dir, "pareto_aggregated.csv"),
            "media_transforms": os.path.join(self.working_dir, "pareto_media_transform_matrix.csv"),
            "all_decomp": os.path.join(self.working_dir, "pareto_alldecomp_matrix.csv"),
        }
