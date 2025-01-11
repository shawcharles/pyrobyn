"""
Export Manager for Robyn MMM Results.

This module handles the export of various model outputs to CSV files, including:
- Pareto hyperparameters
- Aggregated decomposition
- Media transformation matrices
- All decomposition matrices
"""

from pathlib import Path
from typing import Optional
import pandas as pd
from robyn.modeling.pareto.pareto_result import ParetoResult
from robyn.data.entities.mmmdata import MMMData


class ExportManager:
    """Manages the export of model results to various file formats."""

    def __init__(self, output_dir: str):
        """
        Initialize the ExportManager.

        Args:
            output_dir: Directory where output files will be saved
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_pareto_results(
        self,
        pareto_result: ParetoResult,
        mmm_data: MMMData,
        prefix: Optional[str] = None
    ) -> dict:
        """
        Export all Pareto-related results to CSV files.

        Args:
            pareto_result: ParetoResult object containing model outputs
            mmm_data: MMMData object containing model inputs
            prefix: Optional prefix for output filenames

        Returns:
            dict: Dictionary containing paths to exported files
        """
        prefix = prefix or "pareto"
        exported_files = {}

        # 1. Export hyperparameters
        hyperparams_df = pareto_result.get_hyperparameters_df()
        if hyperparams_df is not None:
            filepath = self.output_dir / f"{prefix}_hyperparameters.csv"
            hyperparams_df.to_csv(filepath, index=False)
            exported_files['hyperparameters'] = filepath

        # 2. Export aggregated decomposition
        agg_decomp_df = pareto_result.get_aggregated_decomposition()
        if agg_decomp_df is not None:
            filepath = self.output_dir / f"{prefix}_aggregated.csv"
            agg_decomp_df.to_csv(filepath, index=False)
            exported_files['aggregated'] = filepath

        # 3. Export media transformation matrix
        media_transform_df = pareto_result.get_media_transform_matrix()
        if media_transform_df is not None:
            filepath = self.output_dir / f"{prefix}_media_transform_matrix.csv"
            media_transform_df.to_csv(filepath, index=False)
            exported_files['media_transform'] = filepath

        # 4. Export all decomposition matrix
        all_decomp_df = pareto_result.get_all_decomposition_matrix()
        if all_decomp_df is not None:
            filepath = self.output_dir / f"{prefix}_alldecomp_matrix.csv"
            all_decomp_df.to_csv(filepath, index=False)
            exported_files['all_decomp'] = filepath

        return exported_files

    def get_export_paths(self, prefix: Optional[str] = None) -> dict:
        """
        Get the paths to exported files.

        Args:
            prefix: Optional prefix for filenames

        Returns:
            dict: Dictionary containing paths to exported files
        """
        prefix = prefix or "pareto"
        return {
            'hyperparameters': self.output_dir / f"{prefix}_hyperparameters.csv",
            'aggregated': self.output_dir / f"{prefix}_aggregated.csv",
            'media_transform': self.output_dir / f"{prefix}_media_transform_matrix.csv",
            'all_decomp': self.output_dir / f"{prefix}_alldecomp_matrix.csv"
        }
