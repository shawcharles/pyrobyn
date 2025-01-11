# pyre-strict

from typing import Dict, List
import logging

import pandas as pd
from robyn.common.logger import RobynLogger
from robyn.modeling.entities.modeloutputs import ModelOutputs, Trial
from robyn.modeling.pareto.data import ParetoData


class TrialValidator:
    @staticmethod
    def ensure_trial_ids(trial: Trial) -> Trial:
        """Ensure trial has proper sol_id and all its dataframes have sol_id column."""
        if not trial.sol_id:
            trial.sol_id = f"{trial.trial}_{trial.iter_ng}_{trial.iter_par}"

        # Ensure result_hyp_param has sol_id
        if isinstance(trial.result_hyp_param, pd.DataFrame):
            if "sol_id" not in trial.result_hyp_param.columns:
                trial.result_hyp_param["sol_id"] = trial.sol_id

        # Ensure x_decomp_agg has sol_id
        if isinstance(trial.x_decomp_agg, pd.DataFrame):
            if "sol_id" not in trial.x_decomp_agg.columns:
                trial.x_decomp_agg["sol_id"] = trial.sol_id

        # Ensure decomp_spend_dist has sol_id if it exists
        if isinstance(trial.decomp_spend_dist, pd.DataFrame):
            if "sol_id" not in trial.decomp_spend_dist.columns:
                trial.decomp_spend_dist["sol_id"] = trial.sol_id

        return trial

    @staticmethod
    def validate_model_outputs(model_outputs: ModelOutputs) -> None:
        if not model_outputs.trials:
            raise ValueError("No trials found in model outputs")

        for trial in model_outputs.trials:
            if not isinstance(trial.result_hyp_param, pd.DataFrame):
                raise ValueError(f"Trial {trial.sol_id} has invalid result_hyp_param")
            if not isinstance(trial.x_decomp_agg, pd.DataFrame):
                raise ValueError(f"Trial {trial.sol_id} has invalid x_decomp_agg")


class DataAggregator:
    def __init__(self, model_outputs: ModelOutputs):
        self.model_outputs = model_outputs
        # Setup logger with a single handler
        self.logger = logging.getLogger(__name__)

    def aggregate_model_data(self, calibrated: bool) -> Dict[str, pd.DataFrame]:
        TrialValidator.validate_model_outputs(self.model_outputs)
        self.logger.info("Starting model data aggregation")

        self.model_outputs.trials = [
            TrialValidator.ensure_trial_ids(trial)
            for trial in self.model_outputs.trials
        ]

        hyper_fixed = self.model_outputs.hyper_fixed
        trials = [
            model
            for model in self.model_outputs.trials
            if hasattr(model, "resultCollect")
        ]

        result_hyp_param_list = [
            trial.result_hyp_param for trial in self.model_outputs.trials
        ]
        x_decomp_agg_list = [trial.x_decomp_agg for trial in self.model_outputs.trials]

        result_hyp_param = pd.concat(result_hyp_param_list, ignore_index=True)
        x_decomp_agg = pd.concat(x_decomp_agg_list, ignore_index=True)

        self.logger.debug("Aggregated result_hyp_param:")
        RobynLogger.log_df(self.logger, result_hyp_param)

        self.logger.debug("Aggregated x_decomp_agg:")
        RobynLogger.log_df(self.logger, x_decomp_agg)

        self._check_sol_id(result_hyp_param, x_decomp_agg)

        result_calibration = self._process_calibration_data(trials, calibrated)

        if not hyper_fixed:
            self._add_iterations(result_hyp_param, x_decomp_agg, result_calibration)

        self.logger.debug("Aggregated x_decomp_agg:")
        RobynLogger.log_df(self.logger, result_calibration)

        self._merge_bootstrap_results(x_decomp_agg)

        return {
            "result_hyp_param": result_hyp_param,
            "x_decomp_agg": x_decomp_agg,
            "result_calibration": result_calibration,
        }

    def _check_sol_id(self, result_hyp_param: pd.DataFrame, x_decomp_agg: pd.DataFrame):
        if "sol_id" not in result_hyp_param.columns:
            raise ValueError("sol_id missing from result_hyp_param after aggregation")
        if "sol_id" not in x_decomp_agg.columns:
            raise ValueError("sol_id missing from x_decomp_agg after aggregation")

    def _process_calibration_data(
        self, trials: List[Trial], calibrated: bool
    ) -> pd.DataFrame:
        if calibrated:
            self.logger.info("Processing calibration data")
            return pd.concat([pd.DataFrame(trial.liftCalibration) for trial in trials])
        return None

    def _add_iterations(
        self,
        result_hyp_param: pd.DataFrame,
        x_decomp_agg: pd.DataFrame,
        result_calibration: pd.DataFrame,
    ):
        df_names = [result_hyp_param, x_decomp_agg]
        if result_calibration is not None:
            df_names.append(result_calibration)
        for df in df_names:
            df["iterations"] = (df["iterNG"] - 1) * self.model_outputs.cores + df[
                "iterPar"
            ]

    def _merge_bootstrap_results(self, x_decomp_agg: pd.DataFrame):
        if (
            len(x_decomp_agg["sol_id"].unique()) == 1
            and "boot_mean" not in x_decomp_agg.columns
        ):
            bootstrap = getattr(self.model_outputs, "bootstrap", None)
            if bootstrap is not None:
                self.logger.info("Merging bootstrap results")
                x_decomp_agg = pd.merge(
                    x_decomp_agg, bootstrap, left_on="rn", right_on="variable"
                )

    def get_all_decomposition_matrix(self, solution_ids: List[str]) -> pd.DataFrame:
        """
        Get all decomposition vectors for specified solutions.

        Args:
            solution_ids: List of solution IDs to get decomposition for

        Returns:
            DataFrame containing all decomposition vectors
        """
        decomp_list = []
        for sol_id in solution_ids:
            decomp = self._get_decomposition_vector(sol_id)
            decomp['solID'] = sol_id
            decomp_list.append(decomp)
        
        return pd.concat(decomp_list, ignore_index=True)

    def aggregate_data(self, calibrated: bool = False) -> ParetoData:
        """
        Aggregate model data and prepare it for Pareto optimization.

        Args:
            calibrated: Whether the models are calibrated

        Returns:
            ParetoData: Aggregated data ready for Pareto optimization
        """
        aggregated = self.aggregate_model_data(calibrated)
        
        # Extract decomp_spend_dist from trials
        decomp_spend_dist_list = []
        for trial in self.model_outputs.trials:
            if hasattr(trial, 'decomp_spend_dist') and isinstance(trial.decomp_spend_dist, pd.DataFrame):
                decomp_spend_dist_list.append(trial.decomp_spend_dist)
        
        decomp_spend_dist = pd.concat(decomp_spend_dist_list, ignore_index=True) if decomp_spend_dist_list else pd.DataFrame()
        
        return ParetoData(
            decomp_spend_dist=decomp_spend_dist,
            result_hyp_param=aggregated['result_hyp_param'],
            x_decomp_agg=aggregated['x_decomp_agg'],
            pareto_fronts=[]  # This will be populated later in the optimization process
        )
