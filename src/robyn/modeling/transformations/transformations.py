# pyre-strict

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from robyn.data.entities.enums import AdstockType
from robyn.data.entities.hyperparameters import ChannelHyperparameters, Hyperparameters
from robyn.data.entities.mmmdata import MMMData
from robyn.modeling.feature_engineering import FeaturizedMMMData
from scipy import stats

# Initialize logger
logger = logging.getLogger(__name__)


@dataclass
class AdstockResult:
    x: pd.Series
    x_decayed: pd.Series
    x_imme: pd.Series
    thetaVec: Optional[pd.Series]
    thetaVecCum: pd.Series
    inflation_total: float

    def __str__(self) -> str:
        return f"AdstockResult(inflation_total={self.inflation_total:.4f}, series_length={len(self.x)})"


@dataclass
class TransformationResult:
    dt_modSaturated: pd.DataFrame
    dt_saturatedImmediate: pd.DataFrame
    dt_saturatedCarryover: pd.DataFrame

    def __str__(self) -> str:
        return (
            f"TransformationResult(modSaturated_shape={self.dt_modSaturated.shape}, "
            f"saturatedImmediate_shape={self.dt_saturatedImmediate.shape}, "
            f"saturatedCarryover_shape={self.dt_saturatedCarryover.shape})"
        )


class Transformation:
    def __init__(self, mmm_data: MMMData):
        """Initialize the Transformation class with MMMData."""
        logger.debug("Initializing Transformation with MMMData: %s", mmm_data)
        self.mmm_data = mmm_data

    def normalize(self, x: pd.Series) -> pd.Series:
        """Normalize a series to [0, 1] range."""
        logger.debug("Normalizing series with length: %d", len(x))
        if x.max() == x.min():
            logger.warning("Series has constant values, returning binary normalization")
            return pd.Series([1] + [0] * (len(x) - 1), index=x.index)
        return (x - x.min()) / (x.max() - x.min())

    def mic_men(
        x: Union[float, pd.Series], Vmax: float, Km: float, reverse: bool = False
    ) -> Union[float, pd.Series]:
        """Apply Michaelis-Menten transformation."""
        logger.debug("Applying Michaelis-Menten transform (reverse=%s)", reverse)
        if not reverse:
            return Vmax * x / (Km + x)
        else:
            return x * Km / (Vmax - x)

    def adstock_geometric(self, x: pd.Series, theta: float) -> AdstockResult:
        """Apply geometric adstock transformation."""
        logger.debug("Applying geometric adstock with theta=%f", theta)

        if not np.isscalar(theta):
            logger.error("Invalid theta value: must be scalar")
            raise ValueError("theta must be a single value")

        x_decayed = pd.Series(index=x.index, dtype=float)
        x_decayed.iloc[0] = x.iloc[0]

        logger.debug("Computing geometric decay")
        for xi in range(1, len(x_decayed)):
            x_decayed.iloc[xi] = x.iloc[xi] + theta * x_decayed.iloc[xi - 1]

        thetaVecCum = pd.Series(theta ** np.arange(len(x)), index=x.index)
        inflation_total = x_decayed.sum() / x.sum()

        logger.debug(
            "Completed geometric adstock with inflation_total=%f", inflation_total
        )
        return AdstockResult(x, x_decayed, x, None, thetaVecCum, inflation_total)

    def adstock_weibull(
        self,
        x: pd.Series,
        shape: float,
        scale: float,
        windlen: Optional[int] = None,
        adstockType: AdstockType = AdstockType.WEIBULL_CDF,
    ) -> AdstockResult:
        """Apply Weibull adstock transformation."""
        logger.debug(
            "Applying Weibull adstock (type=%s, shape=%f, scale=%f)",
            adstockType,
            shape,
            scale,
        )

        if not (np.isscalar(shape) and np.isscalar(scale)):
            logger.error("Invalid shape or scale values: must be scalar")
            raise ValueError("shape and scale must be single values")

        windlen = len(x) if windlen is None else windlen
        x_bin = pd.Series(range(1, windlen + 1))
        scaleTrans = int(x_bin.quantile(scale))

        if shape == 0 or scale == 0:
            logger.warning("Shape or scale is 0, returning unmodified series")
            x_decayed = x
            thetaVecCum = thetaVec = pd.Series(np.zeros(windlen), index=x_bin)
            x_imme = x
        else:
            logger.debug("Computing Weibull transformation")
            if adstockType == AdstockType.WEIBULL_CDF:
                thetaVec = pd.Series(
                    [1]
                    + list(
                        1 - stats.weibull_min.cdf(x_bin[:-1], shape, scale=scaleTrans)
                    ),
                    index=x_bin,
                )
                thetaVecCum = thetaVec.cumprod()
            elif adstockType == AdstockType.WEIBULL_PDF:
                thetaVecCum = self.normalize(
                    pd.Series(
                        stats.weibull_min.pdf(x_bin, shape, scale=scaleTrans),
                        index=x_bin,
                    )
                )

            x_decayed = pd.Series(0, index=x.index, dtype=float)
            logger.debug("Applying convolution")
            for i, xi in enumerate(x):
                x_decayed += pd.Series(
                    np.convolve(
                        np.concatenate((np.zeros(i), np.repeat(xi, windlen - i))),
                        thetaVecCum,
                    )[:windlen],
                    index=x.index,
                )

            x_imme = (
                x
                if adstockType == AdstockType.WEIBULL_CDF
                else pd.Series(np.convolve(x, thetaVecCum)[: len(x)], index=x.index)
            )

        inflation_total = x_decayed.sum() / x.sum()
        logger.debug(
            "Completed Weibull adstock (type=%s) with inflation_total=%f",
            adstockType,
            inflation_total,
        )
        return AdstockResult(
            x,
            x_decayed,
            x_imme,
            thetaVec if "thetaVec" in locals() else None,
            thetaVecCum,
            inflation_total,
        )

    def transform_adstock(
        self,
        x: pd.Series,
        adstockType: AdstockType,
        channelHyperparameters: ChannelHyperparameters,
        windlen: Optional[int] = None,
    ) -> AdstockResult:
        """Apply adstock transformation based on the specified method."""
        logger.debug("Starting adstock transformation with type: %s", adstockType)
        logger.debug("Input series length: %d", len(x))

        try:
            if adstockType == AdstockType.GEOMETRIC:
                theta = channelHyperparameters.thetas[0]
                logger.debug("Using geometric adstock with theta=%f", theta)
                return self.adstock_geometric(x, theta)
            elif adstockType in [AdstockType.WEIBULL_CDF, AdstockType.WEIBULL_PDF]:
                shape = channelHyperparameters.shapes[0]
                scale = channelHyperparameters.scales[0]
                logger.debug(
                    "Using Weibull adstock with shape=%f, scale=%f", shape, scale
                )
                return self.adstock_weibull(self, x, shape, scale, windlen, adstockType)
        except Exception as e:
            logger.error("Error in adstock transformation: %s", str(e))
            raise

    def saturation_hill(
        self,
        x: pd.Series,
        alpha: float,
        gamma: float,
        x_marginal: Optional[pd.Series] = None,
    ) -> pd.Series:
        """Apply Hill function for saturation."""
        logger.debug(
            "Applying Hill saturation (alpha=%f, gamma=%f, with_marginal=%s)",
            alpha,
            gamma,
            x_marginal is not None,
        )

        if not (np.isscalar(alpha) and np.isscalar(gamma)):
            logger.error("Invalid alpha or gamma values: must be scalar")
            raise ValueError("alpha and gamma must be single values")

        inflexion = gamma * x.max() + (1 - gamma) * x.min()
        logger.debug("Calculated inflexion point: %f", inflexion)

        result = (
            x**alpha / (x**alpha + inflexion**alpha)
            if x_marginal is None
            else x_marginal**alpha / (x_marginal**alpha + inflexion**alpha)
        )

        logger.debug("Completed Hill saturation")
        return result

    def run_transformations(
        self,
        featurized_data: FeaturizedMMMData,
        hyperparameters: Hyperparameters,
        adstockType: AdstockType,
    ) -> TransformationResult:
        """Run media transformations including adstock and saturation."""
        logger.debug("Starting media transformations")
        logger.debug("Input data shape: %s", featurized_data.dt_mod.shape)

        all_media = self.mmm_data.mmmdata_spec.all_media
        rollingWindowStartWhich = self.mmm_data.mmmdata_spec.rolling_window_start_which
        rollingWindowEndWhich = self.mmm_data.mmmdata_spec.rolling_window_end_which

        # print("Rolling window start which: ", rollingWindowStartWhich)
        # print("Rolling window end which: ", rollingWindowEndWhich)

        # if rollingWindowStartWhich is None or rollingWindowEndWhich is None:
        #     # Use default values if not specified
        #     rollingWindowStartWhich = 7
        #     rollingWindowEndWhich = 163
        #     # print("Set default values")
        dt_modAdstocked = featurized_data.dt_mod.copy()
        if "ds" in dt_modAdstocked.columns:
            logger.debug("Removing 'ds' column from data")
            dt_modAdstocked.drop(columns="ds", inplace=True)

        mediaAdstocked: Dict[str, pd.Series] = {}
        mediaSaturated: Dict[str, pd.Series] = {}
        mediaSaturatedImmediate: Dict[str, pd.Series] = {}
        mediaSaturatedCarryover: Dict[str, pd.Series] = {}

        for media in all_media:
            logger.debug("Processing media channel: %s", media)
            try:
                m = dt_modAdstocked[media]
                channelHyperparameters = hyperparameters.get_hyperparameter(media)

                logger.debug("Applying adstock transformation for %s", media)
                x_list = self.transform_adstock(m, adstockType, channelHyperparameters)

                m_imme = x_list.x_imme if adstockType == AdstockType.WEIBULL_PDF else m
                m_adstocked = x_list.x_decayed
                mediaAdstocked[media] = m_adstocked
                m_carryover = m_adstocked - m_imme

                m_adstockedRollWind = m_adstocked.iloc[
                    rollingWindowStartWhich : rollingWindowEndWhich + 1
                ]
                m_carryoverRollWind = m_carryover.iloc[
                    rollingWindowStartWhich : rollingWindowEndWhich + 1
                ]

                logger.debug("Applying saturation transformations for %s", media)
                mediaSaturated[media] = self.saturation_hill(
                    m_adstockedRollWind,
                    channelHyperparameters.alphas[0],
                    channelHyperparameters.gammas[0],
                )
                mediaSaturatedCarryover[media] = self.saturation_hill(
                    m_adstockedRollWind,
                    channelHyperparameters.alphas[0],
                    channelHyperparameters.gammas[0],
                    x_marginal=m_carryoverRollWind,
                )
                mediaSaturatedImmediate[media] = (
                    mediaSaturated[media] - mediaSaturatedCarryover[media]
                )
                logger.debug("Completed transformations for media channel: %s", media)
            except Exception as e:
                logger.error("Error processing media channel %s: %s", media, str(e))
                raise

        logger.debug("Combining transformed data")
        dt_modAdstocked = pd.concat(
            [dt_modAdstocked.drop(columns=all_media), pd.DataFrame(mediaAdstocked)],
            axis=1,
        )
        dt_modSaturated = pd.concat(
            [
                dt_modAdstocked.iloc[
                    rollingWindowStartWhich : rollingWindowEndWhich + 1
                ].drop(columns=all_media),
                pd.DataFrame(mediaSaturated),
            ],
            axis=1,
        ).reset_index()
        dt_saturatedImmediate = (
            pd.DataFrame(mediaSaturatedImmediate).fillna(0).reset_index()
        )
        dt_saturatedCarryover = (
            pd.DataFrame(mediaSaturatedCarryover).fillna(0).reset_index()
        )

        result = TransformationResult(
            dt_modSaturated, dt_saturatedImmediate, dt_saturatedCarryover
        )
        logger.debug("Completed all transformations: %s", result)
        return result

    def get_media_transforms(self, solution_ids: List[str]) -> pd.DataFrame:
        """
        Get media transformation vectors for specified solutions.

        Args:
            solution_ids: List of solution IDs to get transforms for

        Returns:
            DataFrame containing media transformation vectors
        """
        transforms_list = []
        for sol_id in solution_ids:
            # Get media variables
            for media_var in self.mmm_data.mmmdata_spec.paid_media_vars:
                transforms = self._get_media_transform_vector(sol_id, media_var)
                transforms['solID'] = sol_id
                transforms['media_variable'] = media_var
                transforms_list.append(transforms)
        
        return pd.concat(transforms_list, ignore_index=True)

    def calculate_decomposition_vectors(
        self,
        coef: pd.DataFrame,
        hyppar: pd.DataFrame,
    ) -> Dict:
        """
        Calculate decomposition vectors for each media channel.

        Args:
            coef: Model coefficients
            hyppar: Model hyperparameters

        Returns:
            Dictionary mapping channel names to their decomposition vectors
        """
        x_decomp_vec = {}
        
        for channel in self.mmm_data.media_names:
            # Get channel hyperparameters
            channel_hyppar = hyppar[hyppar["variable"] == channel].iloc[0]
            
            # Get channel coefficient
            channel_coef = coef[coef["variable"] == channel].iloc[0]["coef"]
            
            # Calculate transformation based on adstock type
            if channel_hyppar["adstock"] == "geometric":
                x_decomp = self._geometric_adstock(
                    x=self.mmm_data.media_data[channel].values,
                    theta=channel_hyppar["theta"],
                )
            else:  # weibull
                x_decomp = self._weibull_adstock(
                    x=self.mmm_data.media_data[channel].values,
                    shape=channel_hyppar["shape"],
                    scale=channel_hyppar["scale"],
                )
            
            # Apply Hill transformation
            x_decomp = self._hill_transform(
                x=x_decomp,
                alpha=channel_hyppar["alpha"],
                gamma=channel_hyppar["gamma"],
            )
            
            # Apply coefficient
            x_decomp = x_decomp * channel_coef
            
            x_decomp_vec[channel] = x_decomp
            
        return x_decomp_vec

    def calculate_carryover_percentage(
        self,
        coef: pd.DataFrame,
        hyppar: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate carryover percentages for each media channel.

        Args:
            coef: Model coefficients
            hyppar: Model hyperparameters

        Returns:
            DataFrame with carryover percentages for each channel
        """
        caov_pct_list = []
        
        for channel in self.mmm_data.media_names:
            # Get channel hyperparameters
            channel_hyppar = hyppar[hyppar["variable"] == channel].iloc[0]
            
            # Calculate carryover based on adstock type
            if channel_hyppar["adstock"] == "geometric":
                caov_pct = channel_hyppar["theta"]
            else:  # weibull
                caov_pct = 1 - (1 / (1 + channel_hyppar["scale"]))
            
            caov_pct_list.append({
                "channel": channel,
                "carryover_pct": caov_pct,
            })
        
        return pd.DataFrame(caov_pct_list)

    def _geometric_adstock(
        self,
        x: np.ndarray,
        theta: float,
    ) -> np.ndarray:
        """Apply geometric adstock transformation."""
        y = np.zeros_like(x)
        y[0] = x[0]
        for i in range(1, len(x)):
            y[i] = x[i] + theta * y[i-1]
        return y

    def _weibull_adstock(
        self,
        x: np.ndarray,
        shape: float,
        scale: float,
    ) -> np.ndarray:
        """Apply Weibull adstock transformation."""
        # Generate Weibull weights
        L = len(x)
        weights = np.zeros(L)
        t = np.arange(1, L+1)
        weights = (shape/scale) * (t/scale)**(shape-1) * np.exp(-(t/scale)**shape)
        weights = weights / weights.sum()  # normalize
        
        # Apply convolution
        return np.convolve(x, weights, mode='full')[:L]

    def _hill_transform(
        self,
        x: np.ndarray,
        alpha: float,
        gamma: float,
    ) -> np.ndarray:
        """Apply Hill (S-curve) transformation."""
        x_scaled = x / x.max()  # scale to [0,1]
        return (x_scaled**gamma) / (alpha**gamma + x_scaled**gamma)
