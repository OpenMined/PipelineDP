# Copyright 2023 OpenMined.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Choosing DP Strategy (i.e. noise_kind, partition selection strategy etc)
based on contribution bounding params."""
from dataclasses import dataclass
from typing import List, Optional

import pipeline_dp
from pipeline_dp import aggregate_params
from pipeline_dp import dp_computations
from pipeline_dp import input_validators


@dataclass
class DPStrategy:
    """Represents the chosen DP strategy."""
    noise_kind: Optional[pipeline_dp.NoiseKind]
    partition_selection_strategy: Optional[
        pipeline_dp.PartitionSelectionStrategy]
    post_aggregation_thresholding: bool


class DPStrategySelector:
    """Chooses DPStrategy based on DP budget, computation sensitivites, etc

    It chooses noise_kind to minimize the noise std deviation.
    For non-public partitions it chooses partition selection strategy to
    minimize threshold. For more details see docstring to
    select_partition_selection_strategy().
    """

    def __init__(self, epsilon: float, delta: float,
                 metrics: List[pipeline_dp.Metric], is_public_partitions: bool):
        input_validators.validate_epsilon_delta(epsilon, delta,
                                                "DPStrategySelector")
        if delta == 0 and not is_public_partitions:
            raise ValueError("DPStrategySelector: when private partition "
                             "selection is used, delta must be positive")
        self._epsilon = epsilon
        self._delta = delta
        self._metrics = metrics
        self._is_public_partitions = is_public_partitions

    @property
    def is_public_partitions(self) -> bool:
        return self._is_public_partitions

    @property
    def metrics(self) -> List[pipeline_dp.Metric]:
        return self._metrics

    def get_dp_strategy(
            self, sensitivities: dp_computations.Sensitivities) -> DPStrategy:
        """Chooses DPStrategy for given sensitivities."""
        if not self._metrics:
            # This is Select partitions case.
            return self._get_strategy_for_select_partition(sensitivities.l0)

        n_metrics = len(self._metrics)
        # Having n metrics is equivalent of multiplying of contributing for
        # n times more partitions
        scaled_sensitivities = dp_computations.Sensitivities(
            l0=sensitivities.l0 * n_metrics, linf=sensitivities.linf)

        if self._is_public_partitions:
            return self._get_dp_strategy_for_public_partitions(
                scaled_sensitivities)
        if self.use_post_aggregation_thresholding(self._metrics):
            return self._get_dp_strategy_with_post_aggregation_threshold(
                scaled_sensitivities.l0)
        return self._get_dp_strategy_private_partition(scaled_sensitivities)

    def _get_strategy_for_select_partition(self,
                                           l0_sensitivity: int) -> DPStrategy:
        return DPStrategy(noise_kind=None,
                          partition_selection_strategy=self.
                          select_partition_selection_strategy(
                              self._epsilon, self._delta, l0_sensitivity),
                          post_aggregation_thresholding=False)

    def _get_dp_strategy_for_public_partitions(
            self, sensitivities: dp_computations.Sensitivities) -> DPStrategy:
        noise_kind = self.select_noise_kind(self._epsilon, self._delta,
                                            sensitivities)
        return DPStrategy(noise_kind=noise_kind,
                          partition_selection_strategy=None,
                          post_aggregation_thresholding=False)

    def _get_dp_strategy_with_post_aggregation_threshold(
            self, l0_sensitivity: int) -> DPStrategy:
        assert pipeline_dp.Metrics.PRIVACY_ID_COUNT in self._metrics
        # Half delta goes to the noise, the other half for partition selection.
        # For more details see
        # https://github.com/google/differential-privacy/blob/main/common_docs/Delta_For_Thresholding.pdf
        delta_noise = self._delta / 2
        # linf sensitivity = 1, because metric = PRIVACY_ID_COUNT
        sensitivities = dp_computations.Sensitivities(l0=l0_sensitivity, linf=1)
        noise_kind = self.select_noise_kind(self._epsilon, delta_noise,
                                            sensitivities)
        partition_selection_strategy = aggregate_params.noise_to_thresholding(
            noise_kind).to_partition_selection_strategy()
        return DPStrategy(
            noise_kind=noise_kind,
            partition_selection_strategy=partition_selection_strategy,
            post_aggregation_thresholding=True)

    def _get_dp_strategy_private_partition(
            self, sensitivities: dp_computations.Sensitivities) -> DPStrategy:
        half_epsilon, half_delta = self._epsilon / 2, self._delta / 2
        noise_kind = self.select_noise_kind(half_epsilon, half_delta,
                                            sensitivities)
        partition_selection_strategy = self.select_partition_selection_strategy(
            half_epsilon, half_delta, sensitivities.l0)
        return DPStrategy(
            noise_kind=noise_kind,
            partition_selection_strategy=partition_selection_strategy,
            post_aggregation_thresholding=False)

    def _get_gaussian_std(
            self, epsilon: float, delta: float,
            sensitivities: dp_computations.Sensitivities) -> float:
        return dp_computations.GaussianMechanism.create_from_epsilon_delta(
            epsilon, delta, l2_sensitivity=sensitivities.l2).std

    def _get_laplace_std(self, epsilon: float,
                         sensitivities: dp_computations.Sensitivities) -> float:
        return dp_computations.LaplaceMechanism.create_from_epsilon(
            epsilon, sensitivities.l1).std

    def select_noise_kind(
            self, epsilon: float, delta: float,
            sensitivities: dp_computations.Sensitivities
    ) -> pipeline_dp.NoiseKind:
        """Returns the noise with the minimum standard deviation."""
        if delta == 0:
            return pipeline_dp.NoiseKind.LAPLACE
        gaussian_std = self._get_gaussian_std(epsilon, delta, sensitivities)
        laplace_std = self._get_laplace_std(epsilon, sensitivities)
        if gaussian_std < laplace_std:
            return pipeline_dp.NoiseKind.GAUSSIAN
        return pipeline_dp.NoiseKind.LAPLACE

    def use_post_aggregation_thresholding(
            self, metrics: List[pipeline_dp.Metric]) -> bool:
        return pipeline_dp.Metrics.PRIVACY_ID_COUNT in metrics

    def select_partition_selection_strategy(
            self, epsilon: float, delta: float,
            l0_sensitivity: int) -> pipeline_dp.PartitionSelectionStrategy:
        """Selects partition selection strategy based on Threshold.

        There are many ways how strategies can be compared. For simplicity
        strategies are compared by the number of privacy units, which is needed
        for achieving the probability of releasing partition to be 50%. That is
        number is equal to the threshold for thresholding strategies.

        Args:
            epsilon, delta: DP budget for partition selection
            l0_sensitivity: l0 sensitivity of the query, i.e. the maximum
              number of partitions, which 1 privacy unit can influence.

        Returns:
            the selected strategy.
        """

        def create_mechanism(strategy: pipeline_dp.PartitionSelectionStrategy):
            return dp_computations.ThresholdingMechanism(epsilon,
                                                         delta,
                                                         strategy,
                                                         l0_sensitivity,
                                                         pre_threshold=None)

        laplace_thresholding_mechanism = create_mechanism(
            pipeline_dp.PartitionSelectionStrategy.LAPLACE_THRESHOLDING)
        gaussian_thresholding_mechanism = create_mechanism(
            pipeline_dp.PartitionSelectionStrategy.GAUSSIAN_THRESHOLDING)
        if laplace_thresholding_mechanism.threshold(
        ) < gaussian_thresholding_mechanism.threshold():
            # Truncated geometric strategy is slightly better than Laplace
            # thresholding, so returns it instead.
            # Truncated geometric does not have threshold, that is why
            #
            return pipeline_dp.PartitionSelectionStrategy.TRUNCATED_GEOMETRIC
        return pipeline_dp.PartitionSelectionStrategy.GAUSSIAN_THRESHOLDING


class DPStrategySelectorFactory:

    def create(self, epsilon: float, delta: float,
               metrics: List[pipeline_dp.Metric], is_public_partitions: bool):
        return DPStrategySelector(epsilon, delta, metrics, is_public_partitions)
