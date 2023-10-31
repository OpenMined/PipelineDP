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
"""todo"""
import math
from dataclasses import dataclass
from typing import List, Optional

import pipeline_dp
from pipeline_dp import aggregate_params
from pipeline_dp import dp_computations


@dataclass
class DPStrategy:
    noise_kind: pipeline_dp.NoiseKind
    partition_selection_strategy: Optional[
        pipeline_dp.PartitionSelectionStrategy]
    post_aggregation_thresholding: bool

    def __post_init__(self):
        pass  # todo


class DPStrategySelector:

    def __init__(self, epsilon: float, delta: float, metric: pipeline_dp.Metric,
                 is_public_partitions: bool, pre_threshold: Optional[int]):
        self._epsilon = epsilon
        self._delta = delta
        self._metric = metric
        self._is_public_partitions = is_public_partitions
        self._pre_threshold = pre_threshold
        # todo: check delta > 0 for is_public_partitions = false

    def get_dp_strategy(
            self, sensitivities: dp_computations.Sensitivities) -> DPStrategy:
        if self._is_public_partitions:
            return self._get_dp_strategy_for_public_partitions(sensitivities)
        if self.use_post_aggregation_thresholding():
            return self._get_dp_strategy_with_post_aggregation_threshold(
                sensitivities.l0)
        return self._get_dp_strategy_private_partition(sensitivities)

    def _get_dp_strategy_for_public_partitions(
            self, sensitivities: dp_computations.Sensitivities) -> DPStrategy:
        noise_kind = self.choose_noise_kind(self._epsilon, self._delta,
                                            sensitivities)
        return DPStrategy(noise_kind=noise_kind,
                          partition_selection_strategy=None,
                          post_aggregation_thresholding=False)

    def _get_dp_strategy_with_post_aggregation_threshold(
            self, l0_sensitivity: int) -> DPStrategy:
        delta_noise = self._delta / 2  # todo comment
        # todo: explain why linf = 1,
        noise_kind = self.choose_noise_kind(
            self._epsilon, delta_noise,
            dp_computations.Sensitivities(l0=l0_sensitivity, linf=1))
        partition_selection_strategy = aggregate_params.noise_to_thresholding(
            noise_kind)
        return DPStrategy(
            noise_kind=noise_kind,
            partition_selection_strategy=partition_selection_strategy,
            post_aggregation_thresholding=True)

    def _get_dp_strategy_private_partition(
            self, sensitivities: dp_computations.Sensitivities) -> DPStrategy:
        half_epsilon, half_delta = self._epsilon / 2, self._delta / 2
        noise_kind = self.choose_noise_kind(half_epsilon, half_delta,
                                            sensitivities)
        partition_selection_strategy = self.choose_partition_selection(
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

    def choose_noise_kind(
            self, epsilon: float, delta: float,
            sensitivities: dp_computations.Sensitivities
    ) -> pipeline_dp.NoiseKind:
        if delta == 0:
            return pipeline_dp.NoiseKind.LAPLACE
        gaussian_std = self._get_gaussian_std(epsilon, delta, sensitivities)
        laplace_std = self._get_laplace_std(epsilon, sensitivities)
        if gaussian_std < laplace_std:
            return pipeline_dp.NoiseKind.GAUSSIAN
        return pipeline_dp.NoiseKind.LAPLACE

    def use_post_aggregation_thresholding(self,
                                          metric: pipeline_dp.Metric) -> bool:
        return metric == pipeline_dp.Metrics.PRIVACY_ID_COUNT

    def choose_partition_selection(
            self, epsilon: float, delta: float,
            l0: int) -> pipeline_dp.PartitionSelectionStrategy:

        def create_mechanism(strategy: pipeline_dp.PartitionSelectionStrategy):
            return dp_computations.ThresholdingMechanism(
                epsilon, delta, strategy, l0, self._pre_threshold)

        laplace_thresholding_mechanism = create_mechanism(
            pipeline_dp.PartitionSelectionStrategy.LAPLACE_THRESHOLDING)
        gaussian_thresholding_mechanism = create_mechanism(
            pipeline_dp.PartitionSelectionStrategy.GAUSSIAN_THRESHOLDING)
        if laplace_thresholding_mechanism.threshold(
        ) < gaussian_thresholding_mechanism.threshold():
            # todo: explain why
            return pipeline_dp.PartitionSelectionStrategy.TRUNCATED_GEOMETRIC
        return pipeline_dp.PartitionSelectionStrategy.GAUSSIAN_THRESHOLDING
