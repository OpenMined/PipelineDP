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
from pydp.algorithms import numerical_mechanisms as dp_mechanisms


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

    def get_dp_strategy(self, l0: int, linf: float) -> DPStrategy:
        # todo:validate l0, linf
        # todo: should be used Sensitivities ?
        if self._is_public_partitions:
            return self._get_dp_strategy_for_public_partitions(l0, linf)
        if self.use_post_aggregation_thresholding():
            return self._get_dp_strategy_with_post_aggregation_threshold(l0)
        return self._get_dp_strategy_private_partition(l0, linf)

    def _get_dp_strategy_for_public_partitions(self, l0: int,
                                               linf: float) -> DPStrategy:
        noise_kind = self.choose_noise_kind(self._epsilon, self._delta, l0,
                                            linf)
        return DPStrategy(noise_kind=noise_kind,
                          partition_selection_strategy=None,
                          post_aggregation_thresholding=False)

    def _get_dp_strategy_with_post_aggregation_threshold(self,
                                                         l0: int) -> DPStrategy:
        delta_noise = self._delta / 2  # todo comment
        noise_kind = self.choose_noise_kind(self._epsilon, delta_noise, l0, 1)
        partition_selection_strategy = aggregate_params.noise_to_thresholding(
            noise_kind)
        return DPStrategy(
            noise_kind=noise_kind,
            partition_selection_strategy=partition_selection_strategy,
            post_aggregation_thresholding=True)

    def _get_dp_strategy_private_partition(self, l0: int,
                                           linf: float) -> DPStrategy:
        half_epsilon, half_delta = self._epsilon / 2, self._delta / 2
        noise_kind = self.choose_noise_kind(half_epsilon, half_delta, l0, linf)
        partition_selection_strategy = self.choose_partition_selection(
            half_epsilon, half_delta, l0)
        return DPStrategy(
            noise_kind=noise_kind,
            partition_selection_strategy=partition_selection_strategy,
            post_aggregation_thresholding=False)

    # def _choose_noise_kind(self, epsilon:float, delta:float)->pipeline_dp.NoiseKind:
    def _get_gaussian_std(self, epsilon: float, delta: float, l0: int,
                          linf: float) -> float:
        l2_sensitivity = math.sqrt(l0) * linf
        return dp_mechanisms.GaussianMechanism(epsilon,
                                               delta,
                                               sensitivity=l2_sensitivity).std

    def _get_laplace_std(self, epsilon: float, l0: int, linf: float) -> float:
        l1_sensitivity = l0 * linf
        return dp_mechanisms.LaplaceMechanism(epsilon,
                                              sensitivity=l1_sensitivity).std

    def choose_noise_kind(self, epsilon: float, delta: float, l0: int,
                          linf: float) -> pipeline_dp.NoiseKind:
        gaussian_std = self._get_gaussian_std(epsilon, delta, l0, linf)
        laplace_std = self._get_gaussian_std(epsilon, l0, linf)
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
