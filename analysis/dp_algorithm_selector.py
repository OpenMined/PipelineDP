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
from pipeline_dp import dp_computations
from pydp.algorithms import numerical_mechanisms as dp_mechanisms


@dataclass
class DPStrategy:
    noise_kind: pipeline_dp.NoiseKind
    partition_selection_strategy: Optional[
        pipeline_dp.PartitionSelectionStrategy]
    post_aggregation_thresholding: bool


class AvailableMethods:

    def supported_noise_kind(self) -> List[pipeline_dp.NoiseKind]:
        return [pipeline_dp.NoiseKind.LAPLACE, pipeline_dp.NoiseKind.GAUSSIAN]

    def supported_partition_selection_strategies(
            self) -> List[pipeline_dp.PartitionSelectionStrategy]:
        return [
            pipeline_dp.PartitionSelectionStrategy.TRUNCATED_GEOMETRIC,
            pipeline_dp.PartitionSelectionStrategy.LAPLACE_THRESHOLDING,
            pipeline_dp.PartitionSelectionStrategy.GAUSSIAN_THRESHOLDING
        ]

    def support_post_aggregation_thresholding(self):
        return True


class DPAlgorithmSelector:

    def __init__(self, epsilon: float, delta: float, metric: pipeline_dp.Metric,
                 is_public_partitions: bool, pre_threshold: Optional[int],
                 available_methods: AvailableMethods):
        self._epsilon = epsilon
        self._delta = delta
        self._metric = metric
        self._is_public_partitions = is_public_partitions
        self._pre_threshold = pre_threshold
        self._available_methods = available_methods

    def get_dp_strategy(self, l0: int, linf: float) -> DPStrategy:
        # todo:validate l0, linf
        if self._is_public_partitions:
            return self._get_dp_strategy_for_public_partitions(l0, linf)
        if self._metric == pipeline_dp.Metrics.PRIVACY_ID_COUNT and self._available_methods.support_post_aggregation_thresholding(
        ):
            return self._get_dp_strategy_with_post_aggregation_threshold(l0)
        return self._get_dp_strategy_private_partition(l0, linf)

    def _get_dp_strategy_for_public_partitions(self, l0: int,
                                               linf: float) -> DPStrategy:
        supported_noise_kinds = self._available_methods.supported_noise_kind()
        if len(supported_noise_kinds) == 1:
            return DPStrategy(noise_kind=supported_noise_kinds[0],
                              partition_selection_strategy=None,
                              post_aggregation_thresholding=False)
        gaussian_std = self._get_gaussian_std(self._epsilon, self._delta, l0,
                                              linf)
        laplace_std = self._get_gaussian_std(self._epsilon, l0, linf)
        if gaussian_std < laplace_std:
            noise_kind = pipeline_dp.NoiseKind.GAUSSIAN
        else:
            noise_kind = pipeline_dp.NoiseKind.LAPLACE
        return DPStrategy(noise_kind=noise_kind,
                          partition_selection_strategy=None,
                          post_aggregation_thresholding=False)

    def _get_dp_strategy_with_post_aggregation_threshold(self,
                                                         l0: int) -> DPStrategy:

        def create_mechanism(strategy: pipeline_dp.PartitionSelectionStrategy):
            return dp_computations.ThresholdingMechanism(
                self._epsilon, self._delta, strategy, l0, self._pre_threshold)

        gaussian_thresholding_mechanism = create_mechanism(
            pipeline_dp.PartitionSelectionStrategy.GAUSSIAN_THRESHOLDING)
        laplace_thresholding_mechanism = create_mechanism(
            pipeline_dp.PartitionSelectionStrategy.LAPLACE_THRESHOLDING)
        # todo: if supported
        if gaussian_thresholding_mechanism.noise_std(
        ) < laplace_thresholding_mechanism.noise_std():
            noise_kind = pipeline_dp.NoiseKind.GAUSSIAN
            partition_selection_strategy = pipeline_dp.PartitionSelectionStrategy.GAUSSIAN_THRESHOLDING
        else:
            noise_kind = pipeline_dp.NoiseKind.LAPLACE
            partition_selection_strategy = pipeline_dp.PartitionSelectionStrategy.LAPLACE_THRESHOLDING
        return DPStrategy(
            noise_kind=noise_kind,
            partition_selection_strategy=partition_selection_strategy,
            post_aggregation_thresholding=True)

    def _get_dp_strategy_private_partition(self, l0: int,
                                           linf: float) -> DPStrategy:
        pass

    # def _choose_noise_kind(self, epsilon:float, delta:float)->pipeline_dp.NoiseKind:
    def _get_gaussian_std(self, epsilon: float, delta: float, l0: int,
                          linf: float) -> float:
        l2_sensitivity = math.sqrt(l0) * linf
        return dp_mechanisms.GaussianMechanism(epsilon,
                                               delta,
                                               sensitivity=l2_sensitivity).std

    def _get_laplace_std(self, epsilon: float, l0: int, linf: float) -> float:
        l1_sensitivity = l0 * linf
        return dp_mechanisms.GaussianMechanism(epsilon,
                                               sensitivity=l1_sensitivity).std
