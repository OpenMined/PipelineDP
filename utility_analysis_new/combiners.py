# Copyright 2022 OpenMined.
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
"""UtilityAnalysisCountCombiner."""

from dataclasses import dataclass
from typing import List, Sized, Tuple
import numpy as np

import pipeline_dp
from pipeline_dp import dp_computations


@dataclass
class CountUtilityAnalysisMetrics:
    """Stores metrics for the count utility analysis.

    Args:
        count: actual count of contributions per partition.
        per_partition_contribution_error: the amount of contribution error in a partition.
        expected_cross_partition_error: the estimated amount of error across partitions.
        std_cross_partition_error: the standard deviation of the contribution bounding error.
        std_noise: the noise standard deviation for DP count.
        noise_kind: the type of noise used.
    """
    count: int
    per_partition_contribution_error: int
    expected_cross_partition_error: float
    std_cross_partition_error: float
    std_noise: float
    noise_kind: pipeline_dp.NoiseKind


@dataclass
class UtilityAnalysisCountAccumulator:
    count: int
    per_partition_contribution_error: int
    expected_cross_partition_error: float
    var_cross_partition_error: float


class UtilityAnalysisCountCombiner(pipeline_dp.Combiner):
    """A combiner for utility analysis counts."""
    AccumulatorType = UtilityAnalysisCountAccumulator

    def __init__(self, params: pipeline_dp.combiners.CombinerParams):
        self._params = params

    def create_accumulator(self, data: Tuple[Sized, int]) -> AccumulatorType:
        """Create an accumulator for data.

        Args:
            data is a Tuple containing; 1) a list of the user's contributions for a single partition, and 2) the total
            number of partitions a user contributed to.

        Returns:
            An accumulator with the count of contributions and the contribution error.
        """
        if not data:
            return UtilityAnalysisCountAccumulator(0, 0, 0, 0)
        values, n_partitions = data
        count = len(values)
        max_per_partition = self._params.aggregate_params.max_contributions_per_partition
        max_partitions = self._params.aggregate_params.max_partitions_contributed
        prob_keep_partition = min(1, max_partitions /
                                  n_partitions) if n_partitions > 0 else 0
        per_partition_contribution = min(max_per_partition, count)
        per_partition_contribution_error = max(
            0, count - per_partition_contribution)
        expected_cross_partition_error = per_partition_contribution * (
            1 - prob_keep_partition)
        var_cross_partition_error = per_partition_contribution**2 * prob_keep_partition * (
            1 - prob_keep_partition)

        return UtilityAnalysisCountAccumulator(
            count, per_partition_contribution_error,
            expected_cross_partition_error, var_cross_partition_error)

    def merge_accumulators(self, acc1: AccumulatorType, acc2: AccumulatorType):
        """Merge two accumulators together additively."""
        return UtilityAnalysisCountAccumulator(
            count=acc1.count + acc2.count,
            per_partition_contribution_error=acc1.
            per_partition_contribution_error +
            acc2.per_partition_contribution_error,
            expected_cross_partition_error=acc1.expected_cross_partition_error +
            acc2.expected_cross_partition_error,
            var_cross_partition_error=acc1.var_cross_partition_error +
            acc2.var_cross_partition_error)

    def compute_metrics(self,
                        acc: AccumulatorType) -> CountUtilityAnalysisMetrics:
        """Compute metrics based on the accumulator properties.

        Args:
            acc: the accumulator to compute from.

        Returns:
            A CountUtilityAnalysisMetrics object with computed metrics.
        """
        std_noise = dp_computations.compute_dp_count_noise_std(
            self._params.mean_var_params)
        return CountUtilityAnalysisMetrics(
            count=acc.count,
            per_partition_contribution_error=acc.
            per_partition_contribution_error,
            expected_cross_partition_error=acc.expected_cross_partition_error,
            std_cross_partition_error=np.sqrt(acc.var_cross_partition_error),
            std_noise=std_noise,
            noise_kind=self._params.aggregate_params.noise_kind)

    def metrics_names(self) -> List[str]:
        return [
            'count', 'per_partition_contribution_error',
            'expected_cross_partition_error', 'std_cross_partition_error',
            'std_noise', 'noise_kind'
        ]

    def explain_computation(self):
        pass
