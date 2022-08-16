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
"""Utility Analysis Combiners."""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Sized, Tuple
import numpy as np

import pipeline_dp
from pipeline_dp import dp_computations

MAX_PROBABILITIES_IN_ACCUMULATOR = 100


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
class SumOfRandomVariablesMoments:
    """Stores moments of sum of random independent random variables."""
    count: int
    expectation: float
    variance: float
    third_central_moment: float

    def __add__(
            self, other: 'SumOfRandomVariablesMoments'
    ) -> 'SumOfRandomVariablesMoments':
        return SumOfRandomVariablesMoments(
            self.count + other.count, self.expectation + other.expectation,
            self.variance + other.variance,
            self.third_central_moment + other.third_central_moment)


def _probabilities_to_moments(
        probabilities: List[float]) -> SumOfRandomVariablesMoments:
    """Computes moments of sum of independent bernoulli random variables."""
    exp = np.sum(probabilities)
    var = np.sum([p * (1 - p) for p in probabilities])
    third_central_moment = np.sum(
        [p * (1 - p) * (1 - 2 * p) for p in probabilities])

    return SumOfRandomVariablesMoments(len(probabilities), exp, var,
                                       third_central_moment)


@dataclass
class PartitionSelectionAccumulator:
    """Stores data needed for computing probability of keeping the partition.

    Args:
        probabilities: probabilities that each specific user contributes to the
        partition after contribution bounding.
        moments: contains moments of the sum of independent random
        variables, which represent whether user contributes to the partition.

    Those variables are set mutually exclusive. If len(probabilities) <= 
    MAX_PROBABILITIES_IN_ACCUMULATOR then 'probabilities' are used otherwise
    'moments'. The idea is that when the number of the contributions are small
    the sum of the random variables is far from Normal distribution and exact
    computations are performed, otherwise a Normal approximation based on
    moments is used.
    """
    probabilities: Optional[List[float]] = None
    moments: Optional[SumOfRandomVariablesMoments] = None

    def __post_init__(self):
        assert (self.probabilities is None) != (
            self.moments is
            None), "Only one of probabilities and moments must be set."

    def __add__(self, other):
        probs_self = self.probabilities
        probs_other = other.probabilities
        if probs_self and probs_other and len(probs_self) + len(
                probs_other) <= MAX_PROBABILITIES_IN_ACCUMULATOR:
            return PartitionSelectionAccumulator(probs_self + probs_other)
        moments_self = self.moments
        if moments_self is None:
            moments_self = _probabilities_to_moments(probs_self)

        moments_other = other.moments
        if moments_other is None:
            moments_other = _probabilities_to_moments(probs_other)

        return PartitionSelectionAccumulator(moments=moments_self +
                                             moments_other)


@dataclass
class UtilityAnalysisCountAccumulator:
    count: int
    per_partition_contribution_error: int
    expected_cross_partition_error: float
    var_cross_partition_error: float
    partition_selection_accumulator: Optional[PartitionSelectionAccumulator]

    def __add__(self, other):
        ps_accumulator = None
        if self.partition_selection_accumulator is not None:
            ps_accumulator = self.partition_selection_accumulator + other.partition_selection_accumulator

        return UtilityAnalysisCountAccumulator(
            self.count + other.count, self.per_partition_contribution_error +
            other.per_partition_contribution_error,
            self.expected_cross_partition_error +
            other.expected_cross_partition_error,
            self.var_cross_partition_error + other.var_cross_partition_error,
            ps_accumulator)


class UtilityAnalysisCountCombiner(pipeline_dp.Combiner):
    """A combiner for utility analysis counts."""
    AccumulatorType = UtilityAnalysisCountAccumulator

    def __init__(self, params: pipeline_dp.combiners.CombinerParams,
                 is_public_partitions: bool):
        self._params = params
        self._is_public_partitions = is_public_partitions

    def create_accumulator(self, data: Tuple[Sized, int]) -> AccumulatorType:
        """Creates an accumulator for data.

        Args:
            data is a Tuple containing; 1) a list of the user's contributions for a single partition, and 2) the total
            number of partitions a user contributed to.

        Returns:
            An accumulator with the count of contributions and the contribution error.
        """
        if not data:
            return UtilityAnalysisCountAccumulator(0, 0, 0, 0, None)
        values, n_partitions = data
        count = len(values)
        max_per_partition = self._params.aggregate_params.max_contributions_per_partition
        max_partitions = self._params.aggregate_params.max_partitions_contributed
        prob_keep_partition = min(1, max_partitions /
                                  n_partitions) if n_partitions > 0 else 0
        per_partition_contribution = min(max_per_partition, count)
        per_partition_contribution_error = per_partition_contribution - count
        expected_cross_partition_error = -per_partition_contribution * (
            1 - prob_keep_partition)
        var_cross_partition_error = per_partition_contribution**2 * prob_keep_partition * (
            1 - prob_keep_partition)

        ps_accumulator = None
        if not self._is_public_partitions:
            ps_accumulator = PartitionSelectionAccumulator(
                probabilities=[prob_keep_partition])

        return UtilityAnalysisCountAccumulator(
            count, per_partition_contribution_error,
            expected_cross_partition_error, var_cross_partition_error,
            ps_accumulator)

    def merge_accumulators(self, acc1: AccumulatorType, acc2: AccumulatorType):
        """Merges two accumulators together additively."""
        return acc1 + acc2

    def compute_metrics(self,
                        acc: AccumulatorType) -> CountUtilityAnalysisMetrics:
        """Computes metrics based on the accumulator properties.

        Args:
            acc: the accumulator to compute from.

        Returns:
            A CountUtilityAnalysisMetrics object with computed metrics.
        """
        std_noise = dp_computations.compute_dp_count_noise_std(
            self._params.scalar_noise_params)
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


@dataclass
class SumUtilityAnalysisMetrics:
    """Stores metrics for the sum utility analysis.

    Args:
        sum: actual sum of contributions per partition.
        per_partition_contribution_error_min: the amount of sum contribution error in a partition created by
        contribution min clipping.
        per_partition_contribution_error_max: the amount of sum contribution error in a partition created by
        contribution max clipping.
        expected_cross_partition_error: the estimated amount of error across partitions.
        std_cross_partition_error: the standard deviation of the contribution bounding error.
        std_noise: the noise standard deviation for DP count.
        noise_kind: the type of noise used.
    """
    sum: float
    per_partition_contribution_error_min: float
    per_partition_contribution_error_max: float
    expected_cross_partition_error: float
    std_cross_partition_error: float
    std_noise: float
    noise_kind: pipeline_dp.NoiseKind


@dataclass
class UtilityAnalysisSumAccumulator:
    sum: float
    per_partition_contribution_error_min: float
    per_partition_contribution_error_max: float
    expected_cross_partition_error: float
    var_cross_partition_error: float
    partition_selection_accumulator: Optional[PartitionSelectionAccumulator]

    def __add__(self, other):
        ps_accumulator = None
        if self.partition_selection_accumulator is not None:
            ps_accumulator = self.partition_selection_accumulator +\
                             other.partition_selection_accumulator

        return UtilityAnalysisSumAccumulator(
            self.sum + other.sum, self.per_partition_contribution_error_min +
            other.per_partition_contribution_error_min,
            self.per_partition_contribution_error_max +
            other.per_partition_contribution_error_max,
            self.expected_cross_partition_error +
            other.expected_cross_partition_error,
            self.var_cross_partition_error + other.var_cross_partition_error,
            ps_accumulator)


class UtilityAnalysisSumCombiner(pipeline_dp.Combiner):
    """A combiner for utility analysis sums."""
    AccumulatorType = UtilityAnalysisSumAccumulator

    def __init__(self, params: pipeline_dp.combiners.CombinerParams,
                 is_public_partitions: bool):
        self._params = params
        self._is_public_partitions = is_public_partitions

    def create_accumulator(self, data: Tuple[Sequence, int]) -> AccumulatorType:
        """Creates an accumulator for data.

        Args:
            data is a Tuple containing; 1) a list of the user's contributions for a single partition, and 2) the total
            number of partitions a user contributed to.

        Returns:
            An accumulator with the sum of contributions and the contribution error.
        """
        if not data or not data[0]:
            return UtilityAnalysisSumAccumulator(0, 0, 0, 0, 0, None)
        values, n_partitions = data
        max_partitions = self._params.aggregate_params.max_partitions_contributed
        prob_keep_partition = min(1, max_partitions /
                                  n_partitions) if n_partitions > 0 else 0
        partition_sum = np.sum(values)
        per_partition_contribution = np.clip(
            partition_sum, self._params.aggregate_params.min_sum_per_partition,
            self._params.aggregate_params.max_sum_per_partition)
        per_partition_contribution_error_min = 0
        per_partition_contribution_error_max = 0
        per_partition_contribution_error = partition_sum - per_partition_contribution
        if per_partition_contribution_error > 0:
            per_partition_contribution_error_max = per_partition_contribution_error
        elif per_partition_contribution_error < 0:
            per_partition_contribution_error_min = per_partition_contribution_error
        expected_cross_partition_error = -per_partition_contribution * (
            1 - prob_keep_partition)
        var_cross_partition_error = per_partition_contribution**2 * prob_keep_partition * (
            1 - prob_keep_partition)

        ps_accumulator = None
        if not self._is_public_partitions:
            ps_accumulator = PartitionSelectionAccumulator(
                probabilities=[prob_keep_partition])

        return UtilityAnalysisSumAccumulator(
            partition_sum, per_partition_contribution_error_min,
            per_partition_contribution_error_max,
            expected_cross_partition_error, var_cross_partition_error,
            ps_accumulator)

    def merge_accumulators(self, acc1: AccumulatorType, acc2: AccumulatorType):
        """Merges two accumulators together additively."""
        return acc1 + acc2

    def compute_metrics(self,
                        acc: AccumulatorType) -> SumUtilityAnalysisMetrics:
        """Computes metrics based on the accumulator properties."""
        std_noise = dp_computations.compute_dp_count_noise_std(
            self._params.scalar_noise_params)
        return SumUtilityAnalysisMetrics(
            sum=acc.sum,
            per_partition_contribution_error_min=acc.
            per_partition_contribution_error_min,
            per_partition_contribution_error_max=acc.
            per_partition_contribution_error_max,
            expected_cross_partition_error=acc.expected_cross_partition_error,
            std_cross_partition_error=np.sqrt(acc.var_cross_partition_error),
            std_noise=std_noise,
            noise_kind=self._params.aggregate_params.noise_kind)

    def metrics_names(self) -> List[str]:
        return [
            'sum', 'per_partition_contribution_error_min',
            'per_partition_contribution_error_max',
            'expected_cross_partition_error', 'std_cross_partition_error',
            'std_noise', 'noise_kind'
        ]

    def explain_computation(self):
        pass
