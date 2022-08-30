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
import math

import scipy

import pipeline_dp
from pipeline_dp import dp_computations
from utility_analysis_new import poisson_binomial

import pydp.algorithms.partition_selection as partition_selection

MAX_PROBABILITIES_IN_ACCUMULATOR = 100


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
    exp = sum(probabilities)
    var = sum([p * (1 - p) for p in probabilities])
    third_central_moment = sum(
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

    def compute_probability_to_keep(self, eps: float, delta: float,
                                    max_partitions_contributed: int) -> float:
        """Computes the probability that this partition is kept.

        If self.probabilities is set, then the computed probability is exact,
        otherwise it is an approximation computed from self.moments.
        """
        if self.probabilities:
            pmf = poisson_binomial.compute_pmf(self.probabilities)
        else:
            moments = self.moments
            std = math.sqrt(moments.variance)
            skewness = moments.third_central_moment / std**3
            pmf = poisson_binomial.compute_pmf_approximation(
                moments.expectation, std, skewness, moments.count)

        ps_strategy = partition_selection.create_truncated_geometric_partition_strategy(
            eps, delta, max_partitions_contributed)
        probability = 0
        for i, prob in enumerate(pmf):
            probability += prob * ps_strategy.probability_of_keep(i)
        return probability


class PartitionSelectionCombiner(pipeline_dp.Combiner):
    """A combiner for utility analysis counts."""
    AccumulatorType = PartitionSelectionAccumulator

    def __init__(self, params: pipeline_dp.combiners.CombinerParams):
        self._params = params

    def create_accumulator(self, data: Tuple[Sized, int]) -> AccumulatorType:
        """Creates an accumulator for data.

        Args:
            data is a Tuple containing; 1) a list of the user's contributions
            for a single partition, and 2) the total number of partitions a user
            contributed to.

        Returns:
            An accumulator for computing probability of selecting partition.
        """
        values, n_partitions = data
        max_partitions = self._params.aggregate_params.max_partitions_contributed
        prob_keep_contribution = min(1, max_partitions /
                                     n_partitions) if n_partitions > 0 else 0

        return PartitionSelectionAccumulator(
            probabilities=[prob_keep_contribution])

    def merge_accumulators(self, acc1: AccumulatorType, acc2: AccumulatorType):
        """Merges two accumulators together additively."""
        return acc1 + acc2

    def compute_metrics(self, acc: AccumulatorType) -> float:
        """Computes metrics based on the accumulator properties."""
        params = self._params
        return acc.compute_probability_to_keep(
            params.eps, params.delta,
            params.aggregate_params.max_partitions_contributed)

    def metrics_names(self) -> List[str]:
        return ['probability_to_keep']

    def explain_computation(self):
        pass


@dataclass
class CountUtilityAnalysisMetrics:
    """Stores metrics for the count utility analysis.

    Args:
        count: actual count of contributions per partition.
        per_partition_error: the amount of error due to per-partition contribution bounding.
        expected_cross_partition_error: the expected amount of error due to cross-partition contribution bounding.
        std_cross_partition_error: the standard deviation of the error due to cross-partition contribution bounding.
        std_noise: the noise standard deviation.
        noise_kind: the type of noise used.
    """
    count: int
    per_partition_error: int
    expected_cross_partition_error: float
    std_cross_partition_error: float
    std_noise: float
    noise_kind: pipeline_dp.NoiseKind

    # Absolute error metrics derived from the primitives above
    abs_error_expected: float = None
    abs_error_variance: float = None
    abs_error_99pct: float = None

    # Relative error metrics derived from the primitives above
    rel_error_expected: float = None
    rel_error_variance: float = None
    rel_error_99pct: float = None


@dataclass
class UtilityAnalysisCountAccumulator:
    count: int
    per_partition_error: int
    expected_cross_partition_error: float
    var_cross_partition_error: float

    def __add__(self, other):
        return UtilityAnalysisCountAccumulator(
            self.count + other.count,
            self.per_partition_error + other.per_partition_error,
            self.expected_cross_partition_error +
            other.expected_cross_partition_error,
            self.var_cross_partition_error + other.var_cross_partition_error)


class UtilityAnalysisCountCombiner(pipeline_dp.Combiner):
    """A combiner for utility analysis counts."""
    AccumulatorType = UtilityAnalysisCountAccumulator

    def __init__(self, params: pipeline_dp.combiners.CombinerParams):
        self._params = params

    @property
    def _is_public_partitions(self):
        return self._partition_selection_budget is None

    def create_accumulator(self, data: Tuple[Sized, int]) -> AccumulatorType:
        """Creates an accumulator for data.

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
        per_partition_error = per_partition_contribution - count
        expected_cross_partition_error = -per_partition_contribution * (
            1 - prob_keep_partition)
        var_cross_partition_error = per_partition_contribution**2 * prob_keep_partition * (
            1 - prob_keep_partition)

        return UtilityAnalysisCountAccumulator(count, per_partition_error,
                                               expected_cross_partition_error,
                                               var_cross_partition_error)

    def merge_accumulators(self, acc1: AccumulatorType, acc2: AccumulatorType):
        """Merges two accumulators together additively."""
        return acc1 + acc2

    def compute_metrics(self,
                        acc: AccumulatorType) -> CountUtilityAnalysisMetrics:
        """Computes metrics based on the accumulator properties."""
        std_noise = dp_computations.compute_dp_count_noise_std(
            self._params.scalar_noise_params)
        return CountUtilityAnalysisMetrics(
            count=acc.count,
            per_partition_error=acc.per_partition_error,
            expected_cross_partition_error=acc.expected_cross_partition_error,
            std_cross_partition_error=np.sqrt(acc.var_cross_partition_error),
            std_noise=std_noise,
            noise_kind=self._params.aggregate_params.noise_kind)

    def metrics_names(self) -> List[str]:
        return [
            'count', 'per_partition_error', 'expected_cross_partition_error',
            'std_cross_partition_error', 'std_noise', 'noise_kind'
        ]

    def explain_computation(self):
        pass


@dataclass
class SumUtilityAnalysisMetrics:
    """Stores metrics for the sum utility analysis.

    Args:
        sum: actual sum of contributions per partition.
        per_partition_error_min: the amount of error due to contribution min clipping.
        per_partition_error_max: the amount of error due to contribution max clipping.
        expected_cross_partition_error: the expected amount of error due to cross-partition contribution bounding.
        std_cross_partition_error: the standard deviation of the error due to cross-partition contribution bounding.
        std_noise: the noise standard deviation.
        noise_kind: the type of noise used.
    """
    sum: float
    per_partition_error_min: float
    per_partition_error_max: float
    expected_cross_partition_error: float
    std_cross_partition_error: float
    std_noise: float
    noise_kind: pipeline_dp.NoiseKind


@dataclass
class UtilityAnalysisSumAccumulator:
    sum: float
    per_partition_error_min: float
    per_partition_error_max: float
    expected_cross_partition_error: float
    var_cross_partition_error: float

    def __add__(self, other):
        return UtilityAnalysisSumAccumulator(
            self.sum + other.sum,
            self.per_partition_error_min + other.per_partition_error_min,
            self.per_partition_error_max + other.per_partition_error_max,
            self.expected_cross_partition_error +
            other.expected_cross_partition_error,
            self.var_cross_partition_error + other.var_cross_partition_error)


class UtilityAnalysisSumCombiner(pipeline_dp.Combiner):
    """A combiner for utility analysis sums."""
    AccumulatorType = UtilityAnalysisSumAccumulator

    def __init__(self, params: pipeline_dp.combiners.CombinerParams):
        self._params = params

    def create_accumulator(self, data: Tuple[Sequence, int]) -> AccumulatorType:
        """Creates an accumulator for data.

        Args:
            data is a Tuple containing; 1) a list of the user's contributions for a single partition, and 2) the total
            number of partitions a user contributed to.

        Returns:
            An accumulator with the sum of contributions and the contribution error.
        """
        if not data or not data[0]:
            return UtilityAnalysisSumAccumulator(0, 0, 0, 0, 0)
        values, n_partitions = data
        max_partitions = self._params.aggregate_params.max_partitions_contributed
        prob_keep_partition = min(1, max_partitions /
                                  n_partitions) if n_partitions > 0 else 0
        partition_sum = np.sum(values)
        per_partition_contribution = np.clip(
            partition_sum, self._params.aggregate_params.min_sum_per_partition,
            self._params.aggregate_params.max_sum_per_partition)
        per_partition_error_min = 0
        per_partition_error_max = 0
        per_partition_error = partition_sum - per_partition_contribution
        if per_partition_error > 0:
            per_partition_error_max = per_partition_error
        elif per_partition_error < 0:
            per_partition_error_min = per_partition_error
        expected_cross_partition_error = -per_partition_contribution * (
            1 - prob_keep_partition)
        var_cross_partition_error = per_partition_contribution**2 * prob_keep_partition * (
            1 - prob_keep_partition)

        return UtilityAnalysisSumAccumulator(partition_sum,
                                             per_partition_error_min,
                                             per_partition_error_max,
                                             expected_cross_partition_error,
                                             var_cross_partition_error)

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
            per_partition_error_min=acc.per_partition_error_min,
            per_partition_error_max=acc.per_partition_error_max,
            expected_cross_partition_error=acc.expected_cross_partition_error,
            std_cross_partition_error=np.sqrt(acc.var_cross_partition_error),
            std_noise=std_noise,
            noise_kind=self._params.aggregate_params.noise_kind)

    def metrics_names(self) -> List[str]:
        return [
            'sum', 'per_partition_error_min', 'per_partition_error_max',
            'expected_cross_partition_error', 'std_cross_partition_error',
            'std_noise', 'noise_kind'
        ]

    def explain_computation(self):
        pass


@dataclass
class PrivacyIdCountUtilityAnalysisMetrics:
    """Stores metrics for the privacy ID count utility analysis.

    Args:
        privacy_id_count: actual count of privacy id in a partition.
        expected_cross_partition_error: the estimated amount of error across partitions.
        std_cross_partition_error: the standard deviation of the contribution bounding error.
        std_noise: the noise standard deviation for DP count.
        noise_kind: the type of noise used.
    """
    privacy_id_count: int
    expected_cross_partition_error: float
    std_cross_partition_error: float
    std_noise: float
    noise_kind: pipeline_dp.NoiseKind


@dataclass
class UtilityAnalysisPrivacyIdCountAccumulator:
    privacy_id_count: int
    expected_cross_partition_error: float
    var_cross_partition_error: float

    def __add__(self, other):
        return UtilityAnalysisPrivacyIdCountAccumulator(
            self.privacy_id_count + other.privacy_id_count,
            self.expected_cross_partition_error +
            other.expected_cross_partition_error,
            self.var_cross_partition_error + other.var_cross_partition_error)


class UtilityAnalysisPrivacyIdCountCombiner(pipeline_dp.Combiner):
    """A combiner for utility analysis privacy ID counts."""
    AccumulatorType = UtilityAnalysisPrivacyIdCountAccumulator

    def __init__(self, params: pipeline_dp.combiners.CombinerParams):
        self._params = params

    def create_accumulator(self, data: Tuple[Sized, int]) -> AccumulatorType:
        """Creates an accumulator for data.

        Args:
            data is a Tuple containing; 1) a list of the user's contributions for a single partition, and 2) the total
            number of partitions a user contributed to.

        Returns:
            An accumulator with the privacy ID count of partitions and the contribution error.
        """
        if not data:
            return UtilityAnalysisPrivacyIdCountAccumulator(0, 0, 0)
        values, n_partitions = data
        privacy_id_count = 1 if values else 0
        max_partitions = self._params.aggregate_params.max_partitions_contributed
        prob_keep_partition = min(1, max_partitions /
                                  n_partitions) if n_partitions > 0 else 0
        expected_cross_partition_error = -(1 - prob_keep_partition)
        var_cross_partition_error = prob_keep_partition * (1 -
                                                           prob_keep_partition)

        return UtilityAnalysisPrivacyIdCountAccumulator(
            privacy_id_count, expected_cross_partition_error,
            var_cross_partition_error)

    def merge_accumulators(self, acc1: AccumulatorType, acc2: AccumulatorType):
        """Merges two accumulators together additively."""
        return acc1 + acc2

    def compute_metrics(
            self, acc: AccumulatorType) -> PrivacyIdCountUtilityAnalysisMetrics:
        """Computes metrics based on the accumulator properties."""
        std_noise = dp_computations.compute_dp_count_noise_std(
            self._params.scalar_noise_params)
        return PrivacyIdCountUtilityAnalysisMetrics(
            privacy_id_count=acc.privacy_id_count,
            expected_cross_partition_error=acc.expected_cross_partition_error,
            std_cross_partition_error=np.sqrt(acc.var_cross_partition_error),
            std_noise=std_noise,
            noise_kind=self._params.aggregate_params.noise_kind)

    def metrics_names(self) -> List[str]:
        return [
            'privacy_id_count', 'expected_cross_partition_error',
            'std_cross_partition_error', 'std_noise', 'noise_kind'
        ]

    def explain_computation(self):
        pass


@dataclass
class AggregateErrorMetrics:
    """Stores aggregate metrics for utility analysis."""

    abs_error_expected_avg: float = None
    abs_error_variance_avg: float = None
    abs_error_99pct_avg: float = None
    rel_error_expected_avg: float = None
    rel_error_variance_avg: float = None
    rel_error_99pct_avg: float = None


@dataclass
class AggregateErrorMetricsAccumulator:
    num_partitions: int

    abs_error_expected_sum: float = None
    abs_error_variance_sum: float = None
    abs_error_99pct_sum: float = None
    rel_error_expected_sum: float = None
    rel_error_variance_sum: float = None
    rel_error_99pct_sum: float = None


class CountAggregateErrorMetricsCombiner(pipeline_dp.Combiner):
    """A combiner for aggregating errors across partitions for Count"""
    AccumulatorType = AggregateErrorMetricsAccumulator

    def __init__(self, params: pipeline_dp.combiners.CombinerParams):
        self._params = params

    def create_accumulator(
            self, metrics: CountUtilityAnalysisMetrics) -> AccumulatorType:
        """Create an accumulator for metrics."""
        # Absolute error metrics
        abs_error_expected = metrics.per_partition_error + metrics.expected_cross_partition_error
        abs_error_variance = metrics.std_cross_partition_error**2 + metrics.std_noise**2
        # TODO: Implement Laplace Noise
        assert metrics.noise_kind == pipeline_dp.NoiseKind.GAUSSIAN, "Laplace noise for utility analysis not implemented yet"
        loc_cpe_ne = metrics.expected_cross_partition_error
        std_cpe_ne = math.sqrt(metrics.std_cross_partition_error**2 +
                               metrics.std_noise**2)
        abs_error_99pct = scipy.stats.norm.ppf(
            q=0.01, loc=loc_cpe_ne,
            scale=std_cpe_ne) + metrics.per_partition_error

        # Relative error metrics
        if metrics.count == 0:  # For empty public partitions, to avoid division by 0
            rel_error_expected = float('inf')
            rel_error_variance = float('inf')
            rel_error_99pct = float('inf')
        else:
            rel_error_expected = metrics.abs_error_expected / metrics.count
            rel_error_variance = metrics.abs_error_variance / (metrics.count**2)
            rel_error_99pct = metrics.abs_error_99pct / metrics.count

        return AggregateErrorMetricsAccumulator(
            num_partitions=1,
            abs_error_expected_sum=abs_error_expected,
            abs_error_variance_sum=abs_error_variance,
            abs_error_99pct_sum=abs_error_99pct,
            rel_error_expected_sum=rel_error_expected,
            rel_error_variance_sum=rel_error_variance,
            rel_error_99pct_sum=rel_error_99pct,
        )

    def merge_accumulators(self, acc1: AccumulatorType, acc2: AccumulatorType):
        """Merge two accumulators together additively."""
        return AggregateErrorMetricsAccumulator(
            num_partitions=acc1.num_partitions + acc2.num_partitions,
            abs_error_expected_sum=acc1.abs_error_expected_sum +
            acc2.abs_error_expected_sum,
            abs_error_variance_sum=acc1.abs_error_variance_sum +
            acc2.abs_error_variance_sum,
            abs_error_99pct_sum=acc1.abs_error_99pct_sum +
            acc2.abs_error_99pct_sum,
            rel_error_expected_sum=acc1.rel_error_expected_sum +
            acc2.rel_error_expected_sum,
            rel_error_variance_sum=acc1.rel_error_variance_sum +
            acc2.rel_error_variance_sum,
            rel_error_99pct_sum=acc1.rel_error_99pct_sum +
            acc2.rel_error_99pct_sum)

    def compute_metrics(self, acc: AccumulatorType) -> AggregateErrorMetrics:
        """Compute metrics based on the accumulator properties.
        Args:
            acc: the accumulator to compute from.
        Returns:
            An AggregateErrorMetrics object with computed metrics.
        """
        return AggregateErrorMetrics(
            abs_error_expected_avg=acc.abs_error_expected_sum /
            acc.num_partitions,
            abs_error_variance_avg=acc.abs_error_expected_sum /
            acc.num_partitions,
            abs_error_99pct_avg=acc.abs_error_99pct_sum / acc.num_partitions,
            rel_error_expected_avg=acc.rel_error_expected_sum /
            acc.num_partitions,
            rel_error_variance_avg=acc.rel_error_variance_sum /
            acc.num_partitions,
            rel_error_99pct_avg=acc.rel_error_99pct_sum / acc.num_partitions)

    def metrics_names(self) -> List[str]:
        return [
            'abs_error_expected_avg', 'abs_error_variance_avg',
            'abs_error_99pct_avg', 'rel_error_expected_avg',
            'rel_error_variance_avg', 'rel_error_99pct_avg'
        ]

    def explain_computation(self):
        pass


class PrivatePartitionSelectionAggregateErrorMetricsCombiner(
        pipeline_dp.Combiner):
    """A combiner for aggregating errors across partitions for private partition selection"""
    # TODO: Implement
    AccumulatorType = AggregateErrorMetricsAccumulator

    def __init__(self, params: pipeline_dp.combiners.CombinerParams):
        self._params = params

    def create_accumulator(self, probability_to_keep: float) -> AccumulatorType:
        """Create an accumulator for metrics."""
        return AggregateErrorMetricsAccumulator(num_partitions=1,)

    def merge_accumulators(self, acc1: AccumulatorType, acc2: AccumulatorType):
        """Merge two accumulators together additively."""
        return AggregateErrorMetricsAccumulator(
            num_partitions=acc1.num_partitions + acc2.num_partitions)

    def compute_metrics(self, acc: AccumulatorType) -> AggregateErrorMetrics:
        """Compute metrics based on the accumulator properties.
        Args:
            acc: the accumulator to compute from.
        Returns:
            An AggregateErrorMetrics object with computed metrics.
        """
        return AggregateErrorMetrics()

    def metrics_names(self) -> List[str]:
        return []

    def explain_computation(self):
        pass
