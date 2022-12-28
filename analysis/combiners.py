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

import abc
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple
import numpy as np
import math
import scipy

import pipeline_dp
from pipeline_dp import dp_computations
from pipeline_dp import combiners
from analysis import metrics
from analysis import poisson_binomial
from analysis import probability_computations
from pipeline_dp import partition_selection

MAX_PROBABILITIES_IN_ACCUMULATOR = 100

# It corresponds to the aggregating per (privacy_id, partition_key).
# (count, sum, num_partition_privacy_id_contributes).
PreaggregatedData = Tuple[int, float, int]


class UtilityAnalysisCombiner(pipeline_dp.Combiner):

    @abc.abstractmethod
    def create_accumulator(self, data: Tuple[int, float, int]):
        """Creates an accumulator for data.

        Args:
            data is a Tuple containing:
              1) the count of the user's contributions for a single partition
              2) the sum of the user's contributions for the same partition
              3) the total number of partitions a user contributed to.

            Only COUNT, PRIVACY_ID_COUNT, SUM metrics can be supported with the
            current format of data.

        Returns:
            A tuple which is an accumulator.
        """

    def merge_accumulators(self, acc1: Tuple, acc2: Tuple):
        """Merges two tuples together additively."""
        return tuple(a + b for a, b in zip(acc1, acc2))

    def explain_computation(self):
        """No-op."""


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
class PartitionSelectionCalculator:
    """Computes probability of keeping the partition.

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

    def compute_probability_to_keep(self,
                                    partition_selection_strategy: pipeline_dp.
                                    PartitionSelectionStrategy, eps: float,
                                    delta: float,
                                    max_partitions_contributed: int) -> float:
        """Computes the probability that this partition is kept.

        If self.probabilities is set, then the computed probability is exact,
        otherwise it is an approximation computed from self.moments.
        """
        pmf = self._compute_pmf()
        ps_strategy = partition_selection.create_partition_selection_strategy(
            partition_selection_strategy, eps, delta,
            max_partitions_contributed)
        probability = 0
        for i, prob in enumerate(pmf.probabilities, pmf.start):
            probability += prob * ps_strategy.probability_of_keep(i)
        return probability

    def _compute_pmf(self) -> poisson_binomial.PMF:
        """Computes the pmf of privacy id count in this partition after contribution bounding."""
        if self.probabilities:
            return poisson_binomial.compute_pmf(self.probabilities)

        moments = self.moments
        std = math.sqrt(moments.variance)
        skewness = 0 if std == 0 else moments.third_central_moment / std**3
        return poisson_binomial.compute_pmf_approximation(
            moments.expectation, std, skewness, moments.count)


# PartitionSelectionAccumulator = (probabilities, moments). These two elements
# exclusive:
# If len(probabilities) <= MAX_PROBABILITIES_IN_ACCUMULATOR then 'probabilities'
# are used otherwise 'moments'. For more details see docstring to
# PartitionSelectionCalculator.
PartitionSelectionAccumulator = Tuple[Optional[Tuple[float]],
                                      Optional[SumOfRandomVariablesMoments]]


def _merge_partition_selection_accumulators(
        acc1: PartitionSelectionAccumulator,
        acc2: PartitionSelectionAccumulator) -> PartitionSelectionAccumulator:
    probs1, moments1 = acc1
    probs2, moments2 = acc2
    if probs1 and probs2 and len(probs1) + len(
            probs2) <= MAX_PROBABILITIES_IN_ACCUMULATOR:
        return (probs1 + probs2, None)

    if moments1 is None:
        moments1 = _probabilities_to_moments(probs1)
    if moments2 is None:
        moments2 = _probabilities_to_moments(probs2)

    return (None, moments1 + moments2)


class PartitionSelectionCombiner(UtilityAnalysisCombiner):
    """A combiner for utility analysis counts."""

    def __init__(self, params: pipeline_dp.combiners.CombinerParams):
        self._params = params

    def create_accumulator(
            self, data: Tuple[int, float,
                              int]) -> PartitionSelectionAccumulator:
        count, sum_, n_partitions = data
        max_partitions = self._params.aggregate_params.max_partitions_contributed
        prob_keep_contribution = min(1, max_partitions /
                                     n_partitions) if n_partitions > 0 else 0

        return ((prob_keep_contribution,), None)

    def merge_accumulators(
            self, acc1: PartitionSelectionAccumulator,
            acc2: PartitionSelectionAccumulator
    ) -> PartitionSelectionAccumulator:
        return _merge_partition_selection_accumulators(acc1, acc2)

    def compute_metrics(self, acc: PartitionSelectionAccumulator) -> float:
        """Computes the probability that the partition kept."""
        probs, moments = acc
        params = self._params
        calculator = PartitionSelectionCalculator(probs, moments)
        return calculator.compute_probability_to_keep(
            params.aggregate_params.partition_selection_strategy, params.eps,
            params.delta, params.aggregate_params.max_partitions_contributed)

    def metrics_names(self) -> List[str]:
        return ['probability_to_keep']


class CountCombiner(UtilityAnalysisCombiner):
    """A combiner for utility analysis counts."""
    # (count, per_partition_error, expected_cross_partition_error,
    # var_cross_partition_error)
    AccumulatorType = Tuple[int, int, float, float]

    def __init__(self, params: pipeline_dp.combiners.CombinerParams):
        self._params = params

    @property
    def _is_public_partitions(self):
        return self._partition_selection_budget is None

    def create_accumulator(self, data: PreaggregatedData) -> AccumulatorType:
        """Creates an accumulator for data."""
        count, sum_, n_partitions = data
        max_per_partition = self._params.aggregate_params.max_contributions_per_partition
        max_partitions = self._params.aggregate_params.max_partitions_contributed
        l0_prob_keep_contribution = min(1, max_partitions /
                                        n_partitions) if n_partitions > 0 else 0
        per_partition_contribution = min(max_per_partition, count)
        per_partition_error = per_partition_contribution - count
        expected_cross_partition_error = -per_partition_contribution * (
            1 - l0_prob_keep_contribution)
        var_cross_partition_error = per_partition_contribution**2 * l0_prob_keep_contribution * (
            1 - l0_prob_keep_contribution)

        return (count, per_partition_error, expected_cross_partition_error,
                var_cross_partition_error)

    def compute_metrics(self, acc: AccumulatorType) -> metrics.CountMetrics:
        count, per_partition_error, expected_cross_partition_error, var_cross_partition_error = acc
        std_noise = dp_computations.compute_dp_count_noise_std(
            self._params.scalar_noise_params)
        return metrics.CountMetrics(
            count=count,
            per_partition_error=per_partition_error,
            expected_cross_partition_error=expected_cross_partition_error,
            std_cross_partition_error=math.sqrt(var_cross_partition_error),
            std_noise=std_noise,
            noise_kind=self._params.aggregate_params.noise_kind)

    def metrics_names(self) -> List[str]:
        return [
            'count', 'per_partition_error', 'expected_cross_partition_error',
            'std_cross_partition_error', 'std_noise', 'noise_kind'
        ]


class SumCombiner(UtilityAnalysisCombiner):
    """A combiner for utility analysis sums."""
    # (partition_sum, per_partition_error_min, per_partition_error_max,
    # expected_cross_partition_error, var_cross_partition_error)
    AccumulatorType = Tuple[float, float, float, float, float]

    def __init__(self, params: pipeline_dp.combiners.CombinerParams):
        self._params = params

    def create_accumulator(self, data: PreaggregatedData) -> AccumulatorType:
        count, partition_sum, n_partitions = data
        min_bound = self._params.aggregate_params.min_sum_per_partition
        max_bound = self._params.aggregate_params.max_sum_per_partition
        max_partitions = self._params.aggregate_params.max_partitions_contributed
        l0_prob_keep_contribution = min(1, max_partitions /
                                        n_partitions) if n_partitions > 0 else 0
        per_partition_contribution = np.clip(partition_sum, min_bound,
                                             max_bound).item()
        per_partition_error_min = 0
        per_partition_error_max = 0
        per_partition_error = per_partition_contribution - partition_sum
        if partition_sum < min_bound:
            per_partition_error_min = per_partition_error
        elif partition_sum > max_bound:
            per_partition_error_max = per_partition_error
        expected_cross_partition_error = -per_partition_contribution * (
            1 - l0_prob_keep_contribution)
        var_cross_partition_error = per_partition_contribution**2 * l0_prob_keep_contribution * (
            1 - l0_prob_keep_contribution)

        return (partition_sum, per_partition_error_min, per_partition_error_max,
                expected_cross_partition_error, var_cross_partition_error)

    def compute_metrics(self, acc: AccumulatorType) -> metrics.SumMetrics:
        """Computes metrics based on the accumulator properties."""
        partition_sum, per_partition_error_min, per_partition_error_max, expected_cross_partition_error, var_cross_partition_error = acc
        std_noise = dp_computations.compute_dp_count_noise_std(
            self._params.scalar_noise_params)
        return metrics.SumMetrics(
            sum=partition_sum,
            per_partition_error_min=per_partition_error_min,
            per_partition_error_max=per_partition_error_max,
            expected_cross_partition_error=expected_cross_partition_error,
            std_cross_partition_error=math.sqrt(var_cross_partition_error),
            std_noise=std_noise,
            noise_kind=self._params.aggregate_params.noise_kind)

    def metrics_names(self) -> List[str]:
        return [
            'sum', 'per_partition_error_min', 'per_partition_error_max',
            'expected_cross_partition_error', 'std_cross_partition_error',
            'std_noise', 'noise_kind'
        ]


class PrivacyIdCountCombiner(CountCombiner):
    """A combiner for utility analysis privacy ID counts."""
    # (count, per_partition_error, expected_cross_partition_error,
    # var_cross_partition_error)
    AccumulatorType = Tuple[int, int, float, float]

    def create_accumulator(self, data: PreaggregatedData) -> AccumulatorType:
        count, _sum, n_partitions = data
        count = 1 if count > 0 else 0
        data = count, _sum, n_partitions
        return super().create_accumulator(data)


class CompoundCombiner(pipeline_dp.combiners.CompoundCombiner):
    """Compound combiner for Utility analysis per partition metrics."""

    # For improving memory usage the compound accumulator has 2 modes:
    # 1. Sparse mode (for small datasets): which contains information about each
    # privacy id's aggregated contributions per partition.
    # 2. Dense mode (for large datasets): which contains underlying accumulators
    # from internal combiners.
    # Since the utility analysis can be run for many configurations, there can
    # be hundreds of the internal combiners, as a result the compound
    # accumulator can contain hundreds accumulators. Converting each privacy id
    # contribution to such accumulators leads to memory usage blow-up. That is
    # why sparse mode introduced - until the number of privacy id contributions
    # is small, they are saved instead of creating accumulators.
    SparseAccumulatorType = List[PreaggregatedData]
    DenseAccumulatorType = List[Any]
    AccumulatorType = Tuple[Optional[SparseAccumulatorType],
                            Optional[DenseAccumulatorType]]

    def create_accumulator(self, data: PreaggregatedData) -> AccumulatorType:
        if not data:
            # This is an empty partition, what is None assigned to. It was added
            # because of public partitions.
            return ([(0, 0, 0)], None)
        return ([data], None)

    def _to_dense(self,
                  sparse_acc: SparseAccumulatorType) -> DenseAccumulatorType:
        result = None
        for count_sum_n_partitions in sparse_acc:
            compound_acc = (1, [
                combiner.create_accumulator(count_sum_n_partitions)
                for combiner in self._combiners
            ])
            if result is None:
                result = compound_acc
            else:
                result = super().merge_accumulators(result, compound_acc)
        return result

    def merge_accumulators(self, acc1: AccumulatorType, acc2: AccumulatorType):
        sparse1, dense1 = acc1
        sparse2, dense2 = acc2
        if sparse1 and sparse2:
            sparse1.extend(sparse2)
            # Computes heuristically that the sparse representation is less
            # than dense. For this assume that 1 accumulator is on average
            # has a size of aggregated contributions from 2 privacy ids.
            is_sparse_less_dense = len(sparse1) <= 2 * len(self._combiners)
            if is_sparse_less_dense:
                return (sparse1, None)
            # Dense is smaller, convert to dense.
            return (None, self._to_dense(sparse1))
        dense1 = self._to_dense(sparse1) if sparse1 else dense1
        dense2 = self._to_dense(sparse2) if sparse2 else dense2
        merged_dense = super().merge_accumulators(dense1, dense2)
        return (None, merged_dense)

    def compute_metrics(self, acc: AccumulatorType):
        sparse, dense = acc
        if sparse:
            dense = self._to_dense(sparse)
        return super().compute_metrics(dense)


@dataclass
class AggregateErrorMetricsAccumulator:
    """ Accumulator for AggregateErrorMetrics.

    All fields in this dataclass are sums across partitions, except for
    noise_variance."""
    num_partitions: int
    kept_partitions_expected: float
    total_aggregate: float  # sum, count, privacy_id_count across partitions

    data_dropped_l0: float
    data_dropped_linf: float
    data_dropped_partition_selection: float

    abs_error_l0_expected: float
    abs_error_linf_expected: float
    abs_error_l0_variance: float
    abs_error_variance: float
    abs_error_quantiles: List[float]
    rel_error_l0_expected: float
    rel_error_linf_expected: float
    rel_error_l0_variance: float
    rel_error_variance: float
    rel_error_quantiles: List[float]

    abs_error_expected_w_dropped_partitions: float
    rel_error_expected_w_dropped_partitions: float

    noise_std: float

    def __add__(self, other):
        assert self.noise_std == other.noise_std, "Two AggregateErrorMetricsAccumulators have to have the same noise_std to be mergeable"
        return AggregateErrorMetricsAccumulator(
            num_partitions=self.num_partitions + other.num_partitions,
            kept_partitions_expected=self.kept_partitions_expected +
            other.kept_partitions_expected,
            total_aggregate=self.total_aggregate + other.total_aggregate,
            data_dropped_l0=self.data_dropped_l0 + other.data_dropped_l0,
            data_dropped_linf=self.data_dropped_linf + other.data_dropped_linf,
            data_dropped_partition_selection=self.
            data_dropped_partition_selection +
            other.data_dropped_partition_selection,
            abs_error_l0_expected=self.abs_error_l0_expected +
            other.abs_error_l0_expected,
            abs_error_linf_expected=self.abs_error_linf_expected +
            other.abs_error_linf_expected,
            abs_error_l0_variance=self.abs_error_l0_variance +
            other.abs_error_l0_variance,
            abs_error_variance=self.abs_error_variance +
            other.abs_error_variance,
            abs_error_quantiles=[
                s1 + s2 for (s1, s2) in zip(self.abs_error_quantiles,
                                            other.abs_error_quantiles)
            ],
            rel_error_l0_expected=self.rel_error_l0_expected +
            other.rel_error_l0_expected,
            rel_error_linf_expected=self.rel_error_linf_expected +
            other.rel_error_linf_expected,
            rel_error_l0_variance=self.rel_error_l0_variance +
            other.rel_error_l0_variance,
            rel_error_variance=self.rel_error_variance +
            other.rel_error_variance,
            rel_error_quantiles=[
                s1 + s2 for (s1, s2) in zip(self.rel_error_quantiles,
                                            other.rel_error_quantiles)
            ],
            abs_error_expected_w_dropped_partitions=self.
            abs_error_expected_w_dropped_partitions +
            other.abs_error_expected_w_dropped_partitions,
            rel_error_expected_w_dropped_partitions=self.
            rel_error_expected_w_dropped_partitions +
            other.rel_error_expected_w_dropped_partitions,
            noise_std=self.noise_std)


class AggregateErrorMetricsCompoundCombiner(combiners.CompoundCombiner):
    """A compound combiner for aggregating error metrics across partitions"""
    AccumulatorType = Tuple[int, Tuple]

    def create_accumulator(self, values) -> AccumulatorType:
        probability_to_keep = 1
        if isinstance(values[0], float):
            probability_to_keep = values[0]
        accumulators = []
        for combiner, metrics in zip(self._combiners, values):
            if isinstance(
                    combiner,
                    PrivatePartitionSelectionAggregateErrorMetricsCombiner):
                accumulators.append(combiner.create_accumulator(metrics))
            else:
                accumulators.append(
                    combiner.create_accumulator(metrics, probability_to_keep))
        return 1, tuple(accumulators)


class CountAggregateErrorMetricsCombiner(pipeline_dp.Combiner):
    """A combiner for aggregating errors across partitions for Count"""
    AccumulatorType = AggregateErrorMetricsAccumulator

    def __init__(self, metric_type: metrics.AggregateMetricType,
                 error_quantiles: List[float]):
        self._metric_type = metric_type
        # The contribution bounding error is negative, so quantiles <0.5 for the
        # error distribution (which is the sum of the noise and the contribution
        # bounding error) should be used to come up with the worst error
        # quantiles.
        self._error_quantiles = [(1 - q) for q in error_quantiles]

    def create_accumulator(self,
                           metrics: metrics.CountMetrics,
                           probability_to_keep: float = 1) -> AccumulatorType:
        """Creates an accumulator for metrics."""
        # Data drop ratio metrics
        total_aggregate = metrics.count
        data_dropped_l0 = -metrics.expected_cross_partition_error
        data_dropped_linf = -metrics.per_partition_error
        data_dropped_partition_selection = (1 - probability_to_keep) * (
            metrics.count + metrics.expected_cross_partition_error +
            metrics.per_partition_error)

        # Absolute error metrics
        abs_error_l0_expected = probability_to_keep * metrics.expected_cross_partition_error
        abs_error_linf_expected = probability_to_keep * metrics.per_partition_error
        abs_error_l0_variance = probability_to_keep * metrics.std_cross_partition_error**2
        abs_error_variance = probability_to_keep * (
            metrics.std_cross_partition_error**2 + metrics.std_noise**2)
        loc_cpe_ne = metrics.expected_cross_partition_error
        std_cpe_ne = math.sqrt(metrics.std_cross_partition_error**2 +
                               metrics.std_noise**2)
        abs_error_quantiles = []
        if metrics.noise_kind == pipeline_dp.NoiseKind.GAUSSIAN:
            error_distribution_quantiles = scipy.stats.norm.ppf(
                q=self._error_quantiles, loc=loc_cpe_ne, scale=std_cpe_ne)
        else:
            error_distribution_quantiles = probability_computations.compute_sum_laplace_gaussian_quantiles(
                laplace_b=metrics.std_noise / math.sqrt(2),
                gaussian_sigma=metrics.std_cross_partition_error,
                quantiles=self._error_quantiles,
                num_samples=10**3)
        for quantile in error_distribution_quantiles:
            error_at_quantile = probability_to_keep * (
                float(quantile) + metrics.per_partition_error)
            abs_error_quantiles.append(error_at_quantile)

        abs_error_expected_w_dropped_partitions = probability_to_keep * (
            metrics.expected_cross_partition_error + metrics.per_partition_error
        ) + (1 - probability_to_keep) * -metrics.count

        # Relative error metrics
        if metrics.count == 0:  # For empty public partitions, to avoid division by 0
            rel_error_l0_expected = 0
            rel_error_linf_expected = 0
            rel_error_l0_variance = 0
            rel_error_variance = 0
            rel_error_quantiles = [0] * len(self._error_quantiles)
            rel_error_expected_w_dropped_partitions = 0
        else:
            rel_error_l0_expected = abs_error_l0_expected / metrics.count
            rel_error_linf_expected = abs_error_linf_expected / metrics.count
            rel_error_l0_variance = abs_error_l0_variance / (metrics.count**2)
            rel_error_variance = abs_error_variance / (metrics.count**2)
            rel_error_quantiles = [
                error / metrics.count for error in abs_error_quantiles
            ]
            rel_error_expected_w_dropped_partitions = abs_error_expected_w_dropped_partitions / metrics.count

        # Noise metrics
        noise_variance = metrics.std_noise**2

        return AggregateErrorMetricsAccumulator(
            num_partitions=1,
            kept_partitions_expected=probability_to_keep,
            total_aggregate=total_aggregate,
            data_dropped_l0=data_dropped_l0,
            data_dropped_linf=data_dropped_linf,
            data_dropped_partition_selection=data_dropped_partition_selection,
            abs_error_l0_expected=abs_error_l0_expected,
            abs_error_linf_expected=abs_error_linf_expected,
            abs_error_l0_variance=abs_error_l0_variance,
            abs_error_variance=abs_error_variance,
            abs_error_quantiles=abs_error_quantiles,
            rel_error_l0_expected=rel_error_l0_expected,
            rel_error_linf_expected=rel_error_linf_expected,
            rel_error_l0_variance=rel_error_l0_variance,
            rel_error_variance=rel_error_variance,
            rel_error_quantiles=rel_error_quantiles,
            abs_error_expected_w_dropped_partitions=
            abs_error_expected_w_dropped_partitions,
            rel_error_expected_w_dropped_partitions=
            rel_error_expected_w_dropped_partitions,
            noise_std=noise_variance,
        )

    def merge_accumulators(self, acc1: AccumulatorType, acc2: AccumulatorType):
        """Merges two accumulators together additively."""
        return acc1 + acc2

    def compute_metrics(self,
                        acc: AccumulatorType) -> metrics.AggregateErrorMetrics:
        """Computes metrics based on the accumulator properties."""
        abs_error_l0_expected = acc.abs_error_l0_expected / acc.kept_partitions_expected
        abs_error_linf_expected = acc.abs_error_linf_expected / acc.kept_partitions_expected
        rel_error_l0_expected = acc.rel_error_l0_expected / acc.kept_partitions_expected
        rel_error_linf_expected = acc.rel_error_linf_expected / acc.kept_partitions_expected
        acc.total_aggregate = max(1.0, acc.total_aggregate)
        return metrics.AggregateErrorMetrics(
            metric_type=self._metric_type,
            ratio_data_dropped_l0=acc.data_dropped_l0 / acc.total_aggregate,
            ratio_data_dropped_linf=acc.data_dropped_linf / acc.total_aggregate,
            ratio_data_dropped_partition_selection=acc.
            data_dropped_partition_selection / acc.total_aggregate,
            abs_error_l0_expected=abs_error_l0_expected,
            abs_error_linf_expected=abs_error_linf_expected,
            abs_error_expected=abs_error_l0_expected + abs_error_linf_expected,
            abs_error_l0_variance=acc.abs_error_l0_variance /
            acc.kept_partitions_expected,
            abs_error_variance=acc.abs_error_variance /
            acc.kept_partitions_expected,
            abs_error_quantiles=[
                sum / acc.kept_partitions_expected
                for sum in acc.abs_error_quantiles
            ],
            rel_error_l0_expected=rel_error_l0_expected,
            rel_error_linf_expected=rel_error_linf_expected,
            rel_error_expected=rel_error_l0_expected + rel_error_linf_expected,
            rel_error_l0_variance=acc.rel_error_l0_variance /
            acc.kept_partitions_expected,
            rel_error_variance=acc.rel_error_variance /
            acc.kept_partitions_expected,
            rel_error_quantiles=[
                sum / acc.kept_partitions_expected
                for sum in acc.rel_error_quantiles
            ],
            abs_error_expected_w_dropped_partitions=acc.
            abs_error_expected_w_dropped_partitions / acc.num_partitions,
            rel_error_expected_w_dropped_partitions=acc.
            rel_error_expected_w_dropped_partitions / acc.num_partitions,
            noise_std=acc.noise_std)

    def metrics_names(self) -> List[str]:
        return [
            'metric_type', 'ratio_data_dropped_l0', 'ratio_data_dropped_linf',
            'ratio_data_dropped_partition_selection', 'abs_error_l0_expected',
            'abs_error_linf_expected', 'abs_error_expected',
            'abs_error_l0_variance', 'abs_error_variance',
            'abs_error_quantiles', 'rel_error_l0_expected',
            'rel_error_linf_expected', 'rel_error_expected',
            'rel_error_l0_variance', 'rel_error_variance',
            'rel_error_quantiles', 'noise_std'
        ]

    def explain_computation(self):
        pass


class SumAggregateErrorMetricsCombiner(pipeline_dp.Combiner):
    """A combiner for aggregating errors across partitions for Sum"""
    AccumulatorType = AggregateErrorMetricsAccumulator

    def __init__(self, error_quantiles: List[float]):
        # The contribution bounding error is negative, so quantiles <0.5 for the
        # error distribution (which is the sum of the noise and the contribution
        # bounding error) should be used to come up with the worst error
        # quantiles.
        self._error_quantiles = [(1 - q) for q in error_quantiles]

    def create_accumulator(self,
                           metrics: metrics.SumMetrics,
                           probability_to_keep: float = 1) -> AccumulatorType:
        """Creates an accumulator for metrics."""
        # Data drop ratio metrics
        total_aggregate = metrics.sum
        # TODO: Compute data_dropped metrics
        data_dropped_l0 = 0
        data_dropped_linf = 0
        data_dropped_partition_selection = 0

        # Absolute error metrics
        abs_error_l0_expected = probability_to_keep * metrics.expected_cross_partition_error
        abs_error_linf_expected = probability_to_keep * (
            metrics.per_partition_error_min + metrics.per_partition_error_max)
        abs_error_l0_variance = probability_to_keep * metrics.std_cross_partition_error**2
        abs_error_variance = probability_to_keep * (
            metrics.std_cross_partition_error**2 + metrics.std_noise**2)
        loc_cpe_ne = metrics.expected_cross_partition_error
        std_cpe_ne = math.sqrt(metrics.std_cross_partition_error**2 +
                               metrics.std_noise**2)
        abs_error_quantiles = []
        if metrics.noise_kind == pipeline_dp.NoiseKind.GAUSSIAN:
            error_distribution_quantiles = scipy.stats.norm.ppf(
                q=self._error_quantiles, loc=loc_cpe_ne, scale=std_cpe_ne)
        else:
            error_distribution_quantiles = probability_computations.compute_sum_laplace_gaussian_quantiles(
                laplace_b=metrics.std_noise / math.sqrt(2),
                gaussian_sigma=metrics.std_cross_partition_error,
                quantiles=self._error_quantiles,
                num_samples=10**3)
        for quantile in error_distribution_quantiles:
            error_at_quantile = probability_to_keep * (
                float(quantile) + metrics.per_partition_error_min +
                metrics.per_partition_error_max)
            abs_error_quantiles.append(error_at_quantile)

        abs_error_expected_w_dropped_partitions = probability_to_keep * (
            metrics.expected_cross_partition_error +
            metrics.per_partition_error_min + metrics.per_partition_error_max
        ) + (1 - probability_to_keep) * -metrics.sum

        # Relative error metrics
        if metrics.sum == 0:  # For empty public partitions & partitions with zero sum, to avoid division by 0
            rel_error_l0_expected = 0
            rel_error_linf_expected = 0
            rel_error_l0_variance = 0
            rel_error_variance = 0
            rel_error_quantiles = [0] * len(self._error_quantiles)
            rel_error_expected_w_dropped_partitions = 0
        else:
            # We take the absolute value of the denominator because it can be
            # negative.
            rel_error_l0_expected = abs_error_l0_expected / abs(metrics.sum)
            rel_error_linf_expected = abs_error_linf_expected / abs(metrics.sum)
            rel_error_l0_variance = abs_error_l0_variance / (metrics.sum**2)
            rel_error_variance = abs_error_variance / (metrics.sum**2)
            rel_error_quantiles = [
                error / abs(metrics.sum) for error in abs_error_quantiles
            ]
            rel_error_expected_w_dropped_partitions = abs_error_expected_w_dropped_partitions / abs(
                metrics.sum)

        # Noise metrics
        noise_variance = metrics.std_noise**2

        return AggregateErrorMetricsAccumulator(
            num_partitions=1,
            kept_partitions_expected=probability_to_keep,
            total_aggregate=total_aggregate,
            data_dropped_l0=data_dropped_l0,
            data_dropped_linf=data_dropped_linf,
            data_dropped_partition_selection=data_dropped_partition_selection,
            abs_error_l0_expected=abs_error_l0_expected,
            abs_error_linf_expected=abs_error_linf_expected,
            abs_error_l0_variance=abs_error_l0_variance,
            abs_error_variance=abs_error_variance,
            abs_error_quantiles=abs_error_quantiles,
            rel_error_l0_expected=rel_error_l0_expected,
            rel_error_linf_expected=rel_error_linf_expected,
            rel_error_l0_variance=rel_error_l0_variance,
            rel_error_variance=rel_error_variance,
            rel_error_quantiles=rel_error_quantiles,
            abs_error_expected_w_dropped_partitions=
            abs_error_expected_w_dropped_partitions,
            rel_error_expected_w_dropped_partitions=
            rel_error_expected_w_dropped_partitions,
            noise_std=noise_variance,
        )

    def merge_accumulators(self, acc1: AccumulatorType, acc2: AccumulatorType):
        """Merges two accumulators together additively."""
        return acc1 + acc2

    def compute_metrics(self,
                        acc: AccumulatorType) -> metrics.AggregateErrorMetrics:
        """Computes metrics based on the accumulator properties."""
        abs_error_l0_expected = acc.abs_error_l0_expected / acc.kept_partitions_expected
        abs_error_linf_expected = acc.abs_error_linf_expected / acc.kept_partitions_expected
        rel_error_l0_expected = acc.rel_error_l0_expected / acc.kept_partitions_expected
        rel_error_linf_expected = acc.rel_error_linf_expected / acc.kept_partitions_expected
        acc.total_aggregate = max(1.0, acc.total_aggregate)
        return metrics.AggregateErrorMetrics(
            metric_type=metrics.AggregateMetricType.SUM,
            ratio_data_dropped_l0=acc.data_dropped_l0 / acc.total_aggregate,
            ratio_data_dropped_linf=acc.data_dropped_linf / acc.total_aggregate,
            ratio_data_dropped_partition_selection=acc.
            data_dropped_partition_selection / acc.total_aggregate,
            abs_error_l0_expected=abs_error_l0_expected,
            abs_error_linf_expected=abs_error_linf_expected,
            abs_error_expected=abs_error_l0_expected + abs_error_linf_expected,
            abs_error_l0_variance=acc.abs_error_l0_variance /
            acc.kept_partitions_expected,
            abs_error_variance=acc.abs_error_variance /
            acc.kept_partitions_expected,
            abs_error_quantiles=[
                sum / acc.kept_partitions_expected
                for sum in acc.abs_error_quantiles
            ],
            rel_error_l0_expected=rel_error_l0_expected,
            rel_error_linf_expected=rel_error_linf_expected,
            rel_error_expected=rel_error_l0_expected + rel_error_linf_expected,
            rel_error_l0_variance=acc.rel_error_l0_variance /
            acc.kept_partitions_expected,
            rel_error_variance=acc.rel_error_variance /
            acc.kept_partitions_expected,
            rel_error_quantiles=[
                sum / acc.kept_partitions_expected
                for sum in acc.rel_error_quantiles
            ],
            abs_error_expected_w_dropped_partitions=acc.
            abs_error_expected_w_dropped_partitions / acc.num_partitions,
            rel_error_expected_w_dropped_partitions=acc.
            rel_error_expected_w_dropped_partitions / acc.num_partitions,
            noise_std=acc.noise_std)

    def metrics_names(self) -> List[str]:
        return [
            'metric_type', 'ratio_data_dropped_l0', 'ratio_data_dropped_linf',
            'ratio_data_dropped_partition_selection', 'abs_error_l0_expected',
            'abs_error_linf_expected', 'abs_error_expected',
            'abs_error_l0_variance', 'abs_error_variance',
            'abs_error_quantiles', 'rel_error_l0_expected',
            'rel_error_linf_expected', 'rel_error_expected',
            'rel_error_l0_variance', 'rel_error_variance',
            'rel_error_quantiles', 'noise_std'
        ]

    def explain_computation(self):
        pass


class PrivatePartitionSelectionAggregateErrorMetricsCombiner(
        pipeline_dp.Combiner):
    """A combiner for aggregating errors across partitions for private partition selection"""
    AccumulatorType = PartitionSelectionAccumulator

    def __init__(self, error_quantiles: List[float]):
        self._error_quantiles = error_quantiles

    def create_accumulator(
            self, prob_to_keep: float) -> PartitionSelectionAccumulator:
        """Creates an accumulator for metrics."""
        return ((prob_to_keep,), None)

    def merge_accumulators(
            self, acc1: PartitionSelectionAccumulator,
            acc2: PartitionSelectionAccumulator
    ) -> PartitionSelectionAccumulator:
        """Merges two accumulators together additively."""
        return _merge_partition_selection_accumulators(acc1, acc2)

    def compute_metrics(
        self, acc: PartitionSelectionAccumulator
    ) -> metrics.PartitionSelectionMetrics:
        """Computes metrics based on the accumulator properties."""
        probs, moments = acc
        if moments is None:
            moments = _probabilities_to_moments(probs)
        kept_partitions_expected = moments.expectation
        kept_partitions_variance = moments.variance
        num_partitions = moments.count
        return metrics.PartitionSelectionMetrics(
            num_partitions=num_partitions,
            dropped_partitions_expected=num_partitions -
            kept_partitions_expected,
            dropped_partitions_variance=kept_partitions_variance)

    def metrics_names(self) -> List[str]:
        return []

    def explain_computation(self):
        pass
