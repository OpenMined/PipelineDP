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
from typing import List, Optional, Sequence, Sized, Tuple
import numpy as np
import math
import scipy

import pipeline_dp
from pipeline_dp import dp_computations
from pipeline_dp import combiners
from utility_analysis_new import metrics
from utility_analysis_new import poisson_binomial
from pipeline_dp import partition_selection

MAX_PROBABILITIES_IN_ACCUMULATOR = 100


class UtilityAnalysisCombiner(pipeline_dp.Combiner):

    @abc.abstractmethod
    def create_accumulator(self, data: Tuple[Sized, int]):
        """Creates an accumulator for data.

        Args:
            data is a Tuple containing:
              1) a list of the user's contributions for a single partition
              2) the total number of partitions a user contributed to.

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
        for i, prob in enumerate(pmf):
            probability += prob * ps_strategy.probability_of_keep(i)
        return probability

    def _compute_pmf(self) -> np.ndarray:
        """Computes the pmf of privacy id count in this partition after contribution bounding."""
        if self.probabilities:
            pmf = poisson_binomial.compute_pmf(self.probabilities)
        else:
            moments = self.moments
            std = math.sqrt(moments.variance)
            if std == 0:  # this is a constant random variable
                pmf = np.zeros(moments.count + 1)
                pmf[int(moments.expectation)] = 1
            else:
                skewness = moments.third_central_moment / std**3
                pmf = poisson_binomial.compute_pmf_approximation(
                    moments.expectation, std, skewness, moments.count)
        return pmf


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
            self, data: Tuple[Sized, int]) -> PartitionSelectionAccumulator:
        values, n_partitions = data
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

    def create_accumulator(self, data: Tuple[Sized, int]) -> AccumulatorType:
        """Creates an accumulator for data."""
        if not data:
            return (0, 0, 0, 0)
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
            std_cross_partition_error=np.sqrt(var_cross_partition_error),
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

    def create_accumulator(self, data: Tuple[Sequence, int]) -> AccumulatorType:
        if not data or not data[0]:
            return (0, 0, 0, 0, 0)
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
            std_cross_partition_error=np.sqrt(var_cross_partition_error),
            std_noise=std_noise,
            noise_kind=self._params.aggregate_params.noise_kind)

    def metrics_names(self) -> List[str]:
        return [
            'sum', 'per_partition_error_min', 'per_partition_error_max',
            'expected_cross_partition_error', 'std_cross_partition_error',
            'std_noise', 'noise_kind'
        ]


class PrivacyIdCountCombiner(UtilityAnalysisCombiner):
    """A combiner for utility analysis privacy ID counts."""
    # (privacy_id_count, expected_cross_partition_error,
    # var_cross_partition_error)
    AccumulatorType = Tuple[int, float, float]

    def __init__(self, params: pipeline_dp.combiners.CombinerParams):
        self._params = params

    def create_accumulator(self, data: Tuple[Sized, int]) -> AccumulatorType:
        if not data:
            return (0, 0, 0)
        values, n_partitions = data
        privacy_id_count = 1 if values else 0
        max_partitions = self._params.aggregate_params.max_partitions_contributed
        prob_keep_partition = min(1, max_partitions /
                                  n_partitions) if n_partitions > 0 else 0
        expected_cross_partition_error = -(1 - prob_keep_partition)
        var_cross_partition_error = prob_keep_partition * (1 -
                                                           prob_keep_partition)

        return (privacy_id_count, expected_cross_partition_error,
                var_cross_partition_error)

    def compute_metrics(self,
                        acc: AccumulatorType) -> metrics.PrivacyIdCountMetrics:
        """Computes metrics based on the accumulator properties."""
        privacy_id_count, expected_cross_partition_error, var_cross_partition_error = acc
        std_noise = dp_computations.compute_dp_count_noise_std(
            self._params.scalar_noise_params)
        return metrics.PrivacyIdCountMetrics(
            privacy_id_count=privacy_id_count,
            expected_cross_partition_error=expected_cross_partition_error,
            std_cross_partition_error=np.sqrt(var_cross_partition_error),
            std_noise=std_noise,
            noise_kind=self._params.aggregate_params.noise_kind)

    def metrics_names(self) -> List[str]:
        return [
            'privacy_id_count', 'expected_cross_partition_error',
            'std_cross_partition_error', 'std_noise', 'noise_kind'
        ]


@dataclass
class AggregateErrorMetricsAccumulator:
    """ Accumulator for AggregateErrorMetrics.

    All fields in this dataclass are sums across partitions"""
    kept_partitions_expected: float

    abs_error_expected: float
    abs_error_variance: float
    abs_error_quantiles: List[float]
    rel_error_expected: float
    rel_error_variance: float
    rel_error_quantiles: List[float]

    def __add__(self, other):
        return AggregateErrorMetricsAccumulator(
            kept_partitions_expected=self.kept_partitions_expected +
            other.kept_partitions_expected,
            abs_error_expected=self.abs_error_expected +
            other.abs_error_expected,
            abs_error_variance=self.abs_error_variance +
            other.abs_error_variance,
            abs_error_quantiles=[
                s1 + s2 for (s1, s2) in zip(self.abs_error_quantiles,
                                            other.abs_error_quantiles)
            ],
            rel_error_expected=self.rel_error_expected +
            other.rel_error_expected,
            rel_error_variance=self.rel_error_variance +
            other.rel_error_variance,
            rel_error_quantiles=[
                s1 + s2 for (s1, s2) in zip(self.rel_error_quantiles,
                                            other.rel_error_quantiles)
            ])


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

    def __init__(self, params: pipeline_dp.combiners.CombinerParams,
                 error_quantiles: List[float]):
        self._params = params
        self._error_quantiles = error_quantiles

    def create_accumulator(self,
                           metrics: metrics.CountMetrics,
                           probability_to_keep: float = 1) -> AccumulatorType:
        """Creates an accumulator for metrics."""
        # Absolute error metrics
        abs_error_expected = probability_to_keep * (
            metrics.per_partition_error +
            metrics.expected_cross_partition_error)
        abs_error_variance = probability_to_keep * (
            metrics.std_cross_partition_error**2 + metrics.std_noise**2)
        # TODO: Implement Laplace Noise
        assert metrics.noise_kind == pipeline_dp.NoiseKind.GAUSSIAN, "Laplace noise for utility analysis not implemented yet"
        loc_cpe_ne = metrics.expected_cross_partition_error
        std_cpe_ne = math.sqrt(metrics.std_cross_partition_error**2 +
                               metrics.std_noise**2)
        abs_error_quantiles = []
        for quantile in self._error_quantiles:
            error_at_quantile = probability_to_keep * (scipy.stats.norm.ppf(
                q=1 - quantile, loc=loc_cpe_ne,
                scale=std_cpe_ne) + metrics.per_partition_error)
            abs_error_quantiles.append(error_at_quantile)

        # Relative error metrics
        if metrics.count == 0:  # For empty public partitions, to avoid division by 0
            rel_error_expected = 0
            rel_error_variance = 0
            rel_error_quantiles = [0] * len(self._error_quantiles)
        else:
            rel_error_expected = abs_error_expected / metrics.count
            rel_error_variance = abs_error_variance / (metrics.count**2)
            rel_error_quantiles = [
                error / metrics.count for error in abs_error_quantiles
            ]

        return AggregateErrorMetricsAccumulator(
            kept_partitions_expected=probability_to_keep,
            abs_error_expected=abs_error_expected,
            abs_error_variance=abs_error_variance,
            abs_error_quantiles=abs_error_quantiles,
            rel_error_expected=rel_error_expected,
            rel_error_variance=rel_error_variance,
            rel_error_quantiles=rel_error_quantiles,
        )

    def merge_accumulators(self, acc1: AccumulatorType, acc2: AccumulatorType):
        """Merges two accumulators together additively."""
        return acc1 + acc2

    def compute_metrics(self,
                        acc: AccumulatorType) -> metrics.AggregateErrorMetrics:
        """Computes metrics based on the accumulator properties."""
        return metrics.AggregateErrorMetrics(
            abs_error_expected=acc.abs_error_expected /
            acc.kept_partitions_expected,
            abs_error_variance=acc.abs_error_variance /
            acc.kept_partitions_expected,
            abs_error_quantiles=[
                sum / acc.kept_partitions_expected
                for sum in acc.abs_error_quantiles
            ],
            rel_error_expected=acc.rel_error_expected /
            acc.kept_partitions_expected,
            rel_error_variance=acc.rel_error_variance /
            acc.kept_partitions_expected,
            rel_error_quantiles=[
                sum / acc.kept_partitions_expected
                for sum in acc.rel_error_quantiles
            ])

    def metrics_names(self) -> List[str]:
        return [
            'abs_error_expected', 'abs_error_variance', 'abs_error_quantiles',
            'rel_error_expected', 'rel_error_variance', 'rel_error_quantiles'
        ]

    def explain_computation(self):
        pass


class PrivatePartitionSelectionAggregateErrorMetricsCombiner(
        pipeline_dp.Combiner):
    """A combiner for aggregating errors across partitions for private partition selection"""
    AccumulatorType = PartitionSelectionAccumulator

    def __init__(self, params: pipeline_dp.combiners.CombinerParams,
                 error_quantiles: List[float]):
        self._params = params
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
